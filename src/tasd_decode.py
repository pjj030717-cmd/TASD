"""
TASD: Training-free Structure-Aware Speculative Decoding (with proper KV cache).

Core flow:
1. Pre-fill KV cache for prompt on both target and draft models
2. Draft model generates multi-block tokens incrementally (using its KV cache)
3. Target model verifies draft tokens in one batch forward (using its KV cache)
4. Prefix acceptance with window-based logic
5. Structural Guard checks for risks (token-level trim)
6. Accept safe prefix, TRIM KV caches to accepted length, continue

Key optimization:
- Verification forward already computes KV cache for ALL draft tokens
- We TRIM the cache to accepted length instead of re-forwarding
- This eliminates redundant forward passes

Missing items addressed:
1. draft_blocks / multi-block draft
2. Per-block early stop
3. Block-level accept/reject stats
4. max_new_tokens remaining budget constraint
5. EOS handling with eos_drafted/eos_accepted/stop_reason
6. Repair token recording
7. Prompt seed check
8. Reference not used in decoding (documented)
9. AR/TASD length alignment (handled by caller)
10. Warmup (handled by caller)
11. CUDA synchronize
12. Memory recording (handled by caller)
13. Failure/fallback mechanism
"""
import time
import re
import torch
from transformers import DynamicCache
from .structural_guard import StructuralGuard
from .guard_v2 import GuardV2  # noqa: F401 - optional, used via parameter


def _forward_with_cache(model, input_ids, past_key_values):
    """Run model forward pass, return logits and updated past_key_values as DynamicCache."""
    if past_key_values is None:
        past_key_values = DynamicCache()

    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.logits, outputs.past_key_values


def _greedy_sample(logits):
    """Greedy sampling from logits. Handles both 1D and 2D tensors."""
    if logits.dim() == 1:
        return logits.argmax().item()
    return logits[0, -1].argmax().item()


def _trim_past_key_values(past_key_values, keep_len):
    """
    Trim past_key_values to keep only the first keep_len tokens.
    Works with both DynamicCache and tuple formats.
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(keep_len)
        return past_key_values

    trimmed = []
    for layer_past in past_key_values:
        if isinstance(layer_past, (list, tuple)):
            trimmed_layer = tuple(
                t[:, :, :keep_len, :] if t is not None else None for t in layer_past
            )
            trimmed.append(trimmed_layer)
        elif layer_past is not None:
            trimmed.append(layer_past[:, :, :keep_len, :])
        else:
            trimmed.append(None)

    return tuple(trimmed)


def _check_tokenizer_consistency(target_tokenizer, draft_tokenizer):
    """
    Check that target and draft tokenizers share the same vocab and encoding.
    Returns (same_vocab, same_encoding).
    """
    same_vocab = target_tokenizer.get_vocab() == draft_tokenizer.get_vocab()
    test_str = "def hello_world(x: int) -> str:\n    return str(x)"
    target_ids = target_tokenizer.encode(test_str)
    draft_ids = draft_tokenizer.encode(test_str)
    same_encoding = target_ids == draft_ids
    return same_vocab, same_encoding


def _check_prompt_seed(prompt, structure_type):
    """
    Check if prompt contains enough structural seed for the given structure_type.
    Returns (valid, seed_count, reason).
    """
    if structure_type == "argparse":
        patterns = [r"add_argument", r"ArgumentParser", r"click\.option", r"click\.argument"]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, prompt))
        valid = count >= 2
        reason = f"argparse seed count={count}" if not valid else ""
        return valid, count, reason
    elif structure_type == "dict_config":
        has_dict = "{" in prompt and "}" in prompt
        has_list = "[" in prompt and "]" in prompt
        count = int(has_dict) + int(has_list)
        valid = count >= 1
        reason = "no dict/list structure in prompt" if not valid else ""
        return valid, count, reason
    elif structure_type in ("openmmlab", "openmmlab_config"):
        patterns = [r"model\s*=", r"pipeline\s*=", r"dataloader\s*=", r"train_cfg", r"test_cfg"]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, prompt))
        valid = count >= 1
        reason = f"openmmlab seed count={count}" if not valid else ""
        return valid, count, reason
    else:
        return True, 0, ""


def _cuda_sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


class FailureAwareFallback:
    """
    TASD-F v2: Failure-aware fallback with fixed 2-token AR fallback.

    Monitors runtime failure signals (rolling accept rate, consecutive zero
    accept rounds, recent repairs) and briefly falls back to target-only AR
    for 2 tokens when divergence is detected, then resumes TASD.

    This is the only TASD-F variant used in the final method.
    TASD-F v3 (progressive/boundary/probe fallback) was abandoned as a
    negative result due to severe slowdowns on short Argparse structures.

    Configuration:
        fallback_tokens=2
        fallback_guarded=False
    """
    def __init__(self, guarded=False, accept_threshold=0.5, repair_threshold=2):
        self.fallback_tokens = 2
        self.guarded = guarded
        self.guarded_trim_count = 0
        self.fallback_count = 0
        self.cooldown_rounds = 0
        self.fallback_cooldown = 0
        self.rolling_accept_rate = []
        self.consecutive_zero_accept_rounds = 0
        self.recent_repairs = []
        self.trigger_count = 0
        self.total_fallback_tokens = 0
        self.accept_threshold = accept_threshold
        self.repair_threshold = repair_threshold

    def record_round(self, drafted: int, accepted: int, repair: int):
        accept_rate = accepted / max(drafted, 1)
        self.rolling_accept_rate.append(accept_rate)
        if len(self.rolling_accept_rate) > 5:
            self.rolling_accept_rate.pop(0)
        if accepted == 0:
            self.consecutive_zero_accept_rounds += 1
        else:
            self.consecutive_zero_accept_rounds = 0
        if repair:
            self.recent_repairs.append(1)
        else:
            self.recent_repairs.append(0)
        if len(self.recent_repairs) > 5:
            self.recent_repairs.pop(0)

    def should_trigger(self) -> tuple:
        if self.fallback_cooldown > 0:
            return False, ""
        rolling = sum(self.rolling_accept_rate) / max(len(self.rolling_accept_rate), 1)
        repairs = sum(self.recent_repairs[-3:])
        if rolling < self.accept_threshold or self.consecutive_zero_accept_rounds >= 2 or repairs >= self.repair_threshold:
            return True, f"low_accept_{rolling:.2f}"
        return False, ""

    def start_fallback(self, reason: str):
        self.fallback_count += 1
        self.trigger_count += 1
        self.fallback_cooldown = 1

    def end_fallback(self, generated: int):
        self.total_fallback_tokens += generated

    def tick_cooldown(self):
        if self.fallback_cooldown > 0:
            self.fallback_cooldown -= 1

    def get_summary(self) -> dict:
        return {
            "trigger_count": self.trigger_count,
            "fallback_count": self.fallback_count,
            "fallback_tokens": self.fallback_tokens,
            "total_fallback_tokens": self.total_fallback_tokens,
            "guarded_trim_count": self.guarded_trim_count,
        }


class ProfitGuard:
    """
    Experimental safety analysis / future work. Not part of the final TASD or TASD-F method.

    Runtime profitability monitor for TASD speculative decoding (v2).

    v2 improvements over v1:
    - Structure-specific min_generated thresholds (Argparse: 64, others: 48)
    - Severe-only trigger: only fallback on truly bad cases
    - Expected recovery check: don't fallback if remaining tokens can't
      amortize the speculative overhead already incurred
    - More conservative overall — only triggers on severe divergence

    Trigger conditions (any one, AND must pass recovery check):
    1. Severe low accept: rolling_accept_rate < 0.35 over min_rounds
    2. High repair: repair_count >= 3 over recent 3 rounds
    3. Consecutive zero-accept: 2 rounds with accepted_count == 0

    Fallback: directly use target AR to generate remaining tokens.
    """

    # Structure-specific minimum generated tokens before fallback
    STRUCT_MIN_GENERATED = {
        "argparse": 64,
        "dict_config": 48,
        "openmmlab_config": 48,
        "rich_cli_option_groups": 48,
        "complex_nested_config": 48,
        "pipeline_stage_config": 48,
    }

    def __init__(
        self,
        min_rounds=2,
        min_generated=None,  # None = use structure-specific default
        accept_threshold=0.35,  # Severe-only: much lower than v1's 0.55
        repair_threshold=3,  # Severe-only: higher than v1's 2
        speed_margin=0.95,
        ar_tps_estimate=None,
        mode="fallback_to_ar",
        structure_type=None,  # For structure-specific thresholds
        enable_recovery_check=True,
    ):
        self.min_rounds = min_rounds
        # Structure-specific min_generated
        if min_generated is not None:
            self.min_generated = min_generated
        elif structure_type is not None:
            self.min_generated = self.STRUCT_MIN_GENERATED.get(structure_type, 48)
        else:
            self.min_generated = 48

        self.accept_threshold = accept_threshold
        self.repair_threshold = repair_threshold
        self.speed_margin = speed_margin
        self.ar_tps_estimate = ar_tps_estimate
        self.mode = mode
        self.structure_type = structure_type
        self.enable_recovery_check = enable_recovery_check

        # Rolling stats
        self.recent_accept_rates = []  # per-round accept rates
        self.recent_repair_counts = []  # per-round repair counts (0 or 1)
        self.consecutive_zero_accept = 0

        # State
        self.triggered = False
        self.trigger_reason = None
        self.trigger_step = None
        self.generated_before_fallback = 0
        self.elapsed_before_fallback = 0.0
        self.remaining_tokens = 0

        # Stats recording
        self.rolling_accept_rate_at_trigger = 0.0
        self.repair_count_recent_at_trigger = 0

    def record_round(self, drafted: int, accepted: int, repair: int):
        """Record stats after each TASD round."""
        round_rate = accepted / drafted if drafted > 0 else 0.0
        self.recent_accept_rates.append(round_rate)
        self.recent_repair_counts.append(repair)

        if accepted == 0:
            self.consecutive_zero_accept += 1
        else:
            self.consecutive_zero_accept = 0

    def _check_recovery_feasible(self, generated_count: int, remaining: int, elapsed_time: float) -> bool:
        """
        Check if fallback to AR can actually recover.

        If remaining tokens are too few to amortize the overhead already
        incurred, don't fallback — it would make things worse.

        Returns True if recovery is feasible (should fallback).
        """
        if not self.enable_recovery_check:
            return True

        # If remaining tokens are very few, fallback won't help much
        if remaining < 16:
            return False

        # If we've already generated enough and remaining is substantial,
        # AR fallback is likely to help
        if remaining >= 32:
            return True

        # Middle ground: check if we have enough remaining to offset overhead
        # Estimate: AR time for remaining = remaining / ar_tps
        # If ar_tps is unknown, assume fallback is safe if remaining > 24
        if self.ar_tps_estimate is not None and self.ar_tps_estimate > 0:
            est_ar_remaining_time = remaining / self.ar_tps_estimate
            # If estimated AR time for remaining is less than 1 second,
            # the overhead amortization is minimal
            if est_ar_remaining_time < 0.5:
                return False

        return True

    def should_trigger(self, generated_count: int, elapsed_time: float, remaining: int = 0) -> tuple:
        """
        Check if profit guard should trigger fallback.
        Returns (should_trigger: bool, reason: str).

        v2: Only triggers on SEVERE conditions, not mild degradation.
        """
        if self.triggered:
            return False, ""

        # Don't trigger too early — need enough generated tokens
        if generated_count < self.min_generated:
            return False, ""

        rates = self.recent_accept_rates
        repairs = self.recent_repair_counts

        triggered = False
        reason = ""

        # 1. Severe low accept rate trigger (threshold: 0.35, not 0.55)
        if len(rates) >= self.min_rounds:
            recent = rates[-self.min_rounds:]
            rolling = sum(recent) / len(recent)
            if rolling < self.accept_threshold:
                triggered = True
                reason = "severe_low_accept_rate"

        # 2. Severe repair trigger (threshold: 3, not 2)
        if not triggered and len(repairs) >= 3:
            recent_repairs = sum(repairs[-3:])
            if recent_repairs >= self.repair_threshold:
                triggered = True
                reason = "severe_high_repair"

        # 3. Consecutive zero-accept trigger
        if not triggered and self.consecutive_zero_accept >= 2:
            triggered = True
            reason = "consecutive_zero_accept"

        if not triggered:
            return False, ""

        # Recovery check: only fallback if it can actually help
        if not self._check_recovery_feasible(generated_count, remaining, elapsed_time):
            return False, f"{reason}_but_no_recovery"

        return True, reason

    def trigger(self, reason: str, generated_count: int, elapsed_time: float, remaining: int):
        """Record the fallback trigger."""
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_step = generated_count
        self.generated_before_fallback = generated_count
        self.elapsed_before_fallback = elapsed_time
        self.remaining_tokens = remaining

        # Record stats at trigger point
        if self.recent_accept_rates:
            n = min(3, len(self.recent_accept_rates))
            self.rolling_accept_rate_at_trigger = sum(self.recent_accept_rates[-n:]) / n
        if self.recent_repair_counts:
            n = min(3, len(self.recent_repair_counts))
            self.repair_count_recent_at_trigger = sum(self.recent_repair_counts[-n:])

    def get_summary(self) -> dict:
        return {
            "profit_guard_triggered": self.triggered,
            "profit_guard_reason": self.trigger_reason,
            "profit_guard_trigger_step": self.trigger_step,
            "profit_guard_generated_before_fallback": self.generated_before_fallback,
            "profit_guard_elapsed_before_fallback": round(self.elapsed_before_fallback, 4),
            "profit_guard_rolling_accept_rate": round(self.rolling_accept_rate_at_trigger, 4),
            "profit_guard_repair_count_recent": self.repair_count_recent_at_trigger,
            "profit_guard_ar_tps_estimate": self.ar_tps_estimate,
            "profit_guard_remaining_tokens": self.remaining_tokens,
        }


class CommentStringFallback:
    """
    Lightweight detector for high-risk regions in config generation:
    - Comment lines (starting with #)
    - Long string values containing file paths / dataset names
    - Multi-line comment blocks

    Designed to be conservative: only triggers on genuinely risky patterns,
    not on normal inline strings like type='LoadImage'.

    When triggered, caller should switch to conservative TASD params.
    """
    # Path patterns that indicate file/dataset references (not just _val in key names)
    PATH_PATTERNS = [
        r"'[^']{15,}\.(?:json|pkl|txt|jpg|png|yaml|py)[^']*'",
        r'"[^"]{15,}\.(?:json|pkl|txt|jpg|png|yaml|py)[^"]*"',
        r"'/(?:images|data|annotations|checkpoints|results)/[^']*'",
        r'"/(?:images|data|annotations|checkpoints|results)/[^"]*"',
    ]
    _path_re = [re.compile(p) for p in PATH_PATTERNS]

    def __init__(self, lookback_chars=200, nl_word_threshold=5):
        self.lookback = lookback_chars
        self.nl_word_threshold = nl_word_threshold
        # Stats
        self.trigger_count = 0
        self.fallback_tokens = 0
        self.fallback_regions = []
        self.fallback_reasons = []
        self.accept_rate_before = []
        self.accept_rate_after = []

    def is_high_risk(self, generated_text: str) -> tuple:
        """
        Check if current position is in a high-risk region.
        Returns (is_risk: bool, reason: str).
        """
        recent = generated_text[-self.lookback:] if len(generated_text) > self.lookback else generated_text
        lines = recent.split('\n')
        last_line = lines[-1] if lines else ""

        # 1. Comment line detection (only for actual comment lines, not inline)
        stripped = last_line.lstrip()
        if stripped.startswith('#'):
            return True, "comment_line"

        # 2. Long string with file extension (not short inline strings)
        for pat in self._path_re:
            if pat.search(recent):
                return True, "long_path_string"

        # 3. Multi-line comment block: check if last 3+ lines are comments
        comment_lines = 0
        for line in reversed(lines):
            s = line.strip()
            if s.startswith('#') or s == '':
                if s.startswith('#'):
                    comment_lines += 1
            else:
                break
        if comment_lines >= 2:
            return True, "comment_block"

        # 4. Very long natural-language text (not config keys)
        # Only trigger if there's a long stretch of alphabetic words
        # with NO code symbols at all (not even = or () )
        alpha_words = re.findall(r'\b[a-zA-Z]{5,}\b', recent)
        code_symbols = recent.count('=') + recent.count('(') + recent.count(')') + recent.count('{') + recent.count('}') + recent.count('[') + recent.count(']') + recent.count(':') + recent.count("'")
        if len(alpha_words) >= self.nl_word_threshold and code_symbols == 0:
            return True, "pure_nl_text"

        return False, ""

    def record_trigger(self, reason: str, tokens_in_fallback: int = 0,
                       accept_before: float = 0.0, accept_after: float = 0.0):
        self.trigger_count += 1
        self.fallback_tokens += tokens_in_fallback
        self.fallback_reasons.append(reason)
        self.fallback_regions.append({
            "reason": reason,
            "tokens": tokens_in_fallback,
            "accept_before": round(accept_before, 4),
            "accept_after": round(accept_after, 4),
        })
        if accept_before > 0:
            self.accept_rate_before.append(accept_before)
        if accept_after > 0:
            self.accept_rate_after.append(accept_after)

    def get_summary(self) -> dict:
        return {
            "trigger_count": self.trigger_count,
            "fallback_tokens": self.fallback_tokens,
            "fallback_reasons": self.fallback_reasons,
            "fallback_regions": self.fallback_regions,
            "avg_accept_before": round(sum(self.accept_rate_before) / len(self.accept_rate_before), 4) if self.accept_rate_before else 0,
            "avg_accept_after": round(sum(self.accept_rate_after) / len(self.accept_rate_after), 4) if self.accept_rate_after else 0,
        }


def _is_early_stop_condition(token, tokenizer, structure_type, generated_text_so_far):
    """
    Check if we should stop drafting early based on:
    - EOS token
    - Obvious structure end markers
    - High-risk content detected by lightweight check
    """
    if token == tokenizer.eos_token_id:
        return True, "eos"

    # Check for structure end markers
    decoded = tokenizer.decode([token], skip_special_tokens=True)
    if structure_type == "argparse":
        if decoded.strip().startswith(("def ", "class ", "import ", "from ")):
            return True, "off_structure"
    elif structure_type == "dict_config":
        if decoded.strip().startswith(("def ", "class ", "import ", "from ")):
            return True, "off_structure"

    return False, ""


class ProfitAwareSwitch:
    """
    TASD-FG-P: Profit-aware AR switch for TASD speculative decoding.

    Monitors early-stage (first N tokens) speculative performance. If the
    speculative strategy is clearly losing (below-1.0x bound), switches to
    target AR generation for the remaining tokens to avoid below-AR outcomes.

    Trigger conditions (any one, within the switch window):
    1. estimated_speedup < 1.05  (wall-clock projection)
    2. fallback_count >= 2        (too many failure-aware fallbacks)
    3. guard_trim_count >= 3      (too many structural guard trims)
    4. rolling_accept_rate < 0.4  (poor draft alignment)
    5. consecutive_zero_accept >= 2 (draft model stalled)

    Switch action: use target_model.generate() for remaining tokens.
    """
    def __init__(self, window_tokens=48, ar_tps_estimate=None):
        self.window_tokens = window_tokens
        self.ar_tps_estimate = ar_tps_estimate

        # Rolling stats
        self.rolling_accept_rates = []
        self.consecutive_zero_accept = 0
        self.cumulative_fallback_count = 0
        self.cumulative_guard_trim_count = 0

        # Switch state
        self.switched = False
        self.switch_reason = None
        self.switch_at_token = 0
        self.generated_before_switch = 0
        self.elapsed_before_switch = 0.0
        self.switch_trigger_values = {}

    def record_round(self, drafted: int, accepted: int, fallback_count: int, guard_trim_count: int):
        """Record stats after each TASD round."""
        if drafted > 0:
            rate = accepted / drafted
            self.rolling_accept_rates.append(rate)
            if len(self.rolling_accept_rates) > 5:
                self.rolling_accept_rates.pop(0)

        if accepted == 0:
            self.consecutive_zero_accept += 1
        else:
            self.consecutive_zero_accept = 0

        self.cumulative_fallback_count += fallback_count
        self.cumulative_guard_trim_count += guard_trim_count

    def should_switch(self, generated_count: int, elapsed_time: float, remaining: int) -> tuple:
        """
        Check if we should switch to AR.
        Returns (should_switch: bool, reason: str).
        Only evaluates within the window (first window_tokens generated).
        """
        if self.switched:
            return False, ""

        # Only evaluate within the window
        if generated_count > self.window_tokens:
            return False, ""

        # Need at least a few rounds to gather stats
        if len(self.rolling_accept_rates) < 2:
            return False, ""

        # 1. Estimated speedup from wall-clock
        if self.ar_tps_estimate is not None and self.ar_tps_estimate > 0 and elapsed_time > 0:
            tps_so_far = generated_count / elapsed_time if elapsed_time > 0 else 0
            estimated_speedup = tps_so_far / self.ar_tps_estimate
            if estimated_speedup < 1.05:
                return True, f"est_speedup={estimated_speedup:.3f}_below_1.05"

        # 2. Too many failure-aware fallbacks
        if self.cumulative_fallback_count >= 2:
            return True, f"fallback_count={self.cumulative_fallback_count}"

        # 3. Too many guard trims
        if self.cumulative_guard_trim_count >= 3:
            return True, f"guard_trim={self.cumulative_guard_trim_count}"

        # 4. Poor rolling accept rate
        if self.rolling_accept_rates:
            rolling = sum(self.rolling_accept_rates) / len(self.rolling_accept_rates)
            if rolling < 0.4:
                return True, f"rolling_accept={rolling:.3f}"

        # 5. Consecutive zero accept (draft model stalled)
        if self.consecutive_zero_accept >= 2:
            return True, "consecutive_zero_accept"

        return False, ""

    def trigger_switch(self, reason: str, generated_count: int, elapsed_time: float):
        """Record the AR switch trigger."""
        self.switched = True
        self.switch_reason = reason
        self.switch_at_token = generated_count
        self.generated_before_switch = generated_count
        self.elapsed_before_switch = elapsed_time
        # Record trigger values for diagnostics
        if self.rolling_accept_rates:
            self.switch_trigger_values["rolling_accept"] = \
                round(sum(self.rolling_accept_rates) / len(self.rolling_accept_rates), 4)
        self.switch_trigger_values["fallback_count"] = self.cumulative_fallback_count
        self.switch_trigger_values["guard_trim"] = self.cumulative_guard_trim_count
        self.switch_trigger_values["consecutive_zero"] = self.consecutive_zero_accept

    def get_summary(self) -> dict:
        return {
            "switched_to_ar": self.switched,
            "switch_reason": self.switch_reason,
            "switch_at_token": self.switch_at_token,
            "generated_before_switch": self.generated_before_switch,
            "elapsed_before_switch": round(self.elapsed_before_switch, 4),
            "window_tokens": self.window_tokens,
            "trigger_values": self.switch_trigger_values,
        }


def tasd_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt,
    structure_type="argparse",
    max_new_tokens=128,
    temperature=0.0,
    draft_len=8,
    top_k_accept=3,
    min_token_prob=1e-4,
    prefix_budget=0.2,
    window_len=2,
    draft_tokenizer=None,
    draft_blocks=2,
    enable_guard=True,
    guard_v2=False,  # Use GuardV2 instead of StructuralGuard (comment/string aware)
    guard_calibrated=True,  # Guard-v1.5: downgrade repetition/brackets/import to warnings (default ON)
    enable_relaxed_accept=True,
    adaptive_policy=None,
    enable_comment_string_fallback=False,
    enable_failure_aware_fallback=False,  # TASD-F v2: optional 2-token runtime fallback
    fallback_guarded=False,              # Apply structural guard during fallback
    fallback_accept_threshold=0.5,       # Rolling accept rate threshold for triggering
    fallback_repair_threshold=2,         # Repair count threshold for triggering
    enable_profit_guard=False,
    profit_guard_ar_tps_estimate=None,
    profit_guard_min_rounds=2,
    profit_guard_min_generated=24,
    profit_guard_accept_threshold=0.55,
    profit_guard_repair_threshold=2,
    profit_guard_speed_margin=0.95,
    profit_guard_mode="fallback_to_ar",
    enable_profit_aware_switch=False,     # TASD-FG-P: early-stage profit-aware AR switch
    profit_switch_window=48,              # Evaluate only in first N tokens
    profit_switch_ar_tps_estimate=None,   # AR TPS for speedup estimation
):
    """
    TASD speculative decoding with structural guard, proper KV cache,
    multi-block draft, and comprehensive stats.

    Args:
        enable_guard: If False, skip structural guard check
        enable_relaxed_accept: If False, only accept draft_tok == target argmax (strict)
        enable_profit_guard: If True, enable ProfitGuard to fallback to AR when unprofitable
        profit_guard_ar_tps_estimate: Estimated AR TPS for wall-clock comparison
        profit_guard_min_rounds: Minimum rounds of low accept to trigger
        profit_guard_min_generated: Minimum tokens generated before trigger can fire
        profit_guard_accept_threshold: Rolling accept rate threshold for trigger
        profit_guard_repair_threshold: Repair count threshold for trigger
        profit_guard_speed_margin: Speed margin for wall-clock loss trigger
        profit_guard_mode: Fallback mode (only "fallback_to_ar" supported)
        enable_profit_aware_switch: If True, enable ProfitAwareSwitch (TASD-FG-P)
        profit_switch_window: Max tokens to evaluate before closing the switch window
        profit_switch_ar_tps_estimate: AR TPS estimate for speedup projection

    Note on reference: reference is NOT used in decoding. It is only
    passed to the evaluator for structural quality assessment.
    """
    # --- Tokenizer consistency check ---
    effective_draft_tok = draft_tokenizer if draft_tokenizer is not None else tokenizer
    same_vocab, same_encoding = _check_tokenizer_consistency(tokenizer, effective_draft_tok)
    if not same_vocab or not same_encoding:
        raise ValueError(
            f"Tokenizer mismatch detected: same_vocab={same_vocab}, same_encoding={same_encoding}. "
            "Target and draft models must share the same tokenizer."
        )

    # --- Prompt seed check ---
    seed_valid, prompt_seed_count, seed_reason = _check_prompt_seed(prompt, structure_type)
    invalid_seed = not seed_valid

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    prompt_len = input_ids.shape[1]
    device = target_model.device
    draft_device = next(draft_model.parameters()).device

    if guard_v2:
        guard = GuardV2(structure_type=structure_type)
    else:
        guard = StructuralGuard(structure_type=structure_type, calibrated=guard_calibrated)

    # Comment/string fallback detector
    fallback_detector = None
    if enable_comment_string_fallback:
        fallback_detector = CommentStringFallback()

    # Failure-aware fallback detector (TASD-F v2)
    failure_fallback = None
    if enable_failure_aware_fallback:
        failure_fallback = FailureAwareFallback(
            guarded=fallback_guarded,
            accept_threshold=fallback_accept_threshold,
            repair_threshold=fallback_repair_threshold,
        )

    # Profit guard
    profit_guard = None
    if enable_profit_guard:
        profit_guard = ProfitGuard(
            min_rounds=profit_guard_min_rounds,
            min_generated=profit_guard_min_generated,
            accept_threshold=profit_guard_accept_threshold,
            repair_threshold=profit_guard_repair_threshold,
            speed_margin=profit_guard_speed_margin,
            ar_tps_estimate=profit_guard_ar_tps_estimate,
            mode=profit_guard_mode,
            structure_type=structure_type,
        )

    # Profit-aware switch (TASD-FG-P)
    profit_switch = None
    if enable_profit_aware_switch:
        profit_switch = ProfitAwareSwitch(
            window_tokens=profit_switch_window,
            ar_tps_estimate=profit_switch_ar_tps_estimate,
        )

    # Default TASD params (can be overridden by fallback)
    _default_draft_len = draft_len
    _default_draft_blocks = draft_blocks
    _default_top_k_accept = top_k_accept
    _default_prefix_budget = prefix_budget

    # Stats
    total_drafted = 0
    total_accepted = 0
    token_accept = 0
    prefix_accept = 0
    prefix_budget_used = 0.0
    window_accept_count = 0
    target_model_forwards = 0
    draft_model_forwards = 0
    repair_count = 0
    consecutive_repair_count = 0
    draft_time_total = 0.0
    target_time_total = 0.0
    trim_reasons = []
    repair_reasons = []
    repair_records = []  # List of {token, text, reason}
    failed = False
    error_type = None
    error_msg = None
    stop_reason = None

    # Multi-block draft stats
    requested_draft_blocks = draft_blocks
    actual_draft_blocks_list = []
    early_stop_reasons = []
    block_accept_count = 0
    block_partial_accept_count = 0
    block_reject_count = 0
    first_reject_block_id = -1

    # EOS stats
    eos_drafted = False
    eos_accepted = False

    # Adaptive policy stats
    adaptive_round_drafted = []
    adaptive_round_accepted = []
    adaptive_round_top3_hits = []
    adaptive_round_top5_hits = []
    adaptive_topk_computed = 0

    generated_ids = []
    max_iterations = max_new_tokens + 50

    # --- Pre-fill KV cache for prompt on both models ---
    _cuda_sync()
    wall_start = time.time()

    with torch.no_grad():
        target_logits_prefill, target_past = _forward_with_cache(target_model, input_ids, None)
        draft_input = input_ids.to(draft_device)
        draft_logits_prefill, draft_past = _forward_with_cache(draft_model, draft_input, None)

    target_model_forwards += 1
    draft_model_forwards += 1

    last_target_logit = target_logits_prefill[0, -1, :]
    last_draft_logit = draft_logits_prefill[0, -1, :].to(device)

    try:
        for iteration in range(max_iterations):
            remaining = max_new_tokens - len(generated_ids)
            if remaining <= 0:
                stop_reason = "max_tokens"
                break

            # --- Adaptive policy: update draft_len / top_k_accept ---
            if adaptive_policy is not None:
                draft_len, top_k_accept = adaptive_policy.get_params()

            # --- Comment/string fallback: detect high-risk regions ---
            in_fallback = False
            fallback_reason = ""
            if fallback_detector is not None and generated_ids:
                generated_text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
                is_risk, risk_reason = fallback_detector.is_high_risk(generated_text_so_far)
                if is_risk:
                    in_fallback = True
                    fallback_reason = risk_reason
                    draft_len = 4
                    draft_blocks = 1
                    top_k_accept = 1
                    prefix_budget = 0.0
                else:
                    # Restore defaults when leaving high-risk region
                    draft_len = _default_draft_len
                    draft_blocks = _default_draft_blocks
                    top_k_accept = _default_top_k_accept
                    prefix_budget = _default_prefix_budget

            # --- Failure-aware fallback: detect runtime failures (TASD-F v2) ---
            in_failure_fallback = False
            fb_reason = ""
            if failure_fallback is not None and generated_ids:
                should_fb, fb_reason = failure_fallback.should_trigger()
                if should_fb:
                    in_failure_fallback = True
                    failure_fallback.start_fallback(fb_reason)

            # --- Execute short AR fallback if triggered (TASD-F v2: unguarded 2-token) ---
            if in_failure_fallback:
                fb_tokens = failure_fallback.fallback_tokens
                fb_remaining = min(fb_tokens, remaining)
                if fb_remaining > 0:
                    current_ids = torch.cat(
                        [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                    ) if generated_ids else input_ids
                    with torch.no_grad():
                        ar_output = target_model.generate(
                            current_ids,
                            max_new_tokens=fb_remaining,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_tokens = ar_output[0][current_ids.shape[1]:].tolist()
                    generated_ids.extend(new_tokens)

                    # If guarded, apply structural guard to fallback tokens
                    if failure_fallback.guarded and guard is not None:
                        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if isinstance(guard, GuardV2):
                            is_safe, safe_token_count, risk_type, _rl = guard.check(
                                full_text, generated_ids, tokenizer
                            )
                        else:
                            is_safe, safe_token_count, risk_type = guard.check(
                                full_text, generated_ids, tokenizer
                            )
                        if not is_safe and safe_token_count < len(generated_ids):
                            trimmed_count = len(generated_ids) - max(safe_token_count, 0)
                            generated_ids = generated_ids[:max(safe_token_count, 1)]
                            failure_fallback.guarded_trim_count += trimmed_count
                            guard.trigger_count += 1
                            guard.trim_count += 1

                    # Re-prefill KV caches with full sequence after AR
                    _cuda_sync()
                    full_ids = torch.cat(
                        [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                    )
                    with torch.no_grad():
                        _, target_past = _forward_with_cache(target_model, full_ids, None)
                        target_model_forwards += 1
                        draft_full = full_ids.to(draft_device)
                        _, draft_past = _forward_with_cache(draft_model, draft_full, None)
                        draft_model_forwards += 1

                    # Get last logits from one-token forward
                    if new_tokens:
                        with torch.no_grad():
                            next_tensor = torch.tensor([[new_tokens[-1]]], device=device)
                            repair_logits, target_past = _forward_with_cache(
                                target_model, next_tensor, target_past
                            )
                            last_target_logit = repair_logits[0, -1, :]
                            target_model_forwards += 1
                        with torch.no_grad():
                            draft_next = torch.tensor([[new_tokens[-1]]], device=draft_device)
                            draft_logits_out, draft_past = _forward_with_cache(
                                draft_model, draft_next, draft_past
                            )
                            last_draft_logit = draft_logits_out[0, -1, :].to(device)
                            draft_model_forwards += 1

                    failure_fallback.end_fallback(len(new_tokens))

                    # Check termination
                    if len(generated_ids) >= max_new_tokens:
                        stop_reason = "max_tokens"
                        break
                    if tokenizer.eos_token_id in new_tokens:
                        eos_pos = generated_ids.index(tokenizer.eos_token_id)
                        generated_ids = generated_ids[:eos_pos]
                        stop_reason = "eos"
                        break

            # --- Failure-aware fallback: tick cooldown ---
            if failure_fallback is not None:
                failure_fallback.tick_cooldown()

            # If too many consecutive repairs, degrade to AR
            if consecutive_repair_count >= 5:
                repair_reasons.append(f"consecutive_repair_limit:{consecutive_repair_count}")
                stop_reason = "consecutive_repair_limit"
                current_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                ) if generated_ids else input_ids
                with torch.no_grad():
                    ar_output = target_model.generate(
                        current_ids,
                        max_new_tokens=remaining,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = ar_output[0][current_ids.shape[1]:].tolist()
                generated_ids.extend(new_tokens)
                break

            # --- Multi-block draft phase ---
            draft_start = time.time()
            _cuda_sync()

            all_draft_tokens = []
            all_draft_logits_list = []
            block_boundaries = []  # (start_idx, end_idx) for each block
            actual_blocks = 0
            block_early_stop = False
            block_early_stop_reason = ""

            with torch.no_grad():
                next_token = _greedy_sample(last_draft_logit)

                for block_id in range(draft_blocks):
                    # Check remaining budget
                    already_this_round = len(all_draft_tokens)
                    effective_remaining = remaining - already_this_round
                    if effective_remaining <= 0:
                        block_early_stop = True
                        block_early_stop_reason = "budget_exhausted"
                        break

                    tokens_this_block = min(draft_len, effective_remaining)
                    block_start_idx = len(all_draft_tokens)

                    for _ in range(tokens_this_block):
                        all_draft_tokens.append(next_token)

                        # Check EOS
                        if next_token == tokenizer.eos_token_id:
                            eos_drafted = True
                            break

                        # Check early stop conditions
                        should_stop, stop_reason_str = _is_early_stop_condition(
                            next_token, tokenizer, structure_type,
                            tokenizer.decode(generated_ids + all_draft_tokens, skip_special_tokens=True)
                        )
                        if should_stop:
                            block_early_stop = True
                            block_early_stop_reason = stop_reason_str
                            break

                        # Incremental forward pass with KV cache
                        next_tensor = torch.tensor([[next_token]], device=draft_device)
                        draft_logits, draft_past = _forward_with_cache(
                            draft_model, next_tensor, draft_past
                        )
                        draft_model_forwards += 1
                        all_draft_logits_list.append(draft_logits[0, -1, :].to(device))
                        next_token = _greedy_sample(draft_logits)

                    block_end_idx = len(all_draft_tokens)
                    block_boundaries.append((block_start_idx, block_end_idx))
                    actual_blocks += 1

                    if block_early_stop:
                        break

                    # If we hit EOS in this block, stop drafting more blocks
                    if eos_drafted:
                        block_early_stop_reason = "eos"
                        break

            _cuda_sync()
            draft_time_total += time.time() - draft_start

            actual_draft_blocks_list.append(actual_blocks)
            if block_early_stop:
                early_stop_reasons.append(block_early_stop_reason)

            draft_tokens = all_draft_tokens
            draft_logits_list = all_draft_logits_list

            if not draft_tokens:
                repair_count += 1
                consecutive_repair_count += 1
                repair_start = time.time()
                _cuda_sync()
                with torch.no_grad():
                    repair_logits, target_past = _forward_with_cache(
                        target_model,
                        torch.tensor([[last_target_logit.argmax().item()]], device=device),
                        target_past,
                    )
                    next_token = _greedy_sample(repair_logits)
                _cuda_sync()
                target_time_total += time.time() - repair_start
                target_model_forwards += 1
                generated_ids.append(next_token)
                last_target_logit = repair_logits[0, -1, :]
                last_draft_logit = repair_logits[0, -1, :]
                repair_text = tokenizer.decode([next_token], skip_special_tokens=True)
                repair_records.append({
                    "token": next_token,
                    "text": repair_text,
                    "reason": "all_draft_rejected",
                })
                repair_reasons.append("all_draft_rejected")
                continue

            # --- Target verification with KV cache ---
            target_start = time.time()
            _cuda_sync()

            draft_tensor = torch.tensor([draft_tokens], device=device)
            target_logits, target_past = _forward_with_cache(
                target_model, draft_tensor, target_past
            )
            _cuda_sync()
            target_time_total += time.time() - target_start
            target_model_forwards += 1

            # Verify each draft token
            accept_mask = []
            round_top3_hits = 0
            round_top5_hits = 0
            for i, draft_tok in enumerate(draft_tokens):
                if i == 0:
                    logit_for_token = last_target_logit
                else:
                    logit_idx = i - 1
                    if logit_idx >= target_logits.shape[1]:
                        accept_mask.append(False)
                        continue
                    logit_for_token = target_logits[0, logit_idx]

                target_argmax = logit_for_token.argmax().item()
                if draft_tok == target_argmax:
                    accept_mask.append(True)
                elif enable_relaxed_accept:
                    probs = torch.softmax(logit_for_token, dim=-1)
                    _, topk_indices = torch.topk(probs, top_k_accept)
                    if draft_tok in topk_indices.tolist():
                        accept_mask.append(True)
                    elif probs[draft_tok].item() >= min_token_prob:
                        accept_mask.append(True)
                    else:
                        accept_mask.append(False)

                # --- Top-k hit tracking (for adaptive policy) ---
                if adaptive_policy is not None:
                    probs = torch.softmax(logit_for_token, dim=-1)
                    _, top3_idx = torch.topk(probs, 3)
                    _, top5_idx = torch.topk(probs, 5)
                    if draft_tok in top3_idx.tolist():
                        round_top3_hits += 1
                    if draft_tok in top5_idx.tolist():
                        round_top5_hits += 1

            total_drafted += len(draft_tokens)

            # --- Block-level accept/reject stats ---
            for block_start, block_end in block_boundaries:
                block_accepts = sum(accept_mask[block_start:block_end])
                block_total = block_end - block_start
                if block_total == 0:
                    continue

                if block_accepts == block_total:
                    block_accept_count += 1
                elif block_accepts > 0:
                    block_partial_accept_count += 1
                    if first_reject_block_id < 0:
                        first_reject_block_id = len(actual_draft_blocks_list) - 1
                else:
                    block_reject_count += 1
                    if first_reject_block_id < 0:
                        first_reject_block_id = len(actual_draft_blocks_list) - 1

            # --- Compute accepted prefix ---
            strict_prefix_len = 0
            for i, accepted in enumerate(accept_mask):
                if accepted:
                    strict_prefix_len = i + 1
                else:
                    break

            accepted_tokens = draft_tokens[:strict_prefix_len]
            token_accept += strict_prefix_len

            # Window acceptance beyond strict prefix
            if enable_relaxed_accept and strict_prefix_len < len(draft_tokens):
                window_start = strict_prefix_len
                while window_start < len(draft_tokens):
                    window_end = min(window_start + window_len, len(draft_tokens))
                    window_accepts = sum(accept_mask[window_start:window_end])
                    window_total = window_end - window_start
                    if window_accepts >= window_total * 0.5:
                        accepted_tokens.extend(draft_tokens[window_start:window_end])
                        window_accept_count += 1
                        window_start = window_end
                    else:
                        break

            # Prefix budget acceptance
            if enable_relaxed_accept and len(accepted_tokens) < len(draft_tokens):
                remaining_draft = draft_tokens[len(accepted_tokens):]
                for idx, tok in enumerate(remaining_draft):
                    pos_in_draft = len(accepted_tokens) + idx
                    if pos_in_draft == 0:
                        logit_for_tok = last_target_logit
                    else:
                        logit_idx = pos_in_draft - 1
                        if logit_idx >= target_logits.shape[1]:
                            break
                        logit_for_tok = target_logits[0, logit_idx]

                    log_probs = torch.log_softmax(logit_for_tok, dim=-1)
                    target_best_logprob = log_probs.max().item()
                    draft_token_logprob = log_probs[tok].item()
                    risk = max(0.0, target_best_logprob - draft_token_logprob)

                    if prefix_budget_used + risk <= prefix_budget:
                        accepted_tokens.append(tok)
                        prefix_budget_used += risk
                    else:
                        break

            prefix_accept += len(accepted_tokens)

            # --- Guard check ---
            if enable_guard:
                accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
                if guard_v2:
                    safe, guard_keep_count, risk_type, _risk_level = guard.check(
                        accepted_text, tokens=accepted_tokens, tokenizer=tokenizer
                    )
                    # Adaptive tightening: high risk → strict verification next round
                    if _risk_level == "high":
                        top_k_accept = 1
                else:
                    safe, guard_keep_count, risk_type = guard.check(
                        accepted_text, tokens=accepted_tokens, tokenizer=tokenizer
                    )

                if not safe:
                    guard.trim_count += 1
                    trim_reasons.append(risk_type)
                    accepted_tokens = accepted_tokens[:guard_keep_count]

            accepted_count = len(accepted_tokens)

            # --- Record fallback stats ---
            if in_fallback and fallback_detector is not None:
                round_accept = accepted_count / len(draft_tokens) if draft_tokens else 0
                fallback_detector.record_trigger(
                    reason=fallback_reason,
                    tokens_in_fallback=accepted_count,
                    accept_before=round_accept,
                    accept_after=round_accept,
                )

            # --- Apply accepted tokens or repair ---
            if accepted_tokens:
                generated_ids.extend(accepted_tokens)
                consecutive_repair_count = 0

                # Check if EOS was accepted
                if tokenizer.eos_token_id in accepted_tokens:
                    eos_accepted = True
                    stop_reason = "eos"

                new_cache_len = prompt_len + len(generated_ids)
                target_past = _trim_past_key_values(target_past, new_cache_len)
                draft_past = _trim_past_key_values(draft_past, new_cache_len)

                # Update logits
                if accepted_count <= target_logits.shape[1]:
                    last_target_logit = target_logits[0, accepted_count - 1, :]
                else:
                    last_target_logit = target_logits[0, -1, :]

                if accepted_count <= len(draft_logits_list):
                    last_draft_logit = draft_logits_list[accepted_count - 1]
                else:
                    last_draft_logit = draft_logits_list[-1] if draft_logits_list else last_draft_logit
            else:
                repair_count += 1
                consecutive_repair_count += 1
                repair_start = time.time()
                _cuda_sync()
                with torch.no_grad():
                    next_token_tensor = torch.tensor(
                        [[last_target_logit.argmax().item()]], device=device
                    )
                    repair_logits, target_past = _forward_with_cache(
                        target_model, next_token_tensor, target_past
                    )
                    next_token = _greedy_sample(repair_logits)
                _cuda_sync()
                target_time_total += time.time() - repair_start
                target_model_forwards += 1
                generated_ids.append(next_token)
                last_target_logit = repair_logits[0, -1, :]
                last_draft_logit = repair_logits[0, -1, :]
                repair_text = tokenizer.decode([next_token], skip_special_tokens=True)
                repair_records.append({
                    "token": next_token,
                    "text": repair_text,
                    "reason": "all_draft_rejected",
                })
                repair_reasons.append("all_draft_rejected")

            total_accepted += accepted_count

            # --- Record round stats for adaptive policy ---
            if adaptive_policy is not None:
                # safe is only defined inside enable_guard block; default True if guard disabled
                guard_safe = locals().get("safe", True)
                guard_triggered_this_round = 1 if (enable_guard and not guard_safe) else 0
                has_off_structure = any(
                    r in ("off_structure", "def_class_import", "import_outside", "class_def")
                    for r in trim_reasons[-1:] if trim_reasons
                )
                adaptive_policy.record_round(
                    drafted=len(draft_tokens),
                    accepted=accepted_count,
                    top3_hits=round_top3_hits,
                    top5_hits=round_top5_hits,
                    repair_count=1 if accepted_count == 0 else 0,
                    guard_triggers=guard_triggered_this_round,
                    off_structure=has_off_structure,
                )
                adaptive_topk_computed += 1

            # --- Record round stats for failure-aware fallback ---
            if failure_fallback is not None:
                failure_fallback.record_round(
                    drafted=len(draft_tokens),
                    accepted=accepted_count,
                    repair=1 if accepted_count == 0 else 0,
                )

            # --- Record round stats for profit guard ---
            if profit_guard is not None and not profit_guard.triggered:
                profit_guard.record_round(
                    drafted=len(draft_tokens),
                    accepted=accepted_count,
                    repair=1 if accepted_count == 0 else 0,
                )

            # --- Record round stats for profit-aware switch ---
            if profit_switch is not None and not profit_switch.switched:
                _fb_count = failure_fallback.fallback_count if failure_fallback is not None else 0
                _guard_trim = guard.trim_count if enable_guard else 0
                profit_switch.record_round(
                    drafted=len(draft_tokens),
                    accepted=accepted_count,
                    fallback_count=_fb_count,
                    guard_trim_count=_guard_trim,
                )

            # --- Profit guard: check if we should fallback to AR ---
            if profit_guard is not None and not profit_guard.triggered:
                current_elapsed = time.time() - wall_start
                should_pg, pg_reason = profit_guard.should_trigger(
                    generated_count=len(generated_ids),
                    elapsed_time=current_elapsed,
                    remaining=remaining,
                )
                if should_pg:
                    profit_guard.trigger(
                        reason=pg_reason,
                        generated_count=len(generated_ids),
                        elapsed_time=current_elapsed,
                        remaining=remaining,
                    )
                    # Fallback to AR for remaining tokens
                    current_ids = torch.cat(
                        [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                    ) if generated_ids else input_ids
                    with torch.no_grad():
                        ar_output = target_model.generate(
                            current_ids,
                            max_new_tokens=remaining,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_tokens = ar_output[0][current_ids.shape[1]:].tolist()
                    generated_ids.extend(new_tokens)
                    stop_reason = "profit_guard_fallback_to_ar"
                    break

            # --- Profit-aware switch: check if we should switch to AR ---
            if profit_switch is not None and not profit_switch.switched:
                current_elapsed = time.time() - wall_start
                should_sw, sw_reason = profit_switch.should_switch(
                    generated_count=len(generated_ids),
                    elapsed_time=current_elapsed,
                    remaining=remaining,
                )
                if should_sw:
                    profit_switch.trigger_switch(
                        reason=sw_reason,
                        generated_count=len(generated_ids),
                        elapsed_time=current_elapsed,
                    )
                    # AR generate remaining tokens
                    current_ids = torch.cat(
                        [input_ids, torch.tensor([generated_ids], device=device)], dim=1
                    ) if generated_ids else input_ids
                    with torch.no_grad():
                        ar_output = target_model.generate(
                            current_ids,
                            max_new_tokens=remaining,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_tokens = ar_output[0][current_ids.shape[1]:].tolist()
                    generated_ids.extend(new_tokens)
                    stop_reason = "profit_switch_to_ar"
                    break

            # Check for EOS in generated
            if tokenizer.eos_token_id in generated_ids:
                eos_pos = generated_ids.index(tokenizer.eos_token_id)
                generated_ids = generated_ids[:eos_pos]
                if stop_reason is None:
                    stop_reason = "eos"
                break

            # Check max tokens
            if len(generated_ids) >= max_new_tokens:
                stop_reason = "max_tokens"
                break

    except Exception as e:
        failed = True
        error_type = type(e).__name__
        error_msg = str(e)
        if stop_reason is None:
            stop_reason = f"error:{error_type}"

    if stop_reason is None:
        stop_reason = "max_iterations"

    # --- Compute final metrics ---
    _cuda_sync()
    wall_time = time.time() - wall_start
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_length = len(generated_ids)

    draft_time_share = draft_time_total / wall_time if wall_time > 0 else 0.0
    tps = generated_length / wall_time if wall_time > 0 else 0.0
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

    stats = {
        # Speed
        "wall_time": round(wall_time, 4),
        "tokens_per_second": round(tps, 2),
        "generated_length": generated_length,
        # Draft / Target
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate": round(accept_rate, 4),
        "target_model_forwards": target_model_forwards,
        "draft_model_forwards": draft_model_forwards,
        "draft_time_total": round(draft_time_total, 4),
        "target_time_total": round(target_time_total, 4),
        "draft_time_share": round(draft_time_share, 4),
        # Quality budget
        "token_accept": token_accept,
        "prefix_accept": prefix_accept,
        "prefix_budget_used": round(prefix_budget_used, 6),
        "window_accept_count": window_accept_count,
        "window_len": window_len,
        "top_k_accept": top_k_accept,
        "min_token_prob": min_token_prob,
        # Guard
        "guard_trigger_count": guard.trigger_count,
        "trim_count": guard.trim_count,
        "hard_trim_count": getattr(guard, "hard_trim_count", 0),
        "repetition_warning_count": getattr(guard, "repetition_warning_count", 0),
        "bracket_warning_count": getattr(guard, "bracket_warning_count", 0),
        "import_warning_count": getattr(guard, "import_warning_count", 0),
        "repair_count": repair_count,
        "consecutive_repair_count": consecutive_repair_count,
        "trim_reasons": trim_reasons,
        "repair_reasons": repair_reasons,
        "repair_records": repair_records,
        # GuardV2
        "guard_v2_high_risk_count": guard.high_risk_count if guard_v2 else 0,
        "guard_v2_medium_risk_count": guard.medium_risk_count if guard_v2 else 0,
        # Multi-block draft
        "requested_draft_blocks": requested_draft_blocks,
        "actual_draft_blocks": actual_draft_blocks_list,
        "early_stop_reasons": early_stop_reasons,
        "block_accept_count": block_accept_count,
        "block_partial_accept_count": block_partial_accept_count,
        "block_reject_count": block_reject_count,
        "first_reject_block_id": first_reject_block_id,
        # EOS
        "eos_drafted": eos_drafted,
        "eos_accepted": eos_accepted,
        "stop_reason": stop_reason,
        # Prompt seed
        "prompt_seed_count": prompt_seed_count,
        "invalid_seed": invalid_seed,
        # Exception
        "failed": failed,
        "error_type": error_type,
        "error_msg": error_msg,
        # Adaptive policy
        "adaptive_topk_computed": adaptive_topk_computed,
        "adaptive_policy_summary": adaptive_policy.get_summary() if adaptive_policy is not None else None,
        # Comment/string fallback
        "comment_string_fallback": fallback_detector.get_summary() if fallback_detector is not None else None,
        "failure_aware_fallback": failure_fallback.get_summary() if failure_fallback is not None else None,
        # Profit guard
        "profit_guard": profit_guard.get_summary() if profit_guard is not None else None,
        # Profit-aware switch (TASD-FG-P)
        "profit_aware_switch": profit_switch.get_summary() if profit_switch is not None else None,
    }

    return {
        "generated_text": generated_text,
        "tokens_per_second": round(tps, 2),
        "generated_tokens": generated_length,
        "elapsed_time": round(wall_time, 4),
        "stats": stats,
        "guard_v2_enabled": guard_v2,
    }

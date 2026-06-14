"""
Structural Guard for TASD.
Rule-based guard that checks structural risks in generated code blocks.
Supports: argparse, dict_config, openmmlab_config

Guard-v1.5 calibration: reduces over-trimming by downgrading 3 rules.
- repetition → warning (no trim)
- unbalanced_brackets → delayed trim (depth>3 & 2+ consecutive rounds)
- off_structure:import → warning on DictConfig, hard trim elsewhere
- duplicate_option → hard trim only for argparse
"""
import re


class StructuralGuard:
    """Rule-based structural guard for speculative decoding."""

    def __init__(self, structure_type="argparse", calibrated=False):
        self.structure_type = structure_type
        self.calibrated = calibrated
        self.trigger_count = 0
        self.trim_count = 0
        # Warning counters (always active, populated by calibrated rules)
        self.repetition_warning_count = 0
        self.bracket_warning_count = 0
        self.import_warning_count = 0
        self.hard_trim_count = 0
        # State for delayed unbalanced_brackets rule
        self._bracket_depth = 0
        self._consecutive_unbalanced_rounds = 0

    def check(self, text, tokens=None, tokenizer=None):
        """
        Check text for structural risks.
        Returns: (safe, trim_token_count, risk_type)
        """
        if self.structure_type == "argparse":
            return self._check_argparse(text, tokens, tokenizer)
        elif self.structure_type == "dict_config":
            return self._check_dict_config(text, tokens, tokenizer)
        elif self.structure_type == "openmmlab_config":
            return self._check_openmmlab(text, tokens, tokenizer)
        return True, len(tokens) if tokens else 0, None

    def _safe_line_trim(self, text, char_pos, tokens=None, tokenizer=None):
        """Trim to the last safe line boundary before char_pos."""
        if tokens is None or tokenizer is None:
            return max(1, int(char_pos / max(len(text), 1) * len(tokens))) if tokens else 0
        safe_count = 0
        accumulated = ""
        for i, tok in enumerate(tokens):
            decoded = tokenizer.decode([tok], skip_special_tokens=True)
            accumulated += decoded
            if len(accumulated) > char_pos:
                break
            safe_count = i + 1
        return safe_count

    # ============================================================
    # argparse guard (v1.5 calibrated: duplicate_option → warning, off_structure → hard trim)
    # Calibration: duplicate_option rule is too aggressive — in actual argparse
    # completions, option names appear in help strings, comments, and conditional
    # logic. Hard-trimming on every duplicate causes catastrophic speed regression.
    # Downgraded to warning (increments counter, no trim).
    # ============================================================
    def _check_argparse(self, text, tokens=None, tokenizer=None):
        # Duplicate option
        options = re.findall(r'--[\w-]+', text)
        seen = set()
        for opt in options:
            if opt in seen:
                if self.calibrated:
                    # Warning only — do NOT trim
                    self.trigger_count += 1
                    self.repetition_warning_count += 1
                    break  # count once per check
                else:
                    first_pos = text.find(opt)
                    second_pos = text.find(opt, first_pos + len(opt))
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_count = self._safe_line_trim(text, second_pos, tokens, tokenizer)
                    return False, safe_count, f"duplicate_option:{opt}"
            seen.add(opt)

        # Off-structure: def/class/import — always hard trim (genuine off-structure)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class |import |from .+ import)', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                self.hard_trim_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"

        return True, len(tokens) if tokens else 0, None

    # ============================================================
    # dict_config guard (v1.5 calibrated)
    # Calibration changes:
    #   1. repetition → warning (repetition_warning_count++, no trim)
    #   2. unbalanced_brackets → delayed (depth>3 & 2+ consecutive rounds)
    #   3. off_structure:import → warning for DictConfig (no trim)
    #   4. off_structure:def/class → hard trim (unchanged)
    # ============================================================
    def _check_dict_config(self, text, tokens=None, tokenizer=None):
        # --- Off-structure def/class (hard trim — always safe) ---
        lines = text.split("\n")
        stripped_lines = [l.strip() for l in lines]
        if self.calibrated:
            # Calibrated: split def/class (hard) from import/from (warning)
            for i, stripped in enumerate(stripped_lines):
                if re.match(r'^(def |class )\w+', stripped):
                    pos = text.find(stripped)
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                    return False, safe_count, f"off_structure:{stripped[:30]}"
                if re.match(r'^(import |from .+ import)', stripped):
                    # DictConfig: downgrade to warning
                    self.trigger_count += 1
                    self.import_warning_count += 1
                    # Do NOT trim — just record warning
        else:
            # Original behavior: all off-structure are hard trim
            for i, stripped in enumerate(stripped_lines):
                if re.match(r'^(def |class |import |from .+ import)', stripped):
                    pos = text.find(stripped)
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                    return False, safe_count, f"off_structure:{stripped[:30]}"

        # --- Bracket balance (calibrated: delayed trim) ---
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        open_parens = text.count("(") - text.count(")")
        self._bracket_depth = max(open_braces, open_brackets, open_parens)

        if open_braces < 0 or open_brackets < 0 or open_parens < 0:
            self._consecutive_unbalanced_rounds += 1
            if self.calibrated:
                # Delayed trim: only trigger if depth>3 AND 2+ consecutive rounds
                if self._bracket_depth > 3 and self._consecutive_unbalanced_rounds >= 2:
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_text = text.rstrip()
                    last_newline = safe_text.rfind("\n")
                    safe_count = self._safe_line_trim(text, last_newline if last_newline > 0 else 0,
                                                      tokens, tokenizer)
                    return False, safe_count, "unbalanced_brackets:delayed"
                else:
                    # Warning only
                    self.trigger_count += 1
                    self.bracket_warning_count += 1
            else:
                # Original behavior: immediate trim
                self.trigger_count += 1
                self.hard_trim_count += 1
                safe_text = text.rstrip()
                last_newline = safe_text.rfind("\n")
                safe_count = self._safe_line_trim(text, last_newline if last_newline > 0 else 0,
                                                  tokens, tokenizer)
                return False, safe_count, "unbalanced_brackets"
        else:
            # Brackets balanced this round → reset consecutive counter
            self._consecutive_unbalanced_rounds = 0

        # --- Repetition (calibrated: warning only) ---
        words = text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2] and len(words[i]) > 2:
                if self.calibrated:
                    # Warning only — do NOT trim
                    self.trigger_count += 1
                    self.repetition_warning_count += 1
                    break  # Only count once per check call
                else:
                    # Original behavior: hard trim
                    pos = text.find(words[i], text.find(words[i]) + 1)
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                    return False, safe_count, f"repeated_word:{words[i]}"

        return True, len(tokens) if tokens else 0, None

    # ============================================================
    # openmmlab guard (v1.5 calibrated)
    # Calibration changes:
    #   - def/class: hard trim (genuine off-structure in OpenMMLab configs)
    #   - import/from: calibrated → warning only (may appear in comments, strings, or top-level)
    # ============================================================
    def _check_openmmlab(self, text, tokens=None, tokenizer=None):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class )\w+', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                self.hard_trim_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"
            if re.match(r'^(import |from .+ import)', stripped):
                if self.calibrated:
                    # Warning only — do NOT trim
                    self.trigger_count += 1
                    self.import_warning_count += 1
                else:
                    pos = text.find(stripped)
                    self.trigger_count += 1
                    self.hard_trim_count += 1
                    safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                    return False, safe_count, f"off_structure:{stripped[:30]}"

        return True, len(tokens) if tokens else 0, None

    def get_stats(self):
        return {
            "guard_trigger_count": self.trigger_count,
            "trim_count": self.trim_count,
            "hard_trim_count": self.hard_trim_count,
            "repetition_warning_count": self.repetition_warning_count,
            "bracket_warning_count": self.bracket_warning_count,
            "import_warning_count": self.import_warning_count,
        }

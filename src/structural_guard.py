"""
Structural Guard for TASD.
Rule-based guard that checks structural risks in generated code blocks.
Supports: argparse, dict_config, openmmlab_config

Each guard policy is tuned for its benchmark type to avoid over-trimming.
"""
import re


class StructuralGuard:
    """Rule-based structural guard for speculative decoding."""

    def __init__(self, structure_type="argparse"):
        self.structure_type = structure_type
        self.trigger_count = 0
        self.trim_count = 0

    def check(self, text, tokens=None, tokenizer=None):
        """
        Check text for structural risks.

        Args:
            text: decoded text string
            tokens: list of token IDs (optional, for precise trim)
            tokenizer: tokenizer (optional, for token-level trim)

        Returns:
            (safe, trim_token_count, risk_type)
            - safe: True if text is structurally safe
            - trim_token_count: number of tokens to keep (0 means reject all)
            - risk_type: description of the risk found
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
            # Fallback: return approximate token count
            return max(1, int(char_pos / max(len(text), 1) * len(tokens))) if tokens else 0

        # Decode tokens one by one to find the last token before char_pos
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
    # argparse guard
    # Primary risk: duplicate --option
    # Policy: trim to before the duplicate, not the whole block
    # ============================================================
    def _check_argparse(self, text, tokens=None, tokenizer=None):
        # Find all --option patterns
        options = re.findall(r'--[\w-]+', text)
        seen = set()
        for opt in options:
            if opt in seen:
                # Find position of the duplicate occurrence
                # Find the second occurrence
                first_pos = text.find(opt)
                second_pos = text.find(opt, first_pos + len(opt))
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, second_pos, tokens, tokenizer)
                return False, safe_count, f"duplicate_option:{opt}"
            seen.add(opt)

        # Off-structure: def/class/import at line start (not inside add_argument call)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class |import |from .+ import)', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"

        return True, len(tokens) if tokens else 0, None

    # ============================================================
    # dict_config guard
    # Primary risks: off-structure, unbalanced brackets, repeated words
    # Policy: safe-line trim for off-structure, reject-all for severe issues
    # ============================================================
    def _check_dict_config(self, text, tokens=None, tokenizer=None):
        # Off-structure check: def/class/import at line start
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class |import |from .+ import)', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"

        # Bracket balance: more closing than opening is a risk
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        open_parens = text.count("(") - text.count(")")

        if open_braces < 0 or open_brackets < 0 or open_parens < 0:
            # Find where it becomes unbalanced - trim to last safe line
            self.trigger_count += 1
            # Conservative: keep only up to the last complete line
            safe_text = text.rstrip()
            last_newline = safe_text.rfind("\n")
            if last_newline > 0:
                safe_count = self._safe_line_trim(text, last_newline, tokens, tokenizer)
            else:
                safe_count = 0
            return False, safe_count, "unbalanced_brackets"

        # Consecutive repeated word check (3+ identical tokens in a row)
        words = text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2] and len(words[i]) > 2:
                pos = text.find(words[i], text.find(words[i]) + 1)
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"repeated_word:{words[i]}"

        return True, len(tokens) if tokens else 0, None

    # ============================================================
    # openmmlab guard
    # Policy: extremely conservative, only block obvious def/class/import
    # Avoid over-trimming on dict() calls or config values
    # ============================================================
    def _check_openmmlab(self, text, tokens=None, tokenizer=None):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only flag actual function/class definitions, not dict() calls
            if re.match(r'^(def |class )\w+', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"
            if re.match(r'^(import |from .+ import)', stripped):
                pos = text.find(stripped)
                self.trigger_count += 1
                safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                return False, safe_count, f"off_structure:{stripped[:30]}"

        return True, len(tokens) if tokens else 0, None

    def get_stats(self):
        return {
            "guard_trigger_count": self.trigger_count,
            "trim_count": self.trim_count,
        }

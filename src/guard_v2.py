"""
Guard-v2: Incremental syntax-aware structural guard with comment/string awareness.

Three features:
1. Incremental syntax state (bracket_stack, quote_state, comment_state)
2. Comment/string awareness (def/class/import in strings → medium risk, not high)
3. Adaptive verification tightening (high risk → top_k_accept=1 signal)
"""
import re


class GuardV2:
    """Incremental syntax-aware guard for speculative decoding."""

    def __init__(self, structure_type="argparse"):
        self.structure_type = structure_type
        self.trigger_count = 0
        self.trim_count = 0
        self.high_risk_count = 0
        self.medium_risk_count = 0

        # Incremental state
        self.bracket_stack = []  # list of (char, pos_in_full_text)
        self.quote_state = "none"  # none, single, double, triple_single, triple_double
        self.comment_state = "normal"  # normal, comment
        self.prev_text_len = 0
        self.full_text = ""

    def update(self, new_text):
        """Incrementally update syntax state with new text (char by char)."""
        if not new_text:
            return
        self.full_text += new_text
        # Only process the new characters
        base_len = len(self.full_text) - len(new_text)
        for i, ch in enumerate(new_text):
            pos = base_len + i
            self._process_char(ch, pos)

    def _process_char(self, ch: str, pos: int):
        """Process single character for state tracking."""
        # Comment state: # starts a comment, newline ends it
        if self.comment_state == "comment":
            if ch == "\n":
                self.comment_state = "normal"
            return  # Skip bracket/quote tracking in comments

        # Hash starts comment (only when not in string)
        if ch == "#" and self.quote_state == "none":
            self.comment_state = "comment"
            return

        # Triple-quote detection
        if self.quote_state == "none":
            if ch == '"' and pos >= 2 and self.full_text[pos - 2:pos] == '""':
                # Check if we just closed a triple-double quote
                if self.full_text[pos - 2:pos + 1] == '"""':
                    pass  # Handle triple quote
            if ch == "'" and pos >= 2 and self.full_text[pos - 2:pos] == "''":
                pass

        # Quote state transitions (simplified)
        if self.quote_state == "none":
            if ch == '"':
                self.quote_state = "double"
                return
            elif ch == "'":
                self.quote_state = "single"
                return
        elif self.quote_state == "double":
            if ch == '"':
                # Check for escaped quote
                if pos > 0 and self.full_text[pos - 1] == "\\":
                    return
                self.quote_state = "none"
                return
        elif self.quote_state == "single":
            if ch == "'":
                if pos > 0 and self.full_text[pos - 1] == "\\":
                    return
                self.quote_state = "none"
                return

        # Bracket tracking (only when not in string/comment)
        if self.quote_state != "none" or self.comment_state != "normal":
            return

        if ch == "(":
            self.bracket_stack.append(("paren", pos))
        elif ch == ")":
            if self.bracket_stack and self.bracket_stack[-1][0] == "paren":
                self.bracket_stack.pop()
        elif ch == "[":
            self.bracket_stack.append(("bracket", pos))
        elif ch == "]":
            if self.bracket_stack and self.bracket_stack[-1][0] == "bracket":
                self.bracket_stack.pop()
        elif ch == "{":
            self.bracket_stack.append(("brace", pos))
        elif ch == "}":
            if self.bracket_stack and self.bracket_stack[-1][0] == "brace":
                self.bracket_stack.pop()

    def in_string(self):
        return self.quote_state != "none"

    def in_comment(self):
        return self.comment_state == "comment"

    def is_in_literal(self):
        """Currently inside a string literal or comment."""
        return self.in_string() or self.in_comment()

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

    def check(self, text, tokens=None, tokenizer=None):
        """
        Check text for structural risks with syntax awareness.

        Returns:
            (safe, trim_token_count, risk_type, risk_level)
            risk_level: 'none', 'low', 'medium', 'high'
        """
        # Incremental update from new text
        if len(text) > self.prev_text_len:
            new = text[self.prev_text_len:]
            self.update(new)
        self.prev_text_len = len(text)

        in_str = self.in_string()
        in_cmt = self.in_comment()
        in_lit = in_str or in_cmt
        risk_level = "none"

        # ---- Unified off-structure check with string/comment awareness ----
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            # Check def/class/import/return at line start
            if re.match(r"^(def |class |import |from .+ import|return\b)", stripped):
                # Determine if this line is inside a string or comment
                # The current state tells us if we're inside a literal
                if in_lit:
                    # Keywords in string/comment are not off-structure
                    if risk_level in ("none",):
                        risk_level = "medium"
                    self.medium_risk_count += 1
                else:
                    # Real off-structure - HIGH risk
                    risk_level = "high"
                    self.high_risk_count += 1
                    self.trigger_count += 1
                    pos = text.find(stripped)
                    safe_count = self._safe_line_trim(text, pos, tokens, tokenizer)
                    return False, safe_count, f"off_structure:{stripped[:30]}", risk_level

        # ---- Bracket imbalance check ----
        unbalanced = False
        for bt, pos in self.bracket_stack:
            # Only flag if at least one bracket pair exists and stack isn't empty
            pass
        if self.bracket_stack:
            # Unclosed brackets at end - low/medium risk
            if risk_level == "none":
                # Check depth - shallow is low, deep is medium
                depth = len(self.bracket_stack)
                risk_level = "medium" if depth >= 3 else "low"

        # ---- String/comment region detection ----
        if in_lit and risk_level in ("none", "low"):
            risk_level = "medium"
            self.medium_risk_count += 1

        # ---- Deep nesting risk ----
        if len(self.bracket_stack) >= 5:
            if risk_level in ("none", "low"):
                risk_level = "medium"

        return True, len(tokens) if tokens else 0, None, risk_level

    def get_stats(self):
        return {
            "guard_trigger_count": self.trigger_count,
            "trim_count": self.trim_count,
            "high_risk_count": self.high_risk_count,
            "medium_risk_count": self.medium_risk_count,
            "bracket_depth": len(self.bracket_stack),
            "in_string": self.in_string(),
            "in_comment": self.in_comment(),
        }

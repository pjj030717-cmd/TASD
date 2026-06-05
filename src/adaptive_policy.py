"""
Structure-aware adaptive decoding policy for TASD.

Dynamically adjusts draft_len and top_k_accept based on rolling window
statistics of draft-target alignment.

Maintains a rolling window of recent rounds and applies heuristic
rules to tune decoding parameters for speed/quality trade-off.
"""


class AdaptivePolicyV2:
    """Adaptive TASD v2: refined ruleset.

    Rules:
      1. draft_len:
         - <0.75 accept → 8
         - >=0.98 accept AND no recent repairs AND no recent guard triggers → 20
         - else → 16
      2. top_k_accept:
         - accept < 0.90 AND (top5 - top3) >= 0.08 → 5
         - else → 3
      3. Fallback: if last round was draft_len=20 but TPS didn't improve
         or accepted/drafted < 0.95, revert to 16.
      4. Guard constraint: if guard triggered this round, next round
         draft_len <= 16, top_k_accept = 3.
    """

    def __init__(self, rolling_window=3):
        self.rolling_window = rolling_window
        self.draft_len = 16
        self.top_k_accept = 3

        self.round_drafted = []
        self.round_accepted = []
        self.round_top3_hits = []
        self.round_top5_hits = []
        self.round_repair = []
        self.round_guard_triggers = []
        self.round_off_structure = []
        self.round_tps = []  # per-round TPS for fallback check

        self.draft_len_history = []
        self.top_k_history = []
        self.rolling_accept_rate_history = []
        self.rolling_top3_rate_history = []
        self.rolling_top5_rate_history = []

        self.k5_round_count = 0
        self.short_draft_round_count = 0
        self.long_draft_round_count = 0

        self._last_was_long = False  # track last round was draft_len=20
        self._prev_draft_len = 16
        self._guard_triggered_last = False

    def get_params(self):
        if len(self.round_drafted) < 2:
            self._record_history()
            return 16, 3

        window_drafted = sum(self.round_drafted[-self.rolling_window:])
        window_accepted = sum(self.round_accepted[-self.rolling_window:])
        window_top3 = sum(self.round_top3_hits[-self.rolling_window:])
        window_top5 = sum(self.round_top5_hits[-self.rolling_window:])
        recent_repairs = sum(self.round_repair[-self.rolling_window:])
        recent_guard = sum(self.round_guard_triggers[-self.rolling_window:])
        recent_off = sum(1 for o in self.round_off_structure[-self.rolling_window:] if o)

        roll_accept = window_accepted / window_drafted if window_drafted > 0 else 0.0
        roll_top3 = window_top3 / window_drafted if window_drafted > 0 else 0.0
        roll_top5 = window_top5 / window_drafted if window_drafted > 0 else 0.0

        # Rule 1: draft_len
        if roll_accept < 0.75:
            self.draft_len = 8
        elif (roll_accept >= 0.98 and recent_repairs == 0 and recent_guard == 0
              and recent_off == 0):
            self.draft_len = 20
        else:
            self.draft_len = 16

        # Rule 3: fallback if last round was long but inefficient
        if self._last_was_long:
            last_accepted = self.round_accepted[-1] if self.round_accepted else 0
            last_drafted = self.round_drafted[-1] if self.round_drafted else 1
            last_round_ratio = last_accepted / last_drafted if last_drafted > 0 else 0.0
            if last_round_ratio < 0.95:
                self.draft_len = 16

        # Rule 4: guard constraint
        if self._guard_triggered_last:
            self.draft_len = min(self.draft_len, 16)
            self.top_k_accept = 3
        else:
            # Rule 2: top_k_accept
            top_gap = roll_top5 - roll_top3
            if roll_accept < 0.90 and top_gap >= 0.08:
                self.top_k_accept = 5
            else:
                self.top_k_accept = 3

        self._record_history()

        return self.draft_len, self.top_k_accept

    def record_round(self, *, drafted, accepted, top3_hits=0, top5_hits=0,
                     repair_count=0, guard_triggers=0, off_structure=False,
                     tps=0.0):
        self.round_drafted.append(drafted)
        self.round_accepted.append(accepted)
        self.round_top3_hits.append(top3_hits)
        self.round_top5_hits.append(top5_hits)
        self.round_repair.append(repair_count)
        self.round_guard_triggers.append(guard_triggers)
        self.round_off_structure.append(off_structure)
        self.round_tps.append(tps)

        self._prev_draft_len = self.draft_len
        self._last_was_long = (self.draft_len == 20)
        self._guard_triggered_last = (guard_triggers > 0)

    def _record_history(self):
        self.draft_len_history.append(self.draft_len)
        self.top_k_history.append(self.top_k_accept)
        if self.top_k_accept == 5:
            self.k5_round_count += 1
        if self.draft_len == 8:
            self.short_draft_round_count += 1
        elif self.draft_len == 20:
            self.long_draft_round_count += 1

    def get_summary(self):
        return {
            "draft_len_history": self.draft_len_history,
            "top_k_history": self.top_k_history,
            "rolling_accept_rate_history": self.rolling_accept_rate_history,
            "rolling_top3_rate_history": self.rolling_top3_rate_history,
            "rolling_top5_rate_history": self.rolling_top5_rate_history,
            "adaptive_change_count": sum(
                1 for i in range(1, len(self.draft_len_history))
                if self.draft_len_history[i] != self.draft_len_history[i - 1]
                or self.top_k_history[i] != self.top_k_history[i - 1]
            ),
            "k5_round_count": self.k5_round_count,
            "short_draft_round_count": self.short_draft_round_count,
            "long_draft_round_count": self.long_draft_round_count,
            "average_draft_len": sum(self.draft_len_history) / len(self.draft_len_history)
                if self.draft_len_history else 16.0,
        }

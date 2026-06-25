"""
Official TASD-BR rerun policy.

This is the EXACT same logic that produces the 13.5% rerun ratio
and 84.4% recoverability in Table 1 of the final paper.

Do NOT modify or re-implement in blind review scripts.
Import from here only.
"""


def is_br_rerun(fg_sample):
    """
    Determine if TASD-FG output should trigger AR rerun under BR policy.

    This is the official rule from run_final_paper_tables.py:generate_br_report().
    Produces exactly 65/480 = 13.5% rerun rate.

    Args:
        fg_sample: dict with keys 'bracket_balance' (float) and 'is_truncated' (int)

    Returns:
        bool: True if rerun should be triggered
    """
    bb = fg_sample.get('bracket_balance', 1.0)
    trunc = fg_sample.get('is_truncated', 0)
    return bb < 0.50 and trunc == 0


def br_decode(fg_text, ar_text, fg_sample):
    """
    Return the final TASD-BR output text.

    Uses the official BR rerun policy:
    - High risk (bracket_balance < 0.50 and not truncated): return AR text
    - Low risk: return TASD-FG text

    Args:
        fg_text: TASD-FG generated text
        ar_text: AR generated text
        fg_sample: dict with bracket_balance and is_truncated

    Returns:
        str: final BR output text
        bool: whether rerun was triggered
    """
    if is_br_rerun(fg_sample):
        return ar_text, True
    return fg_text, False

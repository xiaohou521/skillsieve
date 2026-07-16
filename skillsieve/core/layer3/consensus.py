"""Layer 3 consensus logic: aggregate juror verdicts across debate rounds."""
from __future__ import annotations


def compute_consensus(juror_results: list[dict], round_num: int = 1) -> dict:
    """Compute a consensus verdict from a list of juror result dicts.

    Parameters
    ----------
    juror_results:
        Each dict must contain at least a ``"verdict"`` key whose value is a
        case-insensitive string ``"MALICIOUS"`` or ``"SAFE"``/anything else.
    round_num:
        ``1`` for the initial independent round, ``2`` for the post-debate
        round.  The escalation rules differ between rounds.

    Returns
    -------
    dict
        One of:

        * ``{"verdict": "MALICIOUS"|"SAFE", "consensus": "unanimous", "round": round_num}``
          — all jurors agree.
        * ``{"needs_debate": True, "juror_results": juror_results}``
          — round-1 split; caller should run a debate round.
        * ``{"verdict": "MALICIOUS", "consensus": "majority", "round": 2}``
          — round-2 majority (>=2 of 3 MALICIOUS).
        * ``{"verdict": "SAFE", "consensus": "contested", "round": 2,
             "recommend_human_review": True}``
          — round-2 no majority; default to SAFE but flag for human review.
    """
    malicious_count = sum(
        1
        for r in juror_results
        if str(r.get("verdict", "")).upper() == "MALICIOUS"
    )
    total = len(juror_results)

    # --- Unanimous ---
    if malicious_count == total:
        return {"verdict": "MALICIOUS", "consensus": "unanimous", "round": round_num}
    if malicious_count == 0:
        return {"verdict": "SAFE", "consensus": "unanimous", "round": round_num}

    # --- Split ---
    if round_num == 1:
        return {"needs_debate": True, "juror_results": juror_results}

    # --- Round 2 majority (>=2 out of 3 MALICIOUS) ---
    if malicious_count >= 2:
        return {"verdict": "MALICIOUS", "consensus": "majority", "round": 2}

    # --- Round 2: no majority → contested MALICIOUS ---
    # Conservative policy: contested cases are routed to human review and held
    # as MALICIOUS pending confirmation. This matches the paper's threat-model
    # framing that a missed malicious skill costs far more than one extra
    # manual review.
    return {
        "verdict": "MALICIOUS",
        "consensus": "contested",
        "round": 2,
        "recommend_human_review": True,
    }

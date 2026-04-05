"""
app/utils/financial_utils.py
-----------------------------
Extract a rich query string from a user's financial portfolio JSON.

The portfolio can contain any combination of:
  DEPOSIT_V2, EQUITIES, MUTUALFUNDS, INVIT, REIT, SIP, INSURANCE_POLICIES

From each section we pull names, sectors, asset types, and values so we can
build a meaningful embedding query for news recommendation.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _safe(val: Any, default: str = "") -> str:
    return str(val).strip() if val not in (None, "", "null") else default


def _extract_equities(data: Dict) -> List[str]:
    """Pull stock names and descriptions from EQUITIES section."""
    items = []
    for key, account in data.items():
        try:
            holdings = (
                account.get("summary", {})
                .get("investment", {})
                .get("holdings", {})
                .get("holding", [])
            )
            if isinstance(holdings, dict):
                holdings = [holdings]
            for h in holdings:
                name = _safe(h.get("issuerName"))
                desc = _safe(h.get("description"))
                if name:
                    items.append(name)
                if desc:
                    items.append(desc)
        except Exception:
            pass
    return items


def _extract_mutual_funds(data: Dict) -> List[str]:
    """Pull fund names from MUTUALFUNDS / SIP sections."""
    items = []
    for key, account in data.items():
        try:
            holdings = (
                account.get("summary", {})
                .get("investment", {})
                .get("holdings", {})
                .get("holding", [])
            )
            if isinstance(holdings, dict):
                holdings = [holdings]
            for h in holdings:
                name = _safe(h.get("amc") or h.get("issuerName"))
                scheme = _safe(h.get("schemeCode"))
                if name:
                    items.append(name)
                if scheme:
                    items.append(scheme)
        except Exception:
            pass
    return items


def _extract_deposits(data: Dict) -> List[str]:
    """Pull deposit type/facility info from DEPOSIT_V2."""
    items = []
    try:
        for summary_item in data.get("summary", []):
            acc_type = _safe(summary_item.get("accountType"))
            facility = _safe(summary_item.get("facility"))
            desc     = _safe(summary_item.get("description"))
            if acc_type:
                items.append(f"{acc_type} account")
            if facility and facility != "null":
                items.append(facility)
            if desc:
                items.append(desc)
    except Exception:
        pass
    return items


def _extract_reits_invits(data: Dict) -> List[str]:
    """Pull REIT / InvIT holding names."""
    items = []
    for key, account in data.items():
        try:
            holdings = (
                account.get("summary", {})
                .get("investment", {})
                .get("holdings", {})
                .get("holding", [])
            )
            if isinstance(holdings, dict):
                holdings = [holdings]
            for h in holdings:
                name = _safe(h.get("issuerName"))
                if name:
                    items.append(name)
        except Exception:
            pass
    return items


def _extract_insurance(data: Dict) -> List[str]:
    """Pull policy type and name from INSURANCE_POLICIES."""
    items = []
    for key, account in data.items():
        try:
            summary = account.get("summary", {})
            policy_name = _safe(summary.get("policyName"))
            policy_type = _safe(summary.get("policyType"))
            cover_type  = _safe(summary.get("coverType"))
            if policy_name:
                items.append(policy_name)
            if policy_type:
                items.append(policy_type.replace("_", " ").lower())
            if cover_type:
                items.append(cover_type)
        except Exception:
            pass
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SECTION_HANDLERS = {
    "EQUITIES":           _extract_equities,
    "MUTUALFUNDS":        _extract_mutual_funds,
    "SIP":                _extract_mutual_funds,   # same structure
    "REIT":               _extract_reits_invits,
    "INVIT":              _extract_reits_invits,
    "INSURANCE_POLICIES": _extract_insurance,
}


def build_portfolio_query_text(
    portfolio: Dict[str, Any],
    extra_interests: str = "",
) -> str:
    """
    Convert a financial portfolio JSON into a descriptive query string
    suitable for embedding and Qdrant search.

    Example output:
      "SAVINGS account OVERDRAFT Primary Savings Account.
       SIEMENS LIMITED Industrial Equipment. ICICI BANK LTD Private Bank.
       Bandhan Sterling Value Fund-Growth Bandhan Mutual Fund.
       POWERGRID INFRASTRUCTURE INVESTMENT TRUST.
       MINDSPACE BUSINESS PARKS REIT.
       Family Protector Plan term plan LIFE.
       Indian equities mutual funds real estate investment insurance finance"
    """
    parts: List[str] = []

    # Section-specific extraction
    for section_key, handler in SECTION_HANDLERS.items():
        if section_key in portfolio:
            extracted = handler(portfolio[section_key])
            parts.extend(extracted)

    # DEPOSIT_V2 has a different structure (summary is at top level)
    if "DEPOSIT_V2" in portfolio:
        extracted = _extract_deposits(portfolio["DEPOSIT_V2"])
        parts.extend(extracted)

    # Infer broad investment themes for semantic breadth
    themes: List[str] = []
    if "EQUITIES" in portfolio:
        themes.append("Indian stock market equities shares")
    if "MUTUALFUNDS" in portfolio or "SIP" in portfolio:
        themes.append("mutual funds SIP investment")
    if "REIT" in portfolio:
        themes.append("real estate investment trust REIT")
    if "INVIT" in portfolio:
        themes.append("infrastructure investment trust InvIT")
    if "INSURANCE_POLICIES" in portfolio:
        themes.append("insurance life cover financial planning")
    if "DEPOSIT_V2" in portfolio:
        themes.append("bank savings deposit interest rate")

    parts.extend(themes)

    if extra_interests:
        parts.append(extra_interests.strip())

    # Deduplicate while preserving order
    seen = set()
    unique_parts = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return ". ".join(unique_parts)


def summarise_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a brief structured summary of the portfolio for logging/response.
    """
    summary: Dict[str, Any] = {}

    if "DEPOSIT_V2" in portfolio:
        summaries = portfolio["DEPOSIT_V2"].get("summary", [])
        summary["deposits"] = len(summaries)

    for section in ("EQUITIES", "MUTUALFUNDS", "SIP", "REIT", "INVIT", "INSURANCE_POLICIES"):
        if section in portfolio:
            summary[section.lower()] = len(portfolio[section])

    return summary

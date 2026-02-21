"""
RetireLab — Monte Carlo Retirement Planning Simulator
Author: Scott Wallsten
Disclaimer: This tool is for educational and informational purposes only.
It does not constitute financial, tax, or investment advice. Consult a
qualified financial advisor before making retirement planning decisions.
"""

# Built with Streamlit 1.50+ (st.navigation, st.segmented_control, st.dialog)

# ============================================================
# SECTION 0: Imports + page_config
# ============================================================
from __future__ import annotations

import copy
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="RetireLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SECTION 1: Constants
# ============================================================
ASSET_CLASSES = ["Equities", "REIT", "Bonds", "Alternatives", "Cash"]

# Market outlooks — named return/vol assumptions for asset classes
DEFAULT_SCENARIOS = {
    "Catastrophic": {"eq_mu": 0.00, "eq_vol": 0.35, "reit_mu": 0.01, "reit_vol": 0.30, "bond_mu": 0.01, "bond_vol": 0.10, "alt_mu": 0.01, "alt_vol": 0.20, "cash_mu": 0.02},
    "Bad":         {"eq_mu": 0.04, "eq_vol": 0.22, "reit_mu": 0.04, "reit_vol": 0.20, "bond_mu": 0.02, "bond_vol": 0.08, "alt_mu": 0.03, "alt_vol": 0.18, "cash_mu": 0.02},
    "Base":        {"eq_mu": 0.07, "eq_vol": 0.16, "reit_mu": 0.07, "reit_vol": 0.18, "bond_mu": 0.03, "bond_vol": 0.06, "alt_mu": 0.05, "alt_vol": 0.14, "cash_mu": 0.02},
    "Good":        {"eq_mu": 0.09, "eq_vol": 0.15, "reit_mu": 0.09, "reit_vol": 0.18, "bond_mu": 0.04, "bond_vol": 0.06, "alt_mu": 0.07, "alt_vol": 0.14, "cash_mu": 0.025},
    "Excellent":   {"eq_mu": 0.12, "eq_vol": 0.14, "reit_mu": 0.12, "reit_vol": 0.18, "bond_mu": 0.05, "bond_vol": 0.06, "alt_mu": 0.09, "alt_vol": 0.14, "cash_mu": 0.03},
}

RMD_FACTOR = {72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
              80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
              88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9}

BASE_CORR = {
    ("Eq", "REIT"): 0.70,
    ("Eq", "Bond"): -0.10,
    ("REIT", "Bond"): -0.05,
    ("Eq", "Alt"): 0.20,
    ("REIT", "Alt"): 0.20,
    ("Bond", "Alt"): 0.10,
}

# Regime-switching model: 3-state Markov (Bull / Normal / Bear)
REGIME_NAMES = ["Bull", "Normal", "Bear"]
DEFAULT_REGIME_PARAMS = {
    "Bull":   {"eq_mu": 0.10, "eq_vol": 0.12, "reit_mu": 0.10, "reit_vol": 0.14, "bond_mu": 0.035, "bond_vol": 0.04, "alt_mu": 0.07, "alt_vol": 0.10, "cash_mu": 0.03},
    "Normal": {"eq_mu": 0.07, "eq_vol": 0.16, "reit_mu": 0.07, "reit_vol": 0.18, "bond_mu": 0.03, "bond_vol": 0.06, "alt_mu": 0.05, "alt_vol": 0.14, "cash_mu": 0.02},
    "Bear":   {"eq_mu": -0.05, "eq_vol": 0.25, "reit_mu": -0.05, "reit_vol": 0.25, "bond_mu": 0.02, "bond_vol": 0.08, "alt_mu": -0.01, "alt_vol": 0.18, "cash_mu": 0.015},
}
# Transition matrix: rows = from-state, cols = to-state [Bull, Normal, Bear]
DEFAULT_TRANSITION_MATRIX = [
    [0.90, 0.08, 0.02],   # Bull stays Bull 90%, to Normal 8%, to Bear 2%
    [0.15, 0.75, 0.10],   # Normal to Bull 15%, stays Normal 75%, to Bear 10%
    [0.10, 0.30, 0.60],   # Bear to Bull 10%, to Normal 30%, stays Bear 60%
]
DEFAULT_REGIME_INITIAL_PROBS = [0.35, 0.50, 0.15]

# ============================================================
# SECTION 2: Utility functions
# ============================================================
def ss_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value

def fmt_dollars(x: float) -> str:
    return f"${x:,.0f}"

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(w.get(k, 0.0)) for k in ASSET_CLASSES)
    if s <= 0:
        return {k: 0.0 for k in ASSET_CLASSES}
    return {k: float(w.get(k, 0.0)) / s for k in ASSET_CLASSES}

def w_to_arr(w: Dict[str, float]) -> np.ndarray:
    w = normalize_weights(w)
    arr = np.array([w["Equities"], w["REIT"], w["Bonds"], w["Alternatives"], w["Cash"]], float)
    return arr / arr.sum()

# ---------------- CSV parsing ----------------
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def detect_value_column(df: pd.DataFrame) -> str:
    candidates = []
    for c in df.columns:
        nc = _norm(c)
        if any(k in nc for k in ["marketvalue", "value", "amount", "balance", "currentvalue", "mv"]):
            candidates.append(c)
    if candidates:
        candidates.sort(key=lambda x: (0 if "market" in _norm(x) else 1, len(x)))
        return candidates[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("Could not detect a numeric value column.")
    return num_cols[-1]

def classify_asset(row: pd.Series) -> str:
    text = " ".join(str(row.get(c, "")) for c in row.index).lower()
    if any(k in text for k in ["cash", "money market", "mmf", "govt cash", "government cash", "sweep"]):
        return "Cash"
    if any(k in text for k in ["bond", "treasury", "tips", "municipal", "muni", "fixed income", "aggregate", "inflation protected"]):
        return "Bonds"
    if "reit" in text or "real estate" in text:
        return "REIT"
    if any(k in text for k in ["managed futures", "commodity", "commodities", "trend", "buffer", "defined outcome", "vix", "long volatility", "alternative"]):
        return "Alternatives"
    return "Equities"

def _detect_ticker(row: pd.Series) -> str:
    """Try to extract a ticker symbol from a row. Looks for columns named symbol/ticker first, then
    falls back to scanning short uppercase tokens in all columns."""
    for c in row.index:
        cn = _norm(c)
        if cn in ("symbol", "ticker", "sym"):
            val = str(row[c]).strip().upper()
            if val and val not in ("NAN", "NONE", ""):
                return val
    # Fallback: scan all columns for a short uppercase token that looks like a ticker
    for c in row.index:
        val = str(row[c]).strip()
        if 1 <= len(val) <= 6 and val.isalpha() and val == val.upper():
            return val
    return ""

def classify_equity_sub(row: pd.Series) -> str:
    """Classify an equity holding into one of EQUITY_SUB_CLASSES using TICKER_HEURISTICS.
    Returns the sub-class name, or 'US Large Blend' as default for unrecognized tickers."""
    ticker = _detect_ticker(row)
    if ticker:
        for pattern, sub_class in TICKER_HEURISTICS.items():
            if re.search(r"(?:^|(?<=\|))(" + pattern + r")(?:$|(?=\|))", ticker):
                return sub_class
    # Also try name-based heuristics from the row text
    text = " ".join(str(row.get(c, "")) for c in row.index).lower()
    if any(k in text for k in ["small cap", "small-cap", "russell 2000", "smallcap"]):
        return "US Small"
    if any(k in text for k in ["international", "intl", "foreign", "developed market", "ex-us", "ex us", "ftse dev"]):
        return "Intl Developed"
    if any(k in text for k in ["emerging", "em market"]):
        return "Emerging"
    if any(k in text for k in ["value", "dividend", "high div"]):
        return "US Value"
    if any(k in text for k in ["tech", "nasdaq", "semiconductor", "ai ", "artificial intelligence", "innovation"]):
        return "Tech/AI"
    return "US Large Blend"

def equity_sub_from_holdings(df: pd.DataFrame, valcol: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Given a holdings DataFrame, classify each equity holding into a sub-bucket and compute weights.
    Returns (sub_weights_dict, classification_df_for_display).
    The classification_df has columns: Ticker, Name, Value, Sub-Class for user review."""
    tmp = df.copy()
    tmp["_class"] = tmp.apply(classify_asset, axis=1)
    eq_df = tmp[tmp["_class"] == "Equities"].copy()
    if eq_df.empty:
        return {c: EQUITY_SUB_DEFAULTS[c]["weight"] for c in EQUITY_SUB_CLASSES}, pd.DataFrame()

    eq_df["_sub_class"] = eq_df.apply(classify_equity_sub, axis=1)
    eq_df["_ticker"] = eq_df.apply(_detect_ticker, axis=1)

    # Compute dollar totals per sub-class
    by_sub = eq_df.groupby("_sub_class")[valcol].sum()
    eq_total = float(by_sub.sum())
    if eq_total <= 0:
        return {c: EQUITY_SUB_DEFAULTS[c]["weight"] for c in EQUITY_SUB_CLASSES}, pd.DataFrame()

    sub_weights = {}
    for c in EQUITY_SUB_CLASSES:
        sub_weights[c] = float(by_sub.get(c, 0.0)) / eq_total

    # Build display DataFrame for user review
    # Try to find a name/description column
    name_col = None
    for c in eq_df.columns:
        cn = _norm(c)
        if cn in ("name", "description", "desc", "holdingname", "securityname", "security", "secname", "securitydescription"):
            name_col = c
            break
    display_rows = []
    for _, r in eq_df.iterrows():
        display_rows.append({
            "Ticker": r["_ticker"] if r["_ticker"] else "—",
            "Name": str(r[name_col])[:50] if name_col and pd.notna(r.get(name_col)) else "—",
            "Value": float(r[valcol]),
            "Sub-Class": r["_sub_class"],
        })
    display_df = pd.DataFrame(display_rows)
    return sub_weights, display_df

def load_snapshot(uploaded_file):
    df = pd.read_csv(uploaded_file)
    valcol = detect_value_column(df)
    df = df.copy()
    s = df[valcol].astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\((.*)\)", r"-\1", regex=True)
    df[valcol] = pd.to_numeric(s, errors="coerce")
    df = df.dropna(subset=[valcol])
    return df, valcol

def weights_and_dollars(df: pd.DataFrame, valcol: str) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    tmp = df.copy()
    tmp["_class"] = tmp.apply(classify_asset, axis=1)
    by = tmp.groupby("_class")[valcol].sum()
    total = float(by.sum())
    if total <= 0:
        raise ValueError("Computed total of 0; check parsing.")
    dollars = {k: float(by.get(k, 0.0)) for k in ASSET_CLASSES}
    weights = {k: dollars[k] / total for k in ASSET_CLASSES}
    return weights, total, dollars

# ============================================================
# SECTION 3: Computation engine
# ============================================================
def adjusted_ss_from_62(ss62_monthly: float, claim_age: int, fra_age: int = 67) -> float:
    months_early_62 = max(0, (fra_age - 62) * 12)
    red62 = (min(36, months_early_62) * (5/9)/100) + (max(0, months_early_62 - 36) * (5/12)/100)
    ssFRA = ss62_monthly / max(1e-9, (1 - red62))
    if claim_age < fra_age:
        months_early = (fra_age - claim_age) * 12
        red = (min(36, months_early) * (5/9)/100) + (max(0, months_early - 36) * (5/12)/100)
        return ssFRA * (1 - red)
    months_late = min((claim_age - fra_age) * 12, (70 - fra_age) * 12)
    inc = months_late * (2/3)/100
    return ssFRA * (1 + inc)

def make_q_by_age(start_age: int, end_age: int, q55: float, growth: float) -> np.ndarray:
    ages = np.arange(start_age, end_age + 1)
    q = np.zeros_like(ages, dtype=float)
    for i, a in enumerate(ages):
        q[i] = min(0.50, q55 * (growth ** (a - 55)))
    return q

def draw_death_age(rng, start_age: int, end_age: int, q_by_age: np.ndarray) -> int:
    for i, a in enumerate(np.arange(start_age, end_age + 1)):
        if rng.uniform() < q_by_age[i]:
            return int(a)
    return int(end_age + 1)

def mortgage_monthly_payment(balance: float, annual_rate: float, term_years: int) -> float:
    if balance <= 0 or term_years <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = term_years * 12
    if r <= 0:
        return balance / n
    return balance * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def irmaa_monthly_per_person(magi: np.ndarray, base: float, schedule: List[Tuple[float, float]]) -> np.ndarray:
    prem = np.full_like(magi, base, dtype=float)
    for thresh, monthly in schedule:
        prem = np.where(magi >= thresh, monthly, prem)
    return prem


# ---------- Tax Engine (Phase 1) ----------

def _bracket_tax(taxable_income: np.ndarray, brackets: list,
                 infl_factor: np.ndarray) -> np.ndarray:
    """Compute tax on *taxable_income* using inflation-adjusted brackets.
    All inputs are (n,) arrays. Returns (n,) tax owed."""
    tax = np.zeros_like(taxable_income)
    for i in range(len(brackets)):
        thresh = brackets[i][0] * infl_factor
        rate = brackets[i][1]
        if i + 1 < len(brackets):
            next_thresh = brackets[i + 1][0] * infl_factor
        else:
            next_thresh = np.full_like(taxable_income, 1e15)
        bracket_income = np.clip(taxable_income - thresh, 0, next_thresh - thresh)
        tax += bracket_income * rate
    return tax


def _bracket_tax_stacked(base_income: np.ndarray, investment_income: np.ndarray,
                         brackets: list, infl_factor: np.ndarray) -> np.ndarray:
    """Compute tax on *investment_income* that stacks on top of *base_income*.
    Uses LTCG/QD bracket schedule. All (n,) arrays."""
    # Tax on (base + investment) minus tax on base alone
    total = base_income + investment_income
    tax_total = _bracket_tax(total, brackets, infl_factor)
    tax_base = _bracket_tax(base_income, brackets, infl_factor)
    return tax_total - tax_base


def _marginal_bracket(taxable_income: np.ndarray, brackets: list,
                      infl_factor: np.ndarray) -> np.ndarray:
    """Return the marginal ordinary tax rate for each simulation path."""
    rate = np.full_like(taxable_income, brackets[0][1])
    for thresh_base, r in brackets:
        thresh = thresh_base * infl_factor
        rate = np.where(taxable_income >= thresh, r, rate)
    return rate


def compute_ss_taxable(ss_gross: np.ndarray, other_income: np.ndarray,
                       filing_status: str, infl_factor: np.ndarray) -> np.ndarray:
    """Compute the taxable portion of Social Security benefits using
    the provisional-income method. All inputs (n,) arrays."""
    if filing_status == "mfj":
        t1, t2 = SS_PROVISIONAL_MFJ
    else:
        t1, t2 = SS_PROVISIONAL_SINGLE
    t1_adj = t1 * infl_factor
    t2_adj = t2 * infl_factor

    provisional = other_income + 0.5 * ss_gross

    # Tier 1: 50% of excess above t1
    tier1 = np.clip(provisional - t1_adj, 0, t2_adj - t1_adj) * 0.50
    # Tier 2: 85% of excess above t2
    tier2 = np.maximum(0.0, provisional - t2_adj) * 0.85
    taxable = tier1 + tier2
    # Cap at 85% of gross SS
    return np.minimum(taxable, 0.85 * ss_gross)


def compute_federal_tax(
    ordinary_income: np.ndarray,     # wages, trad WD, RMD (taxable), conversions, annuity
    qualified_dividends: np.ndarray, # qual divs from taxable account
    ltcg_realized: np.ndarray,       # LTCG from taxable withdrawals + gain harvesting
    ss_gross: np.ndarray,            # total SS received (before taxation calc)
    filing_status: str,              # "mfj" or "single"
    infl_factor: np.ndarray,         # inflation index for bracket adjustment
    niit_on: bool,
) -> dict:
    """Vectorized federal tax computation. All array inputs are (n,).
    Returns dict with component taxes and diagnostic fields."""
    n = len(ordinary_income)

    # 1. SS taxation via provisional income
    other_for_provisional = ordinary_income + qualified_dividends + ltcg_realized
    ss_taxable = compute_ss_taxable(ss_gross, other_for_provisional,
                                     filing_status, infl_factor)

    # 2. Total ordinary = ordinary_income + ss_taxable
    ordinary_total = ordinary_income + ss_taxable

    # 3. Standard deduction (inflation-adjusted)
    if filing_status == "mfj":
        std_ded = STANDARD_DEDUCTION_MFJ_2024 * infl_factor
        brackets_ord = FED_BRACKETS_MFJ_2024
        brackets_ltcg = LTCG_BRACKETS_MFJ_2024
        niit_thresh = NIIT_THRESHOLD_MFJ
    else:
        std_ded = STANDARD_DEDUCTION_SINGLE_2024 * infl_factor
        brackets_ord = FED_BRACKETS_SINGLE_2024
        brackets_ltcg = LTCG_BRACKETS_SINGLE_2024
        niit_thresh = NIIT_THRESHOLD_SINGLE

    # 4. Taxable ordinary income
    taxable_ordinary = np.maximum(0.0, ordinary_total - std_ded)

    # 5. Federal ordinary income tax (brackets)
    fed_ordinary_tax = _bracket_tax(taxable_ordinary, brackets_ord, infl_factor)

    # 6. LTCG/QD tax — stacks on top of ordinary taxable income
    investment_income = qualified_dividends + ltcg_realized
    fed_ltcg_tax = _bracket_tax_stacked(taxable_ordinary, investment_income,
                                         brackets_ltcg, infl_factor)

    # 7. NIIT: 3.8% on net investment income above threshold
    niit = np.zeros(n)
    if niit_on:
        magi = ordinary_total + qualified_dividends + ltcg_realized
        niit_base = np.maximum(0.0, magi - niit_thresh * infl_factor)
        niit = np.minimum(niit_base, investment_income) * NIIT_RATE

    # 8. Total federal
    fed_total = fed_ordinary_tax + fed_ltcg_tax + niit

    # 9. MAGI for IRMAA / ACA purposes
    magi = ordinary_total + qualified_dividends + ltcg_realized

    # 10. Marginal bracket & effective rate
    marginal = _marginal_bracket(taxable_ordinary, brackets_ord, infl_factor)
    total_income = taxable_ordinary + investment_income
    effective_rate = np.where(total_income > 0, fed_total / total_income, 0.0)

    return {
        "fed_ordinary_tax": fed_ordinary_tax,
        "fed_ltcg_tax": fed_ltcg_tax,
        "niit": niit,
        "fed_total": fed_total,
        "ss_taxable": ss_taxable,
        "taxable_ordinary": taxable_ordinary,
        "magi": magi,
        "marginal_bracket": marginal,
        "effective_rate": effective_rate,
        "std_deduction": std_ded,
    }


def compute_aca_premium(
    magi: np.ndarray, household_size: int,
    benchmark_premium_real: float, infl_factor: np.ndarray,
    subsidy_schedule: list,
) -> dict:
    """Compute ACA net premium using MAGI and FPL-based subsidy schedule.
    All array inputs (n,). Returns dict with gross/subsidy/net arrays."""
    fpl_base = float(FPL_2024.get(household_size, FPL_2024[2]))
    fpl = fpl_base * infl_factor
    fpl_pct = np.where(fpl > 0, magi / fpl, 9999.0)

    # Interpolate contribution rate from schedule
    contribution_rate = np.ones_like(magi)  # default: no subsidy
    for i in range(len(subsidy_schedule) - 1):
        lo_pct, lo_rate = subsidy_schedule[i]
        hi_pct, hi_rate = subsidy_schedule[i + 1]
        in_band = (fpl_pct >= lo_pct) & (fpl_pct < hi_pct)
        if in_band.any():
            frac = np.clip((fpl_pct - lo_pct) / max(0.001, hi_pct - lo_pct), 0, 1)
            contribution_rate[in_band] = (lo_rate + frac * (hi_rate - lo_rate))[in_band]
    # Below 100% FPL: Medicaid gap (no ACA subsidy in practice, treat as $0 premium)
    contribution_rate = np.where(fpl_pct < 1.0, 0.0, contribution_rate)

    household_contribution = contribution_rate * magi
    benchmark_nom = benchmark_premium_real * infl_factor
    subsidy = np.maximum(0.0, benchmark_nom - household_contribution)
    net_premium = np.maximum(0.0, benchmark_nom - subsidy)

    return {
        "gross_premium": benchmark_nom,
        "subsidy": subsidy,
        "net_premium": net_premium,
        "fpl_pct": fpl_pct,
    }

def gk_update(spend_real: np.ndarray,
              base_spend_real: float,
              port_real: np.ndarray,
              initial_wr: float,
              gk_on: bool,
              gk_upper_pct: float, gk_lower_pct: float,
              cut_pct: float, raise_pct: float) -> np.ndarray:
    if not gk_on:
        return spend_real.copy()
    ceil = initial_wr * (1 + gk_upper_pct)
    floor = initial_wr * (1 - gk_lower_pct)
    wr = np.where(port_real > 0, spend_real / port_real, 1e9)
    spend = spend_real.copy()
    spend = np.where(wr > ceil, spend * (1 - cut_pct), spend)
    spend = np.where(wr < floor, spend * (1 + raise_pct), spend)
    spend = np.clip(spend, base_spend_real * 0.80, base_spend_real * 1.20)
    return spend

def build_cov(eq_vol, reit_vol, bond_vol, alt_vol, infl_vol, home_vol,
              corr_eq_infl, corr_reit_infl, corr_bond_infl, corr_alt_infl,
              corr_home_infl, corr_home_eq) -> np.ndarray:
    vol = np.array([eq_vol, reit_vol, bond_vol, alt_vol, 0.002, infl_vol, home_vol], float)
    corr = np.eye(7, dtype=float)

    def setc(i, j, v):
        corr[i, j] = corr[j, i] = v

    setc(0, 1, BASE_CORR[("Eq", "REIT")])
    setc(0, 2, BASE_CORR[("Eq", "Bond")])
    setc(1, 2, BASE_CORR[("REIT", "Bond")])
    setc(0, 3, BASE_CORR[("Eq", "Alt")])
    setc(1, 3, BASE_CORR[("REIT", "Alt")])
    setc(2, 3, BASE_CORR[("Bond", "Alt")])

    setc(0, 5, corr_eq_infl)
    setc(1, 5, corr_reit_infl)
    setc(2, 5, corr_bond_infl)
    setc(3, 5, corr_alt_infl)

    setc(6, 5, corr_home_infl)
    setc(6, 0, corr_home_eq)

    return np.diag(vol) @ corr @ np.diag(vol)

def build_cov_granular(sub_vols: list, reit_vol, bond_vol, alt_vol, infl_vol, home_vol,
                       sub_corr: dict, sub_classes: list,
                       corr_eq_infl, corr_reit_infl, corr_bond_infl, corr_alt_infl,
                       corr_home_infl, corr_home_eq) -> np.ndarray:
    """Build expanded covariance matrix with 6 equity sub-buckets instead of 1.

    Layout: [sub0, sub1, ..., sub5, REIT, Bond, Alt, Cash, Inflation, Home] = 12 assets.
    """
    nsub = len(sub_classes)
    dim = nsub + 6  # 6 equity subs + REIT + Bond + Alt + Cash + Inflation + Home
    vol = np.zeros(dim)
    for i, v in enumerate(sub_vols):
        vol[i] = v
    vol[nsub]     = reit_vol   # REIT
    vol[nsub + 1] = bond_vol   # Bond
    vol[nsub + 2] = alt_vol    # Alt
    vol[nsub + 3] = 0.002      # Cash
    vol[nsub + 4] = infl_vol   # Inflation
    vol[nsub + 5] = home_vol   # Home

    corr = np.eye(dim, dtype=float)

    def setc(i, j, v):
        corr[i, j] = corr[j, i] = v

    # Equity sub-bucket cross-correlations
    for i in range(nsub):
        for j in range(i + 1, nsub):
            pair = (sub_classes[i], sub_classes[j])
            rpair = (sub_classes[j], sub_classes[i])
            c = sub_corr.get(pair, sub_corr.get(rpair, 0.5))
            setc(i, j, c)

    # Sub-buckets to REIT: each sub has ~same corr as aggregate Eq to REIT
    eq_reit_corr = BASE_CORR[("Eq", "REIT")]
    eq_bond_corr = BASE_CORR[("Eq", "Bond")]
    eq_alt_corr  = BASE_CORR[("Eq", "Alt")]
    for i in range(nsub):
        setc(i, nsub,     eq_reit_corr)       # sub to REIT
        setc(i, nsub + 1, eq_bond_corr)       # sub to Bond
        setc(i, nsub + 2, eq_alt_corr)        # sub to Alt
        setc(i, nsub + 4, corr_eq_infl)       # sub to Inflation
        setc(i, nsub + 5, corr_home_eq)       # sub to Home

    # REIT, Bond, Alt inter-correlations
    setc(nsub, nsub + 1, BASE_CORR[("REIT", "Bond")])
    setc(nsub, nsub + 2, BASE_CORR[("REIT", "Alt")])
    setc(nsub + 1, nsub + 2, BASE_CORR[("Bond", "Alt")])

    # REIT, Bond, Alt to Inflation
    setc(nsub,     nsub + 4, corr_reit_infl)
    setc(nsub + 1, nsub + 4, corr_bond_infl)
    setc(nsub + 2, nsub + 4, corr_alt_infl)

    # Home to Inflation, Home to REIT
    setc(nsub + 5, nsub + 4, corr_home_infl)

    # Ensure positive semi-definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    return np.diag(vol) @ corr @ np.diag(vol)


def draw_shocks(rng, n, cov, dist_type: str, t_df: int) -> np.ndarray:
    if dist_type == "t":
        z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)
        g = rng.chisquare(t_df, size=n) / t_df
        return z / np.sqrt(g)[:, None]
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)

def draw_regime_sequence(rng, n: int, T: int, transition: list, initial_probs: list) -> np.ndarray:
    """Draw (n, T) array of regime state indices (0=Bull, 1=Normal, 2=Bear)."""
    states = np.zeros((n, T), dtype=int)
    trans = np.array(transition)  # (3, 3)
    # Initial state
    states[:, 0] = rng.choice(3, size=n, p=initial_probs)
    for t in range(1, T):
        for s in range(3):
            mask = states[:, t-1] == s
            if mask.any():
                count = int(mask.sum())
                states[mask, t] = rng.choice(3, size=count, p=trans[s])
    return states


def scenario_table(scenarios: dict) -> pd.DataFrame:
    rows = []
    for name, p in scenarios.items():
        rows.append({
            "Market Outlook": name,
            "Eq mean": p["eq_mu"], "Eq vol": p["eq_vol"],
            "REIT mean": p["reit_mu"], "REIT vol": p["reit_vol"],
            "Bond mean": p["bond_mu"], "Bond vol": p["bond_vol"],
            "Alt mean": p["alt_mu"], "Alt vol": p["alt_vol"],
            "Cash mean": p["cash_mu"],
        })
    return pd.DataFrame(rows)

def objective_target(objective: str, runway_years: int = 4) -> Dict[str, float]:
    if objective == "Reduce downside risk":
        return {"Equities": 0.45, "REIT": 0.05, "Bonds": 0.40, "Alternatives": 0.05, "Cash": 0.05}
    if objective == "Reduce volatility":
        return {"Equities": 0.50, "REIT": 0.05, "Bonds": 0.40, "Alternatives": 0.05, "Cash": 0.00}
    if objective == "Reduce sequence risk (first 10y)":
        return {"Equities": 0.50, "REIT": 0.05, "Bonds": 0.35, "Alternatives": 0.05, "Cash": 0.05}
    if objective == "Maximize expected return":
        return {"Equities": 0.70, "REIT": 0.10, "Bonds": 0.15, "Alternatives": 0.05, "Cash": 0.00}
    if objective == "Improve tax location":
        return {"Equities": 0.60, "REIT": 0.08, "Bonds": 0.25, "Alternatives": 0.05, "Cash": 0.02}
    return {"Equities": 0.60, "REIT": 0.08, "Bonds": 0.25, "Alternatives": 0.05, "Cash": 0.02}

def dollars_by_bucket(total: float, w: Dict[str, float]) -> Dict[str, float]:
    return {k: total * float(w.get(k, 0.0)) for k in ASSET_CLASSES}

def combined_weights(parts: List[Tuple[float, Dict[str, float]]]) -> Dict[str, float]:
    total = sum(v for v, _ in parts)
    if total <= 0:
        return {k: 0.0 for k in ASSET_CLASSES}
    out = {}
    for k in ASSET_CLASSES:
        out[k] = sum(v * float(w.get(k, 0.0)) for v, w in parts) / total
    return out

def glide_equity_frac(age: int, eq_start: float, eq_end: float,
                      age_start: int, age_end: int) -> float:
    """Return target equity fraction at a given age via linear interpolation."""
    if age <= age_start:
        return eq_start
    if age >= age_end:
        return eq_end
    return eq_start + (eq_end - eq_start) * (age - age_start) / max(1, age_end - age_start)


def _build_glide_weights(eq_frac: float, base_weights: np.ndarray) -> np.ndarray:
    """Build a 5-element weight array for a target equity fraction,
    preserving the sub-asset ratios from the original base_weights.
    Indices: 0=Equities, 1=REIT, 2=Bonds, 3=Alternatives, 4=Cash."""
    eq_orig = base_weights[0] + base_weights[1]
    noneq_orig = base_weights[2] + base_weights[3] + base_weights[4]
    # Equity sub-split
    eq_ratio = base_weights[0] / eq_orig if eq_orig > 0 else 0.9
    # Non-equity sub-split
    if noneq_orig > 0:
        ne_ratios = base_weights[2:5] / noneq_orig
    else:
        ne_ratios = np.array([0.9, 0.0, 0.1])
    return np.array([
        eq_frac * eq_ratio,
        eq_frac * (1 - eq_ratio),
        (1 - eq_frac) * ne_ratios[0],
        (1 - eq_frac) * ne_ratios[1],
        (1 - eq_frac) * ne_ratios[2],
    ])


# 2024 Federal tax brackets (used for dynamic Roth conversions)
FED_BRACKETS_MFJ_2024 = [
    (0, 0.10), (23200, 0.12), (94300, 0.22), (201050, 0.24),
    (383900, 0.32), (487450, 0.35), (731200, 0.37),
]
FED_BRACKETS_SINGLE_2024 = [
    (0, 0.10), (11600, 0.12), (47150, 0.22), (100525, 0.24),
    (191950, 0.32), (243725, 0.35), (609350, 0.37),
]
IRMAA_MAGI_TIERS_MFJ_2024 = [206000, 258000, 322000, 386000, 750000]
IRMAA_MAGI_TIERS_SINGLE_2024 = [103000, 129000, 161000, 193000, 500000]

# 0% LTCG bracket thresholds (2024) — taxable income up to this amount pays 0% on LTCG
LTCG_0PCT_THRESHOLD_MFJ_2024 = 94050
LTCG_0PCT_THRESHOLD_SINGLE_2024 = 47025
STANDARD_DEDUCTION_MFJ_2024 = 29200
STANDARD_DEDUCTION_SINGLE_2024 = 14600

# ---------- Tax Engine Constants (Phase 1) ----------
# LTCG / Qualified Dividend brackets (2024)
LTCG_BRACKETS_MFJ_2024 = [(0, 0.00), (94050, 0.15), (583750, 0.20)]
LTCG_BRACKETS_SINGLE_2024 = [(0, 0.00), (47025, 0.15), (518900, 0.20)]

# Net Investment Income Tax (NIIT)
NIIT_THRESHOLD_MFJ = 250000
NIIT_THRESHOLD_SINGLE = 200000
NIIT_RATE = 0.038

# Social Security provisional income thresholds
SS_PROVISIONAL_MFJ = (32000, 44000)    # (50% threshold, 85% threshold)
SS_PROVISIONAL_SINGLE = (25000, 34000)

# IRMAA Part B surcharges (monthly per person, 2024 MFJ)
IRMAA_PART_B_2024_MFJ = [
    (206000, 244.60), (258000, 349.40), (322000, 454.20),
    (386000, 559.00), (750000, 594.00),
]
IRMAA_PART_B_2024_SINGLE = [
    (103000, 244.60), (129000, 349.40), (161000, 454.20),
    (193000, 559.00), (500000, 594.00),
]
# IRMAA Part D surcharges (monthly per person, 2024)
IRMAA_PART_D_2024_MFJ = [
    (206000, 12.90), (258000, 33.30), (322000, 53.80),
    (386000, 74.20), (750000, 81.00),
]
IRMAA_PART_D_2024_SINGLE = [
    (103000, 12.90), (129000, 33.30), (161000, 53.80),
    (193000, 74.20), (500000, 81.00),
]

# ACA constants (2024)
FPL_2024 = {1: 15060, 2: 20440, 3: 25820, 4: 31200}
ACA_SUBSIDY_SCHEDULE_2024 = [
    (1.00, 0.00),   # ≤100% FPL: Medicaid eligible
    (1.50, 0.02),   # 100-150%: 0-2% of income
    (2.00, 0.04),   # 150-200%: 2-4%
    (2.50, 0.065),  # 200-250%: 4-6.5%
    (3.00, 0.085),  # 250-300%: 6.5-8.5%
    (4.00, 0.085),  # 300-400%: 8.5% (cap)
    (9999, 1.00),   # >400% FPL: no subsidy
]

# Equity sub-bucket constants
EQUITY_SUB_CLASSES = ["US Large Blend", "US Small", "Intl Developed", "Emerging", "US Value", "Tech/AI"]
EQUITY_SUB_DEFAULTS = {
    "US Large Blend": {"mu": 0.07, "vol": 0.16, "weight": 0.45},
    "US Small":       {"mu": 0.08, "vol": 0.20, "weight": 0.10},
    "Intl Developed": {"mu": 0.06, "vol": 0.17, "weight": 0.20},
    "Emerging":       {"mu": 0.08, "vol": 0.24, "weight": 0.10},
    "US Value":       {"mu": 0.075, "vol": 0.17, "weight": 0.10},
    "Tech/AI":        {"mu": 0.10, "vol": 0.25, "weight": 0.05},
}
EQUITY_SUB_CORR = {
    ("US Large Blend", "US Small"): 0.85,
    ("US Large Blend", "Intl Developed"): 0.70,
    ("US Large Blend", "Emerging"): 0.60,
    ("US Large Blend", "US Value"): 0.80,
    ("US Large Blend", "Tech/AI"): 0.85,
    ("US Small", "Intl Developed"): 0.65,
    ("US Small", "Emerging"): 0.55,
    ("US Small", "US Value"): 0.80,
    ("US Small", "Tech/AI"): 0.75,
    ("Intl Developed", "Emerging"): 0.75,
    ("Intl Developed", "US Value"): 0.65,
    ("Intl Developed", "Tech/AI"): 0.60,
    ("Emerging", "US Value"): 0.55,
    ("Emerging", "Tech/AI"): 0.55,
    ("US Value", "Tech/AI"): 0.60,
}
# Ticker heuristic mapping for CSV uploads
TICKER_HEURISTICS = {
    "VTI|VTSAX|SWTSX|ITOT|SPY|IVV|VOO": "US Large Blend",
    "VB|VSMAX|SCHA|IJR|IWM": "US Small",
    "VXUS|VTIAX|IXUS|EFA|IEFA": "Intl Developed",
    "VWO|VEMAX|IEMG|EEM": "Emerging",
    "VTV|VTRIX|VONV|IWD|VLUE": "US Value",
    "QQQ|VGT|XLK|ARKK|SMH|SOXX": "Tech/AI",
}


def bracket_fill_conversion(ordinary_income: np.ndarray, trad_balance: np.ndarray,
                            target_bracket: float, brackets: list,
                            infl_factor: np.ndarray,
                            irmaa_aware: bool, irmaa_limit: float) -> np.ndarray:
    """Compute how much to convert from trad→Roth to fill up to target_bracket.
    All inputs are (n,) arrays except scalars. Returns (n,) conversion amount."""
    n = len(ordinary_income)
    # Find the top of the target bracket (inflation-adjusted)
    bracket_top = np.full(n, 1e12)
    for i in range(len(brackets) - 1):
        _, rate = brackets[i]
        next_thresh, next_rate = brackets[i + 1]
        if next_rate > target_bracket:
            bracket_top = next_thresh * infl_factor
            break
    room = np.maximum(0.0, bracket_top - ordinary_income)
    if irmaa_aware:
        irmaa_nom = irmaa_limit * infl_factor
        room = np.minimum(room, np.maximum(0.0, irmaa_nom - ordinary_income))
    return np.maximum(0.0, np.minimum(room, trad_balance))


@st.cache_data(show_spinner=False, max_entries=8)
def simulate(cfg: dict, hold: dict) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    n = int(cfg["n_sims"])
    start_age = int(cfg["start_age"])
    end_age = int(cfg["end_age"])
    retire_age = int(cfg["retire_age"])
    ages = np.arange(start_age, end_age + 1)
    T = len(ages) - 1

    # holdings
    start_tax = float(hold["total_tax"])
    start_ret = float(hold["total_ret"])
    w_tax = w_to_arr(hold["w_tax"])
    w_ret = w_to_arr(hold["w_ret"])

    # retirement split
    roth_frac = float(cfg["roth_frac"])
    trad0 = start_ret * (1 - roth_frac)
    roth0 = start_ret * roth_frac

    # HSA
    hsa_on = bool(cfg["hsa_on"])
    hsa0 = float(cfg["hsa_balance0"]) if hsa_on else 0.0
    hsa_like_ret = bool(cfg["hsa_like_ret"])
    hsa_med_real = float(cfg["hsa_med_real"]) if hsa_on else 0.0

    taxable = np.zeros((n, T+1)); taxable[:, 0] = start_tax
    trad = np.zeros((n, T+1)); trad[:, 0] = trad0
    roth = np.zeros((n, T+1)); roth[:, 0] = roth0
    hsa = np.zeros((n, T+1)); hsa[:, 0] = hsa0

    basis = np.zeros((n, T+1)); basis[:, 0] = start_tax * float(cfg["basis_frac"])
    infl_index = np.ones((n, T+1))

    # home
    home_on = bool(cfg["home_on"])
    home_value = np.zeros((n, T+1))
    mortgage = np.zeros((n, T+1))
    rent_on = np.zeros(n, dtype=bool)
    if home_on:
        home_value[:, 0] = float(cfg["home_value0"])
        mortgage[:, 0] = float(cfg["mortgage_balance0"])
    annual_pmt = mortgage_monthly_payment(float(cfg["mortgage_balance0"]),
                                          float(cfg["mortgage_rate"]),
                                          int(cfg["mortgage_term_years"])) * 12.0

    # mortality
    mort_on = bool(cfg["mort_on"])
    has_spouse = bool(cfg.get("has_spouse", True))
    spouse_age = int(cfg.get("spouse_age", start_age)) if has_spouse else start_age
    age_gap = start_age - spouse_age

    if mort_on:
        q1 = make_q_by_age(start_age, end_age, float(cfg["q55_1"]), float(cfg["mort_growth"]))
        death1 = np.array([draw_death_age(rng, start_age, end_age, q1) for _ in range(n)], dtype=int)
        if has_spouse:
            sp2_start = spouse_age
            sp2_end = end_age - age_gap
            q2 = make_q_by_age(sp2_start, max(sp2_end, sp2_start), float(cfg["q55_2"]), float(cfg["mort_growth"]))
            death2_sp_age = np.array([draw_death_age(rng, sp2_start, max(sp2_end, sp2_start), q2) for _ in range(n)], dtype=int)
            death2 = death2_sp_age + age_gap
        else:
            death2 = death1.copy()
    else:
        death1 = np.full(n, end_age+1, dtype=int)
        death2 = np.full(n, end_age+1, dtype=int)
        if not has_spouse:
            death2 = death1.copy()

    second_death = np.maximum(death1, death2)

    # inheritance
    inh_on = bool(cfg["inh_on"])
    inh_year = np.full(n, -1, dtype=int)
    inh_amt = np.zeros(n, dtype=float)
    if inh_on:
        prob = float(cfg["inh_prob"])
        horizon = int(cfg["inh_horizon"])
        u = rng.uniform(size=n)
        occurs = u < prob
        t_u = rng.uniform(size=n)
        inh_year[occurs] = start_age + (t_u[occurs] * horizon).astype(int)
        inh_min = float(cfg["inh_min"])
        inh_mean = float(cfg["inh_mean"])
        inh_sigma = float(cfg["inh_sigma"])
        if inh_mean <= inh_min:
            inh_amt[occurs] = inh_min
        else:
            m = np.log(inh_mean) - 0.5 * inh_sigma**2
            draws = np.exp(rng.normal(m, inh_sigma, size=n))
            draws = np.maximum(draws, inh_min)
            inh_amt[occurs] = draws[occurs]

    # SS (REAL annual)
    claim1_age = int(cfg["claim1"])
    claim2_age = int(cfg["claim2"]) if has_spouse else 999
    ss1 = adjusted_ss_from_62(float(cfg["ss62_1"]), claim1_age, int(cfg["fra"])) * 12.0
    if has_spouse:
        ss2 = adjusted_ss_from_62(float(cfg["ss62_2"]), claim2_age, int(cfg["fra"])) * 12.0
    else:
        ss2 = 0.0
    claim2_on_p1_scale = claim2_age + age_gap

    # scenario params
    p = cfg["scenario_params"]
    eq_mu, eq_vol = float(p["eq_mu"]), float(p["eq_vol"])
    reit_mu, reit_vol = float(p["reit_mu"]), float(p["reit_vol"])
    bond_mu, bond_vol = float(p["bond_mu"]), float(p["bond_vol"])
    alt_mu, alt_vol = float(p["alt_mu"]), float(p["alt_vol"])
    cash_mu = float(p["cash_mu"])

    # inflation/home
    infl_mu, infl_vol = float(cfg["infl_mu"]), float(cfg["infl_vol"])
    infl_min, infl_max = float(cfg["infl_min"]), float(cfg["infl_max"])
    home_mu, home_vol = float(cfg["home_mu"]), float(cfg["home_vol"])

    cov = build_cov(eq_vol, reit_vol, bond_vol, alt_vol, infl_vol, home_vol,
                    float(cfg["corr_eq_infl"]), float(cfg["corr_reit_infl"]), float(cfg["corr_bond_infl"]), float(cfg["corr_alt_infl"]),
                    float(cfg["corr_home_infl"]), float(cfg["corr_home_eq"]))

    # Equity sub-bucket granularity (Phase 5)
    equity_granular = bool(cfg.get("equity_granular_on", False))
    sub_weights_raw = cfg.get("equity_sub_weights", {k: v["weight"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    sub_mu_raw = cfg.get("equity_sub_mu", {k: v["mu"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    sub_vol_raw = cfg.get("equity_sub_vol", {k: v["vol"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    tech_bubble_on = bool(cfg.get("tech_bubble_on", False)) and equity_granular
    tech_bubble_prob = float(cfg.get("tech_bubble_prob", 0.03))
    tech_bubble_extra = float(cfg.get("tech_bubble_extra_drop", -0.50))

    if equity_granular:
        nsub = len(EQUITY_SUB_CLASSES)
        sub_weights = np.array([float(sub_weights_raw.get(c, 0.0)) for c in EQUITY_SUB_CLASSES])
        wsum = sub_weights.sum()
        if wsum > 0:
            sub_weights = sub_weights / wsum  # normalize to 1
        sub_mus = np.array([float(sub_mu_raw.get(c, 0.07)) for c in EQUITY_SUB_CLASSES])
        sub_vols_arr = [float(sub_vol_raw.get(c, 0.16)) for c in EQUITY_SUB_CLASSES]
        cov_granular = build_cov_granular(
            sub_vols_arr, reit_vol, bond_vol, alt_vol, infl_vol, home_vol,
            EQUITY_SUB_CORR, EQUITY_SUB_CLASSES,
            float(cfg["corr_eq_infl"]), float(cfg["corr_reit_infl"]),
            float(cfg["corr_bond_infl"]), float(cfg["corr_alt_infl"]),
            float(cfg["corr_home_infl"]), float(cfg["corr_home_eq"]))

        # Weighted-average mu for the blended equity return (used by the rest of the engine)
        eq_mu_blended = float(np.dot(sub_weights, sub_mus))

    # Regime-switching model setup
    use_regime = str(cfg.get("return_model", "standard")) == "regime"
    regime_states = None
    regime_covs = {}
    regime_means = {}
    if use_regime:
        r_params = cfg.get("regime_params", DEFAULT_REGIME_PARAMS)
        r_trans = cfg.get("regime_transition", DEFAULT_TRANSITION_MATRIX)
        r_init = cfg.get("regime_initial_probs", DEFAULT_REGIME_INITIAL_PROBS)
        regime_states = draw_regime_sequence(rng, n, T, r_trans, r_init)
        # Pre-compute covariance matrices and mean vectors per state
        for si, sname in enumerate(REGIME_NAMES):
            sp = r_params.get(sname, DEFAULT_REGIME_PARAMS[sname])
            regime_covs[si] = build_cov(
                float(sp["eq_vol"]), float(sp["reit_vol"]), float(sp["bond_vol"]),
                float(sp["alt_vol"]), infl_vol, home_vol,
                float(cfg["corr_eq_infl"]), float(cfg["corr_reit_infl"]),
                float(cfg["corr_bond_infl"]), float(cfg["corr_alt_infl"]),
                float(cfg["corr_home_infl"]), float(cfg["corr_home_eq"]))
            regime_means[si] = np.array([
                float(sp["eq_mu"]), float(sp["reit_mu"]), float(sp["bond_mu"]),
                float(sp["alt_mu"]), float(sp["cash_mu"])
            ])
    regime_track = np.zeros((n, T+1), dtype=int)  # track which regime

    # taxes — tax engine or legacy flat rates
    use_tax_engine = bool(cfg.get("tax_engine_on", True))
    filing_status = str(cfg.get("filing_status", "mfj"))
    niit_on = bool(cfg.get("niit_on", False))
    state_rate_ord = float(cfg.get("state_rate_ordinary", float(cfg.get("state_ord", 0.05))))
    state_rate_cg = float(cfg.get("state_rate_capgains", float(cfg.get("state_capg", 0.05))))

    # ACA
    aca_mode = str(cfg.get("aca_mode", "simple"))
    aca_household_size = int(cfg.get("aca_household_size", 2))
    aca_benchmark = float(cfg.get("aca_benchmark_premium_real", 20000.0))
    aca_schedule = cfg.get("aca_subsidy_schedule", list(ACA_SUBSIDY_SCHEDULE_2024))

    # Legacy flat rates (fallback when tax engine off)
    eff_capg = min(0.95, max(0.0, float(cfg["fed_capg"]) + float(cfg["state_capg"])))
    eff_div = min(0.95, max(0.0, float(cfg["fed_div"]) + float(cfg["state_capg"])))
    eff_ord = min(0.95, max(0.0, float(cfg["fed_ord"]) + float(cfg["state_ord"])))
    eff_capg_wd = eff_capg * (1 - float(cfg["tlh_reduction"])) if bool(cfg["tlh_on"]) else eff_capg
    taxable_tax_drag = float(cfg["div_yield"]) * eff_div + float(cfg["dist_yield"]) * eff_capg

    # crash + seq stress
    crash_on = bool(cfg["crash_on"])
    crash_prob = float(cfg["crash_prob"])
    crash_eq_extra = float(cfg["crash_eq_extra"])
    crash_reit_extra = float(cfg["crash_reit_extra"])
    crash_alt_extra = float(cfg["crash_alt_extra"])
    crash_home_extra = float(cfg["crash_home_extra"])

    seq_on = bool(cfg["seq_on"])
    seq_drop = float(cfg["seq_drop"])
    seq_years = int(cfg["seq_years"])
    seq_start_age_raw = int(cfg.get("seq_start_age", 0))
    seq_start_age = seq_start_age_raw if seq_start_age_raw > 0 else retire_age

    # spending + GK
    spend_split_on = bool(cfg.get("spend_split_on", False))
    base_spend_real_input = float(cfg["spend_real"])
    if spend_split_on:
        base_essential_real = float(cfg.get("spend_essential_real", 180000.0))
        base_discretionary_real = float(cfg.get("spend_discretionary_real", 120000.0))
        base_spend_real_input = base_essential_real + base_discretionary_real
        essential_real = np.full(n, base_essential_real)
        discretionary_real = np.full(n, base_discretionary_real)
    spend_real = np.full(n, base_spend_real_input)

    total_liquid_start = start_tax + start_ret + (float(cfg["hsa_balance0"]) if bool(cfg["hsa_on"]) else 0.0)
    initial_wr = base_spend_real_input / total_liquid_start if total_liquid_start > 0 else 0.05

    # phases
    p1_end, p2_end = int(cfg["phase1_end"]), int(cfg["phase2_end"])
    p1_mult, p2_mult, p3_mult = float(cfg["phase1_mult"]), float(cfg["phase2_mult"]), float(cfg["phase3_mult"])

    # home costs + survival adjustments
    home_cost_pct = float(cfg["home_cost_pct"])
    home_cost_drop = float(cfg["home_cost_drop_after_death"])
    spend_drop = float(cfg["spend_drop_after_death"])

    # pre-65 health
    pre65_on = bool(cfg["pre65_health_on"])
    pre65_real = float(cfg["pre65_health_real"])
    medicare_age = int(cfg["medicare_age"])

    # Roth conversions
    conv_on = bool(cfg["conv_on"])
    conv_start = int(cfg["conv_start"])
    conv_end = int(cfg["conv_end"])
    conv_real = float(cfg["conv_real"])
    conv_type = str(cfg.get("conv_type", "fixed"))
    conv_target_bracket = float(cfg.get("conv_target_bracket", 0.24))
    conv_irmaa_aware = bool(cfg.get("conv_irmaa_aware", True))
    conv_irmaa_target_tier = int(cfg.get("conv_irmaa_target_tier", 0))
    conv_filing_status = filing_status if use_tax_engine else str(cfg.get("conv_filing_status", "mfj"))
    if conv_filing_status == "mfj":
        conv_brackets = FED_BRACKETS_MFJ_2024
        conv_irmaa_tiers = IRMAA_MAGI_TIERS_MFJ_2024
    else:
        conv_brackets = FED_BRACKETS_SINGLE_2024
        conv_irmaa_tiers = IRMAA_MAGI_TIERS_SINGLE_2024
    conv_irmaa_limit = float(conv_irmaa_tiers[conv_irmaa_target_tier]) if conv_irmaa_aware and conv_irmaa_target_tier < len(conv_irmaa_tiers) else 1e12

    # RMD
    rmd_on = bool(cfg["rmd_on"])
    rmd_age = int(cfg["rmd_start_age"])

    # QCD (Qualified Charitable Distributions)
    qcd_on = bool(cfg.get("qcd_on", False))
    qcd_annual_real = float(cfg.get("qcd_annual_real", 20000.0))
    qcd_start_age = int(cfg.get("qcd_start_age", 70))
    qcd_max_annual = float(cfg.get("qcd_max_annual", 105000.0))

    # Gain harvesting
    gain_harvest_on = bool(cfg.get("gain_harvest_on", False))
    gain_harvest_filing = filing_status if use_tax_engine else str(cfg.get("gain_harvest_filing", "mfj"))
    if gain_harvest_filing == "mfj":
        ltcg_0pct_threshold = LTCG_0PCT_THRESHOLD_MFJ_2024
        std_deduction = STANDARD_DEDUCTION_MFJ_2024
    else:
        ltcg_0pct_threshold = LTCG_0PCT_THRESHOLD_SINGLE_2024
        std_deduction = STANDARD_DEDUCTION_SINGLE_2024

    # fees
    fee_tax = float(cfg["fee_tax"])
    fee_ret = float(cfg["fee_ret"])

    # Annuities
    ann_configs = []
    for prefix in ("ann1", "ann2"):
        if bool(cfg[f"{prefix}_on"]):
            ann_configs.append({
                "purchase_age": int(cfg[f"{prefix}_purchase_age"]),
                "income_start_age": int(cfg[f"{prefix}_income_start_age"]),
                "purchase_amount": float(cfg[f"{prefix}_purchase_amount"]),
                "payout_rate": float(cfg[f"{prefix}_payout_rate"]),
                "cola_on": bool(cfg[f"{prefix}_cola_on"]),
                "cola_rate": float(cfg[f"{prefix}_cola_rate"]),
                "cola_match_inflation": bool(cfg[f"{prefix}_cola_match_inflation"]),
                "purchased": np.zeros(n, dtype=bool),
                "base_payment": np.zeros(n),  # annual payout at purchase (real)
            })
    annuity_income_track = np.zeros((n, T + 1))
    annuity_purchase_track = np.zeros((n, T + 1))

    # Glide path
    glide_on = bool(cfg["glide_on"])
    if glide_on:
        glide_tax_eq_start = float(cfg["glide_tax_eq_start"])
        glide_tax_eq_end = float(cfg["glide_tax_eq_end"])
        glide_tax_start_age = int(cfg["glide_tax_start_age"])
        glide_tax_end_age = int(cfg["glide_tax_end_age"])
        glide_ret_same = bool(cfg["glide_ret_same"])
        if glide_ret_same:
            glide_ret_eq_start = glide_tax_eq_start
            glide_ret_eq_end = glide_tax_eq_end
            glide_ret_start_age = glide_tax_start_age
            glide_ret_end_age = glide_tax_end_age
        else:
            glide_ret_eq_start = float(cfg["glide_ret_eq_start"])
            glide_ret_eq_end = float(cfg["glide_ret_eq_end"])
            glide_ret_start_age = int(cfg["glide_ret_start_age"])
            glide_ret_end_age = int(cfg["glide_ret_end_age"])
        # Store original weights as basis for sub-asset ratio preservation
        base_w_tax = w_tax.copy()
        base_w_ret = w_ret.copy()

    # IRMAA
    irmaa_on = bool(cfg["irmaa_on"])
    irmaa_people = int(cfg["irmaa_people"])
    irmaa_base = float(cfg["irmaa_base"])
    irmaa_schedule = cfg["irmaa_schedule"]

    magi_track = np.zeros((n, T+1))
    tier_track = np.zeros((n, T+1))
    irmaa_track = np.zeros((n, T+1))

    # home sale
    sale_on = bool(cfg["sale_on"])
    sale_age = int(cfg["sale_age"])
    selling_cost = float(cfg["selling_cost_pct"])
    post_sale_mode = str(cfg["post_sale_mode"])
    down_frac = float(cfg["downsize_fraction"])
    rent_real = float(cfg["rent_real"])

    # LTC
    ltc_on = bool(cfg["ltc_on"])
    ltc_start_age = int(cfg["ltc_start_age"]) if ltc_on else 200
    ltc_annual_prob = float(cfg["ltc_annual_prob"]) if ltc_on else 0.0
    ltc_cost_real = float(cfg["ltc_cost_real"]) if ltc_on else 0.0
    ltc_dur_mean = float(cfg["ltc_duration_mean"]) if ltc_on else 0.0
    ltc_dur_sigma = float(cfg["ltc_duration_sigma"]) if ltc_on else 0.0

    ltc_onset_age = np.full(n, 999, dtype=int)
    ltc_end_age = np.full(n, 999, dtype=int)
    if ltc_on and ltc_annual_prob > 0:
        for a in range(ltc_start_age, end_age + 1):
            not_yet = ltc_onset_age >= 999
            if not not_yet.any():
                break
            triggers = not_yet & (rng.uniform(size=n) < ltc_annual_prob)
            ltc_onset_age[triggers] = a
            if triggers.any():
                if ltc_dur_mean > 0 and ltc_dur_sigma > 0:
                    m = np.log(ltc_dur_mean) - 0.5 * (ltc_dur_sigma / ltc_dur_mean) ** 2
                    s = ltc_dur_sigma / ltc_dur_mean
                    dur = np.exp(rng.normal(m, s, size=int(triggers.sum())))
                    dur = np.maximum(1.0, dur)
                else:
                    dur = np.full(int(triggers.sum()), max(1.0, ltc_dur_mean))
                ltc_end_age[triggers] = a + np.ceil(dur).astype(int)

    # Pre-retirement income & contributions
    contrib_on = bool(cfg.get("contrib_on", False)) and (start_age < retire_age)
    pretax_income_1 = float(cfg.get("pretax_income_1", 0.0)) if contrib_on else 0.0
    pretax_income_2 = float(cfg.get("pretax_income_2", 0.0)) if contrib_on else 0.0
    income_growth_real = float(cfg.get("income_growth_real", 0.01))
    contrib_ret = float(cfg.get("contrib_ret_annual", 0.0)) if contrib_on else 0.0
    contrib_roth_401k_frac = float(cfg.get("contrib_roth_401k_frac", 0.0))
    contrib_match = float(cfg.get("contrib_match_annual", 0.0)) if contrib_on else 0.0
    contrib_taxable = float(cfg.get("contrib_taxable_annual", 0.0)) if contrib_on else 0.0
    contrib_hsa = float(cfg.get("contrib_hsa_annual", 0.0)) if (contrib_on and hsa_on) else 0.0
    pre_ret_spend = float(cfg.get("pre_ret_spend_real", 0.0))
    if pre_ret_spend <= 0 and contrib_on:
        pre_ret_spend = float(cfg.get("spend_real", 120000.0))  # default to post-retirement spend
    # Coast FIRE phase
    coast_on = bool(cfg.get("coast_on", False)) and contrib_on
    coast_start_age = int(cfg.get("coast_start_age", 55))
    coast_income_real = float(cfg.get("coast_income_real", 50000.0))
    coast_contrib_ret = float(cfg.get("coast_contrib_ret", 0.0))
    coast_contrib_hsa = float(cfg.get("coast_contrib_hsa", 0.0))

    # Spending events
    spending_events = list(cfg.get("spending_events", []))

    # -------- Decomposition trackers (nominal) --------
    event_expense_track = np.zeros((n, T+1))
    event_income_track = np.zeros((n, T+1))
    ltc_cost_track = np.zeros((n, T+1))
    baseline_core_track = np.zeros((n, T+1))
    core_spend_track = np.zeros((n, T+1))
    ss_nom_track = np.zeros((n, T+1))
    home_cost_track = np.zeros((n, T+1))
    mort_pay_track = np.zeros((n, T+1))
    rent_track = np.zeros((n, T+1))
    health_track = np.zeros((n, T+1))
    irmaa_paid_track = np.zeros((n, T+1))

    medical_nom_track = np.zeros((n, T+1))
    hsa_withdraw_track = np.zeros((n, T+1))
    hsa_end_track = np.zeros((n, T+1))

    total_outflow_track = np.zeros((n, T+1))

    gross_tax_wd_track = np.zeros((n, T+1))
    gross_trad_wd_track = np.zeros((n, T+1))
    gross_roth_wd_track = np.zeros((n, T+1))
    gross_rmd_track = np.zeros((n, T+1))
    taxes_paid_track = np.zeros((n, T+1))
    conv_gross_track = np.zeros((n, T+1))
    qcd_track = np.zeros((n, T+1))
    gain_harvest_track = np.zeros((n, T+1))
    essential_spend_track = np.zeros((n, T+1))
    discretionary_spend_track = np.zeros((n, T+1))
    essential_funded_track = np.zeros((n, T+1))

    # Tax engine trackers
    fed_tax_track = np.zeros((n, T+1))
    state_tax_track = np.zeros((n, T+1))
    niit_track = np.zeros((n, T+1))
    ss_taxable_track = np.zeros((n, T+1))
    effective_rate_track = np.zeros((n, T+1))
    marginal_bracket_track = np.zeros((n, T+1))
    ltcg_realized_track = np.zeros((n, T+1))
    qual_div_track = np.zeros((n, T+1))

    # ACA trackers
    aca_gross_track = np.zeros((n, T+1))
    aca_subsidy_track = np.zeros((n, T+1))
    aca_net_track = np.zeros((n, T+1))

    # IRMAA Part B/D trackers
    irmaa_part_b_track = np.zeros((n, T+1))
    irmaa_part_d_track = np.zeros((n, T+1))

    # Pre-retirement income trackers
    earned_income_track = np.zeros((n, T+1))
    pre_ret_tax_track = np.zeros((n, T+1))
    pre_ret_savings_track = np.zeros((n, T+1))

    for t in range(1, T+1):
        age = int(ages[t-1])
        years_left = int(end_age - age)

        alive1 = age < death1
        alive2 = age < death2
        both_alive = alive1 & alive2
        one_alive = alive1 ^ alive2
        none_alive = ~(alive1 | alive2)

        if equity_granular and not (use_regime and t <= regime_states.shape[1]):
            # Equity sub-bucket model: draw from expanded covariance matrix
            g_shocks = draw_shocks(rng, n, cov_granular, str(cfg["dist_type"]), int(cfg["t_df"]))
            # g_shocks layout: [sub0..sub5, REIT, Bond, Alt, Cash, Inflation, Home]
            sub_returns = np.zeros((n, nsub))
            for si_idx in range(nsub):
                sub_returns[:, si_idx] = sub_mus[si_idx] + g_shocks[:, si_idx]
            r_eq = np.dot(sub_returns, sub_weights)  # weighted-average equity return
            r_reit = reit_mu + g_shocks[:, nsub]
            r_bond = bond_mu + g_shocks[:, nsub + 1]
            r_alt = alt_mu + g_shocks[:, nsub + 2]
            r_cash = cash_mu + g_shocks[:, nsub + 3]
            infl = np.clip(infl_mu + g_shocks[:, nsub + 4], infl_min, infl_max)
            home_app = home_mu + g_shocks[:, nsub + 5]

            # Tech/AI bubble stress overlay
            if tech_bubble_on:
                tech_idx = EQUITY_SUB_CLASSES.index("Tech/AI")
                tech_crash = rng.uniform(size=n) < tech_bubble_prob
                if tech_crash.any():
                    sub_returns[tech_crash, tech_idx] = np.clip(
                        sub_returns[tech_crash, tech_idx] + tech_bubble_extra, -0.95, 2.0)
                    # Recompute blended equity return for affected paths
                    r_eq[tech_crash] = np.dot(sub_returns[tech_crash], sub_weights)

        elif use_regime and t <= regime_states.shape[1]:
            # Regime-switching: draw shocks per-state, combine
            r_eq = np.zeros(n)
            r_reit = np.zeros(n)
            r_bond = np.zeros(n)
            r_alt = np.zeros(n)
            r_cash = np.zeros(n)
            infl_shocks = np.zeros(n)
            home_shocks = np.zeros(n)
            state_t = regime_states[:, t-1]
            regime_track[:, t] = state_t
            for si in range(3):
                mask = state_t == si
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                s_shocks = draw_shocks(rng, cnt, regime_covs[si], str(cfg["dist_type"]), int(cfg["t_df"]))
                r_eq[mask] = regime_means[si][0] + s_shocks[:, 0]
                r_reit[mask] = regime_means[si][1] + s_shocks[:, 1]
                r_bond[mask] = regime_means[si][2] + s_shocks[:, 2]
                r_alt[mask] = regime_means[si][3] + s_shocks[:, 3]
                r_cash[mask] = regime_means[si][4] + s_shocks[:, 4]
                infl_shocks[mask] = s_shocks[:, 5]
                home_shocks[mask] = s_shocks[:, 6]
            infl = np.clip(infl_mu + infl_shocks, infl_min, infl_max)
            home_app = home_mu + home_shocks
        else:
            # Standard model
            shocks = draw_shocks(rng, n, cov, str(cfg["dist_type"]), int(cfg["t_df"]))
            r_eq = eq_mu + shocks[:, 0]
            r_reit = reit_mu + shocks[:, 1]
            r_bond = bond_mu + shocks[:, 2]
            r_alt = alt_mu + shocks[:, 3]
            r_cash = cash_mu + shocks[:, 4]
            infl = np.clip(infl_mu + shocks[:, 5], infl_min, infl_max)
            home_app = home_mu + shocks[:, 6]

        infl_index[:, t] = infl_index[:, t-1] * (1 + infl)

        if crash_on:
            crash = rng.uniform(size=n) < crash_prob
            r_eq[crash] = np.clip(r_eq[crash] + crash_eq_extra, -0.95, 2.0)
            r_reit[crash] = np.clip(r_reit[crash] + crash_reit_extra, -0.95, 2.0)
            r_alt[crash] = np.clip(r_alt[crash] + crash_alt_extra, -0.95, 2.0)
            home_app[crash] = np.clip(home_app[crash] + crash_home_extra, -0.95, 2.0)

        if seq_on and age >= seq_start_age and age < seq_start_age + seq_years:
            r_eq = np.clip(r_eq + seq_drop, -0.95, 2.0)
            r_reit = np.clip(r_reit + seq_drop, -0.95, 2.0)
            r_alt = np.clip(r_alt + 0.5 * seq_drop, -0.95, 2.0)

        # home flows
        home_cost_nom = np.zeros(n)
        mort_pay_nom = np.zeros(n)
        rent_nom = np.zeros(n)

        if home_on:
            still_own = ~rent_on
            hv_prev = home_value[:, t-1]
            hv_new = hv_prev.copy()
            hv_new[still_own] = hv_prev[still_own] * (1 + np.clip(home_app[still_own], -0.50, 0.50))
            home_value[:, t] = hv_new

            bal_prev = mortgage[:, t-1]
            bal_new = bal_prev.copy()
            if annual_pmt > 0:
                interest = bal_prev * float(cfg["mortgage_rate"])
                payment_cap = bal_prev + interest
                payment = np.minimum(annual_pmt, payment_cap)
                principal = np.maximum(0.0, payment - interest)
                bal_new = np.maximum(0.0, bal_prev - principal)
                mort_pay_nom = payment
            mortgage[:, t] = bal_new

            cost_pct_vec = np.full(n, home_cost_pct)
            if mort_on:
                cost_pct_vec[one_alive] *= (1 - home_cost_drop)
                cost_pct_vec[none_alive] = 0.0
            home_cost_nom = cost_pct_vec * home_value[:, t-1]

            if sale_on and rent_real > 0:
                rent_nom[rent_on] = rent_real * infl_index[rent_on, t-1]

            if sale_on and age == sale_age:
                do_sale = still_own & (home_value[:, t-1] > 0)
                sale_price = home_value[:, t-1]
                net = np.maximum(0.0, sale_price * (1 - selling_cost) - mortgage[:, t-1])
                if post_sale_mode == "Sell and rent":
                    rent_on[do_sale] = True
                    home_value[do_sale, t] = 0.0
                    mortgage[do_sale, t] = 0.0
                    taxable[do_sale, t-1] += net[do_sale]
                    basis[do_sale, t-1] += net[do_sale]
                else:
                    new_home = net * down_frac
                    cash_out = np.maximum(0.0, net - new_home)
                    home_value[do_sale, t] = new_home[do_sale]
                    mortgage[do_sale, t] = 0.0
                    taxable[do_sale, t-1] += cash_out[do_sale]
                    basis[do_sale, t-1] += cash_out[do_sale]
        else:
            home_value[:, t] = 0.0
            mortgage[:, t] = 0.0

        home_cost_track[:, t] = home_cost_nom
        mort_pay_track[:, t] = mort_pay_nom
        rent_track[:, t] = rent_nom

        # SS real
        if mort_on:
            b1 = np.where(alive1 & (age >= claim1_age), ss1, 0.0)
            b2 = np.where(alive2 & (age >= claim2_on_p1_scale), ss2, 0.0)
            ss_real = np.zeros(n)
            ss_real[both_alive] = b1[both_alive] + b2[both_alive]
            ss_real[one_alive] = np.maximum(b1[one_alive], b2[one_alive])
            ss_real[none_alive] = 0.0
        else:
            ss_real = (ss1 if age >= claim1_age else 0.0) + (ss2 if age >= claim2_on_p1_scale else 0.0)

        # GK update ONLY after retire_age
        if age >= retire_age and bool(cfg["gk_on"]):
            liquid_prev = taxable[:, t-1] + trad[:, t-1] + roth[:, t-1] + hsa[:, t-1]
            liquid_prev_real = liquid_prev / infl_index[:, t-1]
            if spend_split_on:
                # GK adjusts total spend, then attribute delta to discretionary only
                total_before_gk = spend_real.copy()
                spend_real = gk_update(
                    spend_real,
                    base_spend_real_input,
                    liquid_prev_real,
                    initial_wr,
                    True,
                    float(cfg["gk_upper_pct"]),
                    float(cfg["gk_lower_pct"]),
                    float(cfg["gk_cut"]),
                    float(cfg["gk_raise"]),
                )
                delta = spend_real - total_before_gk
                # Apply delta ONLY to discretionary (essentials never cut)
                discretionary_real = np.maximum(0.0, discretionary_real + delta)
                # Re-derive total from components
                spend_real = essential_real + discretionary_real
            else:
                spend_real = gk_update(
                    spend_real,
                    base_spend_real_input,
                    liquid_prev_real,
                    initial_wr,
                    True,
                    float(cfg["gk_upper_pct"]),
                    float(cfg["gk_lower_pct"]),
                    float(cfg["gk_cut"]),
                    float(cfg["gk_raise"]),
                )
        else:
            spend_real[:] = base_spend_real_input
            if spend_split_on:
                essential_real[:] = base_essential_real
                discretionary_real[:] = base_discretionary_real

        spend_eff = spend_real.copy()
        if mort_on:
            spend_eff[one_alive] *= (1 - float(cfg["spend_drop_after_death"]))
            spend_eff[none_alive] = 0.0

        # phase multiplier — when split on, phases apply only to discretionary
        if age < retire_age:
            phase_mult = 0.0
        else:
            if age < p1_end: phase_mult = p1_mult
            elif age < p2_end: phase_mult = p2_mult
            else: phase_mult = p3_mult

        # pre-65 health insurance (nominal)
        health_nom = np.zeros(n)
        if pre65_on and age >= retire_age and age < medicare_age:
            health_nom[:] = float(cfg["pre65_health_real"]) * infl_index[:, t-1]
            if mort_on:
                health_nom[one_alive] *= 0.60
                health_nom[none_alive] = 0.0
        health_track[:, t] = health_nom

        # Baseline vs GK-adjusted core spending (nominal)
        if spend_split_on and age >= retire_age:
            # Essentials: no phase multiplier. Discretionary: phase multiplier.
            mort_factor = np.ones(n)
            if mort_on:
                mort_factor[one_alive] = 1 - spend_drop
                mort_factor[none_alive] = 0.0
            ess_nom = essential_real * mort_factor * infl_index[:, t-1]
            disc_nom = discretionary_real * phase_mult * mort_factor * infl_index[:, t-1]
            adjusted_core_nom = ess_nom + disc_nom
            baseline_core_nom = (base_spend_real_input * phase_mult) * infl_index[:, t-1]
            essential_spend_track[:, t] = ess_nom
            discretionary_spend_track[:, t] = disc_nom
            # Check if guaranteed income covers essentials (use prev year annuity as proxy)
            ss_nom_est = ss_real * infl_index[:, t-1]
            ann_prev_year = annuity_income_track[:, t-1] if t > 1 else np.zeros(n)
            guaranteed_income = ss_nom_est + ann_prev_year
            essential_funded_track[:, t] = np.where(guaranteed_income >= ess_nom, 1.0, 0.0)
        else:
            baseline_core_nom = (base_spend_real_input * phase_mult) * infl_index[:, t-1]
            adjusted_core_nom = (spend_eff * phase_mult) * infl_index[:, t-1]
        baseline_core_track[:, t] = baseline_core_nom
        core_spend_track[:, t] = adjusted_core_nom

        # Medical (HSA-eligible) spending (nominal) and HSA funding
        medical_nom = np.zeros(n)
        if hsa_on and age >= retire_age:
            medical_nom[:] = hsa_med_real * infl_index[:, t-1]
            if mort_on:
                medical_nom[one_alive] *= 0.60
                medical_nom[none_alive] = 0.0
        medical_nom_track[:, t] = medical_nom

        # HSA withdrawal to cover medical (tax-free)
        hsa_prev = hsa[:, t-1]
        hsa_used_med = np.minimum(hsa_prev, medical_nom)
        hsa_withdraw_track[:, t] = hsa_used_med
        hsa_after_med = hsa_prev - hsa_used_med

        # Remaining medical becomes part of spending need
        medical_short = np.maximum(0.0, medical_nom - hsa_used_med)

        # Long-term care cost
        ltc_nom = np.zeros(n)
        if ltc_on:
            in_ltc = (age >= ltc_onset_age) & (age < ltc_end_age)
            if mort_on:
                in_ltc = in_ltc & (alive1 | alive2)
            ltc_nom[in_ltc] = ltc_cost_real * infl_index[in_ltc, t-1]
        ltc_cost_track[:, t] = ltc_nom

        # Spending events: compute event expenses and income for this age
        evt_expense_nom = np.zeros(n)
        evt_income_nom = np.zeros(n)
        evt_income_taxable_nom = np.zeros(n)
        for evt in spending_events:
            e_start = int(evt.get("start_age", 0))
            e_end = int(evt.get("end_age", e_start))
            if e_start <= age <= e_end:
                amt_nom = float(evt.get("amount_real", 0.0)) * infl_index[:, t-1]
                if mort_on:
                    amt_nom[one_alive] *= (1 - spend_drop)
                    amt_nom[none_alive] = 0.0
                if evt.get("type", "expense") == "expense":
                    evt_expense_nom += amt_nom
                else:
                    evt_income_nom += amt_nom
                    if evt.get("taxable_income", False):
                        evt_income_taxable_nom += amt_nom
        event_expense_track[:, t] = evt_expense_nom
        event_income_track[:, t] = evt_income_nom

        # Total outflow (including event expenses)
        outflow_nom = adjusted_core_nom + home_cost_nom + mort_pay_nom + rent_nom + health_nom + medical_short + ltc_nom + evt_expense_nom
        total_outflow_track[:, t] = outflow_nom

        # inheritance
        inh_add = np.zeros(n)
        if inh_on:
            hit = inh_year == age
            inh_add[hit] = inh_amt[hit]

        tax_prev = taxable[:, t-1] + inh_add
        bas_prev = basis[:, t-1] + inh_add
        trad_prev = trad[:, t-1]
        roth_prev = roth[:, t-1]

        # Annuity purchase and income
        ann_income_this_year = np.zeros(n)
        for ac in ann_configs:
            # Purchase: deduct lump sum from portfolio at purchase age
            if age == ac["purchase_age"] and not ac["purchased"].any():
                purchase_nom = ac["purchase_amount"] * infl_index[:, t-1]
                # Deduct from taxable first
                from_tax = np.minimum(tax_prev, purchase_nom)
                tax_prev -= from_tax
                bas_prev = np.where(tax_prev + from_tax > 0,
                                    bas_prev * (tax_prev / (tax_prev + from_tax)),
                                    0.0)
                remaining_purchase = purchase_nom - from_tax
                # Then from trad (with tax hit accounted for in spending)
                from_trad = np.minimum(trad_prev, remaining_purchase)
                trad_prev -= from_trad
                ac["purchased"][:] = True
                ac["base_payment"][:] = ac["purchase_amount"] * ac["payout_rate"]
                annuity_purchase_track[:, t] = annuity_purchase_track[:, t] + purchase_nom

            # Income: add guaranteed payments after income start age
            if ac["purchased"].any() and age >= ac["income_start_age"]:
                years_paying = age - ac["income_start_age"]
                if ac["cola_on"]:
                    if ac["cola_match_inflation"]:
                        # COLA matches simulated inflation
                        ann_pay = ac["base_payment"] * infl_index[:, t-1]
                    else:
                        # Fixed COLA rate
                        ann_pay = ac["base_payment"] * ((1 + ac["cola_rate"]) ** years_paying)
                        ann_pay = ann_pay * infl_index[:, 0]  # convert to nominal from real base
                else:
                    # No COLA — fixed nominal payment (value erodes with inflation)
                    ann_pay = ac["base_payment"] * infl_index[:, 0]  # nominal from year-0 dollars
                # Annuity stops at death
                if mort_on:
                    ann_pay[none_alive] = 0.0
                    ann_pay[one_alive] *= 0.60  # joint-and-survivor reduction
                ann_income_this_year += ann_pay
        annuity_income_track[:, t] = ann_income_this_year

        # RMD preview (needed by bracket-fill conversions before actual RMD)
        rmd_preview = np.zeros(n)
        if rmd_on and age >= rmd_age:
            factor = float(RMD_FACTOR.get(age, RMD_FACTOR.get(95, 8.9)))
            rmd_preview = np.minimum(trad_prev, trad_prev / max(1.0, factor))

        # conversions
        conv_gross = np.zeros(n)
        if conv_on and (conv_start <= age <= conv_end):
            if conv_type == "bracket_fill":
                # Estimate ordinary income for bracket calculation
                ss_est = ss_real * infl_index[:, t-1]
                div_est = tax_prev * (float(cfg["div_yield"]) + float(cfg["dist_yield"]))
                ordinary_est = ss_est + rmd_preview + div_est + ann_income_this_year + evt_income_taxable_nom
                infl_factor_t = infl_index[:, t-1]
                # Use median inflation factor for bracket inflation
                median_infl = float(np.median(infl_factor_t))
                conv_gross = bracket_fill_conversion(
                    ordinary_est, trad_prev, conv_target_bracket,
                    conv_brackets, median_infl, conv_irmaa_aware, conv_irmaa_limit
                )
            else:
                # Fixed real amount (original behavior)
                conv_gross[:] = conv_real * infl_index[:, t-1]
            if mort_on:
                conv_gross[one_alive] *= 0.60
                conv_gross[none_alive] = 0.0
            conv_gross = np.minimum(conv_gross, trad_prev)
            # Conversion: move from trad to roth. Tax paid separately below.
            trad_prev = np.maximum(0.0, trad_prev - conv_gross)
            roth_prev = roth_prev + conv_gross
        conv_gross_track[:, t] = conv_gross

        ss_nom = ss_real * infl_index[:, t-1]
        ss_nom_track[:, t] = ss_nom
        inflows_nom = ss_nom + ann_income_this_year + evt_income_nom

        need_before = np.maximum(0.0, outflow_nom - inflows_nom)
        if mort_on:
            need_before[none_alive] = 0.0

        # RMD (re-compute on post-conversion trad balance)
        gross_rmd = np.zeros(n)
        if rmd_on and age >= rmd_age:
            factor = float(RMD_FACTOR.get(age, RMD_FACTOR.get(95, 8.9)))
            gross_rmd = np.minimum(trad_prev, trad_prev / max(1.0, factor))

        # QCD: divert part of RMD to charity (no tax, no cash to retiree)
        qcd_amount = np.zeros(n)
        if qcd_on and age >= qcd_start_age and rmd_on and age >= rmd_age:
            qcd_desired = np.minimum(qcd_annual_real * infl_index[:, t-1],
                                     qcd_max_annual * infl_index[:, t-1])
            qcd_amount = np.minimum(qcd_desired, gross_rmd)
            if mort_on:
                qcd_amount[none_alive] = 0.0
        qcd_track[:, t] = qcd_amount
        taxable_rmd = gross_rmd - qcd_amount
        trad_prev = np.maximum(0.0, trad_prev - gross_rmd)
        gross_rmd_track[:, t] = gross_rmd

        # Compute income components for tax engine
        div_yield_val = float(cfg["div_yield"])
        dist_yield_val = float(cfg["dist_yield"])
        qual_divs = tax_prev * div_yield_val           # qualified dividends
        infl_factor_t = infl_index[:, t-1]

        # Gain harvesting: step up basis when in 0% LTCG bracket
        gain_harvested = np.zeros(n)
        if gain_harvest_on and age >= retire_age:
            ss_for_tax = 0.85 * ss_nom
            div_inc = tax_prev * (div_yield_val + dist_yield_val)
            ordinary_income_est = ss_for_tax + taxable_rmd + conv_gross + ann_income_this_year + evt_income_taxable_nom + div_inc
            std_ded_adj = std_deduction * infl_factor_t
            taxable_income_est = np.maximum(0.0, ordinary_income_est - std_ded_adj)
            ltcg_threshold_adj = ltcg_0pct_threshold * infl_factor_t
            room = np.maximum(0.0, ltcg_threshold_adj - taxable_income_est)
            unrealized_gain = np.maximum(0.0, tax_prev - bas_prev)
            gain_harvested = np.minimum(room, unrealized_gain)
            bas_prev += gain_harvested
            if mort_on:
                gain_harvested[none_alive] = 0.0
        gain_harvest_track[:, t] = gain_harvested

        # ================================================================
        # TAX ENGINE: compute taxes, then determine net RMD, IRMAA, and withdrawals
        # ================================================================
        # Phase 1 estimate (before knowing withdrawal amounts): use RMD + conversion income
        ordinary_income_pre = taxable_rmd + conv_gross + ann_income_this_year + evt_income_taxable_nom
        ltcg_pre = tax_prev * dist_yield_val  # cap gains distributions (not yet withdrawal gains)

        if use_tax_engine and age >= retire_age:
            # --- Pass 1: estimate taxes before withdrawal ---
            tax_result_1 = compute_federal_tax(
                ordinary_income_pre, qual_divs, ltcg_pre, ss_nom,
                filing_status, infl_factor_t, niit_on)
            state_tax_1 = state_rate_ord * np.maximum(0.0, tax_result_1["magi"] - tax_result_1["std_deduction"])
            total_tax_1 = tax_result_1["fed_total"] + state_tax_1
            magi_1 = tax_result_1["magi"]
            magi_track[:, t] = magi_1

            # Estimate effective tax on pre-withdrawal income using marginal rate
            marg_1 = tax_result_1["marginal_bracket"]
            eff_ord_est = marg_1 + state_rate_ord

            # Conversion tax: deduct from taxable account or absorb from trad
            conv_tax = conv_gross * eff_ord_est
            pay_from_tax = np.minimum(tax_prev, conv_tax)
            tax_prev -= pay_from_tax
            remaining_conv_tax = conv_tax - pay_from_tax
            trad_prev = np.maximum(0.0, trad_prev - remaining_conv_tax)

            # FICA on taxable event income (self-employment estimate: 15.3%)
            evt_fica = evt_income_taxable_nom * 0.153
            # Event income FICA added to spending need
            need_before = need_before + evt_fica

            # Net RMD cash (after tax at marginal rate)
            rmd_tax_est = taxable_rmd * eff_ord_est
            net_rmd_cash = np.maximum(0.0, taxable_rmd - rmd_tax_est)
            used_rmd = np.minimum(net_rmd_cash, need_before)
            need = np.maximum(0.0, need_before - used_rmd)
            excess = np.maximum(0.0, net_rmd_cash - used_rmd)
            tax_prev += excess
            bas_prev += excess

            # IRMAA (using MAGI from tax engine, 2-year lookback)
            if irmaa_on and age >= medicare_age:
                lookback_magi = magi_track[:, max(0, t-2)]  # 2-year lookback
                irmaa_part_b_sched = cfg.get("irmaa_part_b_schedule", IRMAA_PART_B_2024_MFJ)
                irmaa_part_d_sched = cfg.get("irmaa_part_d_schedule", IRMAA_PART_D_2024_MFJ)
                prem_b = irmaa_monthly_per_person(lookback_magi, irmaa_base, irmaa_part_b_sched)
                prem_d = irmaa_monthly_per_person(lookback_magi, 0.0, irmaa_part_d_sched)
                irmaa_nom = (prem_b + prem_d) * 12.0 * irmaa_people
                if mort_on:
                    irmaa_nom[one_alive] *= 0.60
                    irmaa_nom[none_alive] = 0.0
                need += irmaa_nom
                irmaa_paid_track[:, t] = irmaa_nom
                irmaa_part_b_track[:, t] = prem_b * 12.0 * irmaa_people
                irmaa_part_d_track[:, t] = prem_d * 12.0 * irmaa_people
                tier = np.zeros(n)
                for i, (th, _) in enumerate(irmaa_part_b_sched, start=1):
                    tier = np.where(lookback_magi >= th * infl_factor_t, i, tier)
                tier_track[:, t] = tier

            # ACA: replace flat health premium with subsidy-based for pre-65
            if aca_mode == "aca" and age >= retire_age and age < medicare_age:
                aca_result = compute_aca_premium(magi_1, aca_household_size,
                    aca_benchmark, infl_factor_t, aca_schedule)
                # Replace health_nom with ACA net premium
                aca_net = aca_result["net_premium"]
                if mort_on:
                    aca_net[one_alive] *= 0.60
                    aca_net[none_alive] = 0.0
                # Adjust need: remove old health_nom, add ACA net
                need = need - health_nom + aca_net
                aca_gross_track[:, t] = aca_result["gross_premium"]
                aca_subsidy_track[:, t] = aca_result["subsidy"]
                aca_net_track[:, t] = aca_net
                health_track[:, t] = aca_net  # overwrite

            # --- Withdrawals using estimated marginal tax rates ---
            gain = np.maximum(0.0, tax_prev - bas_prev)
            gain_frac = np.where(tax_prev > 0, np.minimum(1.0, gain / tax_prev), 0.0)
            # Estimate marginal rates for withdrawal tax
            marg_rate = tax_result_1["marginal_bracket"]
            ltcg_thresh = (LTCG_0PCT_THRESHOLD_MFJ_2024 if filing_status == "mfj"
                          else LTCG_0PCT_THRESHOLD_SINGLE_2024) * infl_factor_t
            est_capg_rate = np.where(tax_result_1["taxable_ordinary"] < ltcg_thresh,
                                     0.0, 0.15) + state_rate_cg
            est_ord_rate = marg_rate + state_rate_ord
            if bool(cfg["tlh_on"]):
                est_capg_rate = est_capg_rate * (1 - float(cfg["tlh_reduction"]))
            denom_tax_wd = np.maximum(1e-6, 1 - gain_frac * est_capg_rate)
            denom_trad_wd = np.maximum(1e-6, 1 - est_ord_rate)

            gross_tax = np.zeros(n)
            gross_trad = np.zeros(n)
            gross_roth = np.zeros(n)

            if cfg["wd_strategy"] == "taxable_first":
                take_tax = np.minimum(tax_prev, need / denom_tax_wd)
                gross_tax += take_tax
                need2 = np.maximum(0.0, need - take_tax * denom_tax_wd)
                take_trad = np.minimum(trad_prev, need2 / denom_trad_wd)
                gross_trad += take_trad
                need3 = np.maximum(0.0, need2 - take_trad * denom_trad_wd)
                take_roth = np.minimum(roth_prev, need3)
                gross_roth += take_roth
            else:
                total_avail = tax_prev + trad_prev + roth_prev
                share_tax = np.where(total_avail > 0, tax_prev / total_avail, 0.0)
                share_trad = np.where(total_avail > 0, trad_prev / total_avail, 0.0)
                share_roth = 1 - share_tax - share_trad
                take_tax = np.minimum(tax_prev, (need * share_tax) / denom_tax_wd)
                take_trad = np.minimum(trad_prev, (need * share_trad) / denom_trad_wd)
                take_roth = np.minimum(roth_prev, (need * share_roth))
                gross_tax += take_tax
                gross_trad += take_trad
                gross_roth += take_roth

            # --- Pass 2: recompute tax with actual withdrawal income ---
            ltcg_from_wd = gross_tax * gain_frac
            ordinary_income_final = taxable_rmd + conv_gross + ann_income_this_year + gross_trad + evt_income_taxable_nom
            ltcg_final = ltcg_from_wd + ltcg_pre
            tax_result_2 = compute_federal_tax(
                ordinary_income_final, qual_divs, ltcg_final, ss_nom,
                filing_status, infl_factor_t, niit_on)
            state_tax_2 = state_rate_ord * np.maximum(0.0,
                tax_result_2["taxable_ordinary"]) + state_rate_cg * ltcg_final
            total_tax_2 = tax_result_2["fed_total"] + state_tax_2

            # Record final tax engine outputs
            fed_tax_track[:, t] = tax_result_2["fed_total"]
            state_tax_track[:, t] = state_tax_2
            niit_track[:, t] = tax_result_2["niit"]
            ss_taxable_track[:, t] = tax_result_2["ss_taxable"]
            effective_rate_track[:, t] = tax_result_2["effective_rate"]
            marginal_bracket_track[:, t] = tax_result_2["marginal_bracket"]
            magi_track[:, t] = tax_result_2["magi"]
            ltcg_realized_track[:, t] = ltcg_final
            qual_div_track[:, t] = qual_divs
            taxes_paid_track[:, t] = total_tax_2

        else:
            # Legacy flat-rate tax path (pre-retirement or tax_engine_on=False)
            # Conversion tax (flat rate)
            conv_tax = conv_gross * eff_ord
            pay_from_tax = np.minimum(tax_prev, conv_tax)
            tax_prev -= pay_from_tax
            remaining_conv_tax = conv_tax - pay_from_tax
            trad_prev = np.maximum(0.0, trad_prev - remaining_conv_tax)

            net_rmd_cash = taxable_rmd * (1 - eff_ord)
            used_rmd = np.minimum(net_rmd_cash, need_before)
            need = np.maximum(0.0, need_before - used_rmd)
            excess = np.maximum(0.0, net_rmd_cash - used_rmd)
            tax_prev += excess
            bas_prev += excess

            # IRMAA (legacy)
            if irmaa_on and age >= medicare_age:
                magi_est = taxable_rmd + conv_gross + ann_income_this_year + evt_income_taxable_nom + tax_prev * (div_yield_val + dist_yield_val)
                magi_track[:, t] = magi_est
                prem = irmaa_monthly_per_person(magi_est, irmaa_base, irmaa_schedule)
                irmaa_nom = prem * 12.0 * irmaa_people
                if mort_on:
                    irmaa_nom[one_alive] *= 0.60
                    irmaa_nom[none_alive] = 0.0
                need += irmaa_nom
                irmaa_paid_track[:, t] = irmaa_nom
                tier = np.zeros(n)
                for i, (th, _) in enumerate(irmaa_schedule, start=1):
                    tier = np.where(magi_est >= th, i, tier)
                tier_track[:, t] = tier

            gain = np.maximum(0.0, tax_prev - bas_prev)
            gain_frac = np.where(tax_prev > 0, np.minimum(1.0, gain / tax_prev), 0.0)
            denom_tax_wd = np.maximum(1e-6, 1 - gain_frac * eff_capg_wd)
            denom_trad_wd = np.maximum(1e-6, 1 - eff_ord)

            gross_tax = np.zeros(n)
            gross_trad = np.zeros(n)
            gross_roth = np.zeros(n)

            if cfg["wd_strategy"] == "taxable_first":
                take_tax = np.minimum(tax_prev, need / denom_tax_wd)
                gross_tax += take_tax
                need2 = np.maximum(0.0, need - take_tax * denom_tax_wd)
                take_trad = np.minimum(trad_prev, need2 / denom_trad_wd)
                gross_trad += take_trad
                need3 = np.maximum(0.0, need2 - take_trad * denom_trad_wd)
                take_roth = np.minimum(roth_prev, need3)
                gross_roth += take_roth
            else:
                total_avail = tax_prev + trad_prev + roth_prev
                share_tax = np.where(total_avail > 0, tax_prev / total_avail, 0.0)
                share_trad = np.where(total_avail > 0, trad_prev / total_avail, 0.0)
                share_roth = 1 - share_tax - share_trad
                take_tax = np.minimum(tax_prev, (need * share_tax) / denom_tax_wd)
                take_trad = np.minimum(trad_prev, (need * share_trad) / denom_trad_wd)
                take_roth = np.minimum(roth_prev, (need * share_roth))
                gross_tax += take_tax
                gross_trad += take_trad
                gross_roth += take_roth

            taxes_paid_track[:, t] = (gross_tax * gain_frac * eff_capg_wd) + (gross_trad * eff_ord)

        # --- Common post-withdrawal accounting ---
        gross_tax_wd_track[:, t] = gross_tax
        gross_trad_wd_track[:, t] = gross_trad
        gross_roth_wd_track[:, t] = gross_roth

        tax_before = np.maximum(0.0, tax_prev - gross_tax)
        trad_before = np.maximum(0.0, trad_prev - gross_trad)
        roth_before = np.maximum(0.0, roth_prev - gross_roth)

        frac_withdraw = np.where(tax_prev > 0, np.minimum(1.0, gross_tax / tax_prev), 0.0)
        bas_after = bas_prev * (1 - frac_withdraw)
        bas_after = np.minimum(bas_after, tax_before)

        # returns (with optional glide path)
        if glide_on:
            eq_t = glide_equity_frac(age, glide_tax_eq_start, glide_tax_eq_end,
                                     glide_tax_start_age, glide_tax_end_age)
            wt = _build_glide_weights(eq_t, base_w_tax)
            eq_r = glide_equity_frac(age, glide_ret_eq_start, glide_ret_eq_end,
                                     glide_ret_start_age, glide_ret_end_age)
            wr = _build_glide_weights(eq_r, base_w_ret)
        else:
            wt = w_tax
            wr = w_ret
        rt = (1+r_eq)*wt[0] + (1+r_reit)*wt[1] + (1+r_bond)*wt[2] + (1+r_alt)*wt[3] + (1+r_cash)*wt[4] - 1
        rr = (1+r_eq)*wr[0] + (1+r_reit)*wr[1] + (1+r_bond)*wr[2] + (1+r_alt)*wr[3] + (1+r_cash)*wr[4] - 1
        rt = np.clip(rt - taxable_tax_drag - fee_tax, -0.95, 2.0)
        rr = np.clip(rr - fee_ret, -0.95, 2.0)

        taxable[:, t] = tax_before * (1 + rt)
        trad[:, t] = trad_before * (1 + rr)
        roth[:, t] = roth_before * (1 + rr)
        basis[:, t] = bas_after

        # HSA growth
        if hsa_on:
            if hsa_like_ret:
                hsa[:, t] = hsa_after_med * (1 + rr)
            else:
                hsa_r = 0.8 * r_bond + 0.2 * r_cash
                hsa[:, t] = hsa_after_med * (1 + np.clip(hsa_r, -0.95, 2.0))
            hsa_end_track[:, t] = hsa[:, t]
        else:
            hsa[:, t] = 0.0

        # Pre-retirement income, taxes, contributions, and surplus savings
        if contrib_on and age < retire_age:
            ci = infl_index[:, t]
            _in_coast = coast_on and age >= coast_start_age

            if _in_coast:
                # Coast FIRE: part-time income, reduced/no contributions, no employer match
                gross_income_nom = coast_income_real * ci
                employee_401k_nom = coast_contrib_ret * ci
                roth_401k_nom = employee_401k_nom * contrib_roth_401k_frac
                trad_401k_nom = employee_401k_nom * (1 - contrib_roth_401k_frac)
                match_nom = np.zeros(n)  # no employer match during coast
                hsa_contrib_nom = np.zeros(n)
                if hsa_on and coast_contrib_hsa > 0:
                    hsa_contrib_nom = coast_contrib_hsa * ci
                    hsa[:, t] += hsa_contrib_nom
            else:
                # Full-time: real wage growth on top of inflation
                years_worked = age - start_age
                real_growth_factor = (1 + income_growth_real) ** years_worked
                gross_income_nom = (pretax_income_1 + pretax_income_2) * ci * real_growth_factor
                # Employee 401k contributions (today's $ scaled by inflation)
                employee_401k_nom = contrib_ret * ci
                # Split employee contribution: Roth 401k vs Traditional 401k
                roth_401k_nom = employee_401k_nom * contrib_roth_401k_frac
                trad_401k_nom = employee_401k_nom * (1 - contrib_roth_401k_frac)
                # Employer match is always pre-tax traditional
                match_nom = contrib_match * ci
                # HSA contributions (pre-tax)
                hsa_contrib_nom = np.zeros(n)
                if hsa_on and contrib_hsa > 0:
                    hsa_contrib_nom = np.full(n, contrib_hsa * ci)
                    hsa[:, t] += hsa_contrib_nom

            earned_income_track[:, t] = gross_income_nom
            trad[:, t] += trad_401k_nom + match_nom
            roth[:, t] += roth_401k_nom

            # Taxable income for working years: gross - traditional 401k - HSA
            # (Roth 401k is post-tax, so not deducted)
            taxable_wages = np.maximum(0.0, gross_income_nom - trad_401k_nom - match_nom - hsa_contrib_nom)

            # Estimate federal + state tax on working income
            if use_tax_engine:
                # compute_federal_tax handles standard deduction internally
                pre_ret_tax_result = compute_federal_tax(
                    ordinary_income=taxable_wages,
                    qualified_dividends=np.zeros(n),
                    ltcg_realized=np.zeros(n),
                    ss_gross=np.zeros(n),
                    filing_status=filing_status,
                    infl_factor=ci,
                    niit_on=False,
                )
                fed_tax_working = pre_ret_tax_result["fed_total"]
                state_tax_working = taxable_wages * state_rate_ord
                pre_ret_taxes = fed_tax_working + state_tax_working
            else:
                pre_ret_taxes = taxable_wages * eff_ord  # Flat effective rate fallback
            # Also subtract Roth 401k and FICA (~7.65%) from take-home
            fica = gross_income_nom * 0.0765
            pre_ret_taxes = pre_ret_taxes + fica
            pre_ret_tax_track[:, t] = pre_ret_taxes

            # Pre-retirement spending (nominal)
            pre_ret_spend_nom = pre_ret_spend * ci

            # After-tax take-home: gross - 401k employee - HSA - taxes
            take_home = gross_income_nom - employee_401k_nom - hsa_contrib_nom - pre_ret_taxes

            # Surplus after spending → saved to taxable account
            # Include event expenses in pre-retirement spending, and event income in take-home
            _pre_ret_total_spend = pre_ret_spend_nom + evt_expense_nom
            _pre_ret_take_home = take_home + evt_income_nom
            surplus = np.maximum(0.0, _pre_ret_take_home - _pre_ret_total_spend)
            # Explicit extra taxable savings (if user specified additional amount)
            explicit_taxable = contrib_taxable * ci
            total_new_taxable = surplus + explicit_taxable
            taxable[:, t] += total_new_taxable
            basis[:, t] += total_new_taxable
            pre_ret_savings_track[:, t] = total_new_taxable

            # Track spending and MAGI during working years
            core_spend_track[:, t] = pre_ret_spend_nom
            magi_track[:, t] = gross_income_nom  # MAGI ≈ gross income for working years (simplified)
            taxes_paid_track[:, t] = pre_ret_taxes

    liquid = taxable + trad + roth + hsa
    home_equity = np.maximum(0.0, home_value - mortgage)
    net_worth = liquid + home_equity
    liquid_real = liquid / infl_index
    net_worth_real = net_worth / infl_index

    legacy = np.zeros(n)
    for i in range(n):
        d = int(second_death[i])
        idx = min(max(d - start_age, 0), T)
        legacy[i] = net_worth[i, idx]

    ruin_age = np.full(n, end_age + 1, dtype=int)
    for i in range(n):
        for t_idx in range(1, T + 1):
            a = int(ages[t_idx])
            someone_alive = a < death1[i] or a < death2[i]
            if someone_alive and liquid[i, t_idx] <= 0:
                ruin_age[i] = a
                break

    p10_liquid = np.percentile(liquid, 10, axis=0)
    funded_through_age = int(end_age)
    for t_idx in range(1, T + 1):
        if p10_liquid[t_idx] <= 0:
            funded_through_age = int(ages[t_idx - 1])
            break

    # Convert large arrays to float32 before returning to halve cached memory footprint
    def _f32(a):
        return a.astype(np.float32) if a.dtype == np.float64 else a

    return {
        "ages": ages,
        "liquid": _f32(liquid),
        "net_worth": _f32(net_worth),
        "liquid_real": _f32(liquid_real),
        "net_worth_real": _f32(net_worth_real),
        "home_value": _f32(home_value),
        "mortgage": _f32(mortgage),
        "taxable": _f32(taxable),
        "trad": _f32(trad),
        "roth": _f32(roth),
        "hsa": _f32(hsa),
        "magi": _f32(magi_track),
        "irmaa_tier": tier_track,
        "legacy": _f32(legacy),
        "ruin_age": ruin_age,
        "funded_through_age": funded_through_age,
        "second_death": second_death,
        "decomp": {
            "baseline_core": _f32(baseline_core_track),
            "core_adjusted": _f32(core_spend_track),
            "spend_real_track": _f32(core_spend_track / np.maximum(infl_index, 1e-9)),
            "home_cost": _f32(home_cost_track),
            "mort_pay": _f32(mort_pay_track),
            "rent": _f32(rent_track),
            "health": _f32(health_track),
            "medical_nom": _f32(medical_nom_track),
            "hsa_used_med": _f32(hsa_withdraw_track),
            "ltc_cost": _f32(ltc_cost_track),
            "outflow_total": _f32(total_outflow_track),
            "ss_inflow": _f32(ss_nom_track),
            "irmaa": _f32(irmaa_paid_track),
            "gross_rmd": _f32(gross_rmd_track),
            "gross_tax_wd": _f32(gross_tax_wd_track),
            "gross_trad_wd": _f32(gross_trad_wd_track),
            "gross_roth_wd": _f32(gross_roth_wd_track),
            "taxes_paid": _f32(taxes_paid_track),
            "hsa_end": _f32(hsa_end_track),
            "annuity_income": _f32(annuity_income_track),
            "annuity_purchase": _f32(annuity_purchase_track),
            "conv_gross": _f32(conv_gross_track),
            "qcd": _f32(qcd_track),
            "gain_harvest": _f32(gain_harvest_track),
            "essential_spend": _f32(essential_spend_track),
            "discretionary_spend": _f32(discretionary_spend_track),
            "essential_funded": _f32(essential_funded_track),
            "fed_tax": _f32(fed_tax_track),
            "state_tax": _f32(state_tax_track),
            "niit": _f32(niit_track),
            "ss_taxable": _f32(ss_taxable_track),
            "effective_rate": _f32(effective_rate_track),
            "marginal_bracket": _f32(marginal_bracket_track),
            "ltcg_realized": _f32(ltcg_realized_track),
            "qual_divs": _f32(qual_div_track),
            "aca_gross": _f32(aca_gross_track),
            "aca_subsidy": _f32(aca_subsidy_track),
            "aca_net": _f32(aca_net_track),
            "irmaa_part_b": _f32(irmaa_part_b_track),
            "irmaa_part_d": _f32(irmaa_part_d_track),
            "earned_income": _f32(earned_income_track),
            "pre_ret_tax": _f32(pre_ret_tax_track),
            "pre_ret_savings": _f32(pre_ret_savings_track),
            "event_expense": _f32(event_expense_track),
            "event_income": _f32(event_income_track),
        },
        "infl_index": _f32(infl_index),
        "regime_states": regime_track,
    }

def summarize_end(paths: np.ndarray) -> dict:
    end = paths[:, -1]
    return {
        "p_end_pos": float((end > 0).mean()),
        "p_end_nonpos": float((end <= 0).mean()),
        "p10": float(np.percentile(end, 10)),
        "p50": float(np.percentile(end, 50)),
        "p90": float(np.percentile(end, 90)),
    }

# ============================================================
# SECTION 3b: Policy Optimizer
# ============================================================

OPTIMIZER_CONV_LEVELS = [0, 25000, 50000, 75000, 100000, 150000, 200000]
OPTIMIZER_CONV_LABELS = ["$0", "$25k", "$50k", "$75k", "$100k", "$150k", "$200k"]
OPTIMIZER_WD_STRATEGIES = ["taxable_first", "pro_rata"]
OPTIMIZER_WD_LABELS = {"taxable_first": "Taxable first", "pro_rata": "Pro rata"}
OPTIMIZER_OBJECTIVES = {
    "Minimize ruin probability": "ruin",
    "Minimize spending volatility": "spend_vol",
    "Maximize expected legacy": "legacy",
    "Minimize lifetime taxes": "taxes",
}


def run_optimizer(cfg: dict, hold: dict, objective: str) -> dict:
    """Grid search over conversion and withdrawal policies, validate with MC.

    Returns dict with:
      - best: (conv_level, wd_strategy, score)
      - candidates: list of (conv_level, wd_strategy, mc_results) sorted by objective
      - baseline: MC result for current settings
    """
    import time
    t0 = time.time()
    n_fast = 1000  # simulations per candidate

    # Run baseline
    cfg_base = dict(cfg)
    cfg_base["n_sims"] = n_fast
    out_base = simulate(cfg_base, hold)
    base_score = _eval_objective(out_base, cfg_base, objective)

    # Grid search: conversion levels × withdrawal strategies
    candidates = []
    for conv_level in OPTIMIZER_CONV_LEVELS:
        for wd_strat in OPTIMIZER_WD_STRATEGIES:
            cfg_trial = dict(cfg)
            cfg_trial["n_sims"] = n_fast
            cfg_trial["wd_strategy"] = wd_strat
            if conv_level == 0:
                cfg_trial["conv_on"] = False
            else:
                cfg_trial["conv_on"] = True
                cfg_trial["conv_type"] = "fixed"
                cfg_trial["conv_real"] = float(conv_level)
            # Each combo gets a different seed to reduce noise
            cfg_trial["seed"] = int(cfg.get("seed", 42)) + hash((conv_level, wd_strat)) % 10000
            out_trial = simulate(cfg_trial, hold)
            score = _eval_objective(out_trial, cfg_trial, objective)
            candidates.append({
                "conv_level": conv_level,
                "wd_strategy": wd_strat,
                "score": score,
                "success_rate": _success_rate(out_trial, cfg_trial),
                "median_legacy": float(np.median(out_trial["legacy"])),
                "median_tax": float(np.median(out_trial["decomp"]["taxes_paid"].sum(axis=1))),
                "spend_vol": float(np.std(out_trial["decomp"]["core_adjusted"][:, -1] /
                                          np.maximum(out_trial["infl_index"][:, -1], 1e-9))),
            })

    # Sort by objective (lower is better for ruin/vol/taxes, higher for legacy)
    reverse = objective == "legacy"
    candidates.sort(key=lambda c: c["score"], reverse=reverse)

    elapsed = time.time() - t0
    return {
        "candidates": candidates,
        "baseline": {
            "score": base_score,
            "success_rate": _success_rate(out_base, cfg_base),
            "median_legacy": float(np.median(out_base["legacy"])),
            "median_tax": float(np.median(out_base["decomp"]["taxes_paid"].sum(axis=1))),
        },
        "elapsed": elapsed,
        "objective": objective,
    }


def _success_rate(out: dict, cfg: dict) -> float:
    end_age = int(cfg["end_age"])
    ra = out["ruin_age"]
    return 100.0 - float((ra <= end_age).sum()) / len(ra) * 100


def _eval_objective(out: dict, cfg: dict, objective: str) -> float:
    if objective == "ruin":
        # Lower is better: ruin probability
        return 100.0 - _success_rate(out, cfg)
    elif objective == "spend_vol":
        # Lower is better: std dev of real spending at end
        real_spend = out["decomp"]["core_adjusted"] / np.maximum(out["infl_index"], 1e-9)
        return float(np.std(real_spend[:, -1]))
    elif objective == "legacy":
        # Higher is better: median legacy
        return float(np.median(out["legacy"]))
    elif objective == "taxes":
        # Lower is better: median lifetime taxes
        return float(np.median(out["decomp"]["taxes_paid"].sum(axis=1)))
    return 0.0


# ============================================================
# SECTION 4: CSS / Theme
# ============================================================
def inject_theme_css():
    # NOTE: The actual color theme (light mode, text colors, backgrounds) is set in
    # .streamlit/config.toml via [theme]. This CSS only handles things the theme
    # system can't: nav bar styling, custom metric cards, section titles, and
    # minor polish. We intentionally do NOT override widget text/background colors
    # here because that fights with Streamlit's built-in theme engine.
    st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* ---- Base font ---- */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header .stDeployButton { display: none; }

    /* ---- Top navigation bar ---- */
    /* Let Streamlit's light theme handle the nav bar colors natively.
       We just add a subtle bottom border for polish. */
    header[data-testid="stHeader"] {
        border-bottom: 1px solid #e2e8f0;
    }

    /* ---- Section titles (used via st.html) ---- */
    .pro-section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B2A4A;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #00897B;
        display: inline-block;
    }

    /* ---- Custom metric cards (used via st.html) ---- */
    .metric-card {
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .metric-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.15rem;
    }
    .metric-green  .metric-value { color: #00897B; }
    .metric-amber  .metric-value { color: #FF8F00; }
    .metric-coral  .metric-value { color: #E53935; }
    .metric-navy   .metric-value { color: #1B2A4A; }

    /* ---- Bordered column cards: subtle shadow ---- */
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ---- Expander polish ---- */
    details[data-testid="stExpander"] {
        border-radius: 8px;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 600;
    }
    </style>
    """)


def _metric_card_html(label: str, value: str, sub: str = "", color_class: str = "metric-navy") -> str:
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


# ============================================================
# SECTION 5: Chart builders
# ============================================================
def render_wealth_fan_chart(paths: np.ndarray, ages: np.ndarray, title_label: str = "Wealth ($M)",
                            retire_age: int | None = None, y_max: float | None = None):
    """Altair layered fan chart: 3 bands, median line, zero baseline, retirement line, tooltip."""
    pcts = [5, 10, 25, 50, 75, 90, 95]
    fan = {p: np.percentile(paths, p, axis=0) for p in pcts}
    fan_df = pd.DataFrame({"Age": ages})
    for p in pcts:
        fan_df[f"p{p}"] = (fan[p] / 1e6).round(3)

    # Y-axis scale (shared when comparing side-by-side)
    y_scale = alt.Scale(domain=[fan_df["p5"].min(), y_max]) if y_max is not None else alt.Undefined

    # Bands
    band_outer = alt.Chart(fan_df).mark_area(opacity=0.10, color="#1B2A4A").encode(
        x=alt.X("Age:Q", title="Age"),
        y=alt.Y("p5:Q", title=title_label, scale=y_scale),
        y2="p95:Q",
    )
    band_mid = alt.Chart(fan_df).mark_area(opacity=0.18, color="#1B2A4A").encode(
        x="Age:Q", y="p10:Q", y2="p90:Q",
    )
    band_inner = alt.Chart(fan_df).mark_area(opacity=0.28, color="#1B2A4A").encode(
        x="Age:Q", y="p25:Q", y2="p75:Q",
    )

    # Median line
    median_line = alt.Chart(fan_df).mark_line(color="#00897B", strokeWidth=3).encode(
        x="Age:Q", y=alt.Y("p50:Q", title=title_label),
    )

    # Zero baseline
    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[6, 3], color="#E53935", strokeWidth=1
    ).encode(y="y:Q")

    # Tooltip layer
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Age"], empty=False)
    tooltip_layer = alt.Chart(fan_df).mark_point(opacity=0, size=1).encode(
        x="Age:Q", y="p50:Q",
        tooltip=[
            alt.Tooltip("Age:Q", title="Age", format=".0f"),
            alt.Tooltip("p95:Q", title="p95 (great)", format="$.2f"),
            alt.Tooltip("p90:Q", title="p90", format="$.2f"),
            alt.Tooltip("p75:Q", title="p75", format="$.2f"),
            alt.Tooltip("p50:Q", title="Median", format="$.2f"),
            alt.Tooltip("p25:Q", title="p25", format="$.2f"),
            alt.Tooltip("p10:Q", title="p10", format="$.2f"),
            alt.Tooltip("p5:Q", title="p5 (tough)", format="$.2f"),
        ],
    ).add_params(nearest)

    vrule = alt.Chart(fan_df).mark_rule(color="gray", strokeDash=[4, 2], strokeWidth=0.8).encode(
        x="Age:Q",
    ).transform_filter(nearest)

    median_dot = alt.Chart(fan_df).mark_point(
        filled=True, color="#00897B", size=60,
    ).encode(x="Age:Q", y="p50:Q").transform_filter(nearest)

    layers = band_outer + band_mid + band_inner + median_line + zero_rule + vrule + median_dot + tooltip_layer

    # Retirement age vertical line
    if retire_age is not None:
        retire_df = pd.DataFrame({"x": [retire_age]})
        retire_rule = alt.Chart(retire_df).mark_rule(
            strokeDash=[8, 4], color="#FF8F00", strokeWidth=1.5
        ).encode(x="x:Q")
        retire_text = alt.Chart(retire_df).mark_text(
            align="left", dx=4, dy=-10, fontSize=11, color="#FF8F00", fontWeight="bold"
        ).encode(x="x:Q", text=alt.value("Retire"))
        layers = layers + retire_rule + retire_text

    chart = layers.properties(height=380).interactive(bind_x=False)
    st.altair_chart(chart, use_container_width=True)


def render_spending_fan_chart(out: dict, cfg_run: dict, y_max: float | None = None):
    """Teal-colored fan chart for real spending."""
    spend_real_track = out["decomp"]["spend_real_track"]
    retire_idx = max(0, int(cfg_run["retire_age"]) - int(cfg_run["start_age"]))
    ages = out["ages"][retire_idx:]
    paths = spend_real_track[:, retire_idx:]

    if len(ages) < 2 or paths.shape[1] < 2:
        st.info("Not enough post-retirement years to display spending chart.")
        return

    pcts = [10, 25, 50, 75, 90]
    fan = {p: np.percentile(paths, p, axis=0) for p in pcts}
    df = pd.DataFrame({"Age": ages})
    for p in pcts:
        df[f"p{p}"] = (fan[p] / 1e3).round(1)

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Age"], empty=False)

    sp_y_scale = alt.Scale(domain=[0, y_max]) if y_max is not None else alt.Undefined

    outer = alt.Chart(df).mark_area(opacity=0.12, color="#00897B").encode(
        x=alt.X("Age:Q", title="Age"),
        y=alt.Y("p10:Q", title="Annual spending ($K, today's dollars)", scale=sp_y_scale),
        y2="p90:Q",
    )
    inner = alt.Chart(df).mark_area(opacity=0.25, color="#00897B").encode(
        x="Age:Q", y="p25:Q", y2="p75:Q",
    )
    median = alt.Chart(df).mark_line(color="#004D40", strokeWidth=3).encode(
        x="Age:Q", y=alt.Y("p50:Q", title="Annual spending ($K, today's dollars)"),
    )
    tp = alt.Chart(df).mark_point(opacity=0, size=1).encode(
        x="Age:Q", y="p50:Q",
        tooltip=[
            alt.Tooltip("Age:Q", title="Age", format=".0f"),
            alt.Tooltip("p90:Q", title="p90", format="$,.0f"),
            alt.Tooltip("p75:Q", title="p75", format="$,.0f"),
            alt.Tooltip("p50:Q", title="Median", format="$,.0f"),
            alt.Tooltip("p25:Q", title="p25", format="$,.0f"),
            alt.Tooltip("p10:Q", title="p10", format="$,.0f"),
        ],
    ).add_params(nearest)
    vr = alt.Chart(df).mark_rule(color="gray", strokeDash=[4, 2], strokeWidth=0.8).encode(
        x="Age:Q"
    ).transform_filter(nearest)
    dot = alt.Chart(df).mark_point(filled=True, color="#004D40", size=60).encode(
        x="Age:Q", y="p50:Q"
    ).transform_filter(nearest)

    chart = (outer + inner + median + vr + dot + tp).properties(height=320).interactive(bind_x=False)
    st.altair_chart(chart, use_container_width=True)


def render_tornado_chart(df: pd.DataFrame, top_n: int | None = None):
    """Professional paired tornado chart: each variable has + and - bars."""
    chart_df = df.copy()
    if top_n is not None:
        # Sort by max absolute impact, then take top N variables
        chart_df["_max_abs"] = chart_df[["Up ($M)", "Down ($M)"]].abs().max(axis=1)
        chart_df = chart_df.nlargest(top_n, "_max_abs").drop(columns=["_max_abs"])

    # Sort so biggest impact is at top (Altair reverses Y axis)
    chart_df["_max_abs"] = chart_df[["Up ($M)", "Down ($M)"]].abs().max(axis=1)
    chart_df = chart_df.sort_values("_max_abs", ascending=True).drop(columns=["_max_abs"])

    # Melt into long form for paired bars
    long = chart_df.melt(
        id_vars=["Variable"],
        value_vars=["Up ($M)", "Down ($M)"],
        var_name="Direction",
        value_name="Delta",
    )
    long["Test"] = long["Direction"].map({"Up ($M)": "Increase", "Down ($M)": "Decrease"})
    long["Impact"] = long["Delta"].round(2)
    # Color by OUTCOME (helps vs hurts), not by which test it was
    long["Outcome"] = long["Delta"].apply(lambda x: "Helps" if x >= 0 else "Hurts")

    chart_height = max(220, len(chart_df) * 45)

    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Variable:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=300, labelFontSize=12)),
        x=alt.X("Delta:Q", title="Change in median ending net worth ($M)",
                axis=alt.Axis(format="+.1f")),
        color=alt.Color("Outcome:N",
                         scale=alt.Scale(domain=["Helps", "Hurts"],
                                         range=["#00897B", "#E53935"]),
                         legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[
            alt.Tooltip("Variable:N", title="Variable"),
            alt.Tooltip("Test:N", title="Change"),
            alt.Tooltip("Impact:Q", title="Impact ($M)", format="+.2f"),
        ],
    ).properties(height=chart_height)

    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q")

    st.altair_chart(bars + rule, use_container_width=True)


def _run_sensitivity_tests(cfg_run: dict, hold: dict, n_fast: int = 3000) -> pd.DataFrame:
    """Run paired sensitivity tests and return DataFrame with Up/Down deltas per variable."""
    cfg_fast = dict(cfg_run)
    cfg_fast["n_sims"] = n_fast
    base = simulate(cfg_fast, hold)
    base_med = float(np.percentile(base["net_worth"], 50, axis=0)[-1])

    # Build stock return patches that work for both standard and granular modes
    eq_mu_base = float(cfg_fast["scenario_params"]["eq_mu"])
    _return_bump = 0.02  # ±2 percentage points

    # Standard model: patch scenario_params.eq_mu
    # Granular model: also patch every sub-bucket mu proportionally
    def _build_return_patch(delta: float) -> dict:
        patch = {"scenario_params": dict(cfg_fast["scenario_params"], eq_mu=eq_mu_base + delta)}
        if cfg_fast.get("equity_granular_on", False):
            base_sub_mu = dict(cfg_fast.get("equity_sub_mu", {k: v["mu"] for k, v in EQUITY_SUB_DEFAULTS.items()}))
            patch["equity_sub_mu"] = {k: float(v) + delta for k, v in base_sub_mu.items()}
        return patch

    # Spending perturbation: proportional to current spending, capped at ±10%
    spend_base = float(cfg_fast["spend_real"])
    spend_bump = min(25000.0, round(spend_base * 0.10, -3))  # 10% of spending, max $25k, round to nearest $1k
    spend_bump = max(5000.0, spend_bump)  # at least $5k

    # Retirement age: skip if already retired (start_age >= retire_age)
    start_a = int(cfg_fast["start_age"])
    retire_a = int(cfg_fast["retire_age"])
    can_test_retire_age = retire_a > start_a + 2  # need room to move earlier

    # Paired tests: (variable_label, up_label, up_patch, down_label, down_patch)
    paired_tests = [
        (f"Annual Spending ±${spend_bump/1000:.0f}k",
         f"Spend ${spend_bump/1000:.0f}k more", {"spend_real": spend_base + spend_bump},
         f"Spend ${spend_bump/1000:.0f}k less", {"spend_real": max(30000.0, spend_base - spend_bump)}),
        ("SS Claim Age (Spouse 1)",
         "Claim at 70", {"claim1": 70},
         "Claim at 67", {"claim1": 67}),
        ("Stock Returns ±2%",
         "Returns +2%", _build_return_patch(+_return_bump),
         "Returns −2%", _build_return_patch(-_return_bump)),
        ("Inflation ±1%",
         "Inflation +1%", {"infl_mu": float(cfg_fast["infl_mu"]) + 0.01},
         "Inflation −1%", {"infl_mu": float(cfg_fast["infl_mu"]) - 0.01}),
        ("Medical Costs ±$5k/yr",
         "Med costs +$5k", {"hsa_med_real": float(cfg_fast["hsa_med_real"]) + 5000.0},
         "Med costs −$5k", {"hsa_med_real": max(0.0, float(cfg_fast["hsa_med_real"]) - 5000.0)}),
        ("Equity Glide Path",
         "Rising equity 40→70%", {"glide_on": True, "glide_tax_eq_start": 0.40, "glide_tax_eq_end": 0.70,
                                   "glide_tax_start_age": 55, "glide_tax_end_age": 85, "glide_ret_same": True},
         "Declining equity 70→30%", {"glide_on": True, "glide_tax_eq_start": 0.70, "glide_tax_eq_end": 0.30,
                                      "glide_tax_start_age": 55, "glide_tax_end_age": 85, "glide_ret_same": True}),
        ("$200k SPIA at 65",
         "Buy $200k SPIA", {"ann1_on": True, "ann1_type": "SPIA", "ann1_purchase_age": 65,
                             "ann1_income_start_age": 65, "ann1_purchase_amount": 200000.0, "ann1_payout_rate": 0.065},
         "No annuity", {"ann1_on": False}),
        ("Roth: Bracket-fill 24%",
         "Fill to 24%", {"conv_on": True, "conv_type": "bracket_fill", "conv_target_bracket": 0.24,
                          "conv_start": 62, "conv_end": 72, "conv_irmaa_aware": True, "conv_filing_status": "mfj"},
         "No conversions", {"conv_on": False}),
        ("0% LTCG Gain Harvesting",
         "Harvest 0% gains", {"gain_harvest_on": True, "gain_harvest_filing": "mfj"},
         "No harvesting", {"gain_harvest_on": False}),
        ("$30k QCD",
         "$30k QCD", {"qcd_on": True, "qcd_annual_real": 30000.0, "qcd_start_age": 70, "rmd_on": True},
         "No QCD", {"qcd_on": False}),
    ]

    # Only add retirement age test if there's room to move in both directions
    if can_test_retire_age:
        paired_tests.insert(1, ("Retirement Age ±2 yrs",
            "Retire 2 yrs later", {"retire_age": retire_a + 2},
            "Retire 2 yrs earlier", {"retire_age": retire_a - 2}))

    # Add discretionary spending test only if spend_split is already on
    if bool(cfg_fast.get("spend_split_on", False)):
        disc_base = float(cfg_fast.get("spend_discretionary_real", 0.0))
        ess_base = float(cfg_fast.get("spend_essential_real", 0.0))
        disc_bump = min(15000.0, round(disc_base * 0.15, -3))
        disc_bump = max(5000.0, disc_bump)
        paired_tests.append(
            (f"Discretionary ±${disc_bump/1000:.0f}k",
             f"Disc. +${disc_bump/1000:.0f}k", {
                 "spend_discretionary_real": disc_base + disc_bump,
                 "spend_real": ess_base + disc_base + disc_bump},
             f"Disc. −${disc_bump/1000:.0f}k", {
                 "spend_discretionary_real": max(5000.0, disc_base - disc_bump),
                 "spend_real": ess_base + max(5000.0, disc_base - disc_bump)})
        )

    rows = []
    for var_label, up_name, up_patch, down_name, down_patch in paired_tests:
        # Run the "up" variant
        ctmp_up = dict(cfg_fast)
        ctmp_up.update(up_patch)
        o_up = simulate(ctmp_up, hold)
        med_up = float(np.percentile(o_up["net_worth"], 50, axis=0)[-1])

        # Run the "down" variant
        ctmp_dn = dict(cfg_fast)
        ctmp_dn.update(down_patch)
        o_dn = simulate(ctmp_dn, hold)
        med_dn = float(np.percentile(o_dn["net_worth"], 50, axis=0)[-1])

        rows.append({
            "Variable": var_label,
            "Up ($M)": (med_up - base_med) / 1e6,
            "Down ($M)": (med_dn - base_med) / 1e6,
            "Up Label": up_name,
            "Down Label": down_name,
        })

    tor = pd.DataFrame(rows)
    # Sort by max absolute impact across both directions
    tor["_max_abs"] = tor[["Up ($M)", "Down ($M)"]].abs().max(axis=1)
    tor = tor.sort_values("_max_abs", ascending=False).drop(columns=["_max_abs"])
    return tor


def render_mini_tornado(cfg_run: dict, hold: dict, top_n: int = 6):
    """Quick sensitivity analysis with top N factors displayed."""
    with st.spinner("Analyzing sensitivity..."):
        tor = _run_sensitivity_tests(cfg_run, hold, n_fast=800)
    render_tornado_chart(tor, top_n=top_n)


# ============================================================
# SECTION 6: Dialog functions
# ============================================================
@st.dialog("Correlation Settings", width="large")
def open_correlations_dialog():
    cfg = st.session_state["cfg"]
    st.markdown("Control how different assets move relative to each other and to inflation. "
                "Range: -1 (opposite) to +1 (together). 0 = no relationship.")
    c1, c2 = st.columns(2)
    with c1:
        cfg["corr_eq_infl"] = st.number_input(
            "Stocks vs. inflation", min_value=-0.5, max_value=0.5,
            value=float(cfg["corr_eq_infl"]), step=0.05, format="%.2f", key="dlg_corr_eq_infl")
        cfg["corr_reit_infl"] = st.number_input(
            "Real estate vs. inflation", min_value=-0.5, max_value=0.5,
            value=float(cfg["corr_reit_infl"]), step=0.05, format="%.2f", key="dlg_corr_reit_infl")
        cfg["corr_bond_infl"] = st.number_input(
            "Bonds vs. inflation", min_value=-0.8, max_value=0.8,
            value=float(cfg["corr_bond_infl"]), step=0.05, format="%.2f", key="dlg_corr_bond_infl")
    with c2:
        cfg["corr_alt_infl"] = st.number_input(
            "Alternatives vs. inflation", min_value=-0.5, max_value=0.5,
            value=float(cfg["corr_alt_infl"]), step=0.05, format="%.2f", key="dlg_corr_alt_infl")
        cfg["corr_home_infl"] = st.number_input(
            "Home value vs. inflation", min_value=-0.5, max_value=0.5,
            value=float(cfg["corr_home_infl"]), step=0.05, format="%.2f", key="dlg_corr_home_infl")
        cfg["corr_home_eq"] = st.number_input(
            "Home value vs. stocks", min_value=-0.5, max_value=0.8,
            value=float(cfg["corr_home_eq"]), step=0.05, format="%.2f", key="dlg_corr_home_eq")
    if st.button("Save", key="dlg_corr_save", type="primary"):
        st.session_state["cfg"] = cfg
        st.rerun()


@st.dialog("Crash Overlay Settings", width="large")
def open_crash_dialog():
    cfg = st.session_state["cfg"]
    st.markdown("Configure occasional severe market crashes on top of normal market randomness.")
    cfg["crash_on"] = st.toggle("Enable crash overlay", value=bool(cfg["crash_on"]), key="dlg_crash_on")
    if cfg["crash_on"]:
        cfg["crash_prob"] = st.number_input(
            "Chance of a crash in any given year", min_value=0.0, max_value=0.20,
            value=float(cfg["crash_prob"]), step=0.01, format="%.2f", key="dlg_crash_prob")
        c1, c2 = st.columns(2)
        with c1:
            cfg["crash_eq_extra"] = st.number_input(
                "Extra stock loss", min_value=-0.80, max_value=0.00,
                value=float(cfg["crash_eq_extra"]), step=0.05, format="%.2f", key="dlg_crash_eq")
            cfg["crash_reit_extra"] = st.number_input(
                "Extra REIT loss", min_value=-0.80, max_value=0.00,
                value=float(cfg["crash_reit_extra"]), step=0.05, format="%.2f", key="dlg_crash_reit")
        with c2:
            cfg["crash_alt_extra"] = st.number_input(
                "Extra alternatives return", min_value=-0.50, max_value=0.50,
                value=float(cfg["crash_alt_extra"]), step=0.05, format="%.2f", key="dlg_crash_alt")
            cfg["crash_home_extra"] = st.number_input(
                "Extra home value change", min_value=-0.80, max_value=0.20,
                value=float(cfg["crash_home_extra"]), step=0.05, format="%.2f", key="dlg_crash_home")
    if st.button("Save", key="dlg_crash_save", type="primary"):
        st.session_state["cfg"] = cfg
        st.rerun()


@st.dialog("Market Downturn Stress Test", width="large")
def open_sequence_dialog():
    cfg = st.session_state["cfg"]
    st.markdown("Force a market downturn at any age to test your plan's resilience. "
                "Every simulation gets hit — it's not random.")
    cfg["seq_on"] = st.toggle("Force a market downturn at a specific age", value=bool(cfg["seq_on"]), key="dlg_seq_on")
    if cfg["seq_on"]:
        _seq_start_val = int(cfg.get("seq_start_age", 0))
        _seq_default = int(cfg.get("retire_age", 65)) if _seq_start_val <= 0 else _seq_start_val
        cfg["seq_start_age"] = st.number_input(
            "Downturn starts at age",
            min_value=int(cfg.get("start_age", 55)), max_value=int(cfg.get("end_age", 100)),
            value=_seq_default, step=1, key="dlg_seq_start_age")
        cfg["seq_drop"] = st.number_input(
            "Annual return penalty during stress period",
            min_value=-0.50, max_value=0.0,
            value=float(cfg["seq_drop"]), step=0.01, format="%.2f", key="dlg_seq_drop")
        cfg["seq_years"] = st.number_input(
            "How many years the bad period lasts", min_value=1, max_value=10,
            value=int(cfg["seq_years"]), step=1, key="dlg_seq_years")
    if st.button("Save", key="dlg_seq_save", type="primary"):
        st.session_state["cfg"] = cfg
        st.rerun()


# ============================================================
# SECTION 7: Default config initializer
# ============================================================
def init_defaults():
    """Set ALL defaults into st.session_state['cfg'] and default holdings."""
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = {}
    cfg = st.session_state["cfg"]

    def _d(k, v):
        if k not in cfg:
            cfg[k] = v

    # Market outlook
    _d("scenario", "Base")
    _d("manual_override", False)
    _d("override_params", dict(DEFAULT_SCENARIOS["Base"]))

    # Simulation
    _d("n_sims", 3000)
    _d("seed", 42)

    # Ages
    _d("start_age", 55)
    _d("end_age", 90)
    _d("retire_age", 62)

    # Household
    _d("has_spouse", True)
    _d("spouse_age", 55)

    # Spending
    _d("spend_real", 300000.0)
    _d("spend_split_on", False)
    _d("spend_essential_real", 180000.0)
    _d("spend_discretionary_real", 120000.0)
    _d("phase1_end", 70)
    _d("phase2_end", 80)
    _d("phase1_mult", 1.10)
    _d("phase2_mult", 1.00)
    _d("phase3_mult", 0.85)

    # GK guardrails
    _d("gk_on", True)
    _d("gk_upper_pct", 0.20)
    _d("gk_lower_pct", 0.20)
    _d("gk_cut", 0.10)
    _d("gk_raise", 0.05)

    # Inflation
    _d("infl_mu", 0.025)
    _d("infl_vol", 0.010)
    _d("infl_min", -0.10)
    _d("infl_max", 0.30)

    # Correlations
    _d("corr_eq_infl", 0.10)
    _d("corr_reit_infl", 0.10)
    _d("corr_bond_infl", -0.30)
    _d("corr_alt_infl", 0.00)
    _d("corr_home_infl", 0.10)
    _d("corr_home_eq", 0.20)

    # Return distribution
    _d("dist_type", "t")
    _d("t_df", 7)

    # Regime-switching model
    _d("return_model", "standard")      # "standard" or "regime"
    _d("regime_params", dict(DEFAULT_REGIME_PARAMS))
    _d("regime_transition", [list(row) for row in DEFAULT_TRANSITION_MATRIX])
    _d("regime_initial_probs", list(DEFAULT_REGIME_INITIAL_PROBS))

    # Crash overlay
    _d("crash_on", True)
    _d("crash_prob", 0.05)
    _d("crash_eq_extra", -0.40)
    _d("crash_reit_extra", -0.35)
    _d("crash_alt_extra", -0.10)
    _d("crash_home_extra", -0.20)

    # Sequence / market-downturn stress
    _d("seq_on", False)
    _d("seq_drop", -0.25)
    _d("seq_years", 2)
    _d("seq_start_age", 0)  # 0 = start at retirement age (default); any other value = that specific age

    # Taxes — tax engine
    _d("tax_engine_on", True)
    _d("filing_status", "mfj")
    _d("niit_on", False)
    _d("state_rate_ordinary", 0.05)
    _d("state_rate_capgains", 0.05)

    # Taxes — legacy flat rates (used when tax_engine_on is False)
    _d("fed_capg", 0.20)
    _d("fed_div", 0.20)
    _d("fed_ord", 0.28)
    _d("state_capg", 0.05)
    _d("state_ord", 0.05)
    _d("basis_frac", 0.60)
    _d("div_yield", 0.015)
    _d("dist_yield", 0.005)
    _d("tlh_on", True)
    _d("tlh_reduction", 0.35)
    _d("wd_strategy", "taxable_first")
    _d("rmd_on", True)
    _d("rmd_start_age", 75)
    _d("roth_frac", 0.20)

    # Roth conversions
    _d("conv_on", False)
    _d("conv_start", 62)
    _d("conv_end", 72)
    _d("conv_real", 100000.0)
    _d("conv_type", "fixed")             # "fixed" or "bracket_fill"
    _d("conv_target_bracket", 0.24)
    _d("conv_irmaa_aware", True)
    _d("conv_irmaa_target_tier", 0)
    _d("conv_filing_status", "mfj")

    # Gain harvesting (0% LTCG bracket)
    _d("gain_harvest_on", False)
    _d("gain_harvest_filing", "mfj")

    # ACA / pre-65 healthcare
    _d("aca_mode", "simple")           # "simple" or "aca"
    _d("aca_household_size", 2)
    _d("aca_benchmark_premium_real", 20000.0)
    _d("aca_subsidy_schedule", list(ACA_SUBSIDY_SCHEDULE_2024))

    # Qualified Charitable Distributions
    _d("qcd_on", False)
    _d("qcd_annual_real", 20000.0)
    _d("qcd_start_age", 70)
    _d("qcd_max_annual", 105000.0)

    # Glide path (age-based allocation)
    _d("glide_on", False)
    _d("glide_tax_eq_start", 0.60)
    _d("glide_tax_eq_end", 0.40)
    _d("glide_tax_start_age", 55)
    _d("glide_tax_end_age", 80)
    _d("glide_ret_same", True)
    _d("glide_ret_eq_start", 0.60)
    _d("glide_ret_eq_end", 0.40)
    _d("glide_ret_start_age", 55)
    _d("glide_ret_end_age", 80)

    # Annuity 1 (SPIA or DIA)
    _d("ann1_on", False)
    _d("ann1_type", "SPIA")           # "SPIA" or "DIA"
    _d("ann1_purchase_age", 65)
    _d("ann1_income_start_age", 65)    # for SPIA = purchase age; for DIA can be later
    _d("ann1_purchase_amount", 200000.0)
    _d("ann1_payout_rate", 0.065)      # annual payout as fraction of purchase
    _d("ann1_cola_on", False)
    _d("ann1_cola_rate", 0.02)
    _d("ann1_cola_match_inflation", False)

    # Annuity 2 (optional second annuity)
    _d("ann2_on", False)
    _d("ann2_type", "DIA")
    _d("ann2_purchase_age", 60)
    _d("ann2_income_start_age", 75)
    _d("ann2_purchase_amount", 100000.0)
    _d("ann2_payout_rate", 0.08)
    _d("ann2_cola_on", False)
    _d("ann2_cola_rate", 0.02)
    _d("ann2_cola_match_inflation", False)

    # Fees
    _d("fee_tax", 0.002)
    _d("fee_ret", 0.002)

    # Social Security
    _d("fra", 67)
    _d("ss62_1", 2969.0)
    _d("ss62_2", 2969.0)
    _d("claim1", 62)
    _d("claim2", 62)

    # Mortality
    _d("mort_on", True)
    _d("spend_drop_after_death", 0.25)
    _d("home_cost_drop_after_death", 0.25)
    _d("q55_1", 0.006)
    _d("q55_2", 0.005)
    _d("mort_growth", 1.09)

    # Pre-65 health
    _d("pre65_health_on", True)
    _d("pre65_health_real", 20000.0)
    _d("medicare_age", 65)

    # Home
    _d("home_on", True)
    _d("home_value0", 1_000_000.0)
    _d("home_mu", 0.035)
    _d("home_vol", 0.08)
    _d("home_cost_pct", 0.02)
    _d("mortgage_balance0", 400_000.0)
    _d("mortgage_rate", 0.0450)
    _d("mortgage_term_years", 20)
    _d("sale_on", False)
    _d("sale_age", 75)
    _d("selling_cost_pct", 0.06)
    _d("post_sale_mode", "Downsize and keep owning")
    _d("downsize_fraction", 0.60)
    _d("rent_real", 60_000.0)

    # Inheritance
    _d("inh_on", True)
    _d("inh_min", 1_000_000.0)
    _d("inh_mean", 1_500_000.0)
    _d("inh_sigma", 0.35)
    _d("inh_prob", 0.80)
    _d("inh_horizon", 10)

    # Legacy
    _d("legacy_on", True)

    # IRMAA
    _d("irmaa_on", False)
    _d("irmaa_people", 2)
    _d("irmaa_base", 175.0)
    _d("irmaa_t1", 200000.0)
    _d("irmaa_p1", 250.0)
    _d("irmaa_t2", 300000.0)
    _d("irmaa_p2", 400.0)
    _d("irmaa_t3", 400000.0)
    _d("irmaa_p3", 560.0)
    # Phase 3 IRMAA Part B/D schedules
    _d("irmaa_part_b_schedule", list(IRMAA_PART_B_2024_MFJ))
    _d("irmaa_part_d_schedule", list(IRMAA_PART_D_2024_MFJ))

    # LTC
    _d("ltc_on", False)
    _d("ltc_start_age", 75)
    _d("ltc_annual_prob", 0.03)
    _d("ltc_cost_real", 120000.0)
    _d("ltc_duration_mean", 3.0)
    _d("ltc_duration_sigma", 1.5)

    # HSA
    _d("hsa_on", True)
    _d("hsa_balance0", 34000.0)
    _d("hsa_like_ret", True)
    _d("hsa_med_real", 9000.0)
    # Pre-retirement income & contributions
    _d("contrib_on", True)
    _d("pretax_income_1", 200000.0)    # Spouse 1 gross salary (today's $)
    _d("pretax_income_2", 0.0)         # Spouse 2 gross salary (today's $)
    _d("income_growth_real", 0.01)     # Real wage growth above inflation
    _d("contrib_ret_annual", 23500.0)  # Employee 401k/IRA contribution (today's $)
    _d("contrib_roth_401k_frac", 0.0)  # Fraction of 401k that goes Roth (0 = all traditional)
    _d("contrib_match_annual", 11750.0)  # Employer match (always pre-tax)
    _d("contrib_taxable_annual", 0.0)  # Extra after-tax savings to taxable
    _d("contrib_hsa_annual", 4300.0)
    _d("pre_ret_spend_real", 0.0)      # Pre-retirement annual spending (today's $); 0 = use post-ret spend_real
    # Coast FIRE / part-time income phase
    _d("coast_on", False)
    _d("coast_start_age", 55)
    _d("coast_income_real", 50000.0)
    _d("coast_contrib_ret", 0.0)
    _d("coast_contrib_hsa", 0.0)

    # Spending events (one-time and recurring income/expenses)
    _d("spending_events", [])
    _d("spending_floor", 80000.0)

    # Equity sub-buckets (Phase 5)
    _d("equity_granular_on", False)
    _d("equity_sub_weights", {k: v["weight"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    _d("equity_sub_mu", {k: v["mu"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    _d("equity_sub_vol", {k: v["vol"] for k, v in EQUITY_SUB_DEFAULTS.items()})
    _d("tech_bubble_on", False)
    _d("tech_bubble_prob", 0.03)
    _d("tech_bubble_extra_drop", -0.50)

    st.session_state["cfg"] = cfg

    # Default holdings
    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = []

    if "hold" not in st.session_state:
        eq_pct_tax = 60
        eq_pct_ret = 70
        def _simple_w(eq_pct):
            eq = eq_pct / 100.0
            return {"Equities": eq * 0.90, "REIT": eq * 0.10,
                    "Bonds": (1 - eq) * 0.90, "Alternatives": 0.0,
                    "Cash": (1 - eq) * 0.10}
        w_tax = _simple_w(eq_pct_tax)
        w_ret = _simple_w(eq_pct_ret)
        manual_tax = 1_000_000.0
        manual_ret = 2_000_000.0
        st.session_state["hold"] = {
            "total_tax": manual_tax,
            "total_ret": manual_ret,
            "w_tax": w_tax,
            "w_ret": w_ret,
            "dollars_tax": {k: manual_tax * w_tax[k] for k in ASSET_CLASSES},
            "dollars_ret": {k: manual_ret * w_ret[k] for k in ASSET_CLASSES},
        }



# ---- Section completion tracking (Feature 4) ----
# Which config keys belong to each Plan Setup section, used for completion indicators
_SECTION_KEYS = {
    "Basics": ["start_age", "end_age", "retire_age", "has_spouse", "spouse_age"],
    "Income": ["contrib_on", "pretax_income_1", "pretax_income_2", "income_growth_real",
               "contrib_ret_annual", "contrib_roth_401k_frac", "contrib_match_annual",
               "ss62_1", "claim1", "ss62_2", "claim2",
               "coast_on", "coast_start_age", "coast_income_real", "spending_events"],
    "Spending": ["spend_real", "spend_split_on", "spend_essential_real", "spend_discretionary_real",
                 "gk_on", "pre_ret_spend_real", "spending_events"],
    "Home": ["home_on"],
    "Health": ["health_model", "ltc_on"],
    "Taxes": ["tax_engine_on", "filing_status", "conv_on", "rmd_on", "gain_harvest_on",
              "irmaa_on", "qcd_on"],
    "Market": ["return_model", "scenario", "manual_override"],
    "Allocation": ["glide_on", "equity_granular_on"],
    "Stress Tests": ["crash_on", "seq_on"],
    "Advanced": ["n_sims", "seed", "dist_type", "t_df"],
}

# Default values for tracking what's been customized
_SECTION_DEFAULTS = {
    "start_age": 55, "end_age": 90, "retire_age": 62, "has_spouse": True, "spouse_age": 55,
    "contrib_on": False, "pretax_income_1": 200000.0, "pretax_income_2": 0.0,
    "income_growth_real": 0.01, "contrib_ret_annual": 0.0, "contrib_roth_401k_frac": 0.0,
    "contrib_match_annual": 0.0, "ss62_1": 0.0, "claim1": 67, "ss62_2": 0.0, "claim2": 67,
    "coast_on": False, "coast_start_age": 55, "coast_income_real": 50000.0,
    "spend_real": 300000.0, "spend_split_on": False, "spend_essential_real": 180000.0,
    "spend_discretionary_real": 120000.0, "gk_on": True, "pre_ret_spend_real": 0.0,
    "spending_events": [],
    "home_on": False, "health_model": "aca_marketplace", "ltc_on": False,
    "tax_engine_on": True, "filing_status": "mfj", "conv_on": False, "rmd_on": True,
    "gain_harvest_on": False, "irmaa_on": False, "qcd_on": False,
    "return_model": "standard", "scenario": "Base", "manual_override": False,
    "glide_on": False, "equity_granular_on": False,
    "crash_on": True, "seq_on": False,
    "n_sims": 3000, "seed": 42, "dist_type": "t", "t_df": 7,
}

def _section_is_customized(cfg: dict, section_name: str) -> bool:
    """Check if any key in a section differs from its default value."""
    keys = _SECTION_KEYS.get(section_name, [])
    for k in keys:
        if k in cfg and k in _SECTION_DEFAULTS:
            cv = cfg[k]
            dv = _SECTION_DEFAULTS[k]
            if isinstance(dv, float):
                if abs(float(cv) - dv) > 0.001:
                    return True
            elif cv != dv:
                return True
    return False


def _generate_plan_interpretation(pct_success: float, funded_through: int, end_age_val: int,
                                   cfg: dict, sLr: dict, out: dict) -> str:
    """Generate rule-based plain-English interpretation of simulation results."""
    lines = []

    # Overall assessment
    if pct_success >= 95:
        lines.append("**Your plan looks very strong.** You have money left in the vast majority of scenarios.")
    elif pct_success >= 90:
        lines.append("**Your plan looks solid.** There's a small chance of running short in poor-market scenarios, but the odds are well in your favor.")
    elif pct_success >= 80:
        lines.append("**Your plan is reasonable but has some risk.** About 1 in " + f"{int(round(100/(100-pct_success)))}" + " scenarios run out of money. Consider adjusting spending or building in more buffer.")
    elif pct_success >= 70:
        lines.append("**Your plan has notable risk.** Roughly " + f"{100-pct_success:.0f}%" + " of scenarios run short. You may want to reduce spending, delay retirement, or add income sources.")
    else:
        lines.append("**Your plan needs attention.** More than " + f"{100-pct_success:.0f}%" + " of scenarios run out of money before age " + str(end_age_val) + ". Significant changes are recommended.")

    # Funded-through age insight
    if funded_through < end_age_val:
        shortfall_years = end_age_val - funded_through
        shortfall_msg = f"that is {shortfall_years} years short." if shortfall_years > 0 else "cutting it close."
        lines.append(f"In a bad-luck scenario (worst 10%), your money runs out around age {funded_through} — {shortfall_msg}")

    # Real vs nominal gap
    real_med = sLr.get("p50", 0)
    if real_med < 0:
        lines.append("After inflation, the median scenario ends with negative liquid assets — inflation is a real threat to your plan.")

    # Spending level check
    spend = float(cfg.get("spend_real", 0))
    total_portfolio = float(cfg.get("_total_portfolio_approx", 0))
    if total_portfolio <= 0:
        hold = st.session_state.get("hold", {})
        total_portfolio = float(hold.get("total_tax", 0)) + float(hold.get("total_ret", 0))
    if total_portfolio > 0 and spend > 0:
        wr = spend / total_portfolio * 100
        if wr > 5:
            lines.append(f"Your implied withdrawal rate is **{wr:.1f}%**, which is above the commonly cited 4% guideline. "
                         "Reducing spending or increasing savings would improve your odds.")
        elif wr > 4:
            lines.append(f"Your implied withdrawal rate of **{wr:.1f}%** is slightly above the traditional 4% rule. "
                         "Guardrails and flexible spending help manage this.")

    # Social Security insight
    claim1 = int(cfg.get("claim1", 67))
    ss62 = float(cfg.get("ss62_1", 0))
    if ss62 > 0 and claim1 < 70:
        lines.append(f"You're claiming Social Security at {claim1}. Delaying to 70 increases your benefit by about "
                     f"{int((70-claim1)*8)}% — the sensitivity analysis can show you the dollar impact.")

    # Guardrails note
    if not bool(cfg.get("gk_on", False)):
        lines.append("Adaptive spending guardrails are off. Turning them on lets your spending flex with market performance, "
                     "which significantly improves plan survival in bad scenarios.")

    # Spending floor note (when guardrails are on)
    if bool(cfg.get("gk_on", False)):
        _retire_age_interp = int(cfg.get("retire_age", 62))
        _start_age_interp = int(cfg.get("start_age", 55))
        _ri_interp = max(0, _retire_age_interp - _start_age_interp)
        _spend_real_interp = out["decomp"]["spend_real_track"][:, _ri_interp:]
        if _spend_real_interp.shape[1] > 1:
            _floor_val_interp = float(cfg.get("spending_floor", 80000.0))
            _min_spend_interp = np.min(_spend_real_interp, axis=1)
            _pct_floor_interp = float((_min_spend_interp >= _floor_val_interp).mean()) * 100
            if _pct_floor_interp < 80:
                lines.append(f"With guardrails, you're unlikely to fully run out of money, but there's a "
                             f"{100 - _pct_floor_interp:.0f}% chance your spending drops below "
                             f"${_floor_val_interp:,.0f}/yr at some point.")

    # Spending events summary
    events = cfg.get("spending_events", [])
    if events:
        total_exp = sum(float(e.get("amount_real", 0)) for e in events if e.get("type") == "expense"
                        and e.get("start_age") == e.get("end_age"))
        recurring_exp = [e for e in events if e.get("type") == "expense" and e.get("start_age") != e.get("end_age")]
        income_evts = [e for e in events if e.get("type") == "income"]
        parts = []
        if total_exp > 0:
            parts.append(f"${total_exp:,.0f} in one-time expenses")
        if recurring_exp:
            parts.append(f"{len(recurring_exp)} recurring expense event(s)")
        if income_evts:
            parts.append(f"{len(income_evts)} additional income stream(s)")
        if parts:
            lines.append(f"Your plan includes {', '.join(parts)}.")

    return " ".join(lines)


# ---- Regime-switching market presets (Feature 5) ----
_REGIME_PRESETS = {
    "Historical Average": {
        "desc": "Calibrated to match the Base scenario's ~7% equity expected return, but with realistic bull/bear clustering.",
        "params": dict(DEFAULT_REGIME_PARAMS),
        "transition": [list(row) for row in DEFAULT_TRANSITION_MATRIX],
        "initial": list(DEFAULT_REGIME_INITIAL_PROBS),
    },
    "Conservative": {
        "desc": "Lower returns (~4-5% equity). Longer bear markets, shorter bulls. Good for stress-testing.",
        "params": {
            "Bull":   {"eq_mu": 0.09, "eq_vol": 0.14, "bond_mu": 0.03, "bond_vol": 0.04,
                       "reit_mu": 0.08, "reit_vol": 0.16, "alt_mu": 0.06, "alt_vol": 0.10, "cash_mu": 0.02},
            "Normal": {"eq_mu": 0.06, "eq_vol": 0.18, "bond_mu": 0.025, "bond_vol": 0.05,
                       "reit_mu": 0.06, "reit_vol": 0.18, "alt_mu": 0.04, "alt_vol": 0.12, "cash_mu": 0.015},
            "Bear":   {"eq_mu": -0.04, "eq_vol": 0.26, "bond_mu": 0.015, "bond_vol": 0.07,
                       "reit_mu": -0.04, "reit_vol": 0.24, "alt_mu": -0.01, "alt_vol": 0.16, "cash_mu": 0.005},
        },
        "transition": [[0.80, 0.15, 0.05], [0.10, 0.70, 0.20], [0.08, 0.32, 0.60]],
        "initial": [0.20, 0.50, 0.30],
    },
    "Optimistic": {
        "desc": "Higher returns (~10% equity), shorter bear markets. Reflects a favorable economic outlook.",
        "params": {
            "Bull":   {"eq_mu": 0.12, "eq_vol": 0.11, "bond_mu": 0.04, "bond_vol": 0.035,
                       "reit_mu": 0.12, "reit_vol": 0.13, "alt_mu": 0.08, "alt_vol": 0.08, "cash_mu": 0.03},
            "Normal": {"eq_mu": 0.08, "eq_vol": 0.15, "bond_mu": 0.035, "bond_vol": 0.04,
                       "reit_mu": 0.08, "reit_vol": 0.15, "alt_mu": 0.06, "alt_vol": 0.10, "cash_mu": 0.025},
            "Bear":   {"eq_mu": -0.03, "eq_vol": 0.22, "bond_mu": 0.02, "bond_vol": 0.05,
                       "reit_mu": -0.03, "reit_vol": 0.20, "alt_mu": 0.00, "alt_vol": 0.12, "cash_mu": 0.015},
        },
        "transition": [[0.92, 0.06, 0.02], [0.20, 0.72, 0.08], [0.15, 0.40, 0.45]],
        "initial": [0.40, 0.45, 0.15],
    },
    "Volatile": {
        "desc": "Similar expected returns (~5%) but more frequent regime changes and higher volatility within each state.",
        "params": {
            "Bull":   {"eq_mu": 0.14, "eq_vol": 0.18, "bond_mu": 0.035, "bond_vol": 0.05,
                       "reit_mu": 0.13, "reit_vol": 0.20, "alt_mu": 0.09, "alt_vol": 0.14, "cash_mu": 0.025},
            "Normal": {"eq_mu": 0.06, "eq_vol": 0.20, "bond_mu": 0.025, "bond_vol": 0.06,
                       "reit_mu": 0.06, "reit_vol": 0.20, "alt_mu": 0.04, "alt_vol": 0.14, "cash_mu": 0.015},
            "Bear":   {"eq_mu": -0.06, "eq_vol": 0.30, "bond_mu": 0.015, "bond_vol": 0.08,
                       "reit_mu": -0.06, "reit_vol": 0.28, "alt_mu": -0.02, "alt_vol": 0.20, "cash_mu": 0.005},
        },
        "transition": [[0.75, 0.15, 0.10], [0.15, 0.60, 0.25], [0.12, 0.28, 0.60]],
        "initial": [0.25, 0.45, 0.30],
    },
}


def build_cfg_run(cfg: dict) -> dict:
    """Prepare cfg for simulate() by adding computed fields."""
    scenario_params = dict(cfg["override_params"]) if bool(cfg["manual_override"]) else dict(DEFAULT_SCENARIOS[cfg["scenario"]])
    cfg_run = dict(cfg)
    cfg_run["scenario_params"] = scenario_params
    cfg_run["irmaa_schedule"] = [
        (float(cfg_run["irmaa_t1"]), float(cfg_run["irmaa_p1"])),
        (float(cfg_run["irmaa_t2"]), float(cfg_run["irmaa_p2"])),
        (float(cfg_run["irmaa_t3"]), float(cfg_run["irmaa_p3"])),
    ]
    return cfg_run


# ============================================================
# SECTION 8: Page functions
# ============================================================

# ---- 8a: Dashboard page ----
def dashboard_page():
    cfg = st.session_state["cfg"]
    hold = st.session_state["hold"]

    # Header row — compact, with save/load in popovers instead of file_uploader
    _saved_scenarios = st.session_state["saved_scenarios"]
    _saved_names = [s["name"] for s in _saved_scenarios]
    _active_sc = st.session_state.get("_active_scenario_name")

    hdr_left, hdr_switch, hdr_save_sc, hdr_mid, hdr_right = st.columns([3, 1, 1, 1, 1])
    with hdr_left:
        outlook_name = cfg.get("scenario", "Base")
        badges = f"""<span style="display:inline-block; background:#00897B; color:white; padding:2px 12px;
                     border-radius:12px; font-size:0.8rem; font-weight:600; margin-left:12px;
                     vertical-align:middle;">{outlook_name} market outlook</span>"""
        if _active_sc:
            badges += f"""<span style="display:inline-block; background:#1B2A4A; color:white; padding:2px 12px;
                         border-radius:12px; font-size:0.8rem; font-weight:600; margin-left:6px;
                         vertical-align:middle;">📋 {_active_sc}</span>"""
        st.html(f"""
        <div style="margin-bottom:0.2rem;">
            <span style="font-size:1.8rem; font-weight:700; color:#1B2A4A;">RetireLab</span>
            {badges}
        </div>
        """)
        st.caption("By Scott Wallsten · For educational and informational purposes only. Not financial, tax, or investment advice.")
    with hdr_switch:
        if _saved_names:
            _switch_opts = ["— Unsaved —"] + _saved_names
            _cur_idx = (_switch_opts.index(_active_sc) if _active_sc in _switch_opts else 0)
            _switch_pick = st.selectbox("Switch scenario", _switch_opts, index=_cur_idx,
                                        key="hdr_switch_sc", label_visibility="collapsed")
            if _switch_pick != "— Unsaved —" and _switch_pick != _active_sc:
                sc_load = next(s for s in _saved_scenarios if s["name"] == _switch_pick)
                st.session_state["cfg"] = copy.deepcopy(sc_load["cfg"])
                st.session_state["hold"] = copy.deepcopy(sc_load["hold"])
                st.session_state["_active_scenario_name"] = _switch_pick
                st.rerun()
            elif _switch_pick == "— Unsaved —" and _active_sc is not None:
                st.session_state.pop("_active_scenario_name", None)
                st.rerun()
    with hdr_save_sc:
        _saved = st.session_state["saved_scenarios"]
        _n_saved = len(_saved)
        _existing_names = [s["name"] for s in _saved]
        with st.popover(":material/add_circle: Save as Scenario", use_container_width=True):
            # Show what's already saved
            if _existing_names:
                st.caption(f"Saved: {', '.join(_existing_names)}")
            _default_name = st.session_state.get("_active_scenario_name", f"Scenario {_n_saved + 1}")
            _sc_name = st.text_input("Scenario name", value=_default_name, key="save_sc_name")
            _is_overwrite = _sc_name in _existing_names
            if _n_saved >= 4 and not _is_overwrite:
                st.warning("Maximum 4 saved scenarios. Use an existing name to update, or delete one on the Compare page.")
            else:
                _btn_label = f"Update '{_sc_name}'" if _is_overwrite else "Save"
                if st.button(_btn_label, key="save_sc_btn", use_container_width=True):
                    if _is_overwrite:
                        for i, s in enumerate(_saved):
                            if s["name"] == _sc_name:
                                _saved[i] = {
                                    "name": _sc_name,
                                    "cfg": copy.deepcopy(dict(cfg)),
                                    "hold": copy.deepcopy(dict(hold)),
                                }
                                break
                    else:
                        _saved.append({
                            "name": _sc_name,
                            "cfg": copy.deepcopy(dict(cfg)),
                            "hold": copy.deepcopy(dict(hold)),
                        })
                    st.session_state["_active_scenario_name"] = _sc_name
                    st.rerun()
    with hdr_mid:
        st.download_button(
            ":material/download: Save Settings",
            json.dumps(cfg, indent=2), "retirelab_config.json", "application/json",
            use_container_width=True, key="dash_save",
        )
    with hdr_right:
        with st.popover(":material/upload: Load Settings", use_container_width=True):
            up = st.file_uploader(
                "Upload a saved .json config", type=["json", "txt"], key="dash_load",
                label_visibility="collapsed",
            )
            if up is not None:
                st.session_state["cfg"] = json.loads(up.read().decode("utf-8"))
                st.rerun()

    # ---- Getting Started / Portfolio upload ----
    _using_defaults = (abs(hold["total_tax"] - 1_000_000.0) < 1.0 and abs(hold["total_ret"] - 2_000_000.0) < 1.0
                       and not st.session_state.get("_portfolio_uploaded", False))
    # Check if core inputs are still at defaults (first-time user signal)
    _basics_at_defaults = (int(cfg.get("start_age", 55)) == 55
                           and abs(float(cfg.get("spend_real", 300000)) - 300000.0) < 1.0
                           and float(cfg.get("ss62_1", 0)) == 0.0)
    _show_getting_started = _using_defaults or _basics_at_defaults

    if _show_getting_started and not st.session_state.get("_onboarding_dismissed", False):
        with st.expander("🚀 **Getting Started — enter your key numbers** (everything else has smart defaults)", expanded=True):
            st.markdown("Fill in these 5–6 inputs to get a meaningful projection. "
                        "You can fine-tune everything else later in the **Assumptions** tab.")

            gs_c1, gs_c2, gs_c3 = st.columns(3, border=True)
            with gs_c1:
                st.markdown("**Your ages**")
                cfg["start_age"] = st.number_input("Current age", value=int(cfg["start_age"]), step=1, key="gs_start_age")
                cfg["retire_age"] = st.number_input("Planned retirement age", value=int(cfg["retire_age"]), step=1, key="gs_retire_age")
                cfg["end_age"] = st.number_input("Plan through age", value=int(cfg["end_age"]), step=1, key="gs_end_age")
            with gs_c2:
                st.markdown("**Your money**")
                _gs_tax = st.number_input("Taxable account balance ($)", value=float(hold["total_tax"]),
                    step=50000.0, key="gs_total_tax",
                    help="Brokerage accounts, savings, CDs — anything outside retirement accounts.")
                _gs_ret = st.number_input("Retirement account balance ($)", value=float(hold["total_ret"]),
                    step=50000.0, key="gs_total_ret",
                    help="401(k), IRA, 403(b), TSP, Roth IRA — all retirement accounts combined.")
            with gs_c3:
                st.markdown("**Your spending & income**")
                cfg["spend_real"] = st.number_input("Annual spending in retirement ($)", value=float(cfg["spend_real"]),
                    step=10000.0, key="gs_spend_real",
                    help="Total annual living expenses in today's dollars. Don't include mortgage (tracked separately).")
                cfg["ss62_1"] = st.number_input("Monthly Social Security at 62 ($)", value=float(cfg["ss62_1"]),
                    step=100.0, key="gs_ss62",
                    help="Your estimated monthly benefit at age 62. Find yours at ssa.gov/myaccount. Set to 0 if unknown.")

            # Process manual portfolio entry from Getting Started
            if abs(_gs_tax - hold["total_tax"]) > 1.0 or abs(_gs_ret - hold["total_ret"]) > 1.0:
                # User changed balances — update holdings with existing allocation weights
                final_hold = dict(hold)
                final_hold["total_tax"] = _gs_tax
                final_hold["dollars_tax"] = {k: _gs_tax * hold["w_tax"][k] for k in ASSET_CLASSES}
                final_hold["total_ret"] = _gs_ret
                final_hold["dollars_ret"] = {k: _gs_ret * hold["w_ret"][k] for k in ASSET_CLASSES}
                st.session_state["hold"] = final_hold
                hold = final_hold

            # CSV upload option
            with st.popover("📎 Or upload brokerage CSVs instead"):
                uc1, uc2 = st.columns(2)
                with uc1:
                    dash_tax_file = st.file_uploader(
                        "Taxable account CSV", type=["csv", "txt", "tsv"], key="dash_tax_up",
                        help="CSV export from your taxable brokerage (Schwab, Fidelity, Vanguard, etc.)",
                    )
                with uc2:
                    dash_ret_file = st.file_uploader(
                        "Retirement account CSV", type=["csv", "txt", "tsv"], key="dash_ret_up",
                        help="CSV export from your 401(k), IRA, 403(b), or similar retirement account",
                    )
                dash_tax_ok, dash_ret_ok = False, False
                if dash_tax_file:
                    try:
                        tax_df, tax_val = load_snapshot(dash_tax_file)
                        w_tax_u, total_tax_u, dollars_tax_u = weights_and_dollars(tax_df, tax_val)
                        dash_tax_ok = True
                    except Exception as e:
                        st.error(f"Error parsing taxable CSV: {e}")
                if dash_ret_file:
                    try:
                        ret_df, ret_val = load_snapshot(dash_ret_file)
                        w_ret_u, total_ret_u, dollars_ret_u = weights_and_dollars(ret_df, ret_val)
                        dash_ret_ok = True
                    except Exception as e:
                        st.error(f"Error parsing retirement CSV: {e}")
                if dash_tax_ok or dash_ret_ok:
                    final_hold = dict(hold)
                    if dash_tax_ok:
                        final_hold["total_tax"] = total_tax_u
                        final_hold["w_tax"] = w_tax_u
                        final_hold["dollars_tax"] = dollars_tax_u
                    if dash_ret_ok:
                        final_hold["total_ret"] = total_ret_u
                        final_hold["w_ret"] = w_ret_u
                        final_hold["dollars_ret"] = dollars_ret_u
                    st.session_state["hold"] = final_hold
                    st.session_state["_portfolio_uploaded"] = True
                    hold = final_hold
                    st.success(f"Portfolio loaded — Taxable: {fmt_dollars(final_hold['total_tax'])} | "
                               f"Retirement: {fmt_dollars(final_hold['total_ret'])}")

            if st.button("✓ I've entered my basics — dismiss this panel", key="gs_dismiss"):
                st.session_state["_onboarding_dismissed"] = True
                st.rerun()

    # Build cfg_run and simulate
    cfg_run = build_cfg_run(cfg)
    with st.spinner("Running simulation..."):
        out = simulate(cfg_run, hold)

    # Store result for deep dive
    st.session_state["sim_result"] = out
    st.session_state["cfg_run"] = cfg_run

    ages = out["ages"]
    liquid = out["liquid"]
    net_worth = out["net_worth"]
    liquid_real = out["liquid_real"]
    net_worth_real = out["net_worth_real"]

    sL = summarize_end(liquid)
    sN = summarize_end(net_worth)
    sLr = summarize_end(liquid_real)
    sNr = summarize_end(net_worth_real)

    ruin_age = out["ruin_age"]
    end_age_val = int(cfg["end_age"])
    ran_out_while_alive = (ruin_age <= end_age_val).sum()
    pct_success = 100.0 - float(ran_out_while_alive) / len(ruin_age) * 100
    funded_through = out["funded_through_age"]

    # ---- 4 metric cards ----
    if pct_success >= 90:
        sr_color = "metric-green"
    elif pct_success >= 75:
        sr_color = "metric-amber"
    else:
        sr_color = "metric-coral"

    m1, m2, m3, m4 = st.columns(4, border=True)
    with m1:
        st.html(_metric_card_html(
            "Success Rate", f"{pct_success:.0f}%",
            f"of {int(cfg['n_sims']):,} sims have money at {end_age_val}", sr_color
        ))
    with m2:
        ft_str = f"{funded_through}+" if funded_through >= end_age_val else str(funded_through)
        st.html(_metric_card_html(
            "Funded Through Age", ft_str,
            "in a bad (p10) scenario", "metric-navy"
        ))
    with m3:
        st.html(_metric_card_html(
            f"Liquid Assets at {end_age_val}", f"${sL['p50']/1e6:.1f}M",
            f"nominal (real: ${sLr['p50']/1e6:.1f}M)", "metric-navy"
        ))
    with m4:
        st.html(_metric_card_html(
            f"Net Worth at {end_age_val}", f"${sN['p50']/1e6:.1f}M",
            f"nominal (real: ${sNr['p50']/1e6:.1f}M)", "metric-navy"
        ))

    # ---- Interpretive guidance (Feature 3) ----
    _interpretation = _generate_plan_interpretation(pct_success, funded_through, end_age_val, cfg, sLr, out)
    if pct_success >= 90:
        _interp_icon = "✅"
    elif pct_success >= 75:
        _interp_icon = "⚠️"
    else:
        _interp_icon = "🔴"
    st.info(f"{_interp_icon} {_interpretation}")

    # ---- Spending floor probability ----
    _retire_idx_dash = max(0, int(cfg["retire_age"]) - int(out["ages"][0]))
    _spend_real_post_ret = out["decomp"]["spend_real_track"][:, _retire_idx_dash:]
    if _spend_real_post_ret.shape[1] > 1:
        _min_spend_per_sim = np.min(_spend_real_post_ret, axis=1)
        _floor_val = float(cfg.get("spending_floor", 80000.0))
        _pct_above_floor = float((_min_spend_per_sim >= _floor_val).mean()) * 100
        _floor_col1, _floor_col2 = st.columns([1, 3])
        with _floor_col1:
            cfg["spending_floor"] = st.number_input("Spending floor (real $)",
                value=_floor_val, step=5000.0, key="dash_floor",
                help="Minimum acceptable annual spending in today's dollars. Shows the probability your core spending never drops below this level.")
            _floor_val = float(cfg["spending_floor"])
            _pct_above_floor = float((_min_spend_per_sim >= _floor_val).mean()) * 100
        with _floor_col2:
            _floor_color = "metric-green" if _pct_above_floor >= 90 else "metric-amber" if _pct_above_floor >= 75 else "metric-coral"
            st.html(_metric_card_html("Spending Floor Probability", f"{_pct_above_floor:.0f}%",
                f"chance core spending stays above {fmt_dollars(_floor_val)}/yr (real)", _floor_color))

    # ---- View toggles ----
    tc1, tc2 = st.columns(2)
    with tc1:
        view_choice = st.segmented_control(
            "View", ["Liquid Assets", "Net Worth", "Spending"],
            default="Liquid Assets", key="dash_view"
        )
    with tc2:
        dollar_choice = st.segmented_control(
            "Dollars", ["Nominal", "Real"],
            default="Nominal", key="dash_dollars"
        )

    # ---- Fan chart ----
    if view_choice == "Spending":
        st.html('<div class="pro-section-title">Annual Spending (Today\'s Dollars)</div>')
        render_spending_fan_chart(out, cfg_run)
    else:
        is_liquid = view_choice == "Liquid Assets"
        is_real = dollar_choice == "Real"
        if is_liquid:
            paths = liquid_real if is_real else liquid
        else:
            paths = net_worth_real if is_real else net_worth
        label = f"{'Liquid' if is_liquid else 'Net Worth'} ({'Real' if is_real else 'Nominal'} $M)"
        st.html(f'<div class="pro-section-title">{label} Projection</div>')
        render_wealth_fan_chart(paths, ages, title_label=f"{label}",
                                retire_age=int(cfg["retire_age"]))

        # End-of-plan summary
        end_vals = paths[:, -1] / 1e6
        ec1, ec2, ec3 = st.columns(3, border=True)
        with ec1:
            st.metric("Median at end", f"${np.percentile(end_vals, 50):.1f}M")
        with ec2:
            st.metric("Bad luck (p10)", f"${np.percentile(end_vals, 10):.1f}M")
        with ec3:
            st.metric("Good luck (p90)", f"${np.percentile(end_vals, 90):.1f}M")

    # ---- Quick market outlook comparison ----
    with st.expander("Quick market outlook comparison", expanded=False):
        st.caption("See how your plan looks under different market outlooks (uses fewer simulations for speed).")
        if st.button("Run comparison", key="dash_mkt_compare"):
            comp_rows = []
            with st.spinner("Comparing market outlooks..."):
                for sname, sparams in DEFAULT_SCENARIOS.items():
                    ctmp = dict(cfg_run)
                    ctmp["scenario_params"] = dict(sparams)
                    ctmp["n_sims"] = 1000
                    o = simulate(ctmp, hold)
                    liq_end = o["liquid"][:, -1]
                    nw_end = o["net_worth"][:, -1]
                    ra = o["ruin_age"]
                    pct_ok = 100.0 - float((ra <= end_age_val).sum()) / len(ra) * 100
                    comp_rows.append({
                        "Market Outlook": sname,
                        "Won't Run Out": f"{pct_ok:.0f}%",
                        "Median Liquid": f"${np.median(liq_end)/1e6:.1f}M",
                        "Median Net Worth": f"${np.median(nw_end)/1e6:.1f}M",
                        "p10 Liquid": f"${np.percentile(liq_end, 10)/1e6:.1f}M",
                    })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ---- What Matters Most (mini tornado) ----
    st.html('<div class="pro-section-title">What Matters Most</div>')
    st.caption("Top factors affecting your median ending net worth. Full analysis available in Deep Dive.")
    if st.button("▶ Analyze sensitivity", key="dash_sensitivity"):
        render_mini_tornado(cfg_run, hold, top_n=6)


# ---- Event editor (reusable for both Spending and Income tabs) ----
def _next_event_id(events: list) -> int:
    """Return a unique integer ID not already used by any event."""
    used = {ev.get("_id", 0) for ev in events}
    _id = 1
    while _id in used:
        _id += 1
    return _id


def _render_event_editor(cfg: dict, filter_type: str):
    """Render an add/remove event list filtered to 'expense' or 'income' type.

    Events are stored in cfg["spending_events"] (list of dicts, all types mixed).
    Each event is shown as an editable expander.  Preset buttons add a new event
    with sensible defaults and open it for editing.  Every event carries a stable
    ``_id`` used for widget keys so that adding/removing events never causes
    widget-key collisions.
    """
    all_events = list(cfg.get("spending_events", []))
    _cur_age = int(cfg.get("start_age", 55))
    _ret_age = int(cfg.get("retire_age", 62))
    _end_age_cfg = int(cfg.get("end_age", 90))
    _prefix = "exp" if filter_type == "expense" else "inc"

    # Ensure every event has a stable _id (backfill old configs)
    _id_dirty = False
    for ev in all_events:
        if "_id" not in ev:
            ev["_id"] = _next_event_id(all_events)
            _id_dirty = True
    if _id_dirty:
        cfg["spending_events"] = all_events

    # Which event _id was just added and should be open for editing?
    _just_added_id = st.session_state.pop(f"_evt_just_added_{_prefix}", None)

    # Show existing events of this type — each in an editable expander
    filtered = [(i, ev) for i, ev in enumerate(all_events) if ev.get("type") == filter_type]
    _dirty = False
    if filtered:
        for orig_idx, ev in filtered:
            _eid = ev["_id"]
            _age_range = (str(ev.get("start_age", ""))
                          if ev.get("start_age") == ev.get("end_age")
                          else f"{ev.get('start_age')}–{ev.get('end_age')}")
            _tax_tag = ""
            if filter_type == "income" and ev.get("taxable_income", False):
                _tax_tag = " · taxable"
            _icon = "💸" if filter_type == "expense" else "💰"
            _label = f"{_icon} {ev.get('name', 'Event')} — ${ev.get('amount_real', 0):,.0f}/yr · ages {_age_range}{_tax_tag}"
            _open = (_eid == _just_added_id)
            with st.expander(_label, expanded=_open):
                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    _ed_name = st.text_input("Name", value=ev.get("name", ""),
                                             key=f"ev_n_{_prefix}_{_eid}")
                    _ed_amt = st.number_input("Annual amount (today's $)",
                                              value=float(ev.get("amount_real", 0)),
                                              step=1000.0, min_value=0.0,
                                              key=f"ev_a_{_prefix}_{_eid}")
                with _ec2:
                    _ed_sa = st.number_input("Start age",
                                             value=int(ev.get("start_age", _cur_age)),
                                             step=1, key=f"ev_sa_{_prefix}_{_eid}")
                    _ed_ea = st.number_input("End age (same = one-time)",
                                             value=int(ev.get("end_age", _cur_age)),
                                             step=1, key=f"ev_ea_{_prefix}_{_eid}")
                _ed_tax = ev.get("taxable_income", False)
                if filter_type == "income":
                    _ed_tax = st.toggle("Taxable earned income (income tax + FICA)",
                                        value=bool(ev.get("taxable_income", True)),
                                        key=f"ev_tx_{_prefix}_{_eid}")

                # Apply edits back into the event dict
                _updated = {
                    "_id": _eid,
                    "name": _ed_name.strip() or ev.get("name", "Event"),
                    "type": filter_type,
                    "amount_real": _ed_amt,
                    "start_age": _ed_sa,
                    "end_age": max(_ed_sa, _ed_ea),
                    "taxable_income": _ed_tax if filter_type == "income" else False,
                }
                if _updated != all_events[orig_idx]:
                    all_events[orig_idx] = _updated
                    _dirty = True

                if st.button("Remove", key=f"evt_rm_{_prefix}_{_eid}",
                             use_container_width=True):
                    all_events.pop(orig_idx)
                    cfg["spending_events"] = all_events
                    st.rerun()
    else:
        st.caption(f"No {'expense' if filter_type == 'expense' else 'income'} events yet.")

    if _dirty:
        cfg["spending_events"] = all_events

    # ---- Presets (add event with defaults, open for editing) ----
    if filter_type == "expense":
        _presets = {
            "🚗 New car": {"name": "New car", "amount_real": 45000.0,
                        "start_age": _cur_age + 5, "end_age": _cur_age + 5},
            "🏠 Renovation": {"name": "Home renovation", "amount_real": 80000.0,
                           "start_age": _cur_age + 3, "end_age": _cur_age + 3},
            "🎓 College": {"name": "College tuition", "amount_real": 50000.0,
                        "start_age": _cur_age + 5, "end_age": _cur_age + 8},
            "🏥 Long-term care": {"name": "Long-term care", "amount_real": 24000.0,
                               "start_age": 80, "end_age": _end_age_cfg},
        }
    else:
        _presets = {
            "🏦 Pension": {"name": "Pension", "amount_real": 24000.0,
                        "start_age": 65, "end_age": _end_age_cfg, "taxable_income": True},
            "💼 Consulting": {"name": "Consulting income", "amount_real": 60000.0,
                           "start_age": _ret_age, "end_age": _ret_age + 5, "taxable_income": True},
            "🏘️ Rental": {"name": "Rental income", "amount_real": 24000.0,
                       "start_age": _cur_age, "end_age": 75, "taxable_income": True},
            "⏰ Part-time": {"name": "Part-time work", "amount_real": 30000.0,
                          "start_age": _ret_age, "end_age": _ret_age + 8, "taxable_income": True},
        }

    st.caption("Quick-add a preset (click to add, then edit details):")
    _pr_cols = st.columns(len(_presets))
    for i, (pk, pv) in enumerate(_presets.items()):
        with _pr_cols[i]:
            if st.button(pk, key=f"evt_pr_{_prefix}_{pk}", use_container_width=True):
                _new_id = _next_event_id(all_events)
                _new_evt = {
                    "_id": _new_id,
                    "name": pv["name"],
                    "type": filter_type,
                    "amount_real": pv["amount_real"],
                    "start_age": pv["start_age"],
                    "end_age": max(pv["start_age"], pv["end_age"]),
                    "taxable_income": pv.get("taxable_income", False) if filter_type == "income" else False,
                }
                all_events.append(_new_evt)
                cfg["spending_events"] = all_events
                # Store the _id so the new event's expander opens
                st.session_state[f"_evt_just_added_{_prefix}"] = _new_id
                st.rerun()

    # ---- Blank "add custom" ----
    with st.expander("Add a custom event", expanded=False):
        ae_c1, ae_c2 = st.columns(2)
        with ae_c1:
            _new_name = st.text_input("Event name", key=f"evt_name_{_prefix}")
            _new_amount = st.number_input("Annual amount (today's $)",
                value=25000.0, step=1000.0, min_value=0.0, key=f"evt_amt_{_prefix}")
        with ae_c2:
            _new_start = st.number_input("Start age", value=(_cur_age + 5), step=1, key=f"evt_sa_{_prefix}")
            _new_end = st.number_input("End age (same as start = one-time)",
                value=(_cur_age + 5), step=1, key=f"evt_ea_{_prefix}")
            _new_taxable = False
            if filter_type == "income":
                _new_taxable = st.toggle("Taxable earned income (income tax + FICA)",
                    value=True, key=f"evt_tax_{_prefix}")

        if st.button("Add event", key=f"evt_add_{_prefix}", type="primary"):
            if _new_name.strip():
                _new_id = _next_event_id(all_events)
                all_events.append({
                    "_id": _new_id,
                    "name": _new_name.strip(),
                    "type": filter_type,
                    "amount_real": _new_amount,
                    "start_age": _new_start,
                    "end_age": max(_new_start, _new_end),
                    "taxable_income": _new_taxable if filter_type == "income" else False,
                })
                cfg["spending_events"] = all_events
                st.rerun()
            else:
                st.warning("Please enter an event name.")


# ---- 8b: Plan Setup page ----
def plan_setup_page():
    cfg = st.session_state["cfg"]
    hold = st.session_state["hold"]

    st.html('<div style="font-size:1.8rem; font-weight:700; color:#1B2A4A; margin-bottom:0.5rem;">Plan Setup</div>')

    # Section names with completion indicators (Feature 4)
    # Use bare names for the widget (no checkmarks in labels) to avoid
    # label-mismatch issues that cause the tab to jump back to Basics.
    # Instead, show a small "✓" via a caption below.
    _section_names = ["Basics", "Income", "Spending", "Home", "Health", "Taxes", "Market", "Allocation", "Stress Tests", "Advanced"]

    section = st.segmented_control(
        "Section",
        _section_names,
        default="Basics", key="setup_section"
    )
    if section is None:
        section = "Basics"

    # ================================================================
    # BASICS
    # ================================================================
    if section == "Basics":
        st.html('<div class="pro-section-title">Ages & Household</div>')
        c1, c2, c3 = st.columns(3, border=True)
        with c1:
            cfg["start_age"] = st.number_input("Your current age", value=int(cfg["start_age"]), step=1, key="ps_start_age")
        with c2:
            cfg["retire_age"] = st.number_input("Retirement age", value=int(cfg["retire_age"]), step=1, key="ps_retire_age")
        with c3:
            cfg["end_age"] = st.number_input("Plan through age", value=int(cfg["end_age"]), step=1, key="ps_end_age")

        cfg["has_spouse"] = st.toggle("Planning with a spouse/partner", value=bool(cfg["has_spouse"]), key="ps_spouse")
        if cfg["has_spouse"]:
            cfg["spouse_age"] = st.number_input("Spouse's current age", value=int(cfg.get("spouse_age", cfg["start_age"])), step=1, key="ps_spouse_age")

        st.html('<div class="pro-section-title">Portfolio</div>')
        input_method = st.segmented_control(
            "Entry method", ["Manual", "Upload CSV"],
            default="Manual", key="ps_input_method"
        )

        if input_method == "Upload CSV":
            st.caption("Upload brokerage CSV exports. The tool auto-detects value columns and classifies each holding into asset classes.")
            uc1, uc2 = st.columns(2, border=True)
            with uc1:
                st.html('<div style="font-weight:700; color:#1B2A4A; font-size:1rem; margin-bottom:0.5rem;">Taxable Account</div>')
                tax_file = st.file_uploader(
                    "Taxable account CSV", type=["csv", "txt", "tsv"],
                    key="ps_tax_up",
                    help="CSV export from your taxable brokerage (Schwab, Fidelity, Vanguard, etc.) showing each holding and its market value.",
                )
            with uc2:
                st.html('<div style="font-weight:700; color:#1B2A4A; font-size:1rem; margin-bottom:0.5rem;">Retirement Account</div>')
                ret_file = st.file_uploader(
                    "Retirement account CSV", type=["csv", "txt", "tsv"],
                    key="ps_ret_up",
                    help="CSV export from your 401(k), IRA, 403(b), or similar retirement account.",
                )

            # Process whichever files are uploaded
            tax_ok, ret_ok = False, False
            dollars_tax_u, dollars_ret_u = {}, {}
            w_tax_u, w_ret_u = {}, {}
            total_tax_u, total_ret_u = 0.0, 0.0

            # Also track raw DataFrames for sub-bucket classification
            tax_df_raw, tax_val_raw = None, None
            ret_df_raw, ret_val_raw = None, None

            if tax_file:
                try:
                    tax_df, tax_val = load_snapshot(tax_file)
                    w_tax_u, total_tax_u, dollars_tax_u = weights_and_dollars(tax_df, tax_val)
                    tax_df_raw, tax_val_raw = tax_df, tax_val
                    tax_ok = True
                except Exception as e:
                    st.error(f"Error parsing taxable CSV: {e}")

            if ret_file:
                try:
                    ret_df, ret_val = load_snapshot(ret_file)
                    w_ret_u, total_ret_u, dollars_ret_u = weights_and_dollars(ret_df, ret_val)
                    ret_df_raw, ret_val_raw = ret_df, ret_val
                    ret_ok = True
                except Exception as e:
                    st.error(f"Error parsing retirement CSV: {e}")

            if tax_ok or ret_ok:
                # Use uploaded data for whichever accounts were provided; keep existing for the other
                final_hold = dict(hold)
                if tax_ok:
                    final_hold["total_tax"] = total_tax_u
                    final_hold["w_tax"] = w_tax_u
                    final_hold["dollars_tax"] = dollars_tax_u
                if ret_ok:
                    final_hold["total_ret"] = total_ret_u
                    final_hold["w_ret"] = w_ret_u
                    final_hold["dollars_ret"] = dollars_ret_u
                st.session_state["hold"] = final_hold
                hold = final_hold

                # Show summary table
                dt = final_hold.get("dollars_tax", {k: 0.0 for k in ASSET_CLASSES})
                dr = final_hold.get("dollars_ret", {k: 0.0 for k in ASSET_CLASSES})
                df_display = pd.DataFrame({
                    "Asset class": ASSET_CLASSES,
                    "Taxable": [fmt_dollars(dt.get(k, 0)) for k in ASSET_CLASSES],
                    "Retirement": [fmt_dollars(dr.get(k, 0)) for k in ASSET_CLASSES],
                    "Total": [fmt_dollars(dt.get(k, 0) + dr.get(k, 0)) for k in ASSET_CLASSES],
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                st.success(f"Taxable: {fmt_dollars(final_hold['total_tax'])} | "
                           f"Retirement: {fmt_dollars(final_hold['total_ret'])} | "
                           f"Combined: {fmt_dollars(final_hold['total_tax'] + final_hold['total_ret'])}")

                # --- Equity sub-bucket auto-classification from CSV holdings ---
                all_sub_display = []
                combined_sub_weights = {c: 0.0 for c in EQUITY_SUB_CLASSES}
                combined_eq_total = 0.0

                if tax_ok and tax_df_raw is not None:
                    sub_w_tax, disp_tax = equity_sub_from_holdings(tax_df_raw, tax_val_raw)
                    eq_tax_total = float(dollars_tax_u.get("Equities", 0.0))
                    if not disp_tax.empty:
                        disp_tax.insert(0, "Account", "Taxable")
                        all_sub_display.append(disp_tax)
                    for c in EQUITY_SUB_CLASSES:
                        combined_sub_weights[c] += sub_w_tax.get(c, 0.0) * eq_tax_total
                    combined_eq_total += eq_tax_total

                if ret_ok and ret_df_raw is not None:
                    sub_w_ret, disp_ret = equity_sub_from_holdings(ret_df_raw, ret_val_raw)
                    eq_ret_total = float(dollars_ret_u.get("Equities", 0.0))
                    if not disp_ret.empty:
                        disp_ret.insert(0, "Account", "Retirement")
                        all_sub_display.append(disp_ret)
                    for c in EQUITY_SUB_CLASSES:
                        combined_sub_weights[c] += sub_w_ret.get(c, 0.0) * eq_ret_total
                    combined_eq_total += eq_ret_total

                # Normalize combined weights
                if combined_eq_total > 0:
                    for c in EQUITY_SUB_CLASSES:
                        combined_sub_weights[c] /= combined_eq_total

                    # Show classification review
                    with st.expander("🔍 Equity sub-bucket classification from holdings", expanded=False):
                        st.caption("Each equity holding was classified into a sub-bucket using ticker symbols and description keywords. "
                                   "Review the mapping below — you can adjust weights in the Allocation section.")
                        if all_sub_display:
                            review_df = pd.concat(all_sub_display, ignore_index=True)
                            review_df["Value"] = review_df["Value"].apply(lambda v: fmt_dollars(v))
                            st.dataframe(review_df, use_container_width=True, hide_index=True)

                        # Show computed sub-bucket weight summary
                        sw_df = pd.DataFrame({
                            "Sub-Class": EQUITY_SUB_CLASSES,
                            "Weight from CSV": [f"{combined_sub_weights[c]:.1%}" for c in EQUITY_SUB_CLASSES],
                            "Dollar Amount": [fmt_dollars(combined_sub_weights[c] * combined_eq_total) for c in EQUITY_SUB_CLASSES],
                        })
                        st.dataframe(sw_df, use_container_width=True, hide_index=True)
                        st.info(f"Total equities classified: {fmt_dollars(combined_eq_total)}")

                    # Auto-populate sub-bucket weights in config
                    cfg["equity_sub_weights"] = combined_sub_weights
                    # Auto-enable granular mode when we have CSV-derived sub-bucket data
                    if not cfg.get("equity_granular_on", False):
                        cfg["equity_granular_on"] = True
                        st.toast("📊 Equity sub-bucket detail auto-enabled from CSV holdings", icon="📊")
            elif not tax_file and not ret_file:
                st.info("Upload one or both CSV files above. You can also use Manual entry.")
        else:
            bc1, bc2 = st.columns(2, border=True)
            with bc1:
                manual_tax = st.number_input("Total taxable balance", value=float(hold["total_tax"]), step=50000.0, key="ps_manual_tax")
            with bc2:
                manual_ret = st.number_input("Total retirement balance", value=float(hold["total_ret"]), step=50000.0, key="ps_manual_ret")

            with st.expander("Approximate asset mix"):
                mc1, mc2 = st.columns(2)
                with mc1:
                    eq_tax = st.number_input("Taxable: % in stocks", 0, 100, 60, 5, key="ps_eq_tax")
                with mc2:
                    eq_ret = st.number_input("Retirement: % in stocks", 0, 100, 70, 5, key="ps_eq_ret")

                def _simple_w(eq_pct):
                    eq = eq_pct / 100.0
                    return {"Equities": eq * 0.90, "REIT": eq * 0.10,
                            "Bonds": (1 - eq) * 0.90, "Alternatives": 0.0,
                            "Cash": (1 - eq) * 0.10}
                w_tax_m = _simple_w(eq_tax)
                w_ret_m = _simple_w(eq_ret)

            st.session_state["hold"] = {
                "total_tax": manual_tax, "total_ret": manual_ret,
                "w_tax": w_tax_m, "w_ret": w_ret_m,
                "dollars_tax": {k: manual_tax * w_tax_m[k] for k in ASSET_CLASSES},
                "dollars_ret": {k: manual_ret * w_ret_m[k] for k in ASSET_CLASSES},
            }

    # ================================================================
    # INCOME
    # ================================================================
    elif section == "Income":
        st.html('<div class="pro-section-title">Pre-Retirement Earned Income</div>')
        cfg["contrib_on"] = st.toggle("Still working before retirement", value=bool(cfg["contrib_on"]), key="ps_contrib_on",
            help="If you haven't retired yet, the model uses your earned income to pay taxes, fund spending, make retirement contributions, and save any surplus.")
        if not cfg["contrib_on"]:
            st.caption("Toggle on if you're still working. The model will track your income, taxes, contributions, and surplus savings until retirement.")
        if cfg["contrib_on"]:
            inc1, inc2 = st.columns(2, border=True)
            with inc1:
                cfg["pretax_income_1"] = st.number_input("Spouse 1 gross salary (today's $)", value=float(cfg["pretax_income_1"]), step=10000.0, key="ps_income_1",
                    help="Annual gross salary before taxes and deductions. Grows with inflation plus real wage growth.")
            with inc2:
                if cfg["has_spouse"]:
                    cfg["pretax_income_2"] = st.number_input("Spouse 2 gross salary (today's $)", value=float(cfg["pretax_income_2"]), step=10000.0, key="ps_income_2",
                        help="Spouse 2's annual gross salary. Set to 0 if not working.")
                else:
                    cfg["pretax_income_2"] = 0.0
                    st.info("Single-person plan — no Spouse 2 income.")
            cfg["income_growth_real"] = st.number_input("Real wage growth (above inflation)", value=float(cfg["income_growth_real"]),
                min_value=-0.05, max_value=0.10, step=0.005, format="%.3f", key="ps_income_growth",
                help="Annual real (above-inflation) salary growth rate. 0.01 = 1% real raises per year.")
            pre_ret_sp = float(cfg.get("pre_ret_spend_real", 0.0))
            if pre_ret_sp > 0:
                st.info(f"Pre-retirement spending: **${pre_ret_sp:,.0f}**/yr (set in the **Spending** tab)")
            else:
                post_sp = float(cfg.get("spend_real", 120000.0))
                st.info(f"Pre-retirement spending: **${post_sp:,.0f}**/yr (same as retirement spending — change in the **Spending** tab)")

            st.html('<div class="pro-section-title">Retirement Contributions</div>')
            cc1, cc2 = st.columns(2, border=True)
            with cc1:
                cfg["contrib_ret_annual"] = st.number_input("Employee 401k/IRA (today's $)", value=float(cfg["contrib_ret_annual"]), step=1000.0, key="ps_contrib_ret",
                    help="Your annual 401(k) or IRA contribution. The 2024 limit is $23,500 ($31,000 if 50+).")
                cfg["contrib_roth_401k_frac"] = st.number_input("Roth 401k fraction", value=float(cfg["contrib_roth_401k_frac"]),
                    min_value=0.0, max_value=1.0, step=0.1, format="%.1f", key="ps_roth_401k_frac",
                    help="What fraction of your 401k contribution goes to Roth (post-tax). 0 = all traditional, 1 = all Roth 401k.")
            with cc2:
                cfg["contrib_match_annual"] = st.number_input("Employer match (today's $)", value=float(cfg["contrib_match_annual"]), step=1000.0, key="ps_contrib_match",
                    help="Employer matching contribution. Always goes to traditional (pre-tax) 401k.")
                cfg["contrib_hsa_annual"] = st.number_input("HSA contributions (today's $)", value=float(cfg["contrib_hsa_annual"]), step=500.0, key="ps_contrib_hsa",
                    help="Pre-tax HSA contributions. 2024 family limit is $8,300.")

            cfg["contrib_taxable_annual"] = st.number_input("Extra taxable savings on top of surplus (today's $)", value=float(cfg["contrib_taxable_annual"]), step=1000.0, key="ps_contrib_tax",
                help="Additional fixed amount saved to taxable brokerage beyond the automatic surplus (income minus taxes, spending, and contributions). Usually 0 — the model auto-saves your surplus.")

            # Show estimated pre-retirement cash flow summary
            gross = float(cfg["pretax_income_1"]) + float(cfg["pretax_income_2"])
            emp_401k = float(cfg["contrib_ret_annual"])
            hsa_c = float(cfg["contrib_hsa_annual"])
            match_c = float(cfg["contrib_match_annual"])
            roth_401k_f = float(cfg["contrib_roth_401k_frac"])
            trad_deductions = emp_401k * (1 - roth_401k_f) + match_c + hsa_c
            pre_spend = float(cfg["pre_ret_spend_real"])
            if pre_spend <= 0:
                pre_spend = float(cfg.get("spend_real", 120000.0))
            # Rough tax estimate
            rough_agi = max(0, gross - trad_deductions - (30000 if cfg["has_spouse"] else 15000))
            rough_tax = rough_agi * 0.28 + gross * 0.0765  # ~28% marginal + FICA
            take_home = gross - emp_401k - hsa_c - rough_tax
            surplus = max(0, take_home - pre_spend)
            with st.expander("💰 Estimated pre-retirement cash flow (today's $)", expanded=True):
                cf_data = {
                    "": ["Gross income", "− Employee 401k/IRA", "− HSA contribution",
                         "− Est. taxes (fed+state+FICA)", "= Take-home pay",
                         "− Pre-retirement spending", "= **Surplus → taxable savings**",
                         "", "Employer match → traditional 401k"],
                    "Amount": [fmt_dollars(gross), fmt_dollars(-emp_401k), fmt_dollars(-hsa_c),
                               fmt_dollars(-rough_tax), fmt_dollars(take_home),
                               fmt_dollars(-pre_spend), f"**{fmt_dollars(surplus)}**",
                               "", fmt_dollars(match_c)],
                }
                st.dataframe(pd.DataFrame(cf_data), use_container_width=True, hide_index=True)
                st.caption("This is a rough estimate using today's dollars. The simulation uses the full tax engine with inflation-adjusted brackets.")

            # Coast FIRE / Part-time income phase
            st.html('<div class="pro-section-title">Part-Time / Coast FIRE Phase</div>')
            cfg["coast_on"] = st.toggle("Add a part-time income phase before full retirement",
                value=bool(cfg.get("coast_on", False)), key="ps_coast_on",
                help="Models a transition period where you leave your primary career but earn part-time income "
                     "until full retirement. Common in Coast FIRE and Barista FIRE strategies.")
            if not cfg["coast_on"]:
                st.caption("Toggle on to model a period of reduced income between your full-time career and full retirement. "
                           "During this phase, you earn part-time income with reduced or no retirement contributions.")
            if cfg["coast_on"]:
                co1, co2 = st.columns(2, border=True)
                with co1:
                    cfg["coast_start_age"] = st.number_input("Coast phase starts at age",
                        min_value=int(cfg.get("start_age", 55)),
                        max_value=int(cfg.get("retire_age", 62)) - 1,
                        value=min(int(cfg.get("coast_start_age", 55)), int(cfg.get("retire_age", 62)) - 1),
                        step=1, key="ps_coast_start",
                        help="Age when you switch from full-time to part-time. Must be before retirement age.")
                    cfg["coast_income_real"] = st.number_input("Part-time annual income (today's $)",
                        value=float(cfg.get("coast_income_real", 50000.0)), step=5000.0, key="ps_coast_income",
                        help="Annual gross income from part-time work, consulting, freelancing, etc.")
                with co2:
                    cfg["coast_contrib_ret"] = st.number_input("401k/IRA contributions during coast (today's $)",
                        value=float(cfg.get("coast_contrib_ret", 0.0)), step=1000.0, key="ps_coast_contrib_ret",
                        help="Reduced retirement contributions during part-time phase. Set to 0 if no employer plan.")
                    cfg["coast_contrib_hsa"] = st.number_input("HSA contributions during coast (today's $)",
                        value=float(cfg.get("coast_contrib_hsa", 0.0)), step=500.0, key="ps_coast_contrib_hsa",
                        help="HSA contributions during part-time phase. Set to 0 if no HSA access.")
                _coast_years = max(0, int(cfg.get("retire_age", 62)) - int(cfg.get("coast_start_age", 55)))
                st.info(f"Coast phase: age {cfg['coast_start_age']}–{int(cfg.get('retire_age', 62))-1} "
                        f"({_coast_years} years at ${float(cfg['coast_income_real']):,.0f}/yr). "
                        "No employer match during this phase.")

        st.html('<div class="pro-section-title">Social Security</div>')
        cfg["fra"] = st.number_input("Full retirement age (FRA)", value=int(cfg["fra"]), step=1, key="ps_fra")

        sc1, sc2 = st.columns(2, border=True)
        with sc1:
            st.markdown("**Spouse 1**")
            cfg["ss62_1"] = st.number_input("Monthly benefit at 62 (today's $)", value=float(cfg["ss62_1"]), step=50.0, key="ps_ss62_1")
            cfg["claim1"] = st.number_input("Claim age", value=int(cfg["claim1"]), step=1, key="ps_claim1")
        with sc2:
            if cfg["has_spouse"]:
                st.markdown("**Spouse 2**")
                cfg["ss62_2"] = st.number_input("Monthly benefit at 62 (today's $)", value=float(cfg["ss62_2"]), step=50.0, key="ps_ss62_2")
                cfg["claim2"] = st.number_input("Claim age", value=int(cfg["claim2"]), step=1, key="ps_claim2")
            else:
                st.info("Single-person plan - no Spouse 2 inputs.")

        st.html('<div class="pro-section-title">Expected Windfall / Inheritance</div>')
        cfg["inh_on"] = st.toggle("Include a future windfall", value=bool(cfg["inh_on"]), key="ps_inh_on",
            help="Model a possible future inheritance or windfall with a given probability and amount range.")
        if not cfg["inh_on"]:
            st.caption("Toggle on to model a potential inheritance or other financial windfall arriving during the plan.")
        if cfg["inh_on"]:
            ic1, ic2 = st.columns(2, border=True)
            with ic1:
                cfg["inh_prob"] = st.number_input("Probability", min_value=0.0, max_value=1.0, value=float(cfg["inh_prob"]), step=0.05, format="%.2f", key="ps_inh_prob")
                cfg["inh_horizon"] = st.number_input("Could arrive within (years)", value=int(cfg["inh_horizon"]), step=1, key="ps_inh_horizon")
            with ic2:
                cfg["inh_mean"] = st.number_input("Expected amount (today's $)", value=float(cfg["inh_mean"]), step=100000.0, key="ps_inh_mean")
                cfg["inh_min"] = st.number_input("Minimum amount (today's $)", value=float(cfg["inh_min"]), step=100000.0, key="ps_inh_min")
            cfg["inh_sigma"] = st.number_input("Uncertainty (sigma)", min_value=0.05, max_value=0.80, value=float(cfg["inh_sigma"]), step=0.05, format="%.2f", key="ps_inh_sigma")

        # ---- Annuity 1 (SPIA) ----
        st.html('<div class="pro-section-title">Annuity 1 — Guaranteed Income</div>')
        cfg["ann1_on"] = st.toggle("Purchase an annuity (SPIA or DIA)", value=bool(cfg["ann1_on"]), key="ps_ann1_on",
            help="Model buying a single-premium immediate annuity (SPIA) or deferred income annuity (DIA) that provides guaranteed income for life.")
        if not cfg["ann1_on"]:
            st.caption("Toggle on to model a guaranteed income annuity. The purchase price is deducted from your portfolio at the purchase age, and fixed payments begin at the income start age.")
        if cfg["ann1_on"]:
            cfg["ann1_type"] = st.selectbox("Annuity type",
                ["SPIA", "DIA"], index=["SPIA", "DIA"].index(str(cfg["ann1_type"])), key="ps_ann1_type",
                help="SPIA: payments begin immediately at purchase. DIA: payments begin at a later age (deferred).")
            a1c1, a1c2 = st.columns(2, border=True)
            with a1c1:
                cfg["ann1_purchase_age"] = st.number_input("Purchase age", value=int(cfg["ann1_purchase_age"]), step=1, key="ps_ann1_page")
                cfg["ann1_purchase_amount"] = st.number_input("Purchase amount (today's $)", value=float(cfg["ann1_purchase_amount"]), step=25000.0, key="ps_ann1_amt")
                cfg["ann1_payout_rate"] = st.number_input("Annual payout rate", min_value=0.01, max_value=0.20, value=float(cfg["ann1_payout_rate"]), step=0.005, format="%.3f", key="ps_ann1_pr",
                    help="Annual income as fraction of purchase price. Typical SPIA rates: 6-7% at age 65.")
            with a1c2:
                cfg["ann1_income_start_age"] = st.number_input("Income start age", value=int(cfg["ann1_income_start_age"]), step=1, key="ps_ann1_isa",
                    help="For SPIA, typically same as purchase age. For DIA, payments begin later.")
                cfg["ann1_cola_on"] = st.toggle("Include COLA rider", value=bool(cfg["ann1_cola_on"]), key="ps_ann1_cola")
                if cfg["ann1_cola_on"]:
                    cfg["ann1_cola_match_inflation"] = st.toggle("Match simulated inflation", value=bool(cfg["ann1_cola_match_inflation"]), key="ps_ann1_cola_mi",
                        help="If on, payment grows with actual simulated inflation. If off, grows at the fixed rate below.")
                    if not cfg["ann1_cola_match_inflation"]:
                        cfg["ann1_cola_rate"] = st.number_input("Fixed COLA rate", min_value=0.005, max_value=0.05, value=float(cfg["ann1_cola_rate"]), step=0.005, format="%.3f", key="ps_ann1_cola_r")
            ann1_annual = float(cfg["ann1_purchase_amount"]) * float(cfg["ann1_payout_rate"])
            st.info(f"Estimated annual income: **${ann1_annual:,.0f}** (today's dollars) starting at age {cfg['ann1_income_start_age']}.")

        # ---- Annuity 2 (optional) ----
        st.html('<div class="pro-section-title">Annuity 2 — Optional Second Annuity</div>')
        cfg["ann2_on"] = st.toggle("Purchase a second annuity", value=bool(cfg["ann2_on"]), key="ps_ann2_on",
            help="Model a second annuity, e.g. a DIA for longevity insurance starting at 75 or 80.")
        if not cfg["ann2_on"]:
            st.caption("Toggle on to model a second annuity (e.g., a deferred income annuity for late-life income).")
        if cfg["ann2_on"]:
            cfg["ann2_type"] = st.selectbox("Annuity 2 type",
                ["SPIA", "DIA"], index=["SPIA", "DIA"].index(str(cfg["ann2_type"])), key="ps_ann2_type")
            a2c1, a2c2 = st.columns(2, border=True)
            with a2c1:
                cfg["ann2_purchase_age"] = st.number_input("Purchase age", value=int(cfg["ann2_purchase_age"]), step=1, key="ps_ann2_page")
                cfg["ann2_purchase_amount"] = st.number_input("Purchase amount (today's $)", value=float(cfg["ann2_purchase_amount"]), step=25000.0, key="ps_ann2_amt")
                cfg["ann2_payout_rate"] = st.number_input("Annual payout rate", min_value=0.01, max_value=0.20, value=float(cfg["ann2_payout_rate"]), step=0.005, format="%.3f", key="ps_ann2_pr")
            with a2c2:
                cfg["ann2_income_start_age"] = st.number_input("Income start age", value=int(cfg["ann2_income_start_age"]), step=1, key="ps_ann2_isa")
                cfg["ann2_cola_on"] = st.toggle("Include COLA rider", value=bool(cfg["ann2_cola_on"]), key="ps_ann2_cola")
                if cfg["ann2_cola_on"]:
                    cfg["ann2_cola_match_inflation"] = st.toggle("Match simulated inflation", value=bool(cfg["ann2_cola_match_inflation"]), key="ps_ann2_cola_mi")
                    if not cfg["ann2_cola_match_inflation"]:
                        cfg["ann2_cola_rate"] = st.number_input("Fixed COLA rate", min_value=0.005, max_value=0.05, value=float(cfg["ann2_cola_rate"]), step=0.005, format="%.3f", key="ps_ann2_cola_r")
            ann2_annual = float(cfg["ann2_purchase_amount"]) * float(cfg["ann2_payout_rate"])
            st.info(f"Estimated annual income: **${ann2_annual:,.0f}** (today's dollars) starting at age {cfg['ann2_income_start_age']}.")

        # ---- Additional Income Events ----
        st.html('<div class="pro-section-title">Additional Income Streams</div>')
        st.caption("Model recurring or one-time income at specific ages: pension, consulting, rental income, part-time work, etc. "
                   "Taxable income flows through the tax engine (income tax + 15.3% self-employment FICA).")
        _render_event_editor(cfg, "income")

    # ================================================================
    # SPENDING
    # ================================================================
    elif section == "Spending":
        st.html('<div class="pro-section-title">Retirement Spending</div>')
        st.caption("How much you plan to spend each year **after** you retire. Housing, health insurance, and medical costs are tracked separately.")
        cfg["spend_split_on"] = st.toggle("Split into essential vs. discretionary",
            value=bool(cfg.get("spend_split_on", False)), key="ps_split_on",
            help="Separate must-have expenses (food, utilities, insurance) from nice-to-have (travel, dining out). "
                 "Guardrails only cut discretionary. Phase multipliers apply to discretionary only.")
        if cfg["spend_split_on"]:
            sp1, sp2 = st.columns(2, border=True)
            with sp1:
                cfg["spend_essential_real"] = st.number_input("Essential spending (today's $)",
                    value=float(cfg.get("spend_essential_real", 180000.0)), step=5000.0, key="ps_ess_real",
                    help="Must-have: groceries, utilities, basic transport, insurance premiums, property tax.")
            with sp2:
                cfg["spend_discretionary_real"] = st.number_input("Discretionary spending (today's $)",
                    value=float(cfg.get("spend_discretionary_real", 120000.0)), step=5000.0, key="ps_disc_real",
                    help="Nice-to-have: travel, dining out, entertainment, gifts, hobbies.")
            total_split = float(cfg["spend_essential_real"]) + float(cfg["spend_discretionary_real"])
            cfg["spend_real"] = total_split
            st.info(f"**Total: ${total_split:,.0f}** · Guardrails only cut discretionary. Phase multipliers apply to discretionary only. "
                    "Housing, health insurance, and medical costs are tracked separately.")
        else:
            cfg["spend_real"] = st.number_input(
                "Annual retirement spending (today's dollars)", value=float(cfg["spend_real"]), step=10000.0, key="ps_spend_real",
                help="Everyday costs in retirement. Housing, health insurance, and medical are entered separately."
            )
            st.info("Groceries, dining, utilities, transportation, travel, entertainment, etc. "
                    "Housing, health insurance, and medical costs are tracked separately.")

        # Pre-retirement spending (only relevant if still working)
        if cfg.get("contrib_on", False):
            st.html('<div class="pro-section-title">Pre-Retirement Spending</div>')
            st.caption("How much you spend each year **before** you retire. Often higher than retirement spending "
                       "due to commuting, childcare, work clothes, etc. Set to 0 to use your retirement spending amount.")
            cfg["pre_ret_spend_real"] = st.number_input("Pre-retirement spending (today's $)", value=float(cfg.get("pre_ret_spend_real", 0.0)),
                step=5000.0, key="ps_pre_ret_spend",
                help="Annual spending while still working. Set to 0 to use the same amount as your retirement spending above. "
                     "The surplus (income minus spending, taxes, and contributions) is auto-saved to your taxable account.")
            if float(cfg.get("pre_ret_spend_real", 0.0)) <= 0:
                st.caption(f"Using retirement spending of **${float(cfg['spend_real']):,.0f}** for pre-retirement years too.")

        st.html('<div class="pro-section-title">Spending Phases</div>')
        st.caption("Most retirees spend more in early retirement (travel, hobbies), settle into a routine in the middle years, "
                   "and spend less in late retirement. A multiplier of 1.10 means 10% more than your base spending; 0.85 means 15% less.")
        pc1, pc2, pc3 = st.columns(3, border=True)
        with pc1:
            st.markdown("**Early retirement**")
            cfg["phase1_end"] = st.number_input("Ends at age", value=int(cfg["phase1_end"]), step=1, key="ps_p1_end")
            cfg["phase1_mult"] = st.number_input("Multiplier", min_value=0.0, max_value=2.0, value=float(cfg["phase1_mult"]), step=0.05, format="%.2f", key="ps_p1_mult")
        with pc2:
            st.markdown("**Mid retirement**")
            cfg["phase2_end"] = st.number_input("Ends at age", value=int(cfg["phase2_end"]), step=1, key="ps_p2_end")
            cfg["phase2_mult"] = st.number_input("Multiplier", min_value=0.0, max_value=2.0, value=float(cfg["phase2_mult"]), step=0.05, format="%.2f", key="ps_p2_mult")
        with pc3:
            st.markdown("**Late retirement**")
            st.caption("Continues to end of plan")
            cfg["phase3_mult"] = st.number_input("Multiplier", min_value=0.0, max_value=2.0, value=float(cfg["phase3_mult"]), step=0.05, format="%.2f", key="ps_p3_mult")

        st.html('<div class="pro-section-title">Adaptive Spending Guardrails (Guyton-Klinger)</div>')
        cfg["gk_on"] = st.toggle("Enable adaptive spending", value=bool(cfg["gk_on"]), key="ps_gk_on",
            help="If the market drops early in retirement, this automatically reduces your spending to preserve your portfolio. "
                 "If the market booms, it lets you spend a bit more. Most planners recommend this over fixed withdrawals "
                 "because it dramatically improves the chance your money lasts.")
        if not cfg["gk_on"]:
            st.caption("**Recommended.** Without guardrails, you withdraw the same amount regardless of market performance. "
                       "With them, your spending flexes with your portfolio — cutting ~10% after bad years and raising ~5% after good ones. "
                       "This simple rule can improve your plan's survival rate by 10–15 percentage points.")
        if cfg["gk_on"]:
            gc1, gc2 = st.columns(2, border=True)
            with gc1:
                gk_upper_int = st.slider("Upper guardrail (% above initial WR)", 5, 50,
                    int(round(float(cfg["gk_upper_pct"]) * 100)), 5, format="%d%%", key="ps_gk_upper")
                cfg["gk_upper_pct"] = gk_upper_int / 100.0
                cfg["gk_cut"] = st.number_input("Cut %", value=float(cfg["gk_cut"]), step=0.01, format="%.2f", key="ps_gk_cut")
            with gc2:
                gk_lower_int = st.slider("Lower guardrail (% below initial WR)", 5, 50,
                    int(round(float(cfg["gk_lower_pct"]) * 100)), 5, format="%d%%", key="ps_gk_lower")
                cfg["gk_lower_pct"] = gk_lower_int / 100.0
                cfg["gk_raise"] = st.number_input("Raise %", value=float(cfg["gk_raise"]), step=0.01, format="%.2f", key="ps_gk_raise")

        # ---- Spending Events (expenses only) ----
        st.html('<div class="pro-section-title">Spending Events</div>')
        st.caption("Model one-time or recurring expenses at specific ages: a new car, home renovation, college tuition, etc. "
                   "For additional income streams (pension, consulting, rental), see the **Income** tab.")

        _render_event_editor(cfg, "expense")

    # ================================================================
    # HOME
    # ================================================================
    elif section == "Home":
        st.html('<div class="pro-section-title">Your Home</div>')
        cfg["home_on"] = st.toggle("Include home in plan", value=bool(cfg["home_on"]), key="ps_home_on",
            help="Models home equity, appreciation, maintenance costs, mortgage paydown, and potential sale/downsize.")
        if not cfg["home_on"]:
            st.caption("Toggle on to include your home in the simulation: value, appreciation, costs, mortgage, and sale/downsize options.")
        if cfg["home_on"]:
            hc1, hc2 = st.columns(2, border=True)
            with hc1:
                cfg["home_value0"] = st.number_input("Current home value", value=float(cfg["home_value0"]), step=50000.0, key="ps_home_val")
                cfg["home_mu"] = st.number_input("Expected appreciation", value=float(cfg["home_mu"]), step=0.001, format="%.3f", key="ps_home_mu")
                cfg["home_vol"] = st.number_input("Appreciation volatility", value=float(cfg["home_vol"]), step=0.005, format="%.3f", key="ps_home_vol")
                cfg["home_cost_pct"] = st.number_input("Annual costs (% of value)", value=float(cfg["home_cost_pct"]), step=0.001, format="%.3f", key="ps_home_cost")
            with hc2:
                cfg["mortgage_balance0"] = st.number_input("Mortgage balance", value=float(cfg["mortgage_balance0"]), step=25000.0, key="ps_mort_bal")
                cfg["mortgage_rate"] = st.number_input("Mortgage rate", value=float(cfg["mortgage_rate"]), step=0.0005, format="%.4f", key="ps_mort_rate")
                cfg["mortgage_term_years"] = st.number_input("Years remaining", value=int(cfg["mortgage_term_years"]), step=1, key="ps_mort_term")

            st.html('<div class="pro-section-title">Sale / Downsize</div>')
            cfg["sale_on"] = st.toggle("Plan to sell or downsize", value=bool(cfg["sale_on"]), key="ps_sale_on",
                help="Model selling your home at a specific age and either downsizing or renting.")
            if not cfg["sale_on"]:
                st.caption("Toggle on to model selling or downsizing your home at a future age.")
            if cfg["sale_on"]:
                ds1, ds2 = st.columns(2, border=True)
                with ds1:
                    cfg["sale_age"] = st.number_input("Sale age", value=int(cfg["sale_age"]), step=1, key="ps_sale_age")
                    cfg["selling_cost_pct"] = st.number_input("Selling costs (%)", value=float(cfg["selling_cost_pct"]), step=0.005, format="%.3f", key="ps_sell_cost")
                with ds2:
                    cfg["post_sale_mode"] = st.selectbox("After sale",
                        ["Downsize and keep owning", "Sell and rent"],
                        index=["Downsize and keep owning", "Sell and rent"].index(cfg["post_sale_mode"]),
                        key="ps_post_sale")
                    if cfg["post_sale_mode"] == "Downsize and keep owning":
                        cfg["downsize_fraction"] = st.number_input("Fraction for new home", value=float(cfg["downsize_fraction"]), step=0.05, format="%.2f", key="ps_downsize")
                    else:
                        cfg["rent_real"] = st.number_input("Annual rent (today's $)", value=float(cfg["rent_real"]), step=5000.0, key="ps_rent")

    # ================================================================
    # HEALTH
    # ================================================================
    elif section == "Health":
        st.html('<div class="pro-section-title">Pre-Medicare Health Insurance</div>')
        cfg["pre65_health_on"] = st.toggle("Include pre-Medicare health costs", value=bool(cfg["pre65_health_on"]), key="ps_pre65_on",
            help="If you retire before 65, you need to buy health insurance until Medicare kicks in. This models that cost.")
        if not cfg["pre65_health_on"]:
            st.caption("Toggle on to include the cost of health insurance between retirement and Medicare eligibility (age 65).")
        if cfg["pre65_health_on"]:
            hh1, hh2 = st.columns(2, border=True)
            with hh1:
                cfg["medicare_age"] = st.number_input("Medicare starts at age", value=int(cfg["medicare_age"]), step=1, key="ps_medicare_age")
                aca_modes = ["simple", "aca"]
                cfg["aca_mode"] = st.selectbox("Premium model",
                    aca_modes,
                    index=aca_modes.index(str(cfg.get("aca_mode", "simple"))),
                    format_func=lambda x: "Simple (flat annual premium)" if x == "simple" else "ACA (MAGI-based subsidy)",
                    key="ps_aca_mode",
                    help="Simple: flat annual premium you enter. ACA: dynamically computes your premium subsidy "
                         "based on MAGI vs Federal Poverty Level. Requires the tax engine to be enabled.")
                if cfg["aca_mode"] == "aca" and not cfg.get("tax_engine_on", True):
                    st.warning("ACA mode requires the tax engine. Enable it in the Taxes section.")
                    cfg["aca_mode"] = "simple"
            with hh2:
                if cfg.get("aca_mode", "simple") == "simple":
                    cfg["pre65_health_real"] = st.number_input("Annual health insurance cost (today's $)",
                        value=float(cfg["pre65_health_real"]), step=1000.0, key="ps_pre65_cost")
                else:
                    cfg["aca_benchmark_premium_real"] = st.number_input(
                        "Benchmark Silver plan premium (annual, today's $)",
                        value=float(cfg.get("aca_benchmark_premium_real", 20000.0)),
                        step=1000.0, key="ps_aca_bench",
                        help="The second-lowest-cost Silver plan on your exchange. "
                             "This is used to calculate your subsidy amount.")
                    cfg["aca_household_size"] = st.number_input("ACA household size",
                        value=int(cfg.get("aca_household_size", 2)),
                        min_value=1, max_value=4, step=1, key="ps_aca_hh",
                        help="Number of people in your ACA household (determines the FPL threshold).")
                    _fpl = FPL_2024.get(int(cfg["aca_household_size"]), FPL_2024[2])
                    st.info(f"Federal Poverty Level ({cfg['aca_household_size']}-person household): **${_fpl:,}** (inflation-adjusted)")

        st.html('<div class="pro-section-title">Health Savings Account (HSA)</div>')
        cfg["hsa_on"] = st.toggle("Include HSA", value=bool(cfg["hsa_on"]), key="ps_hsa_on",
            help="Model a Health Savings Account that pays medical costs tax-free and grows tax-free.")
        if not cfg["hsa_on"]:
            st.caption("Toggle on to include your HSA balance, medical spending, and tax-free growth in the simulation.")
        if cfg["hsa_on"]:
            hs1, hs2 = st.columns(2, border=True)
            with hs1:
                cfg["hsa_balance0"] = st.number_input("Current HSA balance", value=float(cfg["hsa_balance0"]), step=1000.0, key="ps_hsa_bal")
                cfg["hsa_med_real"] = st.number_input("Annual medical spending (today's $)", value=float(cfg["hsa_med_real"]), step=500.0, key="ps_hsa_med")
            with hs2:
                cfg["hsa_like_ret"] = st.toggle("Invest HSA like retirement", value=bool(cfg["hsa_like_ret"]), key="ps_hsa_invest")

        st.html('<div class="pro-section-title">Long-Term Care Risk</div>')
        cfg["ltc_on"] = st.toggle("Model LTC risk", value=bool(cfg["ltc_on"]), key="ps_ltc_on",
            help="Simulates the chance you or your spouse need paid long-term care (nursing home, assisted living, in-home aide). "
                 "When triggered, LTC costs add to your spending for a random duration.")
        if not cfg["ltc_on"]:
            st.caption("Toggle on to model the financial risk of needing long-term care. "
                       "You can set the probability, annual cost, and average duration.")
        if cfg["ltc_on"]:
            lt1, lt2 = st.columns(2, border=True)
            with lt1:
                cfg["ltc_start_age"] = st.number_input("LTC risk starts at age", value=int(cfg["ltc_start_age"]), step=1, key="ps_ltc_age")
                cfg["ltc_annual_prob"] = st.number_input("Annual probability", min_value=0.0, max_value=0.15, value=float(cfg["ltc_annual_prob"]), step=0.005, format="%.3f", key="ps_ltc_prob")
            with lt2:
                cfg["ltc_cost_real"] = st.number_input("Annual LTC cost (today's $)", value=float(cfg["ltc_cost_real"]), step=10000.0, key="ps_ltc_cost")
                cfg["ltc_duration_mean"] = st.number_input("Average years of care", value=float(cfg["ltc_duration_mean"]), step=0.5, format="%.1f", key="ps_ltc_dur")
                cfg["ltc_duration_sigma"] = st.number_input("Duration uncertainty", value=float(cfg["ltc_duration_sigma"]), step=0.5, format="%.1f", key="ps_ltc_sig")

    # ================================================================
    # TAXES
    # ================================================================
    elif section == "Taxes":
        st.html('<div class="pro-section-title">Tax Engine</div>')

        cfg["tax_engine_on"] = st.toggle("Full tax engine (bracketed federal taxes, SS taxation, LTCG brackets)",
            value=bool(cfg.get("tax_engine_on", True)), key="ps_tax_engine_on",
            help="When enabled, uses actual federal tax brackets, Social Security provisional income taxation, "
                 "LTCG/qualified dividend brackets, and optional NIIT. When disabled, falls back to flat effective rates.")

        if cfg["tax_engine_on"]:
            te1, te2 = st.columns(2, border=True)
            with te1:
                cfg["filing_status"] = st.selectbox("Filing status",
                    ["mfj", "single"],
                    index=["mfj", "single"].index(str(cfg.get("filing_status", "mfj"))),
                    format_func=lambda x: "Married Filing Jointly" if x == "mfj" else "Single",
                    key="ps_filing_status",
                    help="Used for all tax calculations: federal brackets, LTCG brackets, standard deduction, IRMAA, ACA, and gain harvesting.")
                _fs = cfg["filing_status"]
                _std_ded = STANDARD_DEDUCTION_MFJ_2024 if _fs == "mfj" else STANDARD_DEDUCTION_SINGLE_2024
                st.info(f"Standard deduction ({_fs.upper()}): **${_std_ded:,}** (inflation-adjusted each year)")
            with te2:
                cfg["niit_on"] = st.toggle("Include NIIT (3.8% Net Investment Income Tax)",
                    value=bool(cfg.get("niit_on", False)), key="ps_niit_on",
                    help=f"Applies 3.8% surtax on net investment income when MAGI exceeds "
                         f"${NIIT_THRESHOLD_MFJ:,} (MFJ) or ${NIIT_THRESHOLD_SINGLE:,} (Single).")
                st.markdown("**State tax rates (flat)**")
                cfg["state_rate_ordinary"] = st.number_input("State ordinary income rate",
                    value=float(cfg.get("state_rate_ordinary", 0.05)), step=0.005, format="%.3f", key="ps_state_ord_new")
                cfg["state_rate_capgains"] = st.number_input("State capital gains rate",
                    value=float(cfg.get("state_rate_capgains", 0.05)), step=0.005, format="%.3f", key="ps_state_cg_new")
                # Keep legacy keys in sync
                cfg["state_ord"] = cfg["state_rate_ordinary"]
                cfg["state_capg"] = cfg["state_rate_capgains"]
        else:
            st.caption("Using flat effective tax rates (legacy mode).")
            tc1, tc2 = st.columns(2, border=True)
            with tc1:
                st.markdown("**Federal**")
                cfg["fed_capg"] = st.number_input("Capital gains rate", value=float(cfg["fed_capg"]), step=0.01, format="%.2f", key="ps_fed_capg")
                cfg["fed_div"] = st.number_input("Dividend rate", value=float(cfg["fed_div"]), step=0.01, format="%.2f", key="ps_fed_div")
                cfg["fed_ord"] = st.number_input("Ordinary income rate", value=float(cfg["fed_ord"]), step=0.01, format="%.2f", key="ps_fed_ord")
            with tc2:
                st.markdown("**State**")
                cfg["state_capg"] = st.number_input("Cap gains / dividend rate", value=float(cfg["state_capg"]), step=0.005, format="%.3f", key="ps_state_capg")
                cfg["state_ord"] = st.number_input("Ordinary income rate", value=float(cfg["state_ord"]), step=0.005, format="%.3f", key="ps_state_ord")

        st.html('<div class="pro-section-title">Withdrawal Strategy</div>')
        ws1, ws2 = st.columns(2, border=True)
        with ws1:
            cfg["wd_strategy"] = st.selectbox("Withdrawal order",
                ["taxable_first", "pro_rata"],
                index=["taxable_first", "pro_rata"].index(cfg["wd_strategy"]),
                key="ps_wd_strat")
            cfg["basis_frac"] = st.number_input("Cost basis fraction", value=float(cfg["basis_frac"]), step=0.05, format="%.2f", key="ps_basis")
        with ws2:
            cfg["div_yield"] = st.number_input("Dividend yield", value=float(cfg["div_yield"]), step=0.001, format="%.3f", key="ps_div")
            cfg["dist_yield"] = st.number_input("Cap gains distributions", value=float(cfg["dist_yield"]), step=0.001, format="%.3f", key="ps_dist")

        st.html('<div class="pro-section-title">Tax Optimization</div>')
        tl1, tl2 = st.columns(2, border=True)
        with tl1:
            cfg["tlh_on"] = st.toggle("Tax-loss harvesting", value=bool(cfg["tlh_on"]), key="ps_tlh_on")
            if cfg["tlh_on"]:
                cfg["tlh_reduction"] = st.number_input("TLH reduction", min_value=0.0, max_value=1.0, value=float(cfg["tlh_reduction"]), step=0.05, format="%.2f", key="ps_tlh_red")
        with tl2:
            cfg["roth_frac"] = st.number_input("Roth fraction of retirement", min_value=0.0, max_value=1.0, value=float(cfg["roth_frac"]), step=0.05, format="%.2f", key="ps_roth_frac")

        st.html('<div class="pro-section-title">Gain Harvesting & QCDs</div>')
        gh1, gh2 = st.columns(2, border=True)
        with gh1:
            cfg["gain_harvest_on"] = st.toggle("Harvest gains in 0% LTCG bracket",
                value=bool(cfg["gain_harvest_on"]), key="ps_gh_on",
                help="When taxable income is low enough to qualify for the 0% long-term capital gains rate, "
                     "sell and immediately rebuy assets to step up cost basis — no tax owed.")
            if not cfg["tax_engine_on"] and cfg["gain_harvest_on"]:
                cfg["gain_harvest_filing"] = st.selectbox("Filing status for LTCG brackets",
                    ["mfj", "single"],
                    index=["mfj", "single"].index(str(cfg["gain_harvest_filing"])),
                    format_func=lambda x: "Married Filing Jointly" if x == "mfj" else "Single",
                    key="ps_gh_filing")
            elif not cfg["gain_harvest_on"]:
                st.caption("Toggle on to automatically harvest capital gains when they fall in the 0% LTCG bracket.")
        with gh2:
            cfg["qcd_on"] = st.toggle("Qualified Charitable Distributions (QCDs)",
                value=bool(cfg["qcd_on"]), key="ps_qcd_on",
                help="Donate part of your RMD directly to charity. The QCD satisfies your RMD but isn't taxable income, "
                     "which can lower your MAGI and reduce IRMAA surcharges.")
            if cfg["qcd_on"]:
                cfg["qcd_annual_real"] = st.number_input("Annual QCD amount (today's $)",
                    value=float(cfg["qcd_annual_real"]), step=5000.0, key="ps_qcd_amt",
                    help="How much to donate via QCD each year. Capped at your RMD and the IRS annual limit.")
                cfg["qcd_start_age"] = st.number_input("QCD start age",
                    value=int(cfg["qcd_start_age"]), min_value=70, step=1, key="ps_qcd_age",
                    help="QCDs are available starting at age 70½.")
                cfg["qcd_max_annual"] = st.number_input("IRS annual QCD limit (today's $)",
                    value=float(cfg["qcd_max_annual"]), step=5000.0, key="ps_qcd_max",
                    help="Current IRS maximum per person per year.")
            else:
                st.caption("Toggle on to model QCDs that reduce taxable RMD income and IRMAA exposure.")

        st.html('<div class="pro-section-title">RMDs & Roth Conversions</div>')
        rc1, rc2 = st.columns(2, border=True)
        with rc1:
            cfg["rmd_on"] = st.toggle("Include RMDs", value=bool(cfg["rmd_on"]), key="ps_rmd_on")
            if cfg["rmd_on"]:
                cfg["rmd_start_age"] = st.number_input("RMD start age", value=int(cfg["rmd_start_age"]), step=1, key="ps_rmd_age")
        with rc2:
            cfg["conv_on"] = st.toggle("Plan Roth conversions", value=bool(cfg["conv_on"]), key="ps_conv_on",
                help="Convert traditional IRA/401k money to Roth, paying taxes now at today's rates to avoid higher taxes later. "
                     "Especially valuable in the 'gap years' between retirement and Social Security/RMDs when your income is low. "
                     "The bracket-fill strategy automatically converts just enough to stay within a target tax bracket.")
            if cfg["conv_on"]:
                cfg["conv_start"] = st.number_input("Start age", value=int(cfg["conv_start"]), step=1, key="ps_conv_start")
                cfg["conv_end"] = st.number_input("End age", value=int(cfg["conv_end"]), step=1, key="ps_conv_end")
                cfg["conv_type"] = st.selectbox("Conversion strategy",
                    ["fixed", "bracket_fill"],
                    index=["fixed", "bracket_fill"].index(str(cfg["conv_type"])),
                    format_func=lambda x: "Fixed annual amount" if x == "fixed" else "Fill to tax bracket",
                    key="ps_conv_type",
                    help="Fixed: convert a set dollar amount each year. Bracket fill: dynamically convert up to a target federal tax bracket each year.")
                if cfg["conv_type"] == "fixed":
                    cfg["conv_real"] = st.number_input("Annual amount (today's $)", value=float(cfg["conv_real"]), step=10000.0, key="ps_conv_amt")
                else:
                    bracket_options = {0.12: "12% bracket", 0.22: "22% bracket", 0.24: "24% bracket", 0.32: "32% bracket", 0.35: "35% bracket"}
                    bracket_vals = list(bracket_options.keys())
                    current_bracket = float(cfg["conv_target_bracket"])
                    bracket_idx = bracket_vals.index(current_bracket) if current_bracket in bracket_vals else 2
                    cfg["conv_target_bracket"] = st.selectbox("Fill up to bracket",
                        bracket_vals, index=bracket_idx,
                        format_func=lambda x: bracket_options[x],
                        key="ps_conv_bracket",
                        help="Convert enough each year to fill ordinary income up to the top of this bracket.")
                    if not cfg.get("tax_engine_on", True):
                        cfg["conv_filing_status"] = st.selectbox("Filing status",
                            ["mfj", "single"],
                            index=["mfj", "single"].index(str(cfg["conv_filing_status"])),
                            format_func=lambda x: "Married Filing Jointly" if x == "mfj" else "Single",
                            key="ps_conv_filing")
                    else:
                        st.caption(f"Using global filing status: {'MFJ' if cfg.get('filing_status', 'mfj') == 'mfj' else 'Single'}")
                    cfg["conv_irmaa_aware"] = st.toggle("Cap conversions to avoid IRMAA",
                        value=bool(cfg["conv_irmaa_aware"]), key="ps_conv_irmaa",
                        help="Limit conversions so total MAGI stays below the lowest IRMAA threshold.")

        st.html('<div class="pro-section-title">Fees</div>')
        fc1, fc2 = st.columns(2, border=True)
        with fc1:
            cfg["fee_tax"] = st.number_input("Taxable account fee", value=float(cfg["fee_tax"]), step=0.0005, format="%.4f", key="ps_fee_tax")
        with fc2:
            cfg["fee_ret"] = st.number_input("Retirement account fee", value=float(cfg["fee_ret"]), step=0.0005, format="%.4f", key="ps_fee_ret")

        st.html('<div class="pro-section-title">IRMAA</div>')
        cfg["irmaa_on"] = st.toggle("Include Medicare surcharges", value=bool(cfg["irmaa_on"]), key="ps_irmaa_on",
            help="IRMAA = Income-Related Monthly Adjustment Amount. High-income retirees pay more for Medicare Parts B and D.")
        if not cfg["irmaa_on"]:
            st.caption("Toggle on to model Medicare premium surcharges (IRMAA) that apply when retirement income exceeds thresholds.")
        if cfg["irmaa_on"]:
            ir1, ir2 = st.columns(2, border=True)
            with ir1:
                _irmaa_default = int(cfg["irmaa_people"]) if cfg["has_spouse"] else 1
                cfg["irmaa_people"] = st.selectbox("People on Medicare", [1, 2], index=[1, 2].index(_irmaa_default), key="ps_irmaa_ppl")
                cfg["irmaa_base"] = st.number_input("Base monthly premium", value=float(cfg["irmaa_base"]), step=5.0, key="ps_irmaa_base")
            with ir2:
                cfg["irmaa_t1"] = st.number_input("Tier 1 income threshold", value=float(cfg["irmaa_t1"]), step=10000.0, key="ps_irmaa_t1")
                cfg["irmaa_p1"] = st.number_input("Tier 1 premium", value=float(cfg["irmaa_p1"]), step=10.0, key="ps_irmaa_p1")
                cfg["irmaa_t2"] = st.number_input("Tier 2 income threshold", value=float(cfg["irmaa_t2"]), step=10000.0, key="ps_irmaa_t2")
                cfg["irmaa_p2"] = st.number_input("Tier 2 premium", value=float(cfg["irmaa_p2"]), step=10.0, key="ps_irmaa_p2")
                cfg["irmaa_t3"] = st.number_input("Tier 3 income threshold", value=float(cfg["irmaa_t3"]), step=10000.0, key="ps_irmaa_t3")
                cfg["irmaa_p3"] = st.number_input("Tier 3 premium", value=float(cfg["irmaa_p3"]), step=10.0, key="ps_irmaa_p3")

    # ================================================================
    # MARKET
    # ================================================================
    elif section == "Market":
        st.html('<div class="pro-section-title">Return Model</div>')
        cfg["return_model"] = st.selectbox("Return generation model",
            ["standard", "regime"],
            index=["standard", "regime"].index(str(cfg.get("return_model", "standard"))),
            format_func=lambda x: "Standard (single outlook)" if x == "standard" else "Regime-switching (Bull/Normal/Bear cycles)",
            key="ps_return_model",
            help="**Standard** uses one set of return assumptions for the whole simulation — simple and transparent. "
                 "**Regime-switching** models how markets actually behave: long bull runs, occasional bear markets, and "
                 "transitions between them. This produces more realistic risk estimates because it captures the 'clustering' "
                 "of bad years that simple models miss.")

        if cfg["return_model"] == "regime":
            st.caption("The market alternates between Bull, Normal, and Bear states with different return profiles. "
                       "This better captures real market behavior — long expansions punctuated by sharp downturns — "
                       "compared to a single fixed return assumption.")

            # Preset buttons — write directly to widget keys so Streamlit picks up the values
            st.html('<div class="pro-section-title">Quick Presets</div>')
            st.caption("Pick a preset to populate all parameters, or customize below.")
            preset_cols = st.columns(len(_REGIME_PRESETS))
            for i, (pname, pdata) in enumerate(_REGIME_PRESETS.items()):
                with preset_cols[i]:
                    if st.button(f"**{pname}**", key=f"ps_regime_preset_{i}", use_container_width=True,
                                 help=pdata["desc"]):
                        import copy as _cp
                        cfg["regime_params"] = _cp.deepcopy(pdata["params"])
                        cfg["regime_transition"] = _cp.deepcopy(pdata["transition"])
                        cfg["regime_initial_probs"] = list(pdata["initial"])
                        # Write directly into widget keys (Streamlit ignores value= after first render)
                        for _sn in REGIME_NAMES:
                            _sp = pdata["params"][_sn]
                            for _fld in ["eq_mu", "eq_vol", "reit_mu", "reit_vol", "bond_mu", "bond_vol",
                                         "alt_mu", "alt_vol", "cash_mu"]:
                                st.session_state[f"ps_reg_{_sn}_{_fld}"] = float(_sp[_fld])
                        for _si in range(3):
                            for _ci in range(3):
                                st.session_state[f"ps_trans_{_si}_{_ci}"] = float(pdata["transition"][_si][_ci])
                        for _ci in range(3):
                            st.session_state[f"ps_init_{_ci}"] = float(pdata["initial"][_ci])
                        st.rerun()

            r_params = cfg.get("regime_params", dict(DEFAULT_REGIME_PARAMS))
            r_trans = cfg.get("regime_transition", [list(row) for row in DEFAULT_TRANSITION_MATRIX])
            r_init = cfg.get("regime_initial_probs", list(DEFAULT_REGIME_INITIAL_PROBS))

            st.html('<div class="pro-section-title">State Parameters</div>')
            st.caption("Advanced: edit individual return and volatility assumptions per market state.")
            for si, sname in enumerate(REGIME_NAMES):
                sp = r_params.get(sname, DEFAULT_REGIME_PARAMS[sname])
                with st.expander(f"**{sname}** state", expanded=(si == 1)):
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        sp["eq_mu"] = st.number_input(f"Stocks: return", value=float(sp["eq_mu"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_eq_mu")
                        sp["eq_vol"] = st.number_input(f"Stocks: volatility", value=float(sp["eq_vol"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_eq_vol")
                        sp["reit_mu"] = st.number_input(f"REITs: return", value=float(sp["reit_mu"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_reit_mu")
                        sp["reit_vol"] = st.number_input(f"REITs: volatility", value=float(sp["reit_vol"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_reit_vol")
                        sp["cash_mu"] = st.number_input(f"Cash: return", value=float(sp["cash_mu"]), step=0.005, format="%.3f", key=f"ps_reg_{sname}_cash_mu")
                    with rc2:
                        sp["bond_mu"] = st.number_input(f"Bonds: return", value=float(sp["bond_mu"]), step=0.005, format="%.3f", key=f"ps_reg_{sname}_bond_mu")
                        sp["bond_vol"] = st.number_input(f"Bonds: volatility", value=float(sp["bond_vol"]), step=0.005, format="%.3f", key=f"ps_reg_{sname}_bond_vol")
                        sp["alt_mu"] = st.number_input(f"Alts: return", value=float(sp["alt_mu"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_alt_mu")
                        sp["alt_vol"] = st.number_input(f"Alts: volatility", value=float(sp["alt_vol"]), step=0.01, format="%.3f", key=f"ps_reg_{sname}_alt_vol")
                    r_params[sname] = sp
            cfg["regime_params"] = r_params

            st.html('<div class="pro-section-title">Transition Probabilities</div>')
            st.caption("Probability of moving from one state to another each year. Each row must sum to 1.")
            for si, sname in enumerate(REGIME_NAMES):
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    r_trans[si][0] = st.number_input(f"{sname} → Bull", value=float(r_trans[si][0]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=f"ps_trans_{si}_0")
                with tc2:
                    r_trans[si][1] = st.number_input(f"{sname} → Normal", value=float(r_trans[si][1]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=f"ps_trans_{si}_1")
                with tc3:
                    r_trans[si][2] = st.number_input(f"{sname} → Bear", value=float(r_trans[si][2]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=f"ps_trans_{si}_2")
                row_sum = sum(r_trans[si])
                if abs(row_sum - 1.0) > 0.01:
                    st.warning(f"Row '{sname}' sums to {row_sum:.2f} — should be 1.00")
            cfg["regime_transition"] = r_trans

            st.html('<div class="pro-section-title">Initial State Probabilities</div>')
            ip1, ip2, ip3 = st.columns(3)
            with ip1:
                r_init[0] = st.number_input("Bull", value=float(r_init[0]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="ps_init_0")
            with ip2:
                r_init[1] = st.number_input("Normal", value=float(r_init[1]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="ps_init_1")
            with ip3:
                r_init[2] = st.number_input("Bear", value=float(r_init[2]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="ps_init_2")
            init_sum = sum(r_init)
            if abs(init_sum - 1.0) > 0.01:
                st.warning(f"Initial probabilities sum to {init_sum:.2f} — should be 1.00")
            cfg["regime_initial_probs"] = r_init

        else:
            st.html('<div class="pro-section-title">Market Outlooks</div>')
            tbl = scenario_table(DEFAULT_SCENARIOS).copy()
            for c in ["Eq mean","Eq vol","REIT mean","REIT vol","Bond mean","Bond vol","Alt mean","Alt vol","Cash mean"]:
                tbl[c] = (tbl[c] * 100).round(2).astype(str) + "%"
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            cfg["scenario"] = st.selectbox("Select market outlook",
                list(DEFAULT_SCENARIOS.keys()),
                index=list(DEFAULT_SCENARIOS.keys()).index(cfg["scenario"]),
                key="ps_scenario")

            if not cfg.get("manual_override", False):
                cfg["override_params"] = dict(DEFAULT_SCENARIOS[cfg["scenario"]])

            cfg["manual_override"] = st.toggle("Override with custom returns", value=bool(cfg["manual_override"]), key="ps_manual_override")
            if cfg["manual_override"]:
                p = cfg["override_params"]
                mc1, mc2 = st.columns(2, border=True)
                with mc1:
                    p["eq_mu"] = st.number_input("Stocks: return", value=float(p["eq_mu"]), step=0.005, format="%.4f", key="ps_eq_mu")
                    p["eq_vol"] = st.number_input("Stocks: volatility", value=float(p["eq_vol"]), step=0.01, format="%.4f", key="ps_eq_vol")
                    p["reit_mu"] = st.number_input("REITs: return", value=float(p["reit_mu"]), step=0.005, format="%.4f", key="ps_reit_mu")
                    p["reit_vol"] = st.number_input("REITs: volatility", value=float(p["reit_vol"]), step=0.01, format="%.4f", key="ps_reit_vol")
                    p["cash_mu"] = st.number_input("Cash: return", value=float(p["cash_mu"]), step=0.0025, format="%.4f", key="ps_cash_mu")
                with mc2:
                    p["bond_mu"] = st.number_input("Bonds: return", value=float(p["bond_mu"]), step=0.0025, format="%.4f", key="ps_bond_mu")
                    p["bond_vol"] = st.number_input("Bonds: volatility", value=float(p["bond_vol"]), step=0.005, format="%.4f", key="ps_bond_vol")
                    p["alt_mu"] = st.number_input("Alternatives: return", value=float(p["alt_mu"]), step=0.005, format="%.4f", key="ps_alt_mu")
                    p["alt_vol"] = st.number_input("Alternatives: volatility", value=float(p["alt_vol"]), step=0.01, format="%.4f", key="ps_alt_vol")
                cfg["override_params"] = p

        st.html('<div class="pro-section-title">Inflation</div>')
        inf1, inf2 = st.columns(2, border=True)
        with inf1:
            cfg["infl_mu"] = st.number_input("Expected inflation", value=float(cfg["infl_mu"]), step=0.001, format="%.3f", key="ps_infl_mu")
            cfg["infl_vol"] = st.number_input("Inflation volatility", value=float(cfg["infl_vol"]), step=0.001, format="%.3f", key="ps_infl_vol")
        with inf2:
            cfg["infl_min"] = st.number_input("Minimum inflation", value=float(cfg["infl_min"]), step=0.01, format="%.2f", key="ps_infl_min")
            cfg["infl_max"] = st.number_input("Maximum inflation", value=float(cfg["infl_max"]), step=0.01, format="%.2f", key="ps_infl_max")

    # ================================================================
    # ALLOCATION (Glide Path)
    # ================================================================
    elif section == "Allocation":
        st.html('<div class="pro-section-title">Age-Based Glide Path</div>')
        cfg["glide_on"] = st.toggle("Enable age-based allocation glide path", value=bool(cfg["glide_on"]), key="ps_glide_on",
            help="Gradually shift your portfolio mix as you age. The traditional approach reduces stocks over time to protect against "
                 "a crash when you can't wait for recovery. Some research suggests a 'rising equity' path (starting conservative and "
                 "adding stocks later) may actually work better in retirement since early losses are the most dangerous.")
        if not cfg["glide_on"]:
            st.caption("Without a glide path, your portfolio weights stay constant forever. A glide path lets you start conservative "
                       "in early retirement (when sequence risk is highest) or follow a traditional declining-equity strategy.")
        if cfg["glide_on"]:
            st.html('<div class="pro-section-title">Taxable Account Glide Path</div>')
            gc1, gc2 = st.columns(2, border=True)
            with gc1:
                cfg["glide_tax_eq_start"] = st.number_input("Starting equity %", min_value=0.0, max_value=1.0,
                    value=float(cfg["glide_tax_eq_start"]), step=0.05, format="%.2f", key="ps_glide_tax_start",
                    help="Equity allocation at the start age (as a decimal, e.g. 0.60 = 60%).")
                cfg["glide_tax_start_age"] = st.number_input("Glide start age", value=int(cfg["glide_tax_start_age"]), step=1, key="ps_glide_tax_sage")
            with gc2:
                cfg["glide_tax_eq_end"] = st.number_input("Ending equity %", min_value=0.0, max_value=1.0,
                    value=float(cfg["glide_tax_eq_end"]), step=0.05, format="%.2f", key="ps_glide_tax_end",
                    help="Equity allocation at the end age.")
                cfg["glide_tax_end_age"] = st.number_input("Glide end age", value=int(cfg["glide_tax_end_age"]), step=1, key="ps_glide_tax_eage")
            if float(cfg["glide_tax_eq_end"]) > float(cfg["glide_tax_eq_start"]):
                st.info("📈 **Rising equity glidepath** — this increases stock allocation over time, which some research suggests may improve retirement outcomes.")
            elif float(cfg["glide_tax_eq_end"]) < float(cfg["glide_tax_eq_start"]):
                st.info("📉 **Traditional declining glidepath** — gradually reduces equity exposure as you age.")

            st.html('<div class="pro-section-title">Retirement Account Glide Path</div>')
            cfg["glide_ret_same"] = st.toggle("Same glide path as taxable", value=bool(cfg["glide_ret_same"]), key="ps_glide_ret_same",
                help="If on, retirement accounts follow the same glide path as taxable. Turn off to set a different path.")
            if not cfg["glide_ret_same"]:
                gr1, gr2 = st.columns(2, border=True)
                with gr1:
                    cfg["glide_ret_eq_start"] = st.number_input("Starting equity %", min_value=0.0, max_value=1.0,
                        value=float(cfg["glide_ret_eq_start"]), step=0.05, format="%.2f", key="ps_glide_ret_start")
                    cfg["glide_ret_start_age"] = st.number_input("Glide start age", value=int(cfg["glide_ret_start_age"]), step=1, key="ps_glide_ret_sage")
                with gr2:
                    cfg["glide_ret_eq_end"] = st.number_input("Ending equity %", min_value=0.0, max_value=1.0,
                        value=float(cfg["glide_ret_eq_end"]), step=0.05, format="%.2f", key="ps_glide_ret_end")
                    cfg["glide_ret_end_age"] = st.number_input("Glide end age", value=int(cfg["glide_ret_end_age"]), step=1, key="ps_glide_ret_eage")

        # Equity sub-bucket granularity
        st.html('<div class="pro-section-title">Equity Sub-Bucket Granularity</div>')
        cfg["equity_granular_on"] = st.toggle("Enable equity sub-bucket detail", value=bool(cfg.get("equity_granular_on", False)), key="ps_eq_granular",
            help="Break the single 'Equities' asset class into 6 sub-buckets with independent return/vol/correlation parameters. "
                 "When off, equities use the single return and volatility from your market outlook.")
        if not cfg["equity_granular_on"]:
            st.caption("Toggle on to model US Large, US Small, Intl Developed, Emerging, US Value, and Tech/AI as separate asset classes with distinct correlations.")
        if cfg["equity_granular_on"]:
            if str(cfg.get("return_model", "standard")) == "regime":
                st.warning("⚠️ Equity sub-buckets use the standard model (not regime-switching). Regime model will be overridden for equity returns.")
            sub_w = dict(cfg.get("equity_sub_weights", {k: v["weight"] for k, v in EQUITY_SUB_DEFAULTS.items()}))
            sub_m = dict(cfg.get("equity_sub_mu", {k: v["mu"] for k, v in EQUITY_SUB_DEFAULTS.items()}))
            sub_v = dict(cfg.get("equity_sub_vol", {k: v["vol"] for k, v in EQUITY_SUB_DEFAULTS.items()}))

            # Build editable table
            sub_rows = []
            for cls in EQUITY_SUB_CLASSES:
                sub_rows.append({
                    "Sub-Class": cls,
                    "Weight": float(sub_w.get(cls, EQUITY_SUB_DEFAULTS[cls]["weight"])),
                    "Return (μ)": float(sub_m.get(cls, EQUITY_SUB_DEFAULTS[cls]["mu"])),
                    "Volatility (σ)": float(sub_v.get(cls, EQUITY_SUB_DEFAULTS[cls]["vol"])),
                })
            sub_df = pd.DataFrame(sub_rows)
            edited_sub = st.data_editor(sub_df, use_container_width=True, hide_index=True, key="ps_eq_sub_table",
                column_config={
                    "Sub-Class": st.column_config.TextColumn(disabled=True),
                    "Weight": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=1.0, step=0.05),
                    "Return (μ)": st.column_config.NumberColumn(format="%.3f", min_value=-0.10, max_value=0.30, step=0.005),
                    "Volatility (σ)": st.column_config.NumberColumn(format="%.3f", min_value=0.01, max_value=0.60, step=0.01),
                })
            # Read back edited values
            for _, row in edited_sub.iterrows():
                cls = row["Sub-Class"]
                sub_w[cls] = float(row["Weight"])
                sub_m[cls] = float(row["Return (μ)"])
                sub_v[cls] = float(row["Volatility (σ)"])
            cfg["equity_sub_weights"] = sub_w
            cfg["equity_sub_mu"] = sub_m
            cfg["equity_sub_vol"] = sub_v

            # Show weight summary
            total_w = sum(sub_w.values())
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total_w:.2f} — they will be normalized to 1.0 in the simulation.")
            else:
                st.success(f"Weights sum to {total_w:.2f}")

            # Concentration metric
            w_arr = np.array([sub_w.get(c, 0) for c in EQUITY_SUB_CLASSES])
            if w_arr.sum() > 0:
                w_norm = w_arr / w_arr.sum()
                hhi = float(np.sum(w_norm ** 2))
                n_eff = 1.0 / hhi if hhi > 0 else len(EQUITY_SUB_CLASSES)
                st.caption(f"📊 Concentration: HHI = {hhi:.2f} | Effective # of sub-buckets = {n_eff:.1f}")

            # Blended return/vol display
            w_n = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
            blended_mu = sum(w_n[i] * sub_m.get(EQUITY_SUB_CLASSES[i], 0.07) for i in range(len(EQUITY_SUB_CLASSES)))
            blended_vol = np.sqrt(sum(
                w_n[i] * w_n[j] * sub_v.get(EQUITY_SUB_CLASSES[i], 0.16) * sub_v.get(EQUITY_SUB_CLASSES[j], 0.16) *
                (1.0 if i == j else EQUITY_SUB_CORR.get((EQUITY_SUB_CLASSES[i], EQUITY_SUB_CLASSES[j]),
                 EQUITY_SUB_CORR.get((EQUITY_SUB_CLASSES[j], EQUITY_SUB_CLASSES[i]), 0.5)))
                for i in range(len(EQUITY_SUB_CLASSES)) for j in range(len(EQUITY_SUB_CLASSES))
            ))
            st.info(f"📈 **Blended equity**: μ = {blended_mu:.3f}, σ ≈ {blended_vol:.3f}")

            # Tech/AI bubble stress
            st.html('<div class="pro-section-title">Tech/AI Bubble Stress</div>')
            cfg["tech_bubble_on"] = st.toggle("Enable Tech/AI bubble risk", value=bool(cfg.get("tech_bubble_on", False)), key="ps_tech_bubble",
                help="Each year, there's a small probability of a Tech/AI-specific crash that adds an extra drawdown to the Tech/AI sub-bucket only.")
            if cfg["tech_bubble_on"]:
                tb1, tb2 = st.columns(2, border=True)
                with tb1:
                    cfg["tech_bubble_prob"] = st.number_input("Annual bubble probability", value=float(cfg.get("tech_bubble_prob", 0.03)),
                        min_value=0.0, max_value=0.20, step=0.01, format="%.2f", key="ps_tech_prob",
                        help="Probability that a Tech/AI bubble bursts in any given year.")
                with tb2:
                    cfg["tech_bubble_extra_drop"] = st.number_input("Extra Tech/AI drop", value=float(cfg.get("tech_bubble_extra_drop", -0.50)),
                        min_value=-0.90, max_value=0.0, step=0.05, format="%.2f", key="ps_tech_drop",
                        help="Additional return shock applied to Tech/AI sub-bucket when the bubble bursts (e.g., -0.50 = extra -50%).")

    # ================================================================
    # STRESS TESTS
    # ================================================================
    elif section == "Stress Tests":
        st.html('<div class="pro-section-title">Market Crash Overlay</div>')
        st.caption("Simulate sudden market crashes that can strike in any year, on top of normal volatility.")
        cfg["crash_on"] = st.toggle("Enable crash overlay", value=bool(cfg["crash_on"]), key="ps_crash_on",
            help="Each year, there's a small chance of a severe crash. When triggered, returns drop sharply across stocks, REITs, alternatives, and home values.")
        if not cfg["crash_on"]:
            st.caption("Toggle on to add random market crashes to the simulation.")
        if cfg["crash_on"]:
            cr1, cr2 = st.columns(2, border=True)
            with cr1:
                cfg["crash_prob"] = st.number_input("Annual crash probability", min_value=0.0, max_value=0.20,
                    value=float(cfg["crash_prob"]), step=0.01, format="%.2f", key="ps_crash_prob",
                    help="Chance of a crash in any given year. 0.05 = 5% per year.")
                cfg["crash_eq_extra"] = st.number_input("Stocks: extra drop", min_value=-0.80, max_value=0.0,
                    value=float(cfg["crash_eq_extra"]), step=0.05, format="%.2f", key="ps_crash_eq",
                    help="Additional return shock to equities during a crash (e.g. -0.40 = extra -40%).")
                cfg["crash_reit_extra"] = st.number_input("REITs: extra drop", min_value=-0.80, max_value=0.0,
                    value=float(cfg["crash_reit_extra"]), step=0.05, format="%.2f", key="ps_crash_reit")
            with cr2:
                cfg["crash_alt_extra"] = st.number_input("Alternatives: extra drop", min_value=-0.80, max_value=0.0,
                    value=float(cfg["crash_alt_extra"]), step=0.05, format="%.2f", key="ps_crash_alt")
                cfg["crash_home_extra"] = st.number_input("Home value: extra drop", min_value=-0.80, max_value=0.0,
                    value=float(cfg["crash_home_extra"]), step=0.05, format="%.2f", key="ps_crash_home")

        st.html('<div class="pro-section-title">Market Downturn Stress Test</div>')
        st.caption("Force a market downturn at a specific age to test your plan's resilience. "
                   "This applies to **every** simulation — it's not random. A downturn right before or at retirement "
                   "is the classic danger, but you can test any year.")
        cfg["seq_on"] = st.toggle("Force a market downturn at a specific age", value=bool(cfg["seq_on"]), key="ps_seq_on",
            help="Every simulation gets hit with this shock at the specified age. "
                 "Use this to test whether your plan can survive bad timing.")
        if not cfg["seq_on"]:
            st.caption("Toggle on to force a market drop at a chosen age and see if your plan survives.")
        if cfg["seq_on"]:
            sq1, sq2, sq3 = st.columns(3, border=True)
            with sq1:
                _seq_start_val = int(cfg.get("seq_start_age", 0))
                _seq_default = int(cfg.get("retire_age", 65)) if _seq_start_val <= 0 else _seq_start_val
                cfg["seq_start_age"] = st.number_input("Downturn starts at age",
                    min_value=int(cfg.get("start_age", 55)), max_value=int(cfg.get("end_age", 100)),
                    value=_seq_default, step=1, key="ps_seq_start_age",
                    help="The age when the forced market downturn begins. Can be before, at, or after retirement age.")
            with sq2:
                cfg["seq_drop"] = st.number_input("Annual return penalty", min_value=-0.50, max_value=0.0,
                    value=float(cfg["seq_drop"]), step=0.05, format="%.2f", key="ps_seq_drop",
                    help="Extra negative return added to stocks and REITs each year during the shock window. "
                         "E.g. -0.25 = an additional 25% loss per year on top of normal randomness.")
            with sq3:
                cfg["seq_years"] = st.number_input("Duration (years)", min_value=1, max_value=10,
                    value=int(cfg["seq_years"]), step=1, key="ps_seq_years",
                    help="How many consecutive years the forced downturn lasts.")
            _seq_sa = int(cfg["seq_start_age"]) if int(cfg.get("seq_start_age", 0)) > 0 else int(cfg.get("retire_age", 65))
            _seq_ea = _seq_sa + int(cfg["seq_years"]) - 1
            _ret_a = int(cfg.get("retire_age", 65))
            if _seq_sa < _ret_a:
                st.info(f"Downturn hits ages **{_seq_sa}–{_seq_ea}** (starts **{_ret_a - _seq_sa} years before retirement**). "
                        "This tests the impact on your savings and contributions during working years.")
            elif _seq_sa == _ret_a:
                st.info(f"Downturn hits ages **{_seq_sa}–{_seq_ea}** (right at retirement). "
                        "This is the classic sequence-of-returns risk scenario.")
            else:
                st.info(f"Downturn hits ages **{_seq_sa}–{_seq_ea}** ({_seq_sa - _ret_a} years into retirement).")

    # ================================================================
    # ADVANCED
    # ================================================================
    elif section == "Advanced":
        st.html('<div class="pro-section-title">Simulation Settings</div>')
        ad1, ad2 = st.columns(2, border=True)
        with ad1:
            cfg["n_sims"] = st.slider("Number of simulations", 500, 10000, int(cfg["n_sims"]), 500, key="ps_n_sims")
        with ad2:
            cfg["seed"] = st.number_input("Random seed", value=int(cfg["seed"]), step=1, key="ps_seed")

        st.html('<div class="pro-section-title">Return Distribution</div>')
        rd1, rd2 = st.columns(2, border=True)
        with rd1:
            cfg["dist_type"] = st.selectbox("Distribution type", ["t", "normal"],
                index=["t", "normal"].index(cfg["dist_type"]), key="ps_dist_type")
        with rd2:
            if cfg["dist_type"] == "t":
                cfg["t_df"] = st.number_input("Degrees of freedom", min_value=3, max_value=30, value=int(cfg["t_df"]), step=1, key="ps_t_df")

        st.html('<div class="pro-section-title">Correlations</div>')
        st.caption("How different asset classes and inflation move relative to each other.")
        if st.button("Edit correlations", icon=":material/tune:", use_container_width=False, key="ps_btn_corr"):
            open_correlations_dialog()

        st.html('<div class="pro-section-title">Mortality Settings</div>')
        cfg["mort_on"] = st.toggle("Model mortality", value=bool(cfg["mort_on"]), key="ps_mort_on",
            help="Simulates when each spouse dies, adjusting spending and Social Security accordingly.")
        if not cfg["mort_on"]:
            st.caption("Toggle on to model mortality risk. When off, both spouses are assumed alive through the entire plan.")
        if cfg["mort_on"]:
            mo1, mo2 = st.columns(2, border=True)
            with mo1:
                cfg["spend_drop_after_death"] = st.number_input("Spending drop after death",
                    min_value=0.0, max_value=0.7, value=float(cfg["spend_drop_after_death"]), step=0.05, format="%.2f", key="ps_spend_drop")
                cfg["home_cost_drop_after_death"] = st.number_input("Home cost drop after death",
                    min_value=0.0, max_value=0.8, value=float(cfg["home_cost_drop_after_death"]), step=0.05, format="%.2f", key="ps_home_drop")
            with mo2:
                cfg["q55_1"] = st.number_input("Spouse 1: q(55)",
                    min_value=0.001, max_value=0.03, value=float(cfg["q55_1"]), step=0.001, format="%.3f", key="ps_q55_1")
                if cfg["has_spouse"]:
                    cfg["q55_2"] = st.number_input("Spouse 2: q(55)",
                        min_value=0.001, max_value=0.03, value=float(cfg["q55_2"]), step=0.001, format="%.3f", key="ps_q55_2")
                cfg["mort_growth"] = st.number_input("Mortality growth rate",
                    min_value=1.02, max_value=1.15, value=float(cfg["mort_growth"]), step=0.01, format="%.2f", key="ps_mort_growth")

        cfg["legacy_on"] = st.toggle("Show legacy analysis", value=bool(cfg["legacy_on"]), key="ps_legacy_on")

    st.session_state["cfg"] = cfg


# ---- 8c: Deep Dive page ----
def deep_dive_page():
    cfg = st.session_state["cfg"]
    hold = st.session_state["hold"]

    st.html('<div style="font-size:1.8rem; font-weight:700; color:#1B2A4A; margin-bottom:0.5rem;">Deep Dive Analysis</div>')

    if "sim_result" not in st.session_state or "cfg_run" not in st.session_state:
        st.warning("Run a simulation from **Results** first to see detailed analysis.")
        return

    out = st.session_state["sim_result"]
    cfg_run = st.session_state["cfg_run"]
    ages = out["ages"]

    analysis = st.segmented_control(
        "Analysis",
        ["Cashflows", "Accounts", "Withdrawal Rate", "Sensitivity", "IRMAA", "Legacy", "Annuity", "Roth Strategy", "Tax Brackets", "ACA", "Optimizer", "Retire When?", "Failure Analysis", "Regimes", "Reallocation"],
        default="Cashflows", key="dd_analysis"
    )

    # ================================================================
    # CASHFLOWS
    # ================================================================
    if analysis == "Cashflows":
        st.html('<div class="pro-section-title">Income Sources & Expenses</div>')
        st.caption("Median (50th percentile) values across all simulations, in nominal dollars. Post-retirement years only.")

        de = out["decomp"]
        liq = out["liquid"]
        nw = out["net_worth"]
        hv = out["home_value"]
        mb = out["mortgage"]
        home_eq = np.maximum(0.0, hv - mb)
        hsa_bal = out["hsa"]

        # Stacked area charts for income sources and expenses (Change 3)
        _ret_age_idx = max(0, int(cfg_run.get("retire_age", 62)) - int(ages[0]))
        _post_ret_ages = ages[_ret_age_idx:]
        if len(_post_ret_ages) > 1:
            # Use mean shares to break down the median total income by source.
            # This avoids the "marginal median" problem (independent p50s per
            # source can all be zero even when the median sim has money) and
            # avoids picking a single sim that may be an outlier or go broke.
            # Combine the three portfolio withdrawal types + RMDs into a single
            # "Portfolio Withdrawals" category so the chart is easier to read.
            _raw_keys = ["ss_inflow", "gross_rmd", "gross_tax_wd",
                         "gross_trad_wd", "gross_roth_wd", "annuity_income", "event_income"]
            # Merged groups: portfolio = RMDs + taxable + trad + roth
            _src_groups = {
                "Social Security": ["ss_inflow"],
                "Portfolio Withdrawals": ["gross_rmd", "gross_tax_wd", "gross_trad_wd", "gross_roth_wd"],
                "Annuity Income": ["annuity_income"],
                "Event Income": ["event_income"],
            }
            _src_labels = list(_src_groups.keys())

            # Total income per sim per age (all raw sources)
            _raw_arrays = np.array([de[k] for k in _raw_keys])
            _total_inc = _raw_arrays.sum(axis=0)  # shape (n_sims, T+1)

            # Pre-sum each group per sim: shape (n_groups, n_sims, T+1)
            _group_arrays = np.array([
                sum(de[k] for k in keys) for keys in _src_groups.values()
            ])

            _income_rows = []
            _expense_rows = []
            for i, age in enumerate(_post_ret_ages):
                idx = _ret_age_idx + i
                _med_total = float(np.percentile(_total_inc[:, idx], 50))
                _col_total = _total_inc[:, idx]
                _col_total_safe = np.maximum(_col_total, 1e-6)
                _shares = np.array([
                    float(np.mean(_group_arrays[g, :, idx] / _col_total_safe))
                    for g in range(len(_src_labels))
                ])
                _share_sum = _shares.sum()
                if _share_sum > 0:
                    _shares = _shares / _share_sum
                _amounts = _shares * _med_total

                _income_rows.append({
                    "Age": int(age),
                    **{label: float(amt) / 1e3 for label, amt in zip(_src_labels, _amounts)}
                })

                _hc = (float(np.percentile(de["home_cost"][:, idx], 50)) +
                       float(np.percentile(de["mort_pay"][:, idx], 50)) +
                       float(np.percentile(de["rent"][:, idx], 50)))
                _health_total = (float(np.percentile(de["health"][:, idx], 50)) +
                                 float(np.percentile(de["medical_nom"][:, idx], 50)) +
                                 float(np.percentile(de["ltc_cost"][:, idx], 50)))
                _expense_rows.append({
                    "Age": int(age),
                    "Core Spending": float(np.percentile(de["core_adjusted"][:, idx], 50)) / 1e3,
                    "Housing": _hc / 1e3,
                    "Healthcare": _health_total / 1e3,
                    "Taxes": float(np.percentile(de["taxes_paid"][:, idx], 50)) / 1e3,
                    "IRMAA": float(np.percentile(de["irmaa"][:, idx], 50)) / 1e3,
                    "Event Expenses": float(np.percentile(de["event_expense"][:, idx], 50)) / 1e3,
                })

            _inc_df = pd.DataFrame(_income_rows)
            _exp_df = pd.DataFrame(_expense_rows)

            # Compare total income vs total expenses per age.
            # Where expenses > income, scale down expense categories and add
            # a "Shortfall" band so the user sees the unfunded gap.
            _inc_src_cols = [c for c in _inc_df.columns if c != "Age"]
            _exp_cat_cols = [c for c in _exp_df.columns if c != "Age"]
            _inc_totals = _inc_df[_inc_src_cols].sum(axis=1)
            _exp_totals = _exp_df[_exp_cat_cols].sum(axis=1)

            # Build adjusted expense df: scale each category proportionally,
            # add Shortfall for the unfunded portion
            _exp_adj_rows = []
            for r in range(_exp_df.shape[0]):
                inc_t = _inc_totals.iloc[r]
                exp_t = _exp_totals.iloc[r]
                row = {"Age": _exp_df["Age"].iloc[r]}
                if exp_t > 0 and inc_t < exp_t:
                    scale = inc_t / exp_t
                    for c in _exp_cat_cols:
                        row[c] = _exp_df[c].iloc[r] * scale
                    row["Shortfall"] = (exp_t - inc_t)
                else:
                    for c in _exp_cat_cols:
                        row[c] = _exp_df[c].iloc[r]
                    row["Shortfall"] = 0.0
                _exp_adj_rows.append(row)
            _exp_adj_df = pd.DataFrame(_exp_adj_rows)

            _inc_melt = _inc_df.melt("Age", var_name="Source", value_name="Amount ($K)")
            _exp_adj_melt = _exp_adj_df.melt("Age", var_name="Category", value_name="Amount ($K)")

            _inc_domain = ["Social Security", "Portfolio Withdrawals", "Annuity Income", "Event Income"]
            _inc_colors = ["#FF8F00", "#1B2A4A", "#7E57C2", "#4CAF50"]
            _exp_domain = ["Core Spending", "Housing", "Healthcare", "Taxes", "IRMAA", "Event Expenses", "Shortfall"]
            _exp_colors = ["#1B2A4A", "#8D6E63", "#E57373", "#FF8F00", "#9E9E9E", "#AB47BC", "#D32F2F"]

            st.caption("Median total income broken down by average source share. "
                       "'Portfolio Withdrawals' includes RMDs, taxable, traditional IRA, and Roth withdrawals. "
                       "Red 'Shortfall' on the expense chart shows planned spending that exceeds available income.")
            ch_inc, ch_exp = st.columns(2)
            with ch_inc:
                st.markdown("**Where the Money Comes From**")
                _inc_chart = alt.Chart(_inc_melt).mark_area().encode(
                    x=alt.X("Age:Q", title="Age"),
                    y=alt.Y("Amount ($K):Q", title="Amount ($K nominal)", stack=True),
                    color=alt.Color("Source:N", scale=alt.Scale(
                        domain=_inc_domain,
                        range=_inc_colors)),
                    tooltip=["Age:Q", "Source:N", alt.Tooltip("Amount ($K):Q", format=",.0f")]
                ).properties(height=350)
                st.altair_chart(_inc_chart, use_container_width=True)
            with ch_exp:
                st.markdown("**Where the Money Goes**")
                _exp_chart = alt.Chart(_exp_adj_melt).mark_area().encode(
                    x=alt.X("Age:Q", title="Age"),
                    y=alt.Y("Amount ($K):Q", title="Amount ($K nominal)", stack=True),
                    color=alt.Color("Category:N", scale=alt.Scale(
                        domain=_exp_domain,
                        range=_exp_colors)),
                    tooltip=["Age:Q", "Category:N", alt.Tooltip("Amount ($K):Q", format=",.0f")]
                ).properties(height=350)
                st.altair_chart(_exp_chart, use_container_width=True)

        # Detailed cashflow table in expander
        with st.expander("Detailed cashflow table", expanded=False):
            rows = []
            for idx, age in enumerate(ages):
                idx_next = min(idx + 1, len(ages) - 1)
                rows.append({
                    "Age": int(age),
                    "Liquid (start)": float(np.percentile(liq[:, idx], 50)),
                    "Liquid (end)": float(np.percentile(liq[:, idx_next], 50)),
                    "Net Worth (start)": float(np.percentile(nw[:, idx], 50)),
                    "Net Worth (end)": float(np.percentile(nw[:, idx_next], 50)),
                    "Home Equity": float(np.percentile(home_eq[:, idx], 50)),
                    "Core Spending (planned)": float(np.percentile(de["baseline_core"][:, idx], 50)),
                    "Core Spending (guardrails)": float(np.percentile(de["core_adjusted"][:, idx], 50)),
                    "Essential Spending": float(np.percentile(de["essential_spend"][:, idx], 50)),
                    "Discretionary Spending": float(np.percentile(de["discretionary_spend"][:, idx], 50)),
                    "Home Costs": float(np.percentile(de["home_cost"][:, idx], 50)),
                    "Mortgage": float(np.percentile(de["mort_pay"][:, idx], 50)),
                    "Rent": float(np.percentile(de["rent"][:, idx], 50)),
                    "Health Insurance": float(np.percentile(de["health"][:, idx], 50)),
                    "Medical": float(np.percentile(de["medical_nom"][:, idx], 50)),
                    "HSA Used": float(np.percentile(de["hsa_used_med"][:, idx], 50)),
                    "LTC": float(np.percentile(de["ltc_cost"][:, idx], 50)),
                    "Total Spending": float(np.percentile(de["outflow_total"][:, idx], 50)),
                    "SS Income": float(np.percentile(de["ss_inflow"][:, idx], 50)),
                    "RMDs (gross)": float(np.percentile(de["gross_rmd"][:, idx], 50)),
                    "Annuity Income": float(np.percentile(de["annuity_income"][:, idx], 50)),
                    "Event Income": float(np.percentile(de["event_income"][:, idx], 50)),
                    "Event Expenses": float(np.percentile(de["event_expense"][:, idx], 50)),
                    "Roth Conversions": float(np.percentile(de["conv_gross"][:, idx], 50)),
                    "IRMAA": float(np.percentile(de["irmaa"][:, idx], 50)),
                    "Taxable WD": float(np.percentile(de["gross_tax_wd"][:, idx], 50)),
                    "Trad IRA WD": float(np.percentile(de["gross_trad_wd"][:, idx], 50)),
                    "Roth WD": float(np.percentile(de["gross_roth_wd"][:, idx], 50)),
                    "Taxes": float(np.percentile(de["taxes_paid"][:, idx], 50)),
                    "QCD": float(np.percentile(de["qcd"][:, idx], 50)),
                    "Gain Harvest": float(np.percentile(de["gain_harvest"][:, idx], 50)),
                    "HSA Balance": float(np.percentile(hsa_bal[:, idx], 50)),
                })

            de_df = pd.DataFrame(rows)
            for c in de_df.columns:
                if c != "Age":
                    de_df[c] = de_df[c].apply(fmt_dollars)
            st.dataframe(de_df, use_container_width=True, hide_index=True, height=500)

        # Spending floor distribution chart
        _spend_real_dd = de["spend_real_track"]
        _ret_idx_dd = max(0, int(cfg_run.get("retire_age", 62)) - int(ages[0]))
        _spend_real_post = _spend_real_dd[:, _ret_idx_dd:]
        if _spend_real_post.shape[1] > 1:
            with st.expander("Spending floor risk — minimum real spending distribution", expanded=False):
                _min_spend_dd = np.min(_spend_real_post, axis=1) / 1e3
                _hist_df = pd.DataFrame({"Minimum Real Spending ($K)": _min_spend_dd})
                _hist_chart = alt.Chart(_hist_df).mark_bar(color="#1B2A4A", opacity=0.7).encode(
                    alt.X("Minimum Real Spending ($K):Q", bin=alt.Bin(maxbins=40), title="Minimum Real Spending ($K)"),
                    alt.Y("count():Q", title="Number of Simulations"),
                    tooltip=[alt.Tooltip("Minimum Real Spending ($K):Q", bin=alt.Bin(maxbins=40), title="Spending ($K)"),
                             alt.Tooltip("count():Q", title="Simulations")]
                ).properties(height=250)

                _floor_dd = float(cfg_run.get("spending_floor", 80000.0)) / 1e3
                _floor_rule = alt.Chart(pd.DataFrame({"x": [_floor_dd]})).mark_rule(
                    color="#D32F2F", strokeDash=[6, 3], strokeWidth=2
                ).encode(x="x:Q")
                _floor_label = alt.Chart(pd.DataFrame({"x": [_floor_dd + 2], "y": [0], "label": [f"Floor: ${_floor_dd:.0f}K"]})).mark_text(
                    align="left", fontSize=11, color="#D32F2F", dy=-10
                ).encode(x="x:Q", y="y:Q", text="label:N")

                st.altair_chart(_hist_chart + _floor_rule + _floor_label, use_container_width=True)
                _pct_below = float((_min_spend_dd < _floor_dd).mean()) * 100
                st.caption(f"**{_pct_below:.1f}%** of simulations have at least one year where real core spending drops below the ${_floor_dd:.0f}K floor. "
                           "This captures the tail risk from guardrail spending cuts.")

    # ================================================================
    # ACCOUNTS (Change 1)
    # ================================================================
    elif analysis == "Accounts":
        st.html('<div class="pro-section-title">Account Balances Over Time</div>')

        _acct_dollar = st.segmented_control("Dollars", ["Nominal", "Real"], default="Nominal", key="dd_acct_dollars")
        _acct_infl = out["infl_index"]

        _acct_tax = out["taxable"]
        _acct_trad = out["trad"]
        _acct_roth = out["roth"]
        _acct_hsa = out["hsa"]

        if _acct_dollar == "Real":
            _acct_tax = _acct_tax / _acct_infl
            _acct_trad = _acct_trad / _acct_infl
            _acct_roth = _acct_roth / _acct_infl
            _acct_hsa = _acct_hsa / _acct_infl

        _acct_rows = []
        for i, age in enumerate(ages):
            _acct_rows.append({"Age": int(age), "Taxable": float(np.median(_acct_tax[:, i])) / 1e3,
                               "Traditional IRA": float(np.median(_acct_trad[:, i])) / 1e3,
                               "Roth IRA": float(np.median(_acct_roth[:, i])) / 1e3,
                               "HSA": float(np.median(_acct_hsa[:, i])) / 1e3})
        _acct_df = pd.DataFrame(_acct_rows)
        _acct_melt = _acct_df.melt("Age", var_name="Account", value_name="Balance ($K)")

        _acct_chart = alt.Chart(_acct_melt).mark_area().encode(
            x=alt.X("Age:Q", title="Age"),
            y=alt.Y("Balance ($K):Q", title=f"Balance ($K {'real' if _acct_dollar == 'Real' else 'nominal'})", stack=True),
            color=alt.Color("Account:N", scale=alt.Scale(
                domain=["Taxable", "Traditional IRA", "Roth IRA", "HSA"],
                range=["#1B2A4A", "#E57373", "#00897B", "#FF8F00"])),
            tooltip=["Age:Q", "Account:N", alt.Tooltip("Balance ($K):Q", format=",.0f")]
        ).properties(height=400)
        st.altair_chart(_acct_chart, use_container_width=True)

        # Milestone summary table
        st.html('<div class="pro-section-title">Account Balances at Key Ages</div>')
        _retire_a = int(cfg_run.get("retire_age", 62))
        _end_a = int(cfg_run.get("end_age", 90))
        _milestones = sorted(set([_retire_a, 65, 70, 75, 80, 85, 90, _end_a]) & set(int(a) for a in ages))
        _ms_rows = []
        for ma in _milestones:
            mi = int(ma - ages[0])
            if 0 <= mi < len(ages):
                _ms_rows.append({
                    "Age": ma,
                    "Taxable": fmt_dollars(float(np.median(_acct_tax[:, mi]))),
                    "Traditional": fmt_dollars(float(np.median(_acct_trad[:, mi]))),
                    "Roth": fmt_dollars(float(np.median(_acct_roth[:, mi]))),
                    "HSA": fmt_dollars(float(np.median(_acct_hsa[:, mi]))),
                    "Total": fmt_dollars(float(np.median(_acct_tax[:, mi] + _acct_trad[:, mi] + _acct_roth[:, mi] + _acct_hsa[:, mi]))),
                })
        if _ms_rows:
            st.dataframe(pd.DataFrame(_ms_rows), use_container_width=True, hide_index=True)

    # ================================================================
    # WITHDRAWAL RATE (Change 2)
    # ================================================================
    elif analysis == "Withdrawal Rate":
        st.html('<div class="pro-section-title">Effective Withdrawal Rate Over Time</div>')

        de = out["decomp"]
        _liq = out["liquid"]
        _ret_idx = max(0, int(cfg_run.get("retire_age", 62)) - int(ages[0]))

        # Total withdrawals = taxable + trad + roth withdrawals
        _total_wd = de["gross_tax_wd"] + de["gross_trad_wd"] + de["gross_roth_wd"]

        _wr_rows = []
        for i in range(_ret_idx, len(ages)):
            if i > 0:
                _beg_liq = _liq[:, i - 1]
                _wd = _total_wd[:, i]
                # Avoid division by zero
                _safe_liq = np.maximum(_beg_liq, 1.0)
                _wr = _wd / _safe_liq * 100  # as percentage
                _wr_rows.append({
                    "Age": int(ages[i]),
                    "p25": float(np.percentile(_wr, 25)),
                    "p50": float(np.percentile(_wr, 50)),
                    "p75": float(np.percentile(_wr, 75)),
                })

        if _wr_rows:
            _wr_df = pd.DataFrame(_wr_rows)

            # Planned initial withdrawal rate
            _spend_r = float(cfg_run.get("spend_real", 120000))
            _start_liq = float(np.median(_liq[:, 0]))
            _planned_wr = (_spend_r / max(_start_liq, 1.0)) * 100

            _wr_band = alt.Chart(_wr_df).mark_area(opacity=0.2, color="#00897B").encode(
                x=alt.X("Age:Q", title="Age"),
                y=alt.Y("p25:Q", title="Withdrawal Rate (%)"),
                y2="p75:Q",
            )
            _wr_line = alt.Chart(_wr_df).mark_line(color="#00897B", strokeWidth=3).encode(
                x="Age:Q",
                y=alt.Y("p50:Q"),
                tooltip=["Age:Q", alt.Tooltip("p50:Q", title="Median WR %", format=".1f")]
            )
            _four_pct = alt.Chart(pd.DataFrame({"y": [4.0]})).mark_rule(
                color="green", strokeDash=[6, 3], strokeWidth=1.5
            ).encode(y="y:Q")
            _four_label = alt.Chart(pd.DataFrame({"Age": [int(ages[_ret_idx]) + 1], "y": [4.3], "label": ["4% rule"]})).mark_text(
                align="left", fontSize=11, color="green"
            ).encode(x="Age:Q", y="y:Q", text="label:N")
            _planned_rule = alt.Chart(pd.DataFrame({"y": [_planned_wr]})).mark_rule(
                color="#FF8F00", strokeDash=[6, 3], strokeWidth=1.5
            ).encode(y="y:Q")
            _planned_label = alt.Chart(pd.DataFrame({"Age": [int(ages[_ret_idx]) + 1], "y": [_planned_wr + 0.3],
                                                      "label": [f"Planned: {_planned_wr:.1f}%"]})).mark_text(
                align="left", fontSize=11, color="#FF8F00"
            ).encode(x="Age:Q", y="y:Q", text="label:N")

            _wr_chart = (_wr_band + _wr_line + _four_pct + _four_label + _planned_rule + _planned_label).properties(height=400)
            st.altair_chart(_wr_chart, use_container_width=True)
            st.caption("Your effective withdrawal rate changes each year based on portfolio performance, spending adjustments from guardrails, "
                       "and RMD requirements. The 4% line is a common benchmark, not a target. The shaded band shows the p25–p75 range.")
        else:
            st.info("No post-retirement years to analyze.")

    # ================================================================
    # SENSITIVITY
    # ================================================================
    elif analysis == "Sensitivity":
        st.html('<div class="pro-section-title">Full Sensitivity Analysis</div>')
        st.caption("Each variable shows the impact of both an increase and decrease. Uses 1,000 simulations per test for speed.")
        with st.spinner("Running sensitivity tests..."):
            tor = _run_sensitivity_tests(cfg_run, hold, n_fast=1000)
        render_tornado_chart(tor)
        # Show clean table
        display_df = tor[["Variable", "Up ($M)", "Down ($M)", "Up Label", "Down Label"]].copy()
        display_df["Up ($M)"] = display_df["Up ($M)"].map(lambda x: f"{x:+.2f}")
        display_df["Down ($M)"] = display_df["Down ($M)"].map(lambda x: f"{x:+.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ================================================================
    # IRMAA
    # ================================================================
    elif analysis == "IRMAA":
        st.html('<div class="pro-section-title">Medicare IRMAA Surcharges (Part B + Part D)</div>')
        if not bool(cfg_run.get("irmaa_on", False)):
            st.info("IRMAA tracking is off. Enable it in Plan Setup > Taxes to see this analysis.")
        else:
            st.caption("IRMAA uses MAGI from **2 years prior** to determine surcharge tiers. "
                       "Median values across simulations, nominal $.")
            de = out["decomp"]
            medicare_age = int(cfg_run.get("medicare_age", 65))
            irmaa_rows = []
            for idx, age in enumerate(ages):
                if age < medicare_age:
                    continue
                # Lookback MAGI (2 years prior)
                lb_idx = max(0, idx - 2)
                lookback_magi = float(np.median(out["magi"][:, lb_idx]))
                current_magi = float(np.median(out["magi"][:, idx]))
                part_b = float(np.median(de["irmaa_part_b"][:, idx]))
                part_d = float(np.median(de["irmaa_part_d"][:, idx]))
                total_irmaa = float(np.median(de["irmaa_part_b"][:, idx] + de["irmaa_part_d"][:, idx]))
                tier = float(np.median(out["irmaa_tier"][:, idx]))
                irmaa_rows.append({
                    "Age": int(age),
                    "Lookback MAGI (age−2)": lookback_magi,
                    "Current MAGI": current_magi,
                    "Tier": f"{tier:.0f}",
                    "Part B Surcharge": part_b,
                    "Part D Surcharge": part_d,
                    "Total IRMAA": total_irmaa,
                })
            if irmaa_rows:
                irmaa_df = pd.DataFrame(irmaa_rows)
                for c in ["Lookback MAGI (age−2)", "Current MAGI", "Part B Surcharge", "Part D Surcharge", "Total IRMAA"]:
                    irmaa_df[c] = irmaa_df[c].apply(fmt_dollars)
                st.dataframe(irmaa_df, use_container_width=True, hide_index=True, height=500)

                # Stacked bar chart: Part B vs Part D surcharges by age
                chart_rows = []
                for idx, age in enumerate(ages):
                    if age < medicare_age:
                        continue
                    chart_rows.append({"Age": int(age), "Component": "Part B",
                                       "Surcharge": float(np.median(de["irmaa_part_b"][:, idx]))})
                    chart_rows.append({"Age": int(age), "Component": "Part D",
                                       "Surcharge": float(np.median(de["irmaa_part_d"][:, idx]))})
                chart_df = pd.DataFrame(chart_rows)
                if not chart_df.empty and chart_df["Surcharge"].sum() > 0:
                    irmaa_chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X("Age:O", title="Age"),
                        y=alt.Y("Surcharge:Q", title="Annual IRMAA Surcharge ($)", stack="zero"),
                        color=alt.Color("Component:N", scale=alt.Scale(
                            domain=["Part B", "Part D"],
                            range=["#1B2A4A", "#00897B"])),
                        tooltip=[
                            alt.Tooltip("Age:O"),
                            alt.Tooltip("Component:N"),
                            alt.Tooltip("Surcharge:Q", format="$,.0f"),
                        ],
                    ).properties(height=300, title="IRMAA Surcharges by Age (Median)")
                    st.altair_chart(irmaa_chart, use_container_width=True)
                else:
                    st.success("No IRMAA surcharges triggered in the median simulation path.")
            else:
                st.info("No Medicare-eligible ages in the simulation.")

    # ================================================================
    # LEGACY
    # ================================================================
    elif analysis == "Legacy":
        st.html('<div class="pro-section-title">What You Leave Behind</div>')
        if not bool(cfg_run.get("legacy_on", True)):
            st.info("Legacy analysis is off. Enable it in Plan Setup > Advanced.")
        else:
            st.caption("Estimated estate value at second death, nominal $. Includes home equity. Assumes stepped-up cost basis.")
            legacy = out["legacy"]
            lc1, lc2, lc3 = st.columns(3, border=True)
            with lc1:
                st.html(_metric_card_html("Median Legacy", f"${np.median(legacy)/1e6:.1f}M", "", "metric-navy"))
            with lc2:
                st.html(_metric_card_html("p10 (Low)", f"${np.percentile(legacy, 10)/1e6:.1f}M", "", "metric-coral"))
            with lc3:
                st.html(_metric_card_html("p90 (High)", f"${np.percentile(legacy, 90)/1e6:.1f}M", "", "metric-green"))

            # Distribution histogram
            leg_m = legacy / 1e6
            hist_df = pd.DataFrame({"Legacy ($M)": leg_m})
            hist_chart = alt.Chart(hist_df).mark_bar(color="#1B2A4A", opacity=0.7).encode(
                x=alt.X("Legacy ($M):Q", bin=alt.Bin(maxbins=40), title="Legacy ($M)"),
                y=alt.Y("count()", title="Number of simulations"),
            ).properties(height=280)
            st.altair_chart(hist_chart, use_container_width=True)

    # ================================================================
    # ANNUITY BREAKEVEN
    # ================================================================
    elif analysis == "Annuity":
        st.html('<div class="pro-section-title">Annuity Breakeven Analysis</div>')
        ann1_on = bool(cfg_run.get("ann1_on", False))
        ann2_on = bool(cfg_run.get("ann2_on", False))
        if not ann1_on and not ann2_on:
            st.info("No annuities configured. Enable an annuity in Assumptions > Income to see this analysis.")
        else:
            de = out["decomp"]
            for prefix, label in [("ann1", "Annuity 1"), ("ann2", "Annuity 2")]:
                if not bool(cfg_run.get(f"{prefix}_on", False)):
                    continue
                st.html(f'<div class="pro-section-title">{label} — Breakeven & Return on Premium</div>')
                purchase_amt = float(cfg_run[f"{prefix}_purchase_amount"])
                payout_rate = float(cfg_run[f"{prefix}_payout_rate"])
                income_start_age = int(cfg_run[f"{prefix}_income_start_age"])
                annual_payment = purchase_amt * payout_rate

                # Cumulative payments vs purchase price
                paying_ages = [a for a in ages if a >= income_start_age]
                if not paying_ages:
                    st.warning(f"{label} income hasn't started yet in this simulation horizon.")
                    continue

                cum_payments = []
                for yr_count, a in enumerate(paying_ages, 1):
                    if bool(cfg_run.get(f"{prefix}_cola_on", False)):
                        cola_r = float(cfg_run.get(f"{prefix}_cola_rate", 0.02))
                        cum = sum(annual_payment * (1 + cola_r) ** y for y in range(yr_count))
                    else:
                        cum = annual_payment * yr_count
                    cum_payments.append({"Age": int(a), "Cumulative Payments": cum, "Purchase Price": purchase_amt})

                be_df = pd.DataFrame(cum_payments)
                breakeven_age = None
                for _, row in be_df.iterrows():
                    if row["Cumulative Payments"] >= row["Purchase Price"]:
                        breakeven_age = int(row["Age"])
                        break

                bc1, bc2, bc3 = st.columns(3, border=True)
                with bc1:
                    st.html(_metric_card_html("Purchase Price", f"${purchase_amt:,.0f}", "", "metric-navy"))
                with bc2:
                    st.html(_metric_card_html("Annual Payment", f"${annual_payment:,.0f}", "(today's $)", "metric-green"))
                with bc3:
                    be_text = f"Age {breakeven_age}" if breakeven_age else "Beyond plan"
                    st.html(_metric_card_html("Breakeven Age", be_text, "(nominal)", "metric-coral"))

                # Chart
                chart_data = be_df.melt(id_vars=["Age"], value_vars=["Cumulative Payments", "Purchase Price"],
                                        var_name="Series", value_name="Dollars")
                be_chart = alt.Chart(chart_data).mark_line(strokeWidth=2).encode(
                    x=alt.X("Age:Q", title="Age"),
                    y=alt.Y("Dollars:Q", title="Dollars", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Series:N", scale=alt.Scale(range=["#00897B", "#E57373"])),
                    strokeDash=alt.StrokeDash("Series:N"),
                ).properties(height=280)
                st.altair_chart(be_chart, use_container_width=True)

                # Return on premium at various survival ages
                st.markdown("**Return on Premium by Survival Age**")
                rop_rows = []
                for survive_to in [75, 80, 85, 90, 95]:
                    years_paying = max(0, survive_to - income_start_age)
                    if years_paying <= 0:
                        continue
                    if bool(cfg_run.get(f"{prefix}_cola_on", False)):
                        cola_r = float(cfg_run.get(f"{prefix}_cola_rate", 0.02))
                        total_received = sum(annual_payment * (1 + cola_r) ** y for y in range(years_paying))
                    else:
                        total_received = annual_payment * years_paying
                    rop = (total_received - purchase_amt) / purchase_amt * 100
                    rop_rows.append({"Survive To": survive_to, "Years Receiving": years_paying,
                                     "Total Received": f"${total_received:,.0f}",
                                     "Return on Premium": f"{rop:+.1f}%"})
                if rop_rows:
                    st.dataframe(pd.DataFrame(rop_rows), use_container_width=True, hide_index=True)

    # ================================================================
    # ROTH STRATEGY COMPARISON
    # ================================================================
    elif analysis == "Roth Strategy":
        st.html('<div class="pro-section-title">Roth Conversion Strategy Comparison</div>')
        # Use the live cfg so we pick up any changes made since last dashboard run
        _roth_cfg = build_cfg_run(cfg)
        if not bool(_roth_cfg.get("conv_on", False)):
            st.info("Roth conversions are off. Enable them in Assumptions > Taxes, then come back here.")
        else:
            st.caption("Compares your current Roth conversion strategy against no conversions.")
            with st.spinner("Running comparison simulations..."):
                # Run with conversions (current live settings)
                cfg_with = dict(_roth_cfg)
                cfg_with["n_sims"] = 1000
                out_with = simulate(cfg_with, hold)

                # Run without conversions
                cfg_without = dict(_roth_cfg)
                cfg_without["n_sims"] = 1000
                cfg_without["conv_on"] = False
                out_without = simulate(cfg_without, hold)

            # Comparison metrics
            rc1, rc2, rc3 = st.columns(3, border=True)
            with rc1:
                sr_with = float((out_with["liquid"][:, -1] > 0).mean()) * 100
                sr_without = float((out_without["liquid"][:, -1] > 0).mean()) * 100
                delta_sr = sr_with - sr_without
                st.html(_metric_card_html("Success Rate", f"{sr_with:.1f}%", f"vs {sr_without:.1f}% without ({delta_sr:+.1f}pp)", "metric-green" if delta_sr >= 0 else "metric-coral"))
            with rc2:
                med_with = float(np.percentile(out_with["net_worth"][:, -1], 50)) / 1e6
                med_without = float(np.percentile(out_without["net_worth"][:, -1], 50)) / 1e6
                st.html(_metric_card_html("Median Net Worth", f"${med_with:.1f}M", f"vs ${med_without:.1f}M without", "metric-navy"))
            with rc3:
                leg_with = float(np.median(out_with["legacy"])) / 1e6
                leg_without = float(np.median(out_without["legacy"])) / 1e6
                st.html(_metric_card_html("Median Legacy", f"${leg_with:.1f}M", f"vs ${leg_without:.1f}M without", "metric-navy"))

            # Cumulative taxes comparison
            st.html('<div class="pro-section-title">Cumulative Taxes Paid</div>')
            taxes_with = np.cumsum(np.percentile(out_with["decomp"]["taxes_paid"], 50, axis=0))
            taxes_without = np.cumsum(np.percentile(out_without["decomp"]["taxes_paid"], 50, axis=0))
            tax_df = pd.DataFrame({
                "Age": list(out_with["ages"]) + list(out_without["ages"]),
                "Cumulative Taxes ($)": list(taxes_with) + list(taxes_without),
                "Strategy": ["With conversions"] * len(out_with["ages"]) + ["Without conversions"] * len(out_without["ages"]),
            })
            tax_chart = alt.Chart(tax_df).mark_line(strokeWidth=2).encode(
                x=alt.X("Age:Q", title="Age"),
                y=alt.Y("Cumulative Taxes ($):Q", title="Cumulative Taxes", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Strategy:N", scale=alt.Scale(range=["#00897B", "#E57373"])),
            ).properties(height=280)
            st.altair_chart(tax_chart, use_container_width=True)

            # Median conversion amounts by age
            st.html('<div class="pro-section-title">Annual Roth Conversions (Median)</div>')
            conv_med = np.percentile(out_with["decomp"]["conv_gross"], 50, axis=0)
            conv_df = pd.DataFrame({"Age": out_with["ages"], "Conversion ($)": conv_med})
            conv_df = conv_df[conv_df["Conversion ($)"] > 0]
            if len(conv_df) > 0:
                conv_chart = alt.Chart(conv_df).mark_bar(color="#00897B").encode(
                    x=alt.X("Age:O", title="Age"),
                    y=alt.Y("Conversion ($):Q", title="Conversion Amount", axis=alt.Axis(format="$,.0f")),
                ).properties(height=250)
                st.altair_chart(conv_chart, use_container_width=True)
            else:
                st.info("No conversions occurred in the simulation.")

    # ================================================================
    # TAX BRACKETS
    # ================================================================
    elif analysis == "Tax Brackets":
        st.html('<div class="pro-section-title">Tax Bracket Projection</div>')
        de = out["decomp"]
        infl_idx = out.get("infl_index", None)
        _tax_engine_was_on = bool(cfg_run.get("tax_engine_on", True))

        if infl_idx is None:
            st.info("Run the simulation again to see tax bracket projections.")
        elif _tax_engine_was_on:
            st.caption("Actual tax engine output (median across simulations, nominal dollars). Federal brackets are inflation-adjusted each year, "
                       "with SS provisional income and LTCG/QD stacking.")
            retire_age = int(cfg_run["retire_age"])
            tb_rows = []
            for idx, age in enumerate(ages):
                if age < retire_age:
                    continue
                tb_rows.append({
                    "Age": int(age),
                    "SS Gross": float(np.median(de["ss_inflow"][:, idx])),
                    "SS Taxable": float(np.median(de["ss_taxable"][:, idx])),
                    "Conversions": float(np.median(de["conv_gross"][:, idx])),
                    "Trad WD": float(np.median(de["gross_trad_wd"][:, idx])),
                    "Qual Divs": float(np.median(de["qual_divs"][:, idx])),
                    "LTCG": float(np.median(de["ltcg_realized"][:, idx])),
                    "Fed Ordinary": float(np.median(de["fed_tax"][:, idx]) - np.median(de.get("niit", np.zeros_like(de["fed_tax"]))[:, idx])),
                    "Fed LTCG": float(np.median(de["fed_tax"][:, idx])) - float(np.median(de["fed_tax"][:, idx]) - np.median(de.get("niit", np.zeros_like(de["fed_tax"]))[:, idx])),
                    "NIIT": float(np.median(de["niit"][:, idx])),
                    "State Tax": float(np.median(de["state_tax"][:, idx])),
                    "Total Tax": float(np.median(de["taxes_paid"][:, idx])),
                    "Marginal": f"{float(np.median(de['marginal_bracket'][:, idx])):.0%}",
                    "Eff Rate": f"{float(np.median(de['effective_rate'][:, idx])):.1%}",
                    "MAGI": float(np.median(out["magi"][:, idx])),
                })
            if tb_rows:
                tb_df = pd.DataFrame(tb_rows)
                dollar_cols = [c for c in tb_df.columns if c not in ("Age", "Marginal", "Eff Rate")]
                for c in dollar_cols:
                    tb_df[c] = tb_df[c].apply(fmt_dollars)
                st.dataframe(tb_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("No retirement-age data to display.")
        else:
            # Legacy mode: approximate brackets
            st.caption("Estimated income sources and marginal bracket (legacy flat-rate mode). Median values across simulations, nominal $.")
            retire_age = int(cfg_run["retire_age"])
            filing = str(cfg_run.get("conv_filing_status", cfg_run.get("gain_harvest_filing", "mfj")))
            if filing == "mfj":
                brackets = FED_BRACKETS_MFJ_2024
                std_ded_base = STANDARD_DEDUCTION_MFJ_2024
            else:
                brackets = FED_BRACKETS_SINGLE_2024
                std_ded_base = STANDARD_DEDUCTION_SINGLE_2024

            tb_rows = []
            for idx, age in enumerate(ages):
                if age < retire_age:
                    continue
                med_infl = float(np.median(infl_idx[:, idx]))
                ss_inc = float(np.percentile(de["ss_inflow"][:, idx], 50))
                ss_taxable = 0.85 * ss_inc
                rmd_inc = float(np.percentile(de["gross_trad_wd"][:, idx], 50))
                conv_inc = float(np.percentile(de["conv_gross"][:, idx], 50))
                ann_inc = float(np.percentile(de["annuity_income"][:, idx], 50))
                qcd_inc = float(np.percentile(de["qcd"][:, idx], 50))
                total_ordinary = ss_taxable + rmd_inc + conv_inc + ann_inc - qcd_inc
                std_ded_adj = std_ded_base * med_infl
                taxable_income = max(0.0, total_ordinary - std_ded_adj)
                marginal_rate = 0.10
                for i in range(len(brackets) - 1, -1, -1):
                    thresh, rate = brackets[i]
                    if taxable_income >= thresh * med_infl:
                        marginal_rate = rate
                        break
                tax_est = 0.0
                for i in range(len(brackets)):
                    thresh, rate = brackets[i]
                    adj_thresh = thresh * med_infl
                    next_thresh = brackets[i + 1][0] * med_infl if i + 1 < len(brackets) else 1e12
                    if taxable_income > adj_thresh:
                        tax_est += (min(taxable_income, next_thresh) - adj_thresh) * rate
                eff_rate = tax_est / max(1.0, taxable_income)
                tb_rows.append({
                    "Age": int(age), "SS (taxable)": ss_taxable, "RMD/Trad WD": rmd_inc,
                    "Conversions": conv_inc, "Annuity": ann_inc, "QCD": qcd_inc,
                    "Total Ordinary": total_ordinary, "Std Deduction": std_ded_adj,
                    "Taxable Income": taxable_income,
                    "Marginal Bracket": f"{marginal_rate:.0%}",
                    "Est. Effective Rate": f"{eff_rate:.1%}",
                })
            if tb_rows:
                tb_df = pd.DataFrame(tb_rows)
                for c in ["SS (taxable)", "RMD/Trad WD", "Conversions", "Annuity", "QCD",
                           "Total Ordinary", "Std Deduction", "Taxable Income"]:
                    tb_df[c] = tb_df[c].apply(fmt_dollars)
                st.dataframe(tb_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("No retirement-age data to display.")

    # ================================================================
    # ACA
    # ================================================================
    elif analysis == "ACA":
        st.html('<div class="pro-section-title">ACA Premium & Subsidy Projection (Nominal $)</div>')
        de = out["decomp"]
        _aca_mode_run = str(cfg_run.get("aca_mode", "simple"))
        if _aca_mode_run != "aca":
            st.info("ACA mode is not enabled. Enable it in Plan Setup > Health to see subsidy projections. "
                    "The tax engine must also be enabled.")
        else:
            retire_age_val = int(cfg_run["retire_age"])
            medicare_age_val = int(cfg_run.get("medicare_age", 65))
            aca_rows = []
            for idx, age in enumerate(ages):
                if age < retire_age_val or age >= medicare_age_val:
                    continue
                aca_rows.append({
                    "Age": int(age),
                    "MAGI": float(np.median(out["magi"][:, idx])),
                    "FPL %": f"{float(np.median(de['aca_gross'][:, idx])) / max(1, float(np.median(de['aca_gross'][:, idx]))) * 100:.0f}%" if de["aca_gross"][:, idx].any() else "—",
                    "Gross Premium": float(np.median(de["aca_gross"][:, idx])),
                    "Subsidy": float(np.median(de["aca_subsidy"][:, idx])),
                    "Net Premium": float(np.median(de["aca_net"][:, idx])),
                })
            if aca_rows:
                aca_df = pd.DataFrame(aca_rows)
                for c in ["MAGI", "Gross Premium", "Subsidy", "Net Premium"]:
                    aca_df[c] = aca_df[c].apply(fmt_dollars)
                st.dataframe(aca_df, use_container_width=True, hide_index=True)

                # Line chart: net premium by age
                chart_data = pd.DataFrame([{
                    "Age": r["Age"],
                    "Net Premium": float(np.median(de["aca_net"][:, ages.tolist().index(r["Age"])])),
                    "Subsidy": float(np.median(de["aca_subsidy"][:, ages.tolist().index(r["Age"])])),
                } for r in aca_rows if isinstance(r["Age"], int)])
                if not chart_data.empty:
                    chart_melt = chart_data.melt("Age", var_name="Component", value_name="Amount")
                    ch = alt.Chart(chart_melt).mark_bar().encode(
                        x=alt.X("Age:O"),
                        y=alt.Y("Amount:Q", title="Annual ($)"),
                        color=alt.Color("Component:N", scale=alt.Scale(
                            domain=["Net Premium", "Subsidy"],
                            range=["#E53935", "#00897B"])),
                    ).properties(height=300, title="ACA Costs by Age (Median)")
                    st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No ACA-eligible ages in the simulation.")

    # ================================================================
    # OPTIMIZER
    # ================================================================
    elif analysis == "Optimizer":
        st.html('<div class="pro-section-title">Policy Optimizer</div>')
        st.caption("Grid-searches over Roth conversion amounts and withdrawal strategies to find the best policy "
                   "for your chosen objective. Each candidate runs 2,000 simulations — this takes 15-30 seconds.")

        obj_label = st.selectbox("Optimization objective",
            list(OPTIMIZER_OBJECTIVES.keys()),
            index=0, key="dd_opt_objective")
        obj_key = OPTIMIZER_OBJECTIVES[obj_label]

        if st.button("Run Optimizer", key="dd_opt_run", type="primary"):
            with st.spinner("Running policy optimizer..."):
                opt_result = run_optimizer(cfg_run, hold, obj_key)
            st.session_state["_opt_result"] = opt_result
            st.session_state["_opt_objective_label"] = obj_label

        opt_result = st.session_state.get("_opt_result")
        if opt_result:
            obj_label_saved = st.session_state.get("_opt_objective_label", obj_label)
            elapsed = opt_result["elapsed"]
            baseline = opt_result["baseline"]
            candidates = opt_result["candidates"]

            st.success(f"Evaluated {len(candidates)} policies in {elapsed:.0f}s")

            # Best candidate
            best = candidates[0]
            conv_lbl = f"${best['conv_level']:,}/yr" if best["conv_level"] > 0 else "No conversions"
            wd_lbl = OPTIMIZER_WD_LABELS.get(best["wd_strategy"], best["wd_strategy"])

            st.markdown(f"### Recommended: {conv_lbl}, {wd_lbl}")

            # Comparison metrics: baseline vs best
            mc1, mc2, mc3, mc4 = st.columns(4, border=True)
            with mc1:
                delta_sr = best["success_rate"] - baseline["success_rate"]
                st.metric("Success Rate", f"{best['success_rate']:.1f}%",
                          delta=f"{delta_sr:+.1f}pp" if abs(delta_sr) > 0.1 else "same")
            with mc2:
                st.metric("Median Legacy", f"${best['median_legacy']/1e6:.2f}M",
                          delta=f"${(best['median_legacy'] - baseline['median_legacy'])/1e6:+.2f}M")
            with mc3:
                st.metric("Lifetime Taxes", f"${best['median_tax']/1e6:.2f}M",
                          delta=f"${(best['median_tax'] - baseline['median_tax'])/1e6:+.2f}M",
                          delta_color="inverse")
            with mc4:
                st.metric("Spending Vol", f"${best['spend_vol']:,.0f}",
                          delta=f"${best['spend_vol'] - candidates[-1]['spend_vol']:+,.0f}" if len(candidates) > 1 else "")

            # Comparison table — top 10
            st.markdown("### All Policies Ranked")
            rows = []
            for c in candidates[:10]:
                rows.append({
                    "Conv Amount": f"${c['conv_level']:,}/yr" if c["conv_level"] > 0 else "None",
                    "Withdrawal": OPTIMIZER_WD_LABELS.get(c["wd_strategy"], c["wd_strategy"]),
                    "Success %": f"{c['success_rate']:.1f}%",
                    "Median Legacy": fmt_dollars(c["median_legacy"]),
                    "Lifetime Tax": fmt_dollars(c["median_tax"]),
                    "Rank": "⭐" if c is best else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Bar chart: success rate by conversion level
            sr_data = []
            for c in candidates:
                sr_data.append({
                    "Conv Level": f"${c['conv_level']//1000}k" if c['conv_level'] > 0 else "$0",
                    "Strategy": OPTIMIZER_WD_LABELS.get(c["wd_strategy"], c["wd_strategy"]),
                    "Success Rate": c["success_rate"],
                })
            sr_df = pd.DataFrame(sr_data)
            sr_chart = alt.Chart(sr_df).mark_bar().encode(
                x=alt.X("Conv Level:N", sort=None, title="Roth Conversion Amount"),
                y=alt.Y("Success Rate:Q", title="Success Rate (%)"),
                color=alt.Color("Strategy:N"),
                xOffset="Strategy:N",
            ).properties(height=350, title="Success Rate by Policy")
            st.altair_chart(sr_chart, use_container_width=True)

            # Baseline comparison
            with st.expander("Baseline (your current settings)"):
                st.write(f"Success rate: **{baseline['success_rate']:.1f}%**")
                st.write(f"Median legacy: **{fmt_dollars(baseline['median_legacy'])}**")
                st.write(f"Median lifetime taxes: **{fmt_dollars(baseline['median_tax'])}**")

    # ================================================================
    # RETIRE WHEN? (Change 4)
    # ================================================================
    elif analysis == "Retire When?":
        st.html('<div class="pro-section-title">Earliest Retirement Age Finder</div>')
        st.caption("Find the earliest age you can retire and still hit your target success rate.")

        _rw_target = st.slider("Target success rate (%)", 70, 99, 90, 1, key="dd_rw_target")

        if st.button("▶ Find earliest retirement age", key="dd_rw_find"):
            _rw_start = int(cfg_run.get("start_age", 55))
            _rw_end = int(cfg_run.get("end_age", 90))
            _rw_results = []

            with st.spinner("Searching retirement ages..."):
                # Linear scan (more informative than binary search since we show a table)
                _rw_best = None
                for _rw_age in range(_rw_start + 1, min(_rw_end, _rw_start + 30)):
                    _rw_cfg = dict(cfg_run)
                    _rw_cfg["retire_age"] = _rw_age
                    _rw_cfg["n_sims"] = 1000
                    _rw_out = simulate(_rw_cfg, hold)
                    _rw_ruin = _rw_out["ruin_age"]
                    _rw_pct = 100.0 - float((_rw_ruin <= _rw_end).sum()) / len(_rw_ruin) * 100
                    _rw_results.append({"Retirement Age": _rw_age, "Success Rate": _rw_pct})
                    if _rw_best is None and _rw_pct >= _rw_target:
                        _rw_best = _rw_age

            if _rw_best is not None:
                st.html(_metric_card_html("You can retire at age", str(_rw_best),
                    f"with {_rw_target}% success rate target", "metric-green"))
            else:
                st.warning(f"No retirement age found that achieves {_rw_target}% success rate within the plan horizon.")

            # Show table around the best age
            _rw_df = pd.DataFrame(_rw_results)
            _rw_df["Success Rate"] = _rw_df["Success Rate"].map(lambda x: f"{x:.0f}%")
            st.dataframe(_rw_df, use_container_width=True, hide_index=True)

            # Line chart
            _rw_chart_df = pd.DataFrame(_rw_results)
            _rw_line = alt.Chart(_rw_chart_df).mark_line(color="#00897B", strokeWidth=3).encode(
                x=alt.X("Retirement Age:Q", title="Retirement Age"),
                y=alt.Y("Success Rate:Q", title="Success Rate (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Retirement Age:Q", alt.Tooltip("Success Rate:Q", format=".0f")]
            )
            _rw_target_rule = alt.Chart(pd.DataFrame({"y": [float(_rw_target)]})).mark_rule(
                color="#FF8F00", strokeDash=[6, 3], strokeWidth=1.5
            ).encode(y="y:Q")
            st.altair_chart((_rw_line + _rw_target_rule).properties(height=350), use_container_width=True)

    # ================================================================
    # FAILURE ANALYSIS (Change 5)
    # ================================================================
    elif analysis == "Failure Analysis":
        st.html('<div class="pro-section-title">Failure Analysis</div>')

        _fa_ruin = out["ruin_age"]
        _fa_end = int(cfg_run.get("end_age", 90))
        _fa_failed = _fa_ruin <= _fa_end
        _fa_n_failed = int(_fa_failed.sum())
        _fa_n_total = len(_fa_ruin)

        if _fa_n_failed == 0:
            st.success("No simulations ran out of money — nothing to analyze here. Your plan survives all tested scenarios.")
        else:
            _fa_pct = _fa_n_failed / _fa_n_total * 100
            _fa_ruin_ages = _fa_ruin[_fa_failed]
            _fa_med_ruin = int(np.median(_fa_ruin_ages))

            _fa_ret_idx = max(0, int(cfg_run.get("retire_age", 62)) - int(ages[0]))
            _fa_liq_at_ret = out["liquid"][_fa_failed, _fa_ret_idx] if _fa_ret_idx < out["liquid"].shape[1] else np.zeros(_fa_n_failed)
            _fa_med_liq_ret = float(np.median(_fa_liq_at_ret))

            # Metric cards
            fm1, fm2, fm3 = st.columns(3, border=True)
            with fm1:
                st.html(_metric_card_html("Failed Simulations", f"{_fa_n_failed:,}",
                    f"{_fa_pct:.1f}% of {_fa_n_total:,} total", "metric-coral"))
            with fm2:
                st.html(_metric_card_html("Median Failure Age", str(_fa_med_ruin),
                    "age when money runs out", "metric-coral"))
            with fm3:
                st.html(_metric_card_html("Liquid at Retirement", f"${_fa_med_liq_ret/1e6:.1f}M",
                    "median of failed sims", "metric-navy"))

            # When do failures happen? - histogram
            st.html('<div class="pro-section-title">When Do Failures Happen?</div>')
            _fa_hist_df = pd.DataFrame({"Failure Age": _fa_ruin_ages.astype(int)})
            _fa_hist = alt.Chart(_fa_hist_df).mark_bar(color="#E53935", opacity=0.8).encode(
                x=alt.X("Failure Age:Q", bin=alt.Bin(maxbins=20), title="Age When Money Runs Out"),
                y=alt.Y("count():Q", title="Number of Simulations"),
                tooltip=["Failure Age:Q", "count():Q"]
            ).properties(height=300)
            st.altair_chart(_fa_hist, use_container_width=True)

            # What went wrong? - sequence risk analysis
            st.html('<div class="pro-section-title">What Went Wrong?</div>')
            st.caption("Failed scenarios typically experience poor investment returns in the early retirement years — "
                       "this is sequence-of-returns risk.")

            _fa_success = ~_fa_failed
            _fa_liq = out["liquid"]
            _n_post_years = min(5, len(ages) - _fa_ret_idx - 1)
            if _n_post_years > 0 and _fa_ret_idx > 0:
                _fa_compare_rows = []
                for yr in range(1, _n_post_years + 1):
                    _idx = _fa_ret_idx + yr
                    if _idx < _fa_liq.shape[1] and _fa_ret_idx < _fa_liq.shape[1]:
                        _beg_failed = _fa_liq[_fa_failed, _fa_ret_idx]
                        _end_failed = _fa_liq[_fa_failed, _idx]
                        _cum_ret_failed = np.median((_end_failed / np.maximum(_beg_failed, 1.0) - 1.0) * 100)

                        _beg_success = _fa_liq[_fa_success, _fa_ret_idx]
                        _end_success = _fa_liq[_fa_success, _idx]
                        _cum_ret_success = np.median((_end_success / np.maximum(_beg_success, 1.0) - 1.0) * 100)

                        _fa_compare_rows.append({"Year After Retirement": yr, "Group": "Failed", "Cumulative Return (%)": _cum_ret_failed})
                        _fa_compare_rows.append({"Year After Retirement": yr, "Group": "Survived", "Cumulative Return (%)": _cum_ret_success})

                if _fa_compare_rows:
                    _fa_cmp_df = pd.DataFrame(_fa_compare_rows)
                    _fa_cmp_chart = alt.Chart(_fa_cmp_df).mark_bar().encode(
                        x=alt.X("Year After Retirement:O", title="Year After Retirement"),
                        y=alt.Y("Cumulative Return (%):Q", title="Cumulative Portfolio Change (%)"),
                        color=alt.Color("Group:N", scale=alt.Scale(
                            domain=["Failed", "Survived"],
                            range=["#E53935", "#00897B"])),
                        xOffset="Group:N",
                        tooltip=["Year After Retirement:O", "Group:N",
                                 alt.Tooltip("Cumulative Return (%):Q", format=".1f")]
                    ).properties(height=300)
                    st.altair_chart(_fa_cmp_chart, use_container_width=True)

    # ================================================================
    # REGIMES
    # ================================================================
    elif analysis == "Regimes":
        # Check the SIMULATION config (cfg_run), not live config, to see what model was actually used
        _rmodel_run = str(cfg_run.get("return_model", "standard"))
        _rmodel_live = str(cfg.get("return_model", "standard"))
        if _rmodel_run != "regime":
            if _rmodel_live == "regime":
                st.warning("You've enabled regime-switching, but the current results were simulated with the standard model. "
                           "Go to the Dashboard and re-run the simulation to see regime analysis.")
            else:
                st.info("Regime-switching is off. Enable it in Assumptions > Market, then re-run the simulation to see regime analysis.")
        else:
            st.html('<div class="pro-section-title">Regime Analysis</div>')
            st.caption("Distribution of market regimes across simulations by age.")
            r_states = out.get("regime_states", None)
            # Verify actual regime diversity (not all zeros from a standard run)
            has_diversity = r_states is not None and r_states.shape[1] > 1 and int(r_states[:, 1:].max()) > 0
            if has_diversity:
                # Average time in each state
                total_steps = r_states[:, 1:].size  # exclude t=0
                bull_pct = float((r_states[:, 1:] == 0).sum()) / total_steps * 100
                normal_pct = float((r_states[:, 1:] == 1).sum()) / total_steps * 100
                bear_pct = float((r_states[:, 1:] == 2).sum()) / total_steps * 100
                rm1, rm2, rm3 = st.columns(3, border=True)
                with rm1:
                    st.html(_metric_card_html("Bull", f"{bull_pct:.1f}%", "avg time in state", "metric-green"))
                with rm2:
                    st.html(_metric_card_html("Normal", f"{normal_pct:.1f}%", "avg time in state", "metric-navy"))
                with rm3:
                    st.html(_metric_card_html("Bear", f"{bear_pct:.1f}%", "avg time in state", "metric-coral"))

                # Stacked area chart: regime distribution by age
                reg_rows = []
                for idx in range(1, len(ages)):
                    col = r_states[:, idx]
                    n_total = len(col)
                    for si, sname in enumerate(REGIME_NAMES):
                        reg_rows.append({
                            "Age": int(ages[idx]),
                            "Regime": sname,
                            "Fraction": float((col == si).sum()) / n_total,
                        })
                reg_df = pd.DataFrame(reg_rows)
                regime_chart = alt.Chart(reg_df).mark_area().encode(
                    x=alt.X("Age:Q", title="Age"),
                    y=alt.Y("Fraction:Q", title="Fraction of Simulations", stack="normalize"),
                    color=alt.Color("Regime:N",
                        scale=alt.Scale(domain=["Bull", "Normal", "Bear"],
                                        range=["#00897B", "#1B2A4A", "#E53935"]),
                        legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Age:Q", "Regime:N", alt.Tooltip("Fraction:Q", format=".1%")],
                ).properties(height=350, title="Regime Distribution by Age")
                st.altair_chart(regime_chart, use_container_width=True)
            else:
                st.info("No regime data available. Run a simulation with regime-switching enabled.")

    # ================================================================
    # REALLOCATION
    # ================================================================
    elif analysis == "Reallocation":
        st.html('<div class="pro-section-title">Reallocation Suggestions</div>')
        st.caption("Compare your current allocation to suggested targets based on your investment priority.")

        objective = st.selectbox("Investment priority",
            ["Reduce downside risk", "Reduce volatility", "Reduce sequence risk (first 10y)",
             "Maximize expected return", "Improve tax location"],
            index=0, key="dd_objective")
        runway_years = st.slider("Years of safe spending (sequence risk)", 0, 8, 4, 1, key="dd_runway")

        target_w = normalize_weights(objective_target(objective, runway_years=runway_years))
        w_tax = hold["w_tax"]
        w_ret = hold["w_ret"]
        tax_total = float(hold["total_tax"])
        ret_total = float(hold["total_ret"])
        hsa_total = float(cfg_run.get("hsa_balance0", 0)) if bool(cfg_run.get("hsa_on", False)) else 0.0

        hsa_w = w_ret if bool(cfg_run.get("hsa_like_ret", True)) else {"Equities": 0, "REIT": 0, "Bonds": 0.8, "Alternatives": 0, "Cash": 0.2}
        w_comb = combined_weights([(tax_total, w_tax), (ret_total, w_ret), (hsa_total, hsa_w)])
        tot_all = tax_total + ret_total + hsa_total

        curr_dollars = dollars_by_bucket(tot_all, w_comb)
        tgt_dollars = dollars_by_bucket(tot_all, target_w)
        delta = {k: tgt_dollars[k] - curr_dollars[k] for k in ASSET_CLASSES}

        rec_df = pd.DataFrame({
            "Asset Class": ASSET_CLASSES,
            "Current": [fmt_dollars(curr_dollars[k]) for k in ASSET_CLASSES],
            "Current %": [f"{w_comb.get(k, 0)*100:.1f}%" for k in ASSET_CLASSES],
            "Target": [fmt_dollars(tgt_dollars[k]) for k in ASSET_CLASSES],
            "Target %": [f"{target_w.get(k, 0)*100:.1f}%" for k in ASSET_CLASSES],
            "Change": [("+" if delta[k] >= 0 else "") + fmt_dollars(delta[k]) for k in ASSET_CLASSES],
        })
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.markdown("**Implementation guidance:**")
        bullets = ["Rebalance inside **retirement accounts and HSA first** to avoid triggering taxable capital gains."]
        if objective == "Improve tax location":
            bullets.append("Keep bonds, REITs, and alternatives in retirement/HSA; hold broad stock index funds in taxable.")
        else:
            bullets.append("If reducing stocks significantly, sell in retirement/HSA first; only sell in taxable if necessary.")
        if objective == "Reduce sequence risk (first 10y)":
            bullets.append(f"Hold **{runway_years} years of spending** in bonds + cash within retirement/HSA as a safety buffer.")
        for b in bullets:
            st.markdown(f"- {b}")


# ---- 8d: Compare Scenarios page ----

# Keys to display in assumptions diff, with human-readable labels
_DIFF_KEYS = [
    ("start_age", "Start Age"), ("end_age", "End Age"), ("retire_age", "Retire Age"),
    ("has_spouse", "Has Spouse"), ("spouse_age", "Spouse Age"),
    ("spend_real", "Annual Spending"), ("spend_split_on", "Spending Split"),
    ("spend_essential_real", "Essential Spending"), ("spend_discretionary_real", "Discretionary Spending"),
    ("phase1_mult", "Go-Go Multiplier"), ("phase2_mult", "Slow-Go Multiplier"), ("phase3_mult", "No-Go Multiplier"),
    ("gk_on", "Guardrails On"), ("gk_cut", "Guardrail Cut"), ("gk_raise", "Guardrail Raise"),
    ("scenario", "Market Outlook"), ("manual_override", "Manual Override"),
    ("return_model", "Return Model"),
    ("infl_mu", "Inflation Mean"), ("infl_vol", "Inflation Vol"),
    ("tax_engine_on", "Tax Engine"), ("filing_status", "Filing Status"),
    ("niit_on", "NIIT"), ("state_rate_ordinary", "State Ordinary Rate"),
    ("state_rate_capgains", "State Cap Gains Rate"),
    ("aca_mode", "ACA Mode"), ("aca_benchmark_premium_real", "ACA Benchmark Premium"),
    ("fed_ord", "Federal Ordinary Rate"), ("state_ord", "State Ordinary Rate"),
    ("fed_capg", "Federal Cap Gains Rate"), ("basis_frac", "Cost Basis Fraction"),
    ("wd_strategy", "Withdrawal Strategy"), ("rmd_start_age", "RMD Start Age"),
    ("conv_on", "Roth Conversions"), ("conv_start", "Roth Conv Start"),
    ("conv_end", "Roth Conv End"), ("conv_real", "Roth Conv Amount"),
    ("conv_type", "Roth Conv Type"), ("conv_target_bracket", "Target Bracket"),
    ("gain_harvest_on", "Gain Harvesting"), ("qcd_on", "QCDs"),
    ("qcd_annual_real", "QCD Annual"), ("qcd_start_age", "QCD Start Age"),
    ("glide_on", "Glide Path"), ("glide_tax_eq_start", "Glide Tax Eq Start"),
    ("glide_tax_eq_end", "Glide Tax Eq End"),
    ("ann1_on", "Annuity 1"), ("ann1_type", "Ann 1 Type"),
    ("ann1_purchase_amount", "Ann 1 Purchase"), ("ann1_payout_rate", "Ann 1 Payout Rate"),
    ("ann2_on", "Annuity 2"),
    ("ss62_1", "SS Benefit (primary)"), ("claim1", "SS Claim Age (primary)"),
    ("ss62_2", "SS Benefit (spouse)"), ("claim2", "SS Claim Age (spouse)"),
    ("home_on", "Home"), ("home_value0", "Home Value"), ("mortgage_balance0", "Mortgage Balance"),
    ("sale_on", "Home Sale"), ("sale_age", "Sale Age"),
    ("inh_on", "Inheritance"), ("inh_mean", "Inheritance Mean"),
    ("ltc_on", "LTC Risk"), ("ltc_cost_real", "LTC Annual Cost"),
    ("hsa_on", "HSA"), ("hsa_balance0", "HSA Balance"),
    ("contrib_on", "Pre-Ret Income"), ("pretax_income_1", "Salary S1"), ("pretax_income_2", "Salary S2"),
    ("income_growth_real", "Wage Growth"), ("pre_ret_spend_real", "Pre-Ret Spending"),
    ("contrib_ret_annual", "Contrib Annual"), ("contrib_roth_401k_frac", "Roth 401k %"),
    ("fee_tax", "Fee (Taxable)"), ("fee_ret", "Fee (Retirement)"),
    ("irmaa_on", "IRMAA"), ("mort_on", "Mortality"),
    ("crash_on", "Crash Overlay"), ("seq_on", "Sequence Stress"), ("seq_start_age", "Shock Start Age"),
    ("legacy_on", "Legacy"),
    ("equity_granular_on", "Equity Sub-Buckets"), ("tech_bubble_on", "Tech/AI Bubble"),
    ("spending_events", "Spending Events"), ("spending_floor", "Spending Floor"),
]


def _fmt_val(v):
    """Format a config value for display in the diff table."""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, list):
        if not v:
            return "None"
        return f"{len(v)} event(s)"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"${v:,.0f}"
        if abs(v) < 1:
            return f"{v:.1%}"
        return f"{v:.2f}"
    return str(v)


def compare_page():
    """Side-by-side scenario comparison page."""
    st.html('<div style="font-size:1.8rem; font-weight:700; color:#1B2A4A; margin-bottom:0.5rem;">Compare Scenarios</div>')

    saved = st.session_state["saved_scenarios"]
    if len(saved) < 2:
        st.info(
            f"You have **{len(saved)}** saved scenario(s). Save at least **2** from the Results page "
            "using the **:material/add_circle: Save as Scenario** button, then return here to compare."
        )
        if saved:
            st.markdown("**Saved so far:**")
            for i, s in enumerate(saved):
                st.markdown(f"- {s['name']}")
        return

    # ---- Scenario selector (pick exactly 2) ----
    names = [s["name"] for s in saved]
    sel_col, load_col, del_col = st.columns([3, 1, 1])
    with sel_col:
        picked = st.multiselect("Select exactly 2 scenarios to compare", names, default=names[:2],
                                max_selections=2, key="compare_pick")
    with load_col:
        st.markdown("")  # spacing
        load_name = st.selectbox("Load into Results", ["—"] + names, key="compare_load",
                                 help="Load a saved scenario's settings so you can edit and re-save it")
        if load_name != "—":
            sc_load = next(s for s in saved if s["name"] == load_name)
            st.session_state["cfg"] = copy.deepcopy(sc_load["cfg"])
            st.session_state["hold"] = copy.deepcopy(sc_load["hold"])
            st.session_state["_active_scenario_name"] = load_name
            st.rerun()
    with del_col:
        st.markdown("")  # spacing
        del_name = st.selectbox("Delete a scenario", ["—"] + names, key="compare_del")
        if del_name != "—":
            st.session_state["saved_scenarios"] = [s for s in saved if s["name"] != del_name]
            # Clear active name if we deleted the active one
            if st.session_state.get("_active_scenario_name") == del_name:
                st.session_state.pop("_active_scenario_name", None)
            st.rerun()

    if len(picked) != 2:
        st.warning("Select exactly 2 scenarios to compare.")
        return

    sc_a = next(s for s in saved if s["name"] == picked[0])
    sc_b = next(s for s in saved if s["name"] == picked[1])

    # ---- Assumptions diff table ----
    st.html('<div class="pro-section-title">Assumptions That Differ</div>')
    diff_rows = []
    for key, label in _DIFF_KEYS:
        va = sc_a["cfg"].get(key)
        vb = sc_b["cfg"].get(key)
        if va != vb:
            diff_rows.append({"Assumption": label, picked[0]: _fmt_val(va), picked[1]: _fmt_val(vb)})
    # Also compare holdings
    for hkey, hlabel in [("total_tax", "Taxable Balance"), ("total_ret", "Retirement Balance")]:
        va = sc_a["hold"].get(hkey)
        vb = sc_b["hold"].get(hkey)
        if va != vb:
            diff_rows.append({"Assumption": hlabel, picked[0]: _fmt_val(va), picked[1]: _fmt_val(vb)})

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
    else:
        st.success("These two scenarios have identical assumptions.")

    # ---- Run both simulations ----
    with st.spinner("Running simulations for both scenarios..."):
        cfg_a = build_cfg_run(sc_a["cfg"])
        cfg_b = build_cfg_run(sc_b["cfg"])
        out_a = simulate(cfg_a, sc_a["hold"])
        out_b = simulate(cfg_b, sc_b["hold"])

    # ---- Side-by-side metric cards ----
    st.html('<div class="pro-section-title">Key Metrics</div>')

    def _compute_metrics(out, cfg):
        """Compute standard metrics from simulation output."""
        end_age_val = int(cfg["end_age"])
        ruin_age = out["ruin_age"]
        ran_out = (ruin_age <= end_age_val).sum()
        pct_success = 100.0 - float(ran_out) / len(ruin_age) * 100
        sL = summarize_end(out["liquid"])
        sN = summarize_end(out["net_worth"])
        sLr = summarize_end(out["liquid_real"])
        sNr = summarize_end(out["net_worth_real"])
        funded = out["funded_through_age"]
        return {
            "pct_success": pct_success,
            "funded_through": funded,
            "liq_p50": sL["p50"],
            "nw_p50": sN["p50"],
            "liq_real_p50": sLr["p50"],
            "nw_real_p50": sNr["p50"],
            "end_age": end_age_val,
        }

    met_a = _compute_metrics(out_a, sc_a["cfg"])
    met_b = _compute_metrics(out_b, sc_b["cfg"])

    col_a, col_b = st.columns(2, border=True)
    for col, met, name in [(col_a, met_a, picked[0]), (col_b, met_b, picked[1])]:
        with col:
            st.markdown(f"**{name}**")
            sr_color = "metric-green" if met["pct_success"] >= 90 else ("metric-amber" if met["pct_success"] >= 75 else "metric-coral")
            st.html(_metric_card_html("Success Rate", f"{met['pct_success']:.0f}%",
                                      f"of sims have money at {met['end_age']}", sr_color))
            ft_str = f"{met['funded_through']}+" if met["funded_through"] >= met["end_age"] else str(met["funded_through"])
            st.html(_metric_card_html("Funded Through Age", ft_str,
                                      "in a bad (p10) scenario", "metric-navy"))
            st.html(_metric_card_html(f"Liquid Assets at {met['end_age']}", f"${met['liq_p50']/1e6:.1f}M",
                                      f"median nominal · real: ${met['liq_real_p50']/1e6:.1f}M", "metric-navy"))
            st.html(_metric_card_html(f"Net Worth at {met['end_age']}", f"${met['nw_p50']/1e6:.1f}M",
                                      f"median nominal · real: ${met['nw_real_p50']/1e6:.1f}M", "metric-navy"))

    # ---- Success rate comparison bar ----
    st.html('<div class="pro-section-title">Success Rate Comparison</div>')
    bar_df = pd.DataFrame({
        "Scenario": [picked[0], picked[1]],
        "Success Rate (%)": [met_a["pct_success"], met_b["pct_success"]],
    })
    bar_chart = alt.Chart(bar_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Scenario:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Success Rate (%):Q", scale=alt.Scale(domain=[0, 100]), title="Success Rate (%)"),
        color=alt.Color("Scenario:N", scale=alt.Scale(
            domain=[picked[0], picked[1]], range=["#1B2A4A", "#00897B"]
        ), legend=None),
        tooltip=["Scenario:N", alt.Tooltip("Success Rate (%):Q", format=".1f")],
    ).properties(height=250)
    bar_text = alt.Chart(bar_df).mark_text(dy=-12, fontSize=16, fontWeight="bold").encode(
        x="Scenario:N",
        y="Success Rate (%):Q",
        text=alt.Text("Success Rate (%):Q", format=".0f"),
        color=alt.Color("Scenario:N", scale=alt.Scale(
            domain=[picked[0], picked[1]], range=["#1B2A4A", "#00897B"]
        ), legend=None),
    )
    st.altair_chart(bar_chart + bar_text, use_container_width=True)

    # ---- Impact attribution: which differences matter most? ----
    # Group related config keys so we attribute impact to logical changes, not individual knobs
    _ATTRIB_GROUPS = [
        ("Ages", ["start_age", "end_age", "retire_age"]),
        ("Household", ["has_spouse", "spouse_age"]),
        ("Spending", ["spend_real", "spend_split_on", "spend_essential_real", "spend_discretionary_real", "spending_events"]),
        ("Spending Phases", ["phase1_end", "phase2_end", "phase1_mult", "phase2_mult", "phase3_mult"]),
        ("Guardrails", ["gk_on", "gk_upper_pct", "gk_lower_pct", "gk_cut", "gk_raise"]),
        ("Market Outlook", ["scenario", "manual_override", "override_params"]),
        ("Return Model", ["return_model", "regime_params", "regime_transition", "regime_initial_probs"]),
        ("Inflation", ["infl_mu", "infl_vol", "infl_min", "infl_max"]),
        ("Tax Engine", ["tax_engine_on", "filing_status", "niit_on",
                        "state_rate_ordinary", "state_rate_capgains"]),
        ("Tax Rates", ["fed_capg", "fed_div", "fed_ord", "state_capg", "state_ord"]),
        ("Tax Strategy", ["basis_frac", "div_yield", "dist_yield", "tlh_on", "tlh_reduction",
                          "wd_strategy", "rmd_on", "rmd_start_age", "roth_frac"]),
        ("Roth Conversions", ["conv_on", "conv_start", "conv_end", "conv_real", "conv_type",
                              "conv_target_bracket", "conv_irmaa_aware", "conv_irmaa_target_tier", "conv_filing_status"]),
        ("Gain Harvesting", ["gain_harvest_on", "gain_harvest_filing"]),
        ("QCDs", ["qcd_on", "qcd_annual_real", "qcd_start_age", "qcd_max_annual"]),
        ("Glide Path", ["glide_on", "glide_tax_eq_start", "glide_tax_eq_end",
                        "glide_tax_start_age", "glide_tax_end_age", "glide_ret_same",
                        "glide_ret_eq_start", "glide_ret_eq_end", "glide_ret_start_age", "glide_ret_end_age"]),
        ("Annuity 1", ["ann1_on", "ann1_type", "ann1_purchase_age", "ann1_income_start_age",
                        "ann1_purchase_amount", "ann1_payout_rate", "ann1_cola_on", "ann1_cola_rate", "ann1_cola_match_inflation"]),
        ("Annuity 2", ["ann2_on", "ann2_type", "ann2_purchase_age", "ann2_income_start_age",
                        "ann2_purchase_amount", "ann2_payout_rate", "ann2_cola_on", "ann2_cola_rate", "ann2_cola_match_inflation"]),
        ("Social Security", ["fra", "ss62_1", "ss62_2", "claim1", "claim2"]),
        ("Mortality", ["mort_on", "spend_drop_after_death", "home_cost_drop_after_death",
                       "q55_1", "q55_2", "mort_growth"]),
        ("Health Care", ["pre65_health_on", "pre65_health_real", "medicare_age",
                         "aca_mode", "aca_benchmark_premium_real", "aca_household_size",
                         "ltc_on", "ltc_start_age", "ltc_annual_prob", "ltc_cost_real", "ltc_duration_mean", "ltc_duration_sigma"]),
        ("Home", ["home_on", "home_value0", "home_mu", "home_vol", "home_cost_pct",
                  "mortgage_balance0", "mortgage_rate", "mortgage_term_years",
                  "sale_on", "sale_age", "selling_cost_pct", "post_sale_mode", "downsize_fraction", "rent_real"]),
        ("Inheritance", ["inh_on", "inh_min", "inh_mean", "inh_sigma", "inh_prob", "inh_horizon"]),
        ("HSA", ["hsa_on", "hsa_balance0", "hsa_like_ret", "hsa_med_real"]),
        ("Pre-Ret Income & Contributions", ["contrib_on", "pretax_income_1", "pretax_income_2",
            "income_growth_real", "pre_ret_spend_real", "contrib_ret_annual", "contrib_roth_401k_frac",
            "contrib_match_annual", "contrib_taxable_annual", "contrib_hsa_annual"]),
        ("Fees", ["fee_tax", "fee_ret"]),
        ("IRMAA", ["irmaa_on", "irmaa_people", "irmaa_base", "irmaa_t1", "irmaa_p1", "irmaa_t2", "irmaa_p2", "irmaa_t3", "irmaa_p3"]),
        ("Crash Overlay", ["crash_on", "crash_prob", "crash_eq_extra", "crash_reit_extra", "crash_alt_extra", "crash_home_extra"]),
        ("Sequence Stress", ["seq_on", "seq_drop", "seq_years", "seq_start_age"]),
        ("Equity Sub-Buckets", ["equity_granular_on", "equity_sub_weights", "equity_sub_mu",
                                "equity_sub_vol", "tech_bubble_on", "tech_bubble_prob", "tech_bubble_extra_drop"]),
        ("Allocation", []),  # placeholder — holdings checked separately
    ]

    # Find which groups actually differ
    def _group_differs(group_keys, cfg_x, cfg_y):
        for k in group_keys:
            if cfg_x.get(k) != cfg_y.get(k):
                return True
        return False

    def _hold_differs(hold_x, hold_y):
        for k in ["total_tax", "total_ret", "w_tax", "w_ret"]:
            if hold_x.get(k) != hold_y.get(k):
                return True
        return False

    differing_groups = []
    for gname, gkeys in _ATTRIB_GROUPS:
        if gname == "Allocation":
            if _hold_differs(sc_a["hold"], sc_b["hold"]):
                differing_groups.append(("Allocation", []))
        elif _group_differs(gkeys, sc_a["cfg"], sc_b["cfg"]):
            differing_groups.append((gname, gkeys))

    if differing_groups and len(differing_groups) <= 15:
        with st.expander("Impact attribution — which differences matter most?", expanded=True):
            st.caption(
                f"Starting from **{picked[0]}**, each bar shows how much the success rate changes "
                f"when switching that one group of assumptions to **{picked[1]}**'s values. "
                "Uses 1,000 sims per test for speed."
            )
            base_success = met_a["pct_success"]
            attrib_rows = []
            # Build a fast version of cfg_a
            cfg_a_fast = dict(cfg_a)
            cfg_a_fast["n_sims"] = 1000
            with st.spinner("Running attribution analysis..."):
                for gname, gkeys in differing_groups:
                    # Start from A, swap this group to B's values
                    cfg_test = dict(cfg_a_fast)
                    hold_test = copy.deepcopy(sc_a["hold"])
                    if gname == "Allocation":
                        hold_test = copy.deepcopy(sc_b["hold"])
                    else:
                        for k in gkeys:
                            if k in sc_b["cfg"]:
                                cfg_test[k] = copy.deepcopy(sc_b["cfg"][k])
                        # Rebuild scenario_params if we changed market outlook keys
                        if gname == "Market Outlook":
                            cfg_test = build_cfg_run(cfg_test)
                            cfg_test["n_sims"] = 1000
                    out_test = simulate(cfg_test, hold_test)
                    ra_test = out_test["ruin_age"]
                    end_age_test = int(cfg_test.get("end_age", sc_a["cfg"]["end_age"]))
                    pct_test = 100.0 - float((ra_test <= end_age_test).sum()) / len(ra_test) * 100
                    delta = pct_test - base_success
                    attrib_rows.append({"Assumption Group": gname, "Δ Success Rate (pp)": round(delta, 1)})

            if attrib_rows:
                att_df = pd.DataFrame(attrib_rows)
                att_df["abs_delta"] = att_df["Δ Success Rate (pp)"].abs()
                att_df = att_df.sort_values("abs_delta", ascending=False).drop(columns="abs_delta")
                att_df = att_df[att_df["Δ Success Rate (pp)"] != 0.0].reset_index(drop=True)

                if len(att_df) == 0:
                    st.info("No single assumption group produces a measurable change in success rate (differences may be too small or offsetting).")
                else:
                    att_df["color"] = att_df["Δ Success Rate (pp)"].apply(lambda x: "Helps" if x > 0 else "Hurts")
                    bar = alt.Chart(att_df).mark_bar(cornerRadius=4).encode(
                        y=alt.Y("Assumption Group:N", sort=None, title=None,
                                axis=alt.Axis(labelLimit=0, labelOverlap=False)),
                        x=alt.X("Δ Success Rate (pp):Q", title=f"Change in success rate (pp) from {picked[0]} → {picked[1]}"),
                        color=alt.Color("color:N", scale=alt.Scale(
                            domain=["Helps", "Hurts"], range=["#00897B", "#E53935"]
                        ), legend=alt.Legend(title=None, orient="top")),
                        tooltip=["Assumption Group:N", alt.Tooltip("Δ Success Rate (pp):Q", format="+.1f")],
                    ).properties(height=max(len(att_df) * 40, 150))
                    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                        color="#666", strokeDash=[4, 2]
                    ).encode(x="x:Q")
                    labels = alt.Chart(att_df).mark_text(
                        align=alt.expr(alt.expr.if_(alt.datum["Δ Success Rate (pp)"] >= 0, "left", "right")),
                        dx=alt.expr(alt.expr.if_(alt.datum["Δ Success Rate (pp)"] >= 0, 4, -4)),
                        fontSize=12, fontWeight="bold",
                    ).encode(
                        y=alt.Y("Assumption Group:N", sort=None),
                        x="Δ Success Rate (pp):Q",
                        text=alt.Text("Δ Success Rate (pp):Q", format="+.1f"),
                        color=alt.Color("color:N", scale=alt.Scale(
                            domain=["Helps", "Hurts"], range=["#00897B", "#E53935"]
                        ), legend=None),
                    )
                    st.altair_chart(bar + zero_rule + labels, use_container_width=True)

                    st.caption("**Note:** Individual impacts may not sum to the total difference because assumptions can interact.")
    elif len(differing_groups) > 15:
        st.info("Too many assumption groups differ to run attribution analysis. Try comparing scenarios with fewer differences.")

    # ---- Side-by-side fan charts (shared y-axis) ----
    st.html('<div class="pro-section-title">Wealth Projections (Liquid Assets, Nominal)</div>')

    # Compute shared y_max for wealth charts
    p95_a = np.percentile(out_a["liquid"], 95, axis=0).max() / 1e6
    p95_b = np.percentile(out_b["liquid"], 95, axis=0).max() / 1e6
    wealth_y_max = max(p95_a, p95_b) * 1.05

    fc_a, fc_b = st.columns(2)
    with fc_a:
        st.markdown(f"**{picked[0]}**")
        render_wealth_fan_chart(out_a["liquid"], out_a["ages"],
                                title_label="Liquid ($M)",
                                retire_age=int(sc_a["cfg"]["retire_age"]),
                                y_max=wealth_y_max)
    with fc_b:
        st.markdown(f"**{picked[1]}**")
        render_wealth_fan_chart(out_b["liquid"], out_b["ages"],
                                title_label="Liquid ($M)",
                                retire_age=int(sc_b["cfg"]["retire_age"]),
                                y_max=wealth_y_max)

    # ---- Side-by-side spending fan charts ----
    st.html('<div class="pro-section-title">Spending Projections (Real Dollars)</div>')

    # Compute shared y_max for spending charts
    def _spending_p90_max(out, cfg):
        spend = out["decomp"]["spend_real_track"]
        ridx = max(0, int(cfg["retire_age"]) - int(cfg["start_age"]))
        post = spend[:, ridx:]
        if post.shape[1] < 2:
            return 0
        return np.percentile(post, 90, axis=0).max() / 1e3

    sp_max_a = _spending_p90_max(out_a, sc_a["cfg"])
    sp_max_b = _spending_p90_max(out_b, sc_b["cfg"])
    spend_y_max = max(sp_max_a, sp_max_b) * 1.10 if max(sp_max_a, sp_max_b) > 0 else None

    sc_a_col, sc_b_col = st.columns(2)
    with sc_a_col:
        st.markdown(f"**{picked[0]}**")
        render_spending_fan_chart(out_a, cfg_a, y_max=spend_y_max)
    with sc_b_col:
        st.markdown(f"**{picked[1]}**")
        render_spending_fan_chart(out_b, cfg_b, y_max=spend_y_max)



# ---- 8e: Methodology page (Feature 6) ----
def methodology_page():
    st.html('<div style="font-size:1.8rem; font-weight:700; color:#1B2A4A; margin-bottom:0.5rem;">How RetireLab Works</div>')
    st.caption("A plain-English guide to what's under the hood.")

    st.html('<div class="pro-section-title">Monte Carlo Simulation</div>')
    st.markdown(
        "RetireLab doesn't predict one future — it simulates **thousands** of possible futures. "
        "Each simulation randomly draws investment returns, inflation rates, and other variables "
        "from realistic distributions, then plays out your entire financial life year by year. "
        "The result is a probability distribution: instead of saying 'you'll have $X,' it says "
        "'in 90% of scenarios, you still have money at age 95.'\n\n"
        "This is the same methodology used by institutional pension funds and major financial planning firms. "
        "The key insight: a plan that works in the *average* case can still fail badly if the market drops "
        "at the wrong time (this is called **sequence-of-returns risk**)."
    )

    st.html('<div class="pro-section-title">Reading the Fan Charts</div>')
    st.markdown(
        "The colored bands on the projection charts show the range of outcomes across all simulations:\n\n"
        "- **Dark center band** — the middle 20% of outcomes (40th to 60th percentile). This is the 'most likely' range.\n"
        "- **Medium band** — the middle 60% (20th to 80th percentile). Most scenarios fall here.\n"
        "- **Light outer band** — the middle 90% (5th to 95th percentile). Only extreme outliers fall outside this.\n"
        "- **Solid line** — the median (50th percentile), the single most likely path.\n\n"
        "A plan with a tight, upward-sloping fan is robust. A plan with a wide fan that dips below zero "
        "has meaningful risk of running out of money."
    )

    st.html('<div class="pro-section-title">How Taxes Are Computed</div>')
    st.markdown(
        "RetireLab uses a detailed tax engine that models:\n\n"
        "- **Federal income tax** with inflation-adjusted brackets (10%, 12%, 22%, 24%, 32%, 35%, 37%)\n"
        "- **Standard deduction** adjusted for age (extra deduction for 65+)\n"
        "- **Social Security taxation** (up to 85% of benefits can be taxable, based on provisional income)\n"
        "- **Capital gains** at preferential rates (0%, 15%, 20%) depending on income\n"
        "- **NIIT** (3.8% Net Investment Income Tax) for high earners\n"
        "- **State income tax** as a flat rate you specify\n"
        "- **IRMAA** Medicare surcharges for high-income retirees\n\n"
        "All brackets and thresholds are inflation-adjusted each year using simulated inflation. "
        "The tax engine runs every year of every simulation, so tax-efficient strategies like Roth "
        "conversions and gain harvesting are modeled realistically."
    )

    st.html('<div class="pro-section-title">Withdrawal Order</div>')
    st.markdown(
        "When you need money in retirement, RetireLab withdraws in a tax-efficient order:\n\n"
        "1. **Social Security and annuity income** first (these are fixed and automatic)\n"
        "2. **Taxable account** — preferring to spend from here early to let tax-advantaged accounts grow\n"
        "3. **Traditional IRA/401(k)** — taxed as ordinary income when withdrawn\n"
        "4. **Roth IRA** — tax-free withdrawals, saved for last to maximize tax-free growth\n\n"
        "Required Minimum Distributions (RMDs) from traditional accounts are taken automatically starting at age 73."
    )

    st.html('<div class="pro-section-title">Adaptive Guardrails</div>')
    st.markdown(
        "The Guyton-Klinger guardrail system adjusts your spending based on portfolio performance:\n\n"
        "- Each year, it calculates your current withdrawal rate (spending ÷ portfolio)\n"
        "- If the withdrawal rate rises too high (portfolio shrank), spending is **cut** by a set percentage\n"
        "- If the withdrawal rate drops too low (portfolio grew), spending is **raised** by a set percentage\n"
        "- Essential spending is never cut — only discretionary spending is reduced\n\n"
        "This mimics how real retirees behave: tightening the belt in bad years and enjoying more in good years. "
        "Research shows this improves plan survival by 10–15 percentage points compared to rigid withdrawals."
    )

    st.html('<div class="pro-section-title">Regime-Switching Model</div>')
    st.markdown(
        "The optional regime-switching model (under Market settings) generates more realistic return sequences "
        "by modeling three market states:\n\n"
        "- **Bull** — above-average returns, low volatility (typical of long expansions)\n"
        "- **Normal** — moderate returns and volatility (the baseline)\n"
        "- **Bear** — negative returns, high volatility (recessions and crises)\n\n"
        "Each year, the market can transition between states with calibrated probabilities. "
        "This captures the 'clustering' of bad years (bear markets last multiple years) and the long "
        "bull runs that simple models miss. The result is more realistic tail risk estimates."
    )

    st.html('<div class="pro-section-title">What the Success Rate Means</div>')
    st.markdown(
        "The success rate is the percentage of simulations where you still have liquid assets at "
        "your plan end age. A few guidelines:\n\n"
        "- **95%+** — Very strong. Your plan handles most bad scenarios.\n"
        "- **85–95%** — Solid. Some risk in worst-case scenarios, but reasonable.\n"
        "- **75–85%** — Moderate risk. Consider adjustments to spending or income.\n"
        "- **Below 75%** — Significant risk. Changes are strongly recommended.\n\n"
        "Note: 100% is not the goal — that usually means you're spending too little and leaving money on the table. "
        "Most planners target 85–95% as a reasonable balance between security and enjoyment."
    )

    st.html('<div class="pro-section-title">Sensitivity Analysis (Tornado Charts)</div>')
    st.markdown(
        "The 'What Matters Most' tornado chart shows which assumptions have the biggest impact on your outcome. "
        "For each variable, the model runs two variants (e.g., spending +$25k and -$25k) and shows how much "
        "the median ending net worth changes compared to your baseline.\n\n"
        "The variables with the longest bars are the ones most worth your attention. If 'Annual Spending' "
        "dominates the chart, your spending level is the most impactful lever. If 'Stock Returns' is largest, "
        "your outcome depends heavily on market performance — and diversification or guardrails become important."
    )

    st.html('<div class="pro-section-title">Key Limitations</div>')
    st.markdown(
        "RetireLab is a planning tool, not a crystal ball:\n\n"
        "- **Returns are random** — the model doesn't predict markets, it tests your plan against a wide range of scenarios.\n"
        "- **Tax law changes** — current brackets and rules may change. The model uses today's tax code projected forward.\n"
        "- **Spending is simplified** — real spending is irregular (car repairs, vacations, medical events). The model uses smooth annual averages.\n"
        "- **No behavioral modeling** — the model assumes you follow the plan. In reality, people panic-sell or overspend.\n"
        "- **Inflation uncertainty** — the model simulates variable inflation, but extreme scenarios (hyperinflation) are unlikely but possible.\n\n"
        "Use RetireLab to understand the *range* of possibilities and which decisions matter most — not as a precise prediction."
    )

    st.divider()
    st.caption("RetireLab is for educational and informational purposes only. It is not financial, tax, or investment advice. "
               "Consult a qualified financial professional for personalized guidance.")


# ============================================================
# SECTION 9: Navigation entrypoint
# ============================================================
inject_theme_css()
init_defaults()

pg = st.navigation(
    [
        st.Page(dashboard_page, title="Results", icon=":material/dashboard:", default=True),
        st.Page(plan_setup_page, title="Assumptions", icon=":material/tune:"),
        st.Page(deep_dive_page, title="Deep Dive", icon=":material/analytics:"),
        st.Page(compare_page, title="Compare", icon=":material/compare_arrows:"),
        st.Page(methodology_page, title="How It Works", icon=":material/help_outline:"),
    ],
    position="top",
)
pg.run()

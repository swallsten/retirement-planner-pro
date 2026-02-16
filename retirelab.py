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
    "Bull":   {"eq_mu": 0.15, "eq_vol": 0.12, "reit_mu": 0.14, "reit_vol": 0.14, "bond_mu": 0.04, "bond_vol": 0.04, "alt_mu": 0.10, "alt_vol": 0.10, "cash_mu": 0.03},
    "Normal": {"eq_mu": 0.07, "eq_vol": 0.16, "reit_mu": 0.07, "reit_vol": 0.18, "bond_mu": 0.03, "bond_vol": 0.06, "alt_mu": 0.05, "alt_vol": 0.14, "cash_mu": 0.02},
    "Bear":   {"eq_mu": -0.10, "eq_vol": 0.28, "reit_mu": -0.08, "reit_vol": 0.30, "bond_mu": 0.02, "bond_vol": 0.10, "alt_mu": -0.03, "alt_vol": 0.22, "cash_mu": 0.02},
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


@st.cache_data(show_spinner=False)
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
    hsa_allow_nonmed_65 = bool(cfg["hsa_allow_nonmed_65"])

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

    # taxes
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
    conv_filing_status = str(cfg.get("conv_filing_status", "mfj"))
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
    gain_harvest_filing = str(cfg.get("gain_harvest_filing", "mfj"))
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

    # Pre-retirement contributions
    contrib_on = bool(cfg.get("contrib_on", False)) and (start_age < retire_age)
    contrib_ret = float(cfg.get("contrib_ret_annual", 0.0)) if contrib_on else 0.0
    contrib_match = float(cfg.get("contrib_match_annual", 0.0)) if contrib_on else 0.0
    contrib_taxable = float(cfg.get("contrib_taxable_annual", 0.0)) if contrib_on else 0.0
    contrib_hsa = float(cfg.get("contrib_hsa_annual", 0.0)) if (contrib_on and hsa_on) else 0.0

    # -------- Decomposition trackers (nominal) --------
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
    gross_hsa_wd_track = np.zeros((n, T+1))
    taxes_paid_track = np.zeros((n, T+1))
    conv_gross_track = np.zeros((n, T+1))
    qcd_track = np.zeros((n, T+1))
    gain_harvest_track = np.zeros((n, T+1))
    essential_spend_track = np.zeros((n, T+1))
    discretionary_spend_track = np.zeros((n, T+1))
    essential_funded_track = np.zeros((n, T+1))

    for t in range(1, T+1):
        age = int(ages[t-1])
        years_left = int(end_age - age)

        alive1 = age < death1
        alive2 = age < death2
        both_alive = alive1 & alive2
        one_alive = alive1 ^ alive2
        none_alive = ~(alive1 | alive2)

        if use_regime and t <= regime_states.shape[1]:
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

        if seq_on and age >= retire_age and age < retire_age + seq_years:
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

        # Total outflow
        outflow_nom = adjusted_core_nom + home_cost_nom + mort_pay_nom + rent_nom + health_nom + medical_short + ltc_nom
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
                ordinary_est = ss_est + rmd_preview + div_est + ann_income_this_year
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
            conv_tax = conv_gross * eff_ord
            pay_from_tax = np.minimum(tax_prev, conv_tax)
            tax_prev -= pay_from_tax
            remaining = conv_tax - pay_from_tax
            trad_prev = np.maximum(0.0, trad_prev - remaining)
            trad_prev = np.maximum(0.0, trad_prev - conv_gross)
            roth_prev = roth_prev + conv_gross
        conv_gross_track[:, t] = conv_gross

        inflows_nom = ss_real * infl_index[:, t-1]
        ss_nom_track[:, t] = inflows_nom
        # Add annuity income to inflows
        inflows_nom = inflows_nom + ann_income_this_year

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

        net_rmd_cash = taxable_rmd * (1 - eff_ord)
        used_rmd = np.minimum(net_rmd_cash, need_before)
        need = np.maximum(0.0, need_before - used_rmd)
        excess = np.maximum(0.0, net_rmd_cash - used_rmd)
        tax_prev += excess
        bas_prev += excess
        trad_prev = np.maximum(0.0, trad_prev - gross_rmd)

        # Gain harvesting: step up basis when in 0% LTCG bracket
        gain_harvested = np.zeros(n)
        if gain_harvest_on and age >= retire_age:
            # Estimate taxable ordinary income
            ss_for_tax = 0.85 * ss_real * infl_index[:, t-1]
            div_inc = tax_prev * (float(cfg["div_yield"]) + float(cfg["dist_yield"]))
            ordinary_income_est = ss_for_tax + taxable_rmd + conv_gross + ann_income_this_year + div_inc
            # Subtract inflation-adjusted standard deduction
            std_ded_adj = std_deduction * infl_index[:, t-1]
            taxable_income_est = np.maximum(0.0, ordinary_income_est - std_ded_adj)
            # Room in 0% LTCG bracket
            ltcg_threshold_adj = ltcg_0pct_threshold * infl_index[:, t-1]
            room = np.maximum(0.0, ltcg_threshold_adj - taxable_income_est)
            # Harvest: step up basis (no balance change, no cash)
            unrealized_gain = np.maximum(0.0, tax_prev - bas_prev)
            gain_harvested = np.minimum(room, unrealized_gain)
            bas_prev += gain_harvested  # step up basis
            if mort_on:
                gain_harvested[none_alive] = 0.0
        gain_harvest_track[:, t] = gain_harvested

        # IRMAA
        if irmaa_on and age >= medicare_age:
            magi = taxable_rmd + conv_gross + ann_income_this_year + tax_prev * (float(cfg["div_yield"]) + float(cfg["dist_yield"]))
            magi_track[:, t] = magi
            prem = irmaa_monthly_per_person(magi, irmaa_base, irmaa_schedule)
            irmaa_nom = prem * 12.0 * irmaa_people
            if mort_on:
                irmaa_nom[one_alive] *= 0.60
                irmaa_nom[none_alive] = 0.0
            need += irmaa_nom
            irmaa_paid_track[:, t] = irmaa_nom
            tier = np.zeros(n)
            for i, (th, _) in enumerate(irmaa_schedule, start=1):
                tier = np.where(magi >= th, i, tier)
            tier_track[:, t] = tier

        # Optional: allow non-medical HSA withdrawals after 65
        gross_hsa_nonmed = np.zeros(n)
        if hsa_on and hsa_allow_nonmed_65 and age >= 65:
            pass

        # withdrawals (taxable/trad/roth)
        gain = np.maximum(0.0, tax_prev - bas_prev)
        gain_frac = np.where(tax_prev > 0, np.minimum(1.0, gain / tax_prev), 0.0)
        denom_tax = np.maximum(1e-6, 1 - gain_frac * eff_capg_wd)
        denom_trad = np.maximum(1e-6, 1 - eff_ord)

        gross_tax = np.zeros(n)
        gross_trad = np.zeros(n)
        gross_roth = np.zeros(n)

        if cfg["wd_strategy"] == "taxable_first":
            take_tax = np.minimum(tax_prev, need / denom_tax)
            gross_tax += take_tax
            need2 = np.maximum(0.0, need - take_tax * denom_tax)

            take_trad = np.minimum(trad_prev, need2 / denom_trad)
            gross_trad += take_trad
            need3 = np.maximum(0.0, need2 - take_trad * denom_trad)

            take_roth = np.minimum(roth_prev, need3)
            gross_roth += take_roth
        else:
            total_avail = tax_prev + trad_prev + roth_prev
            share_tax = np.where(total_avail > 0, tax_prev / total_avail, 0.0)
            share_trad = np.where(total_avail > 0, trad_prev / total_avail, 0.0)
            share_roth = 1 - share_tax - share_trad

            take_tax = np.minimum(tax_prev, (need * share_tax) / denom_tax)
            take_trad = np.minimum(trad_prev, (need * share_trad) / denom_trad)
            take_roth = np.minimum(roth_prev, (need * share_roth))

            gross_tax += take_tax
            gross_trad += take_trad
            gross_roth += take_roth

        gross_tax_wd_track[:, t] = gross_tax
        gross_trad_wd_track[:, t] = gross_trad
        gross_roth_wd_track[:, t] = gross_roth
        gross_hsa_wd_track[:, t] = gross_hsa_nonmed

        taxes_paid_track[:, t] = (gross_tax * gain_frac * eff_capg_wd) + (gross_trad * eff_ord) + (gross_hsa_nonmed * eff_ord)

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

        # Pre-retirement contributions
        if contrib_on and age < retire_age:
            ci = infl_index[:, t]
            total_ret_contrib = (contrib_ret + contrib_match) * ci
            trad[:, t] += total_ret_contrib * (1 - roth_frac)
            roth[:, t] += total_ret_contrib * roth_frac
            if contrib_taxable > 0:
                taxable[:, t] += contrib_taxable * ci
                basis[:, t] += contrib_taxable * ci
            if hsa_on and contrib_hsa > 0:
                hsa[:, t] += contrib_hsa * ci

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

    return {
        "ages": ages,
        "liquid": liquid,
        "net_worth": net_worth,
        "liquid_real": liquid_real,
        "net_worth_real": net_worth_real,
        "home_value": home_value,
        "mortgage": mortgage,
        "hsa": hsa,
        "magi": magi_track,
        "irmaa_tier": tier_track,
        "legacy": legacy,
        "ruin_age": ruin_age,
        "funded_through_age": funded_through_age,
        "second_death": second_death,
        "decomp": {
            "baseline_core": baseline_core_track,
            "core_adjusted": core_spend_track,
            "spend_real_track": core_spend_track / np.maximum(infl_index, 1e-9),
            "home_cost": home_cost_track,
            "mort_pay": mort_pay_track,
            "rent": rent_track,
            "health": health_track,
            "medical_nom": medical_nom_track,
            "hsa_used_med": hsa_withdraw_track,
            "ltc_cost": ltc_cost_track,
            "outflow_total": total_outflow_track,
            "ss_inflow": ss_nom_track,
            "irmaa": irmaa_paid_track,
            "gross_tax_wd": gross_tax_wd_track,
            "gross_trad_wd": gross_trad_wd_track,
            "gross_roth_wd": gross_roth_wd_track,
            "taxes_paid": taxes_paid_track,
            "hsa_end": hsa_end_track,
            "annuity_income": annuity_income_track,
            "annuity_purchase": annuity_purchase_track,
            "conv_gross": conv_gross_track,
            "qcd": qcd_track,
            "gain_harvest": gain_harvest_track,
            "essential_spend": essential_spend_track,
            "discretionary_spend": discretionary_spend_track,
            "essential_funded": essential_funded_track,
        },
        "infl_index": infl_index,
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

    # Paired tests: (variable_label, up_label, up_patch, down_label, down_patch)
    paired_tests = [
        ("Annual Spending ±$50k",
         "Spend $50k more", {"spend_real": float(cfg_fast["spend_real"]) + 50000.0},
         "Spend $50k less", {"spend_real": max(50000.0, float(cfg_fast["spend_real"]) - 50000.0)}),
        ("Retirement Age ±2 yrs",
         "Retire 2 yrs later", {"retire_age": int(cfg_fast["retire_age"]) + 2},
         "Retire 2 yrs earlier", {"retire_age": max(int(cfg_fast["start_age"]) + 1, int(cfg_fast["retire_age"]) - 2)}),
        ("SS Claim Age (Spouse 1)",
         "Claim at 70", {"claim1": 70},
         "Claim at 67", {"claim1": 67}),
        ("Stock Returns ±2%",
         "Returns +2%", {"scenario_params": dict(cfg_fast["scenario_params"], eq_mu=float(cfg_fast["scenario_params"]["eq_mu"]) + 0.02)},
         "Returns −2%", {"scenario_params": dict(cfg_fast["scenario_params"], eq_mu=float(cfg_fast["scenario_params"]["eq_mu"]) - 0.02)}),
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
        ("Discretionary Spending ±$50k",
         "Discretionary +$50k", {"spend_split_on": True,
                                  "spend_essential_real": float(cfg_fast.get("spend_essential_real", 180000.0)),
                                  "spend_discretionary_real": float(cfg_fast.get("spend_discretionary_real", 120000.0)) + 50000.0,
                                  "spend_real": float(cfg_fast["spend_real"]) + 50000.0},
         "Discretionary −$50k", {"spend_split_on": True,
                                  "spend_essential_real": float(cfg_fast.get("spend_essential_real", 180000.0)),
                                  "spend_discretionary_real": max(10000.0, float(cfg_fast.get("spend_discretionary_real", 120000.0)) - 50000.0),
                                  "spend_real": max(float(cfg_fast.get("spend_essential_real", 180000.0)) + 10000.0, float(cfg_fast["spend_real"]) - 50000.0)}),
        ("0% LTCG Gain Harvesting",
         "Harvest 0% gains", {"gain_harvest_on": True, "gain_harvest_filing": "mfj"},
         "No harvesting", {"gain_harvest_on": False}),
        ("$30k QCD",
         "$30k QCD", {"qcd_on": True, "qcd_annual_real": 30000.0, "qcd_start_age": 70, "rmd_on": True},
         "No QCD", {"qcd_on": False}),
    ]

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
        tor = _run_sensitivity_tests(cfg_run, hold, n_fast=2000)
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


@st.dialog("Early-Retirement Bad-Luck Shock", width="large")
def open_sequence_dialog():
    cfg = st.session_state["cfg"]
    st.markdown("What if the market crashes right when you retire? Force a downturn in the first years of retirement to see if your plan survives.")
    cfg["seq_on"] = st.toggle("Test a bad market right after you retire", value=bool(cfg["seq_on"]), key="dlg_seq_on")
    if cfg["seq_on"]:
        cfg["seq_drop"] = st.number_input(
            "Annual return penalty during stress period",
            value=float(cfg["seq_drop"]), step=0.01, format="%.2f", key="dlg_seq_drop")
        cfg["seq_years"] = st.number_input(
            "How many years the bad period lasts",
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
    _d("n_sims", 10000)
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

    # Sequence stress
    _d("seq_on", False)
    _d("seq_drop", -0.25)
    _d("seq_years", 2)

    # Taxes
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
    _d("hsa_allow_nonmed_65", False)

    # Pre-retirement contributions
    _d("contrib_on", True)
    _d("contrib_ret_annual", 23500.0)
    _d("contrib_match_annual", 11750.0)
    _d("contrib_taxable_annual", 0.0)
    _d("contrib_hsa_annual", 4300.0)

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

    # ---- Portfolio upload (compact) ----
    _using_defaults = (abs(hold["total_tax"] - 1_000_000.0) < 1.0 and abs(hold["total_ret"] - 2_000_000.0) < 1.0
                       and not st.session_state.get("_portfolio_uploaded", False))
    if _using_defaults:
        st.warning("**Using placeholder portfolio** ($1M taxable + $2M retirement). "
                   "Upload your brokerage CSVs below, or go to **Assumptions → Basics** to enter manually.", icon="⚠️")
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
        # Process uploads
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
            f"median nominal · real: ${sLr['p50']/1e6:.1f}M", "metric-navy"
        ))
    with m4:
        st.html(_metric_card_html(
            f"Net Worth at {end_age_val}", f"${sN['p50']/1e6:.1f}M",
            f"median nominal · real: ${sNr['p50']/1e6:.1f}M", "metric-navy"
        ))

    st.markdown("")

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
        comp_rows = []
        for sname, sparams in DEFAULT_SCENARIOS.items():
            ctmp = dict(cfg_run)
            ctmp["scenario_params"] = dict(sparams)
            ctmp["n_sims"] = 2000
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
    render_mini_tornado(cfg_run, hold, top_n=6)


# ---- 8b: Plan Setup page ----
def plan_setup_page():
    cfg = st.session_state["cfg"]
    hold = st.session_state["hold"]

    st.html('<div style="font-size:1.8rem; font-weight:700; color:#1B2A4A; margin-bottom:0.5rem;">Plan Setup</div>')

    section = st.segmented_control(
        "Section",
        ["Basics", "Income", "Spending", "Home", "Health", "Taxes", "Market", "Allocation", "Stress Tests", "Advanced"],
        default="Basics", key="setup_section"
    )

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

            if tax_file:
                try:
                    tax_df, tax_val = load_snapshot(tax_file)
                    w_tax_u, total_tax_u, dollars_tax_u = weights_and_dollars(tax_df, tax_val)
                    tax_ok = True
                except Exception as e:
                    st.error(f"Error parsing taxable CSV: {e}")

            if ret_file:
                try:
                    ret_df, ret_val = load_snapshot(ret_file)
                    w_ret_u, total_ret_u, dollars_ret_u = weights_and_dollars(ret_df, ret_val)
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

        st.html('<div class="pro-section-title">Pre-Retirement Savings</div>')
        cfg["contrib_on"] = st.toggle("Still saving before retirement", value=bool(cfg["contrib_on"]), key="ps_contrib_on",
            help="If you haven't retired yet, the model adds your annual savings contributions to your accounts each year until retirement.")
        if not cfg["contrib_on"]:
            st.caption("Toggle on if you're still working and contributing to 401(k), IRA, taxable, or HSA accounts.")
        if cfg["contrib_on"]:
            cc1, cc2 = st.columns(2, border=True)
            with cc1:
                cfg["contrib_ret_annual"] = st.number_input("Annual 401k/IRA contributions", value=float(cfg["contrib_ret_annual"]), step=1000.0, key="ps_contrib_ret")
                cfg["contrib_match_annual"] = st.number_input("Employer match (annual)", value=float(cfg["contrib_match_annual"]), step=1000.0, key="ps_contrib_match")
            with cc2:
                cfg["contrib_taxable_annual"] = st.number_input("Extra taxable savings", value=float(cfg["contrib_taxable_annual"]), step=1000.0, key="ps_contrib_tax")
                cfg["contrib_hsa_annual"] = st.number_input("Annual HSA contributions", value=float(cfg["contrib_hsa_annual"]), step=500.0, key="ps_contrib_hsa")

    # ================================================================
    # INCOME
    # ================================================================
    elif section == "Income":
        st.html('<div class="pro-section-title">Social Security</div>')
        cfg["fra"] = st.number_input("Full retirement age (FRA)", value=int(cfg["fra"]), step=1, key="ps_fra")

        sc1, sc2 = st.columns(2, border=True)
        with sc1:
            st.markdown("**Spouse 1**")
            cfg["ss62_1"] = st.number_input("Monthly benefit at 62", value=float(cfg["ss62_1"]), step=50.0, key="ps_ss62_1")
            cfg["claim1"] = st.number_input("Claim age", value=int(cfg["claim1"]), step=1, key="ps_claim1")
        with sc2:
            if cfg["has_spouse"]:
                st.markdown("**Spouse 2**")
                cfg["ss62_2"] = st.number_input("Monthly benefit at 62", value=float(cfg["ss62_2"]), step=50.0, key="ps_ss62_2")
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
                cfg["inh_mean"] = st.number_input("Expected amount", value=float(cfg["inh_mean"]), step=100000.0, key="ps_inh_mean")
                cfg["inh_min"] = st.number_input("Minimum amount", value=float(cfg["inh_min"]), step=100000.0, key="ps_inh_min")
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

    # ================================================================
    # SPENDING
    # ================================================================
    elif section == "Spending":
        st.html('<div class="pro-section-title">Annual Core Spending</div>')
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
                "Annual core living expenses (today's dollars)", value=float(cfg["spend_real"]), step=10000.0, key="ps_spend_real",
                help="Everyday costs only. Housing, health insurance, and medical are entered separately."
            )
            st.info("This covers groceries, dining, utilities, transportation, travel, entertainment, etc. "
                    "Housing, health insurance, and medical costs are tracked separately.")

        st.html('<div class="pro-section-title">Spending Phases</div>')
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
            help="Guyton-Klinger guardrails automatically cut or raise spending based on portfolio performance.")
        if not cfg["gk_on"]:
            st.caption("Toggle on to enable dynamic spending rules that adjust your withdrawals up or down based on how your portfolio is doing.")
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
            with hh2:
                cfg["pre65_health_real"] = st.number_input("Annual health insurance cost", value=float(cfg["pre65_health_real"]), step=1000.0, key="ps_pre65_cost")

        st.html('<div class="pro-section-title">Health Savings Account (HSA)</div>')
        cfg["hsa_on"] = st.toggle("Include HSA", value=bool(cfg["hsa_on"]), key="ps_hsa_on",
            help="Model a Health Savings Account that pays medical costs tax-free and grows tax-free.")
        if not cfg["hsa_on"]:
            st.caption("Toggle on to include your HSA balance, medical spending, and tax-free growth in the simulation.")
        if cfg["hsa_on"]:
            hs1, hs2 = st.columns(2, border=True)
            with hs1:
                cfg["hsa_balance0"] = st.number_input("Current HSA balance", value=float(cfg["hsa_balance0"]), step=1000.0, key="ps_hsa_bal")
                cfg["hsa_med_real"] = st.number_input("Annual medical spending", value=float(cfg["hsa_med_real"]), step=500.0, key="ps_hsa_med")
            with hs2:
                cfg["hsa_like_ret"] = st.toggle("Invest HSA like retirement", value=bool(cfg["hsa_like_ret"]), key="ps_hsa_invest")
                cfg["hsa_allow_nonmed_65"] = st.toggle("Allow non-medical HSA after 65", value=bool(cfg["hsa_allow_nonmed_65"]), key="ps_hsa_nonmed")

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
                cfg["ltc_cost_real"] = st.number_input("Annual LTC cost", value=float(cfg["ltc_cost_real"]), step=10000.0, key="ps_ltc_cost")
                cfg["ltc_duration_mean"] = st.number_input("Average years of care", value=float(cfg["ltc_duration_mean"]), step=0.5, format="%.1f", key="ps_ltc_dur")
                cfg["ltc_duration_sigma"] = st.number_input("Duration uncertainty", value=float(cfg["ltc_duration_sigma"]), step=0.5, format="%.1f", key="ps_ltc_sig")

    # ================================================================
    # TAXES
    # ================================================================
    elif section == "Taxes":
        st.html('<div class="pro-section-title">Tax Rates</div>')
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
            if cfg["gain_harvest_on"]:
                cfg["gain_harvest_filing"] = st.selectbox("Filing status for LTCG brackets",
                    ["mfj", "single"],
                    index=["mfj", "single"].index(str(cfg["gain_harvest_filing"])),
                    format_func=lambda x: "Married Filing Jointly" if x == "mfj" else "Single",
                    key="ps_gh_filing")
            else:
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
                cfg["qcd_max_annual"] = st.number_input("IRS annual QCD limit",
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
            cfg["conv_on"] = st.toggle("Plan Roth conversions", value=bool(cfg["conv_on"]), key="ps_conv_on")
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
                    cfg["conv_filing_status"] = st.selectbox("Filing status",
                        ["mfj", "single"],
                        index=["mfj", "single"].index(str(cfg["conv_filing_status"])),
                        format_func=lambda x: "Married Filing Jointly" if x == "mfj" else "Single",
                        key="ps_conv_filing")
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
            format_func=lambda x: "Standard (t-distribution)" if x == "standard" else "Regime-switching (Markov)",
            key="ps_return_model",
            help="Standard: single set of return parameters with optional crash overlay. "
                 "Regime-switching: returns vary depending on whether we're in a Bull, Normal, or Bear market state.")

        if cfg["return_model"] == "regime":
            st.info("When regime-switching is on, the market outlook table is ignored. "
                    "Returns are drawn from state-specific distributions with Markov transitions between Bull, Normal, and Bear markets.")
            r_params = cfg.get("regime_params", dict(DEFAULT_REGIME_PARAMS))
            r_trans = cfg.get("regime_transition", [list(row) for row in DEFAULT_TRANSITION_MATRIX])
            r_init = cfg.get("regime_initial_probs", list(DEFAULT_REGIME_INITIAL_PROBS))

            st.html('<div class="pro-section-title">State Parameters</div>')
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
            help="Gradually shift your portfolio from stocks to bonds as you age. The equity percentage changes linearly between a start and end age.")
        if not cfg["glide_on"]:
            st.caption("Toggle on to automatically shift your allocation over time. Without this, your portfolio weights stay constant throughout the simulation.")
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

        st.html('<div class="pro-section-title">Early-Retirement Bad-Luck Shock</div>')
        st.caption("What if the market tanks right when you retire? This is the most dangerous scenario for retirees — "
                   "withdrawing from a shrinking portfolio in year one can permanently damage your plan.")
        cfg["seq_on"] = st.toggle("Test a bad market right after you retire", value=bool(cfg["seq_on"]), key="ps_seq_on",
            help="Forces a market downturn in the first N years of retirement. Every simulation gets hit with this shock — "
                 "it's not random. This tests whether your plan can survive the worst-case timing.")
        if not cfg["seq_on"]:
            st.caption("Toggle on to force a market drop in the first years of retirement and see if your plan survives.")
        if cfg["seq_on"]:
            sq1, sq2 = st.columns(2, border=True)
            with sq1:
                cfg["seq_drop"] = st.number_input("Annual return penalty during shock", min_value=-0.50, max_value=0.0,
                    value=float(cfg["seq_drop"]), step=0.05, format="%.2f", key="ps_seq_drop",
                    help="Extra negative return added to stocks and REITs each year during the shock window. "
                         "E.g. -0.25 = an additional 25% loss per year on top of normal randomness.")
            with sq2:
                cfg["seq_years"] = st.number_input("How many years the bad period lasts", min_value=1, max_value=10,
                    value=int(cfg["seq_years"]), step=1, key="ps_seq_years",
                    help="Number of years after retirement that the forced downturn applies.")

    # ================================================================
    # ADVANCED
    # ================================================================
    elif section == "Advanced":
        st.html('<div class="pro-section-title">Simulation Settings</div>')
        ad1, ad2 = st.columns(2, border=True)
        with ad1:
            cfg["n_sims"] = st.slider("Number of simulations", 1000, 50000, int(cfg["n_sims"]), 1000, key="ps_n_sims")
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
        st.warning("Run a simulation from the Dashboard first to see detailed analysis.")
        if st.button("Go to Dashboard", key="dd_go_dash"):
            st.switch_page("Dashboard")
        return

    out = st.session_state["sim_result"]
    cfg_run = st.session_state["cfg_run"]
    ages = out["ages"]

    analysis = st.segmented_control(
        "Analysis",
        ["Cashflows", "Sensitivity", "IRMAA", "Legacy", "Annuity", "Roth Strategy", "Tax Brackets", "Regimes", "Reallocation"],
        default="Cashflows", key="dd_analysis"
    )

    # ================================================================
    # CASHFLOWS
    # ================================================================
    if analysis == "Cashflows":
        st.html('<div class="pro-section-title">Year-by-Year Cashflow Breakdown</div>')
        st.caption("Median (50th percentile) values across all simulations, in nominal dollars.")

        de = out["decomp"]
        liq = out["liquid"]
        nw = out["net_worth"]
        hv = out["home_value"]
        mb = out["mortgage"]
        home_eq = np.maximum(0.0, hv - mb)
        hsa_bal = out["hsa"]

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
                "Annuity Income": float(np.percentile(de["annuity_income"][:, idx], 50)),
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

    # ================================================================
    # SENSITIVITY
    # ================================================================
    elif analysis == "Sensitivity":
        st.html('<div class="pro-section-title">Full Sensitivity Analysis</div>')
        st.caption("Each variable shows the impact of both an increase and decrease. Uses 3,000 simulations per test for speed.")
        with st.spinner("Running sensitivity tests..."):
            tor = _run_sensitivity_tests(cfg_run, hold, n_fast=3000)
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
        st.html('<div class="pro-section-title">Medicare Surcharge Tiers</div>')
        if not bool(cfg_run.get("irmaa_on", False)):
            st.info("IRMAA tracking is off. Enable it in Plan Setup > Taxes to see this analysis.")
        else:
            st.caption("Median simulated income and resulting IRMAA tier at each age.")
            magi_med = np.percentile(out["magi"], 50, axis=0)
            tier_med = np.percentile(out["irmaa_tier"], 50, axis=0)
            df = pd.DataFrame({"Age": ages, "Estimated MAGI": magi_med, "IRMAA Tier": tier_med})
            df = df[df["Age"] >= int(cfg_run.get("medicare_age", 65))].copy()
            df["Estimated MAGI"] = df["Estimated MAGI"].apply(fmt_dollars)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ================================================================
    # LEGACY
    # ================================================================
    elif analysis == "Legacy":
        st.html('<div class="pro-section-title">What You Leave Behind</div>')
        if not bool(cfg_run.get("legacy_on", True)):
            st.info("Legacy analysis is off. Enable it in Plan Setup > Advanced.")
        else:
            st.caption("Estimated estate value at second death (includes home equity). Assumes stepped-up cost basis.")
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
            st.caption("Compares your current Roth conversion strategy against no conversions. Uses 3,000 simulations for speed.")
            with st.spinner("Running comparison simulations..."):
                # Run with conversions (current live settings)
                cfg_with = dict(_roth_cfg)
                cfg_with["n_sims"] = 3000
                out_with = simulate(cfg_with, hold)

                # Run without conversions
                cfg_without = dict(_roth_cfg)
                cfg_without["n_sims"] = 3000
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
        st.caption("Estimated income sources, deductions, and marginal bracket by age. Median values across simulations.")

        de = out["decomp"]
        infl_idx = out.get("infl_index", None)
        if infl_idx is None:
            st.info("Run the simulation again to see tax bracket projections.")
        else:
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
                # Approximate taxable SS as 85%
                ss_taxable = 0.85 * ss_inc
                rmd_inc = float(np.percentile(de["gross_trad_wd"][:, idx], 50))  # approx
                conv_inc = float(np.percentile(de["conv_gross"][:, idx], 50))
                ann_inc = float(np.percentile(de["annuity_income"][:, idx], 50))
                qcd_inc = float(np.percentile(de["qcd"][:, idx], 50))

                total_ordinary = ss_taxable + rmd_inc + conv_inc + ann_inc - qcd_inc
                std_ded_adj = std_ded_base * med_infl
                taxable_income = max(0.0, total_ordinary - std_ded_adj)

                # Find marginal bracket
                marginal_rate = 0.10
                for i in range(len(brackets) - 1, -1, -1):
                    thresh, rate = brackets[i]
                    if taxable_income >= thresh * med_infl:
                        marginal_rate = rate
                        break

                # Effective rate estimate
                tax_est = 0.0
                for i in range(len(brackets)):
                    thresh, rate = brackets[i]
                    adj_thresh = thresh * med_infl
                    if i + 1 < len(brackets):
                        next_thresh = brackets[i + 1][0] * med_infl
                    else:
                        next_thresh = 1e12
                    if taxable_income > adj_thresh:
                        bracket_income = min(taxable_income, next_thresh) - adj_thresh
                        tax_est += bracket_income * rate
                eff_rate = tax_est / max(1.0, taxable_income)

                tb_rows.append({
                    "Age": int(age),
                    "SS (taxable)": ss_taxable,
                    "RMD/Trad WD": rmd_inc,
                    "Conversions": conv_inc,
                    "Annuity": ann_inc,
                    "QCD": qcd_inc,
                    "Total Ordinary": total_ordinary,
                    "Std Deduction": std_ded_adj,
                    "Taxable Income": taxable_income,
                    "Marginal Bracket": f"{marginal_rate:.0%}",
                    "Est. Effective Rate": f"{eff_rate:.1%}",
                })

            if tb_rows:
                tb_df = pd.DataFrame(tb_rows)
                # Format dollar columns
                dollar_cols = ["SS (taxable)", "RMD/Trad WD", "Conversions", "Annuity", "QCD",
                               "Total Ordinary", "Std Deduction", "Taxable Income"]
                for c in dollar_cols:
                    tb_df[c] = tb_df[c].apply(fmt_dollars)
                st.dataframe(tb_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("No retirement-age data to display.")

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
    ("contrib_on", "Pre-Ret Contributions"), ("contrib_ret_annual", "Contrib Annual"),
    ("fee_tax", "Fee (Taxable)"), ("fee_ret", "Fee (Retirement)"),
    ("irmaa_on", "IRMAA"), ("mort_on", "Mortality"),
    ("crash_on", "Crash Overlay"), ("seq_on", "Sequence Stress"),
    ("legacy_on", "Legacy"),
]


def _fmt_val(v):
    """Format a config value for display in the diff table."""
    if isinstance(v, bool):
        return "Yes" if v else "No"
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
        ("Spending", ["spend_real", "spend_split_on", "spend_essential_real", "spend_discretionary_real"]),
        ("Spending Phases", ["phase1_end", "phase2_end", "phase1_mult", "phase2_mult", "phase3_mult"]),
        ("Guardrails", ["gk_on", "gk_upper_pct", "gk_lower_pct", "gk_cut", "gk_raise"]),
        ("Market Outlook", ["scenario", "manual_override", "override_params"]),
        ("Return Model", ["return_model", "regime_params", "regime_transition", "regime_initial_probs"]),
        ("Inflation", ["infl_mu", "infl_vol", "infl_min", "infl_max"]),
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
                         "ltc_on", "ltc_start_age", "ltc_annual_prob", "ltc_cost_real", "ltc_duration_mean", "ltc_duration_sigma"]),
        ("Home", ["home_on", "home_value0", "home_mu", "home_vol", "home_cost_pct",
                  "mortgage_balance0", "mortgage_rate", "mortgage_term_years",
                  "sale_on", "sale_age", "selling_cost_pct", "post_sale_mode", "downsize_fraction", "rent_real"]),
        ("Inheritance", ["inh_on", "inh_min", "inh_mean", "inh_sigma", "inh_prob", "inh_horizon"]),
        ("HSA", ["hsa_on", "hsa_balance0", "hsa_like_ret", "hsa_med_real", "hsa_allow_nonmed_65"]),
        ("Contributions", ["contrib_on", "contrib_ret_annual", "contrib_match_annual"]),
        ("Fees", ["fee_tax", "fee_ret"]),
        ("IRMAA", ["irmaa_on", "irmaa_people", "irmaa_base", "irmaa_t1", "irmaa_p1", "irmaa_t2", "irmaa_p2", "irmaa_t3", "irmaa_p3"]),
        ("Crash Overlay", ["crash_on", "crash_prob", "crash_eq_extra", "crash_reit_extra", "crash_alt_extra", "crash_home_extra"]),
        ("Sequence Stress", ["seq_on", "seq_drop", "seq_years"]),
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
                "Uses 2,000 sims per test for speed."
            )
            base_success = met_a["pct_success"]
            attrib_rows = []
            # Build a fast version of cfg_a
            cfg_a_fast = dict(cfg_a)
            cfg_a_fast["n_sims"] = 2000
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
                            cfg_test["n_sims"] = 2000
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
    ],
    position="top",
)
pg.run()

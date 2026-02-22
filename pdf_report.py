"""
pdf_report.py — Generates a polished PDF retirement plan summary for RetireLab.

Uses reportlab for PDF generation and matplotlib for embedded charts.
All chart images are rendered to PNG via BytesIO, then embedded via ImageReader.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Color palette ──────────────────────────────────────────────────────
NAVY = colors.HexColor("#1B2A4A")
TEAL = colors.HexColor("#00897B")
LIGHT_GRAY = colors.HexColor("#F5F5F5")
WHITE = colors.white
MED_GRAY = colors.HexColor("#CCCCCC")
DARK_TEXT = colors.HexColor("#222222")

NAVY_HEX = "#1B2A4A"
TEAL_HEX = "#00897B"
LIGHT_BLUE_HEX = "#B3D4FC"
RED_HEX = "#E53935"
GREEN_HEX = "#00897B"

# ── Formatting helpers ─────────────────────────────────────────────────
def _fmt_dollars(val: float, force_k: bool = False) -> str:
    """Format dollars: $1.2M for millions, $123K for thousands, $1,234 otherwise."""
    if val is None or np.isnan(val):
        return "N/A"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.1f}M"
    if abs(val) >= 10000 or force_k:
        return f"${val / 1e3:,.0f}K"
    return f"${val:,.0f}"


def _fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def _fmt_pct_whole(val: float) -> str:
    return f"{val:.0f}%"


# ── Matplotlib chart helpers ───────────────────────────────────────────
def _chart_to_image(fig, dpi: int = 150) -> io.BytesIO:
    """Convert a matplotlib figure to a BytesIO PNG buffer for reportlab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _style_ax(ax, show_grid: bool = False):
    """Apply clean minimal styling to a matplotlib axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=8)
    if show_grid:
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax.set_axisbelow(True)


def _dollar_formatter(x, _):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


# ── IRMAA constants (mirrored from retirelab.py) ──────────────────────
IRMAA_MAGI_TIERS_MFJ = [206000, 258000, 322000, 386000, 750000]
IRMAA_MAGI_TIERS_SINGLE = [103000, 129000, 161000, 193000, 500000]
IRMAA_PART_B_MFJ = [(206000, 244.60), (258000, 349.40), (322000, 454.20),
                     (386000, 559.00), (750000, 594.00)]
IRMAA_PART_B_SINGLE = [(103000, 244.60), (129000, 349.40), (161000, 454.20),
                        (193000, 559.00), (500000, 594.00)]
IRMAA_PART_D_MFJ = [(206000, 12.90), (258000, 33.30), (322000, 53.80),
                     (386000, 74.20), (750000, 81.00)]
IRMAA_PART_D_SINGLE = [(103000, 12.90), (129000, 33.30), (161000, 53.80),
                        (193000, 74.20), (500000, 81.00)]


# ── Table builder helpers ──────────────────────────────────────────────
def _build_table(data: list[list], col_widths: list[float] | None = None,
                 header: bool = True) -> Table:
    """Build a styled reportlab Table with alternating row colors."""
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_TEXT),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, MED_GRAY),
    ]
    if header and len(data) > 0:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
        ]
        # Alternating row colors starting from row 1
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _build_kv_table(pairs: list[tuple[str, str]], col_widths=None) -> Table:
    """Build a two-column key-value table (no header row)."""
    if col_widths is None:
        col_widths = [2.2 * inch, 1.3 * inch]
    tbl = Table(pairs, colWidths=col_widths, hAlign="LEFT")
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_TEXT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, MED_GRAY),
    ]
    for i in range(len(pairs)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Paragraph styles ──────────────────────────────────────────────────
def _styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
                                textColor=NAVY, spaceAfter=4, alignment=TA_LEFT,
                                leading=22),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9,
                                   textColor=colors.HexColor("#666666"), spaceAfter=10,
                                   leading=12),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14,
                             textColor=NAVY, spaceBefore=14, spaceAfter=8,
                             leading=18),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11,
                             textColor=TEAL, spaceBefore=10, spaceAfter=6,
                             leading=14),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                               textColor=DARK_TEXT, leading=12, spaceAfter=4),
        "body_small": ParagraphStyle("body_small", fontName="Helvetica", fontSize=8,
                                     textColor=colors.HexColor("#555555"), leading=10),
        "verdict": ParagraphStyle("verdict", fontName="Helvetica-Bold", fontSize=11,
                                  textColor=NAVY, alignment=TA_CENTER,
                                  spaceBefore=8, spaceAfter=8, leading=15),
        "metric_label": ParagraphStyle("metric_label", fontName="Helvetica", fontSize=8,
                                       textColor=colors.HexColor("#666666"),
                                       alignment=TA_CENTER),
        "metric_value": ParagraphStyle("metric_value", fontName="Helvetica-Bold",
                                       fontSize=16, textColor=NAVY, alignment=TA_CENTER),
        "footer": ParagraphStyle("footer", fontName="Helvetica", fontSize=7,
                                 textColor=colors.HexColor("#999999")),
        "warning": ParagraphStyle("warning", fontName="Helvetica", fontSize=8,
                                  textColor=colors.HexColor("#E53935"), leading=10,
                                  spaceBefore=2, spaceAfter=2),
    }


# ══════════════════════════════════════════════════════════════════════
# PAGE BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _page1_executive_summary(cfg, hold, out, ages, styles) -> list:
    """Page 1: Executive Summary."""
    elements = []
    de = out["decomp"]

    # ── Header ──
    elements.append(Paragraph("Retirement Plan Summary", styles["title"]))

    filing = str(cfg.get("conv_filing_status", "mfj")).upper()
    if filing == "MFJ":
        filing = "Married Filing Jointly"
    elif filing == "SINGLE":
        filing = "Single"
    sub = (f"Generated {date.today().strftime('%B %d, %Y')}  &bull;  "
           f"Age {int(cfg['start_age'])} &rarr; Retirement at {int(cfg['retire_age'])} "
           f"&rarr; Plan through age {int(cfg['end_age'])}  &bull;  {filing}")
    elements.append(Paragraph(sub, styles["subtitle"]))
    elements.append(Spacer(1, 6))

    # ── Key Metrics ──
    end_age = int(cfg["end_age"])
    ruin = out["ruin_age"]
    success_rate = 100.0 - float((ruin <= end_age).sum()) / len(ruin) * 100.0
    spend = float(cfg["spend_real"])
    legacy_med = float(np.median(out["legacy"]))
    taxes_med = float(np.median(de["taxes_paid"].sum(axis=1)))
    irmaa_on = bool(cfg.get("irmaa_on", False))
    irmaa_med = float(np.median(de["irmaa"].sum(axis=1))) if irmaa_on else None

    floor_val = float(cfg.get("spending_floor", 80000))
    real_spend = de["core_adjusted"] / np.maximum(out["infl_index"], 1e-9)
    # Only check retirement years
    retire_age = int(cfg["retire_age"])
    retire_idx = max(0, int(retire_age - ages[0]))
    if retire_idx < real_spend.shape[1]:
        breached = (real_spend[:, retire_idx:] < floor_val).any(axis=1)
        pct_breached = float(breached.sum()) / len(breached) * 100.0
    else:
        pct_breached = 0.0

    # Build metrics as a table for clean layout
    metric_data = [
        ("Success Rate", _fmt_pct(success_rate)),
        ("Annual Spending", _fmt_dollars(spend)),
        ("Median Legacy", _fmt_dollars(legacy_med)),
        ("Lifetime Taxes", _fmt_dollars(taxes_med)),
    ]
    if irmaa_on and irmaa_med and irmaa_med > 0:
        metric_data.append(("Lifetime IRMAA", _fmt_dollars(irmaa_med)))
    metric_data.append(("Floor Breach Prob.", _fmt_pct(pct_breached)))

    # Create a prominent metrics box
    m_header = [m[0] for m in metric_data]
    m_values = [m[1] for m in metric_data]
    n_cols = len(metric_data)
    col_w = 7.0 * inch / n_cols

    metrics_tbl = Table([m_values, m_header], colWidths=[col_w] * n_cols, hAlign="CENTER")
    metrics_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 14),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("ALIGNMENT", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, 1), 7),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#666666")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 1, TEAL),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, MED_GRAY),
    ]))
    elements.append(metrics_tbl)
    elements.append(Spacer(1, 10))

    # ── Verdict sentence ──
    verdict = (f"Your plan has a <b>{success_rate:.1f}%</b> probability of funding "
               f"<b>{_fmt_dollars(spend)}/yr</b> (inflation-adjusted) "
               f"through age <b>{end_age}</b>.")
    elements.append(Paragraph(verdict, styles["verdict"]))
    elements.append(Spacer(1, 10))

    # ── Spending Floor ──
    if floor_val > 0:
        floor_text = (f"Spending floor: {_fmt_dollars(floor_val)}/yr. "
                      f"Probability of breaching: {_fmt_pct(pct_breached)}.")
        elements.append(Paragraph(floor_text, styles["body"]))
        elements.append(Spacer(1, 6))

    # ── Current Portfolio ──
    elements.append(Paragraph("Current Portfolio", styles["h3"]))
    total_tax = float(hold.get("total_tax", 0))
    total_ret = float(hold.get("total_ret", 0))
    roth_frac = float(cfg.get("roth_frac", 0.2))
    trad_val = total_ret * (1 - roth_frac)
    roth_val = total_ret * roth_frac
    hsa_on = bool(cfg.get("hsa_on", False))
    hsa_bal = float(cfg.get("hsa_balance0", 0)) if hsa_on else 0

    port_rows = [["Account", "Balance"]]
    port_rows.append(["Taxable", _fmt_dollars(total_tax)])
    port_rows.append(["Tax-Deferred", _fmt_dollars(trad_val)])
    port_rows.append(["Roth", _fmt_dollars(roth_val)])
    if hsa_on and hsa_bal > 0:
        port_rows.append(["HSA", _fmt_dollars(hsa_bal)])
    grand_total = total_tax + total_ret + hsa_bal
    port_rows.append(["Total", _fmt_dollars(grand_total)])

    port_tbl = _build_table(port_rows, col_widths=[2.5 * inch, 1.5 * inch])
    # Bold the total row
    port_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, len(port_rows) - 1), (-1, len(port_rows) - 1), "Helvetica-Bold"),
        ("LINEABOVE", (0, len(port_rows) - 1), (-1, len(port_rows) - 1), 1, NAVY),
    ]))
    elements.append(port_tbl)

    return elements


def _page2_trajectory(cfg, hold, out, ages, styles) -> list:
    """Page 2: Portfolio Trajectory."""
    elements = []
    elements.append(Paragraph("Portfolio Trajectory", styles["h2"]))

    # ── Wealth chart ──
    liquid = out["liquid"]  # (n_sims, T+1)
    med = np.median(liquid, axis=0)
    p10 = np.percentile(liquid, 10, axis=0)
    p90 = np.percentile(liquid, 90, axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.fill_between(ages, p10, p90, alpha=0.2, color=LIGHT_BLUE_HEX, label="10th–90th percentile")
    ax.plot(ages, med, color=NAVY_HEX, linewidth=2, label="Median")
    ax.set_xlabel("Age", fontsize=9, color="#555555")
    ax.set_ylabel("Liquid Wealth", fontsize=9, color="#555555")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _style_ax(ax)

    # Annotate key events
    retire_age = int(cfg["retire_age"])
    claim1 = int(cfg.get("claim1", 67))
    claim2 = int(cfg.get("claim2", 67))
    medicare_age = int(cfg.get("medicare_age", 65))
    rmd_age = int(cfg.get("rmd_start_age", 73))
    has_spouse = str(cfg.get("filing_type", cfg.get("conv_filing_status", "mfj"))).lower() == "mfj"

    ymax = float(p90.max()) if p90.max() > 0 else 1.0
    annotations = []
    if ages[0] <= retire_age <= ages[-1]:
        annotations.append((retire_age, "Retire"))
    if ages[0] <= claim1 <= ages[-1]:
        annotations.append((claim1, "SS Claim" + (" (1)" if has_spouse else "")))
    if has_spouse and ages[0] <= claim2 <= ages[-1] and claim2 != claim1:
        annotations.append((claim2, "SS Claim (2)"))
    if ages[0] <= medicare_age <= ages[-1]:
        annotations.append((medicare_age, "Medicare"))
    if ages[0] <= rmd_age <= ages[-1]:
        annotations.append((rmd_age, "RMDs"))

    used_y_offsets = []
    for evt_age, evt_label in annotations:
        ax.axvline(evt_age, color="#AAAAAA", linestyle="--", linewidth=0.7, alpha=0.7)
        y_pos = ymax * 0.92
        # Stagger labels to avoid overlap
        for uy in used_y_offsets:
            if abs(y_pos - uy) < ymax * 0.08:
                y_pos -= ymax * 0.08
        used_y_offsets.append(y_pos)
        ax.text(evt_age + 0.3, y_pos, evt_label, fontsize=7, color="#666666",
                rotation=0, va="top")

    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    img = _chart_to_image(fig)
    elements.append(Image(img, width=7.0 * inch, height=4.0 * inch))
    elements.append(Spacer(1, 8))

    # ── Key Assumptions table (two-column layout) ──
    elements.append(Paragraph("Key Assumptions", styles["h3"]))

    sp = cfg.get("scenario_params", {})
    eq_return = _fmt_pct(float(sp.get("eq_mu", 0.07)) * 100)
    bond_return = _fmt_pct(float(sp.get("bond_mu", 0.03)) * 100)
    inflation = _fmt_pct(float(cfg.get("infl_mu", 0.025)) * 100)
    spend_real = _fmt_dollars(float(cfg.get("spend_real", 80000)))

    ss1_monthly = float(cfg.get("ss62_1", 0))
    ss1_annual = _fmt_dollars(ss1_monthly * 12) if ss1_monthly > 0 else "N/A"
    ss2_monthly = float(cfg.get("ss62_2", 0))
    ss2_annual = _fmt_dollars(ss2_monthly * 12) if has_spouse and ss2_monthly > 0 else "N/A"

    # Get equity allocation from hold weights
    w_tax = hold.get("w_tax", {})
    w_ret = hold.get("w_ret", {})
    eq_tax = float(w_tax.get("Equities", 0.6)) * 100
    eq_ret = float(w_ret.get("Equities", 0.6)) * 100

    conv_on = bool(cfg.get("conv_on", False))
    conv_type = str(cfg.get("conv_type", "fixed"))
    if conv_on:
        if conv_type == "bracket_fill":
            conv_desc = f"Bracket-fill {_fmt_pct_whole(float(cfg.get('conv_target_bracket', 0.24)) * 100)}"
        else:
            conv_desc = f"Fixed {_fmt_dollars(float(cfg.get('conv_amount', 0)))}/yr"
    else:
        conv_desc = "Off"

    n_sims = int(cfg.get("n_sims", 3000))
    filing_str = "MFJ" if has_spouse else "Single"

    # Two-column key-value table
    left_pairs = [
        ("Equity Return", eq_return),
        ("Bond Return", bond_return),
        ("Inflation", inflation),
        ("Spending (real)", spend_real),
        ("Equity % (Tax.)", _fmt_pct_whole(eq_tax)),
        ("Equity % (Ret.)", _fmt_pct_whole(eq_ret)),
    ]
    right_pairs = [
        ("Filing Status", filing_str),
        ("SS at 62 (Sp. 1)", ss1_annual),
        ("SS at 62 (Sp. 2)", ss2_annual if has_spouse else "N/A"),
        ("Roth Conversions", conv_desc),
        ("Simulations", f"{n_sims:,}"),
        ("Plan Horizon", f"Age {int(cfg['start_age'])}–{int(cfg['end_age'])}"),
    ]

    # Pad to equal length
    max_len = max(len(left_pairs), len(right_pairs))
    while len(left_pairs) < max_len:
        left_pairs.append(("", ""))
    while len(right_pairs) < max_len:
        right_pairs.append(("", ""))

    combined = []
    for (lk, lv), (rk, rv) in zip(left_pairs, right_pairs):
        combined.append([lk, lv, rk, rv])

    col_w = [1.5 * inch, 1.0 * inch, 1.5 * inch, 1.0 * inch]
    kv_tbl = Table(combined, colWidths=col_w, hAlign="LEFT")
    kv_style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_TEXT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, MED_GRAY),
    ]
    for i in range(len(combined)):
        if i % 2 == 0:
            kv_style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    kv_tbl.setStyle(TableStyle(kv_style))
    elements.append(kv_tbl)

    return elements


def _page3_income_strategy(cfg, hold, out, ages, styles) -> list:
    """Page 3: Income Strategy Timeline."""
    elements = []
    elements.append(Paragraph("Income Strategy Timeline", styles["h2"]))

    de = out["decomp"]
    retire_age = int(cfg["retire_age"])
    retire_idx = max(0, int(retire_age - ages[0]))
    ret_ages = ages[retire_idx:]

    if len(ret_ages) < 2:
        elements.append(Paragraph("Not enough post-retirement years to display.", styles["body"]))
        return elements

    # Gather series — use ordered list to control stacking order (bottom to top)
    # Traditional IRA = discretionary withdrawals + RMDs (stored separately in decomp)
    trad_plus_rmd = de["gross_trad_wd"] + de.get("gross_rmd", np.zeros_like(de["gross_trad_wd"]))
    series_defs = [
        ("Social Security",  de["ss_inflow"]),
        ("Taxable WD",       de["gross_tax_wd"]),
        ("Traditional/RMD",  trad_plus_rmd),
        ("Roth WD",          de["gross_roth_wd"]),
        ("Annuity",          de["annuity_income"]),
    ]
    if bool(cfg.get("hsa_on", False)):
        series_defs.append(("HSA (Medical)", de["hsa_used_med"]))

    # Conversions shown as overlay line, not stacked
    conv_raw = np.median(de["conv_gross"], axis=0)[retire_idx:]

    # Build active stacked series — clip negatives and filter zeros
    active_names = []
    active_data = []
    for name, raw_arr in series_defs:
        med = np.clip(np.median(raw_arr, axis=0)[retire_idx:], 0, None)
        if np.any(med > 0):
            active_names.append(name)
            active_data.append(med)

    # Distinct colors — maximally separated hues
    color_map = {
        "Social Security":  "#1565C0",  # blue
        "Taxable WD":       "#F9A825",  # amber
        "Traditional/RMD":  "#E65100",  # deep orange
        "Roth WD":          "#6A1B9A",  # purple
        "Annuity":          "#2E7D32",  # green
        "HSA (Medical)":    "#00838F",  # teal
        "Roth Conversions": "#AD1457",  # pink
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    if active_data:
        stack_arr = np.array(active_data)
        ax.stackplot(ret_ages, stack_arr,
                     labels=active_names,
                     colors=[color_map.get(n, "#888888") for n in active_names],
                     alpha=0.85)

    # Overlay conversions as a separate line
    if np.any(conv_raw > 0):
        ax.plot(ret_ages, conv_raw, color=color_map["Roth Conversions"],
                linewidth=1.5, linestyle="--", label="Roth Conversions")

    ax.set_xlabel("Age", fontsize=9, color="#555555")
    ax.set_ylabel("Annual Income (Nominal $)", fontsize=9, color="#555555")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _style_ax(ax)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=2)
    fig.tight_layout()
    img = _chart_to_image(fig)
    elements.append(Image(img, width=7.0 * inch, height=4.0 * inch))
    elements.append(Spacer(1, 8))

    # ── Account Depletion Timeline ──
    elements.append(Paragraph("Account Depletion Timeline", styles["h3"]))

    depletion_items = []
    for label, arr_key in [("Taxable", "taxable"), ("Traditional", "trad"),
                            ("Roth", "roth"), ("HSA", "hsa")]:
        if arr_key == "hsa" and not bool(cfg.get("hsa_on", False)):
            continue
        arr = out.get(arr_key)
        if arr is None:
            continue
        med_bal = np.median(arr, axis=0)
        depl_age = None
        for i, age in enumerate(ages):
            if age >= retire_age and med_bal[i] <= 0:
                depl_age = int(age)
                break
        if depl_age is not None:
            depletion_items.append(f"{label}: depleted at age {depl_age}")
        else:
            depletion_items.append(f"{label}: not depleted")

    depl_text = "  |  ".join(depletion_items)
    elements.append(Paragraph(depl_text, styles["body"]))

    return elements


def _page4_tax_strategy(cfg, hold, out, ages, styles) -> list:
    """Page 4: Tax Strategy."""
    elements = []
    elements.append(Paragraph("Tax Strategy", styles["h2"]))

    de = out["decomp"]
    retire_age = int(cfg["retire_age"])
    retire_idx = max(0, int(retire_age - ages[0]))
    ret_ages = ages[retire_idx:]

    if len(ret_ages) < 2:
        elements.append(Paragraph("Not enough post-retirement years to display.", styles["body"]))
        return elements

    # ── Dual-axis chart: MAGI + Effective Rate ──
    med_magi = np.median(out["magi"], axis=0)[retire_idx:]
    med_eff_rate = np.median(de["effective_rate"], axis=0)[retire_idx:]
    irmaa_on = bool(cfg.get("irmaa_on", False))
    is_mfj = str(cfg.get("conv_filing_status", "mfj")).lower() == "mfj"

    fig, ax1 = plt.subplots(figsize=(7.0, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(ret_ages, med_magi, color=NAVY_HEX, linewidth=1.8, label="MAGI (median)")
    ax1.set_xlabel("Age", fontsize=9, color="#555555")
    ax1.set_ylabel("MAGI ($)", fontsize=9, color=NAVY_HEX)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))

    ax2.plot(ret_ages, med_eff_rate * 100, color=TEAL_HEX, linewidth=1.8,
             linestyle="-.", label="Effective Tax Rate")
    ax2.set_ylabel("Effective Tax Rate (%)", fontsize=9, color=TEAL_HEX)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # IRMAA tier thresholds
    if irmaa_on:
        tiers = IRMAA_MAGI_TIERS_MFJ if is_mfj else IRMAA_MAGI_TIERS_SINGLE
        med_infl = np.median(out["infl_index"], axis=0)[retire_idx:]
        for i, threshold in enumerate(tiers[:4]):  # Skip the top tier for readability
            adj_threshold = threshold * med_infl
            ax1.plot(ret_ages, adj_threshold, color="#AAAAAA", linestyle=":",
                     linewidth=0.7, alpha=0.7)
            # Label at the start of the line
            if len(ret_ages) > 0:
                ax1.text(ret_ages[0] - 0.5, float(adj_threshold[0]),
                         f"Tier {i + 1}", fontsize=6, color="#999999", va="bottom")

    # Roth conversion shaded region
    conv_on = bool(cfg.get("conv_on", False))
    if conv_on:
        conv_start = int(cfg.get("conv_start", retire_age))
        conv_end = int(cfg.get("conv_end", retire_age + 10))
        if conv_start <= ret_ages[-1] and conv_end >= ret_ages[0]:
            ax1.axvspan(max(conv_start, ret_ages[0]), min(conv_end, ret_ages[-1]),
                        alpha=0.08, color=TEAL_HEX, label="Roth Conv. Period")

    # Styling
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelsize=8)
    ax1.spines["left"].set_color(NAVY_HEX)
    ax1.spines["bottom"].set_color("#CCCCCC")
    ax2.spines["right"].set_color(TEAL_HEX)

    # Merged legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
               loc="upper right", framealpha=0.8)

    fig.tight_layout()
    img = _chart_to_image(fig)
    elements.append(Image(img, width=7.0 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 8))

    # ── Tax Summary Table ──
    elements.append(Paragraph("Tax Summary", styles["h3"]))

    fed_tax = float(np.median(de["fed_tax"].sum(axis=1)))
    state_tax = float(np.median(de["state_tax"].sum(axis=1)))
    irmaa_total = float(np.median(de["irmaa"].sum(axis=1))) if irmaa_on else 0
    niit_total = float(np.median(de["niit"].sum(axis=1)))
    conv_total = float(np.median(de["conv_gross"].sum(axis=1)))

    # Marginal bracket range over retirement
    marginal = np.median(de["marginal_bracket"], axis=0)[retire_idx:]
    if len(marginal) > 0:
        bracket_min = float(marginal.min()) * 100
        bracket_max = float(marginal.max()) * 100
        bracket_range = f"{bracket_min:.0f}%–{bracket_max:.0f}%"
    else:
        bracket_range = "N/A"

    tax_rows = [["Metric", "Median Value (Lifetime)"]]
    tax_rows.append(["Federal Tax", _fmt_dollars(fed_tax)])
    tax_rows.append(["State Tax", _fmt_dollars(state_tax)])
    if irmaa_on:
        tax_rows.append(["IRMAA Surcharges", _fmt_dollars(irmaa_total)])
    if niit_total > 100:
        tax_rows.append(["NIIT", _fmt_dollars(niit_total)])
    tax_rows.append(["Total Roth Conversions", _fmt_dollars(conv_total)])
    tax_rows.append(["Marginal Bracket Range", bracket_range])

    tax_tbl = _build_table(tax_rows, col_widths=[2.8 * inch, 2.0 * inch])
    elements.append(tax_tbl)
    elements.append(Spacer(1, 8))

    # ── IRMAA Cliff Warnings ──
    if irmaa_on:
        medicare_age = int(cfg.get("medicare_age", 65))
        tiers = IRMAA_MAGI_TIERS_MFJ if is_mfj else IRMAA_MAGI_TIERS_SINGLE
        part_b = IRMAA_PART_B_MFJ if is_mfj else IRMAA_PART_B_SINGLE
        part_d = IRMAA_PART_D_MFJ if is_mfj else IRMAA_PART_D_SINGLE
        med_infl = np.median(out["infl_index"], axis=0)

        warnings_found = False
        for idx, age in enumerate(ages):
            if age < medicare_age:
                continue
            lb_idx = max(0, idx - 2)
            lookback_magi = float(np.median(out["magi"][:, lb_idx]))
            for tier_i, threshold in enumerate(tiers):
                adj_thresh = threshold * float(med_infl[idx])
                gap = adj_thresh - lookback_magi
                if 0 < gap <= 10000:
                    # Compute annual surcharge for crossing this tier
                    b_surch = float(part_b[tier_i][1]) * 12
                    d_surch = float(part_d[tier_i][1]) * 12
                    total_surch = b_surch + d_surch
                    if is_mfj:
                        total_surch *= 2
                    if not warnings_found:
                        elements.append(Paragraph("IRMAA Cliff Warnings", styles["h3"]))
                        warnings_found = True
                    warn = (f"Age {int(age)}: MAGI ~{_fmt_dollars(gap)} below "
                            f"IRMAA Tier {tier_i + 1} ({_fmt_dollars(adj_thresh)}). "
                            f"Crossing adds {_fmt_dollars(total_surch)}/yr in Medicare surcharges.")
                    elements.append(Paragraph(warn, styles["warning"]))
                    break  # One warning per age

    return elements


def _page5_risk(cfg, hold, out, ages, styles, tornado_df) -> list:
    """Page 5: Risk Analysis."""
    elements = []
    elements.append(Paragraph("Risk Analysis", styles["h2"]))

    # ── Tornado chart ──
    elements.append(Paragraph("Sensitivity Analysis", styles["h3"]))

    if tornado_df is not None and len(tornado_df) > 0:
        df = tornado_df.copy()
        # Sort by max absolute impact
        df["_max_abs"] = df[["Up ($M)", "Down ($M)"]].abs().max(axis=1)
        df = df.sort_values("_max_abs", ascending=True).drop(columns=["_max_abs"])

        variables = df["Variable"].tolist()
        up_vals = df["Up ($M)"].tolist()
        down_vals = df["Down ($M)"].tolist()

        fig, ax = plt.subplots(figsize=(7.0, max(3.0, len(variables) * 0.35)))
        y_pos = range(len(variables))

        # Color by outcome (helps vs hurts)
        up_colors = [GREEN_HEX if v >= 0 else RED_HEX for v in up_vals]
        down_colors = [GREEN_HEX if v >= 0 else RED_HEX for v in down_vals]

        ax.barh(y_pos, up_vals, height=0.35, align="edge",
                color=up_colors, alpha=0.85, label="Increase scenario")
        ax.barh([y - 0.35 for y in y_pos], down_vals, height=0.35, align="edge",
                color=down_colors, alpha=0.60, label="Decrease scenario")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables, fontsize=7)
        ax.set_xlabel("Change in Median Ending Net Worth ($M)", fontsize=8, color="#555555")
        ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}"))
        _style_ax(ax, show_grid=True)

        # Custom legend for helps/hurts
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=GREEN_HEX, alpha=0.85, label="Helps"),
            Patch(facecolor=RED_HEX, alpha=0.85, label="Hurts"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right", framealpha=0.8)

        fig.tight_layout()
        img = _chart_to_image(fig)
        chart_h = max(3.0, len(variables) * 0.35)
        elements.append(Image(img, width=7.0 * inch, height=chart_h * inch))
    else:
        elements.append(Paragraph(
            "<i>Run the Sensitivity Analysis in the app to include it in the report.</i>",
            styles["body"]))

    elements.append(Spacer(1, 10))

    # ── Spending floor analysis ──
    elements.append(Paragraph("Spending Floor Analysis", styles["h3"]))
    de = out["decomp"]
    floor_val = float(cfg.get("spending_floor", 80000))
    real_spend = de["core_adjusted"] / np.maximum(out["infl_index"], 1e-9)
    retire_age = int(cfg["retire_age"])
    retire_idx = max(0, int(retire_age - ages[0]))

    if retire_idx < real_spend.shape[1]:
        breached = (real_spend[:, retire_idx:] < floor_val).any(axis=1)
        pct_breached = float(breached.sum()) / len(breached) * 100.0
    else:
        pct_breached = 0.0

    floor_text = (f"There is a <b>{_fmt_pct(pct_breached)}</b> probability that real annual "
                  f"spending drops below <b>{_fmt_dollars(floor_val)}/yr</b> at some point "
                  f"during retirement.")
    elements.append(Paragraph(floor_text, styles["body"]))

    return elements


def _page6_scenarios(cfg, hold, out, ages, styles, scenarios) -> list:
    """Page 6: Scenario Comparison (optional)."""
    if not scenarios or len(scenarios) < 2:
        return []

    elements = []
    elements.append(Paragraph("Scenario Comparison", styles["h2"]))

    header = ["Metric"] + [s.get("name", f"Scenario {i+1}") for i, s in enumerate(scenarios)]
    rows = [header]

    for s in scenarios:
        s_cfg = s.get("cfg", cfg)
        s_out = s.get("out")
        if s_out is None:
            continue

    # Build comparison rows
    metric_defs = [
        ("Retirement Age", lambda c, o: str(int(c.get("retire_age", 65)))),
        ("Annual Spending", lambda c, o: _fmt_dollars(float(c.get("spend_real", 80000)))),
        ("Success Rate", lambda c, o: _fmt_pct(
            100.0 - float((o["ruin_age"] <= int(c["end_age"])).sum()) / len(o["ruin_age"]) * 100.0
        )),
        ("Median Legacy", lambda c, o: _fmt_dollars(float(np.median(o["legacy"])))),
        ("Lifetime Taxes", lambda c, o: _fmt_dollars(float(np.median(o["decomp"]["taxes_paid"].sum(axis=1))))),
    ]

    for metric_name, metric_fn in metric_defs:
        row = [metric_name]
        for s in scenarios:
            s_cfg = s.get("cfg", cfg)
            s_out = s.get("out")
            if s_out is not None:
                try:
                    row.append(metric_fn(s_cfg, s_out))
                except Exception:
                    row.append("N/A")
            else:
                row.append("N/A")
        rows.append(row)

    n_cols = len(header)
    col_w = 7.0 * inch / n_cols
    tbl = _build_table(rows, col_widths=[col_w] * n_cols)
    elements.append(tbl)

    return elements


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def generate_report_pdf(
    cfg: dict,
    hold: dict,
    out: dict,
    ages: np.ndarray,
    tornado_df: Optional[pd.DataFrame] = None,
    scenarios: Optional[list[dict]] = None,
) -> bytes:
    """Generate a PDF retirement plan report. Returns raw PDF bytes."""
    buf = io.BytesIO()
    sty = _styles()

    # Track pages for numbering
    page_count = [0]  # mutable for closure
    total_pages = [0]

    def _footer(canvas, doc):
        """Draw footer on every page."""
        canvas.saveState()
        page_count[0] += 1

        # Footer left
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#999999"))
        footer_text = (f"Generated by RetireLab  |  {date.today().strftime('%B %d, %Y')}  |  "
                       "This is a planning estimate, not financial advice.")
        canvas.drawString(0.75 * inch, 0.45 * inch, footer_text)

        # Page number right
        canvas.drawRightString(7.75 * inch, 0.45 * inch,
                               f"Page {page_count[0]}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    # Build all elements
    all_elements = []

    # Page 1: Executive Summary
    all_elements.extend(_page1_executive_summary(cfg, hold, out, ages, sty))
    all_elements.append(PageBreak())

    # Page 2: Portfolio Trajectory
    all_elements.extend(_page2_trajectory(cfg, hold, out, ages, sty))
    all_elements.append(PageBreak())

    # Page 3: Income Strategy Timeline
    all_elements.extend(_page3_income_strategy(cfg, hold, out, ages, sty))
    all_elements.append(PageBreak())

    # Page 4: Tax Strategy
    all_elements.extend(_page4_tax_strategy(cfg, hold, out, ages, sty))
    all_elements.append(PageBreak())

    # Page 5: Risk Analysis
    all_elements.extend(_page5_risk(cfg, hold, out, ages, sty, tornado_df))

    # Page 6: Scenario Comparison (optional)
    scenario_elements = _page6_scenarios(cfg, hold, out, ages, sty, scenarios)
    if scenario_elements:
        all_elements.append(PageBreak())
        all_elements.extend(scenario_elements)

    doc.build(all_elements, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()

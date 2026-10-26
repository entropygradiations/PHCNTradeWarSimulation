"""
Philippines-China Trade War Model
Educational model - not a forecast.
"""

import sys
import io

# --- Fix console encoding on Windows/GBK before anything else ---
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass  # non-standard stream; continue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


# ======================================================================
# 0. PARAMETERS
# ======================================================================
@dataclass
class Params:
    # Baseline trade & GDP (USD billions)
    PH_exports_to_CN: float = 10.0
    PH_imports_from_CN: float = 30.0
    PH_GDP: float = 440.0
    CN_GDP: float = 17700.0

    # Demand elasticities (short-run, central)
    el_PH_demand_for_CN: float = 1.2
    el_CN_demand_for_PH: float = 1.5

    # Elasticity ramp (short-run -> long-run)
    el_ramp_start: float = 1.0
    el_ramp_end: float = 2.2
    el_ramp_years: int = 4

    # Tariff phase-in schedule (5 years)
    phase_in: Tuple[float, ...] = (0.50, 0.80, 0.95, 1.00, 1.00)

    # Fiscal block
    MPC: float = 0.65
    tax_rate: float = 0.20
    MPM: float = 0.35
    revenue_recycling_efficiency: float = 0.80
    revenue_offset_share: float = 0.30  # cap on how much revenue offsets DWL

    # Social discounting
    social_discount_rate: float = 0.05

    # Diversion block
    diversion_base: float = 0.08
    diversion_tariff_slope: float = 0.30
    diversion_time_ramp: float = 0.08
    diversion_cap: float = 0.40
    absorption_capacity: float = 0.80
    absorption_elasticity: float = 0.30

    # FX block (structured)
    fx_base_depreciation_per_1pct_netloss: float = 1.20
    fx_interest_parity_weight: float = 0.30
    fx_cb_intervention_strength: float = 0.25
    fx_capital_flight_sensitivity: float = 0.50
    fx_max_depreciation: float = 25.0

    # Pass-through
    export_pass_through_of_depreciation: float = 0.50
    import_price_pass_through: float = 0.30

    # Firm heterogeneity (extensive margin)
    extensive_margin_share: float = 0.25

    # GVC effective-rate-of-protection multiplier
    gvc_erp_multiplier: float = 1.15


# ======================================================================
# 1. SCENARIOS
# ======================================================================
SCENARIOS = {
    "Mild Dispute":    {"ph_tariff_on_cn": 0.10, "cn_tariff_on_ph": 0.10},
    "Moderate War":    {"ph_tariff_on_cn": 0.25, "cn_tariff_on_ph": 0.25},
    "Severe War":      {"ph_tariff_on_cn": 0.50, "cn_tariff_on_ph": 0.40},
    "Full Decoupling": {"ph_tariff_on_cn": 0.80, "cn_tariff_on_ph": 0.80},
}
ELASTICITY_BANDS = {"low": 0.7, "mid": 1.0, "high": 1.4}
DIVERSION_BANDS  = {"low": 0.5, "mid": 1.0, "high": 1.5}
BAND_ORDER = ["low", "mid", "high"]


# ======================================================================
# 2. CORE ENGINE — ONE YEAR
# ======================================================================
def annual_step(t_ph: float, t_cn: float, year: int,
                p: Params,
                el_mult: float = 1.0,
                div_mult: float = 1.0,
                fx_mult: float = 1.0) -> Dict:
    idx = year - 1
    phase = p.phase_in[idx]
    t_ph_eff = t_ph * phase
    t_cn_eff = t_cn * phase

    # Elasticity ramp
    ramp_frac = min(idx / max(p.el_ramp_years - 1, 1), 1.0)
    el_ramp = p.el_ramp_start + (p.el_ramp_end - p.el_ramp_start) * ramp_frac
    el_demand_CN = p.el_PH_demand_for_CN * el_ramp * el_mult
    el_demand_PH = p.el_CN_demand_for_PH * el_ramp * el_mult

    # Pre-FX trade volumes (tariff effect)
    ph_exp_pre = max(p.PH_exports_to_CN * (1 - el_demand_PH * t_cn_eff), 0.0)
    ph_imp_pre = max(p.PH_imports_from_CN * (1 - el_demand_CN * t_ph_eff), 0.0)

    # Structured FX loop
    peso_dep = 0.0
    for _ in range(10):
        fx_exp_boost = p.export_pass_through_of_depreciation * (peso_dep / 100.0)
        fx_imp_drag  = p.import_price_pass_through * (peso_dep / 100.0)

        ph_exp_fx = min(ph_exp_pre * (1 + fx_exp_boost), p.PH_exports_to_CN)
        ph_imp_fx = max(ph_imp_pre * (1 - fx_imp_drag), 0.0)

        net_loss_usd = ((p.PH_exports_to_CN - ph_exp_fx)
                        - (p.PH_imports_from_CN - ph_imp_fx))
        net_loss_pct = 100.0 * net_loss_usd / p.PH_GDP

        trade_pressure = p.fx_base_depreciation_per_1pct_netloss * abs(net_loss_pct)
        cap_flight = p.fx_capital_flight_sensitivity * abs(net_loss_pct)
        cb_drag = p.fx_cb_intervention_strength * trade_pressure
        ip_offset = p.fx_interest_parity_weight * trade_pressure

        peso_new = max((trade_pressure + cap_flight - cb_drag - ip_offset) * fx_mult, 0.0)
        peso_new = min(peso_new, p.fx_max_depreciation)

        if abs(peso_new - peso_dep) < 1e-5:
            peso_dep = peso_new
            break
        peso_dep = peso_new

    # Diversion with absorption capacity
    avg_tariff = 0.5 * (t_ph_eff + t_cn_eff)
    div = (p.diversion_base
           + p.diversion_tariff_slope * avg_tariff
           + p.diversion_time_ramp * idx)
    capacity = min(p.absorption_capacity
                   + p.absorption_elasticity * avg_tariff, 1.0)
    div = min(div * div_mult, p.diversion_cap * capacity)

    ph_exp_eff = ph_exp_fx + (p.PH_exports_to_CN - ph_exp_fx) * div
    ph_imp_eff = ph_imp_fx + (p.PH_imports_from_CN - ph_imp_fx) * div

    # Asymmetric DWL
    ph_dwl_import_side = 0.5 * t_ph_eff * (p.PH_imports_from_CN - ph_imp_eff)
    cn_dwl_export_side = 0.5 * t_cn_eff * (p.PH_exports_to_CN - ph_exp_eff)

    ph_dwl_import_side *= p.gvc_erp_multiplier
    cn_dwl_export_side *= p.gvc_erp_multiplier

    ph_dwl_extensive = p.extensive_margin_share * ph_dwl_import_side
    cn_dwl_extensive = p.extensive_margin_share * cn_dwl_export_side

    ph_dwl = ph_dwl_import_side + ph_dwl_extensive
    cn_dwl = cn_dwl_export_side + cn_dwl_extensive

    # Tariff revenue (informational)
    ph_revenue = t_ph_eff * ph_imp_eff
    cn_revenue = t_cn_eff * ph_exp_eff

    # Revenue can only PARTIALLY offset DWL (it's a transfer, not a gain).
    ph_revenue_offset = ph_revenue * p.revenue_recycling_efficiency * p.revenue_offset_share
    cn_revenue_offset = cn_revenue * p.revenue_recycling_efficiency * p.revenue_offset_share

    ph_net_welfare = max(ph_dwl - ph_revenue_offset, 0.0)
    cn_net_welfare = max(cn_dwl - cn_revenue_offset, 0.0)

    # Welfare decomposition (PH)
    ph_consumer_loss = t_ph_eff * ph_imp_eff
    ph_producer_loss = 0.5 * t_cn_eff * (p.PH_exports_to_CN - ph_exp_eff)
    ph_tot = (p.export_pass_through_of_depreciation * (peso_dep / 100.0) * p.PH_exports_to_CN
              - p.import_price_pass_through * (peso_dep / 100.0) * p.PH_imports_from_CN)

    # Open-economy multiplier
    multiplier = 1.0 / (1.0 - p.MPC * (1 - p.tax_rate) + p.MPM)
    # GDP effect from DWL, always positive (a loss magnitude)
    ph_gdp_effect = ph_dwl * multiplier

    return {
        "Year": year,
        "phase_in": round(phase, 2),
        "elast_mult": round(el_ramp * el_mult, 2),
        "diversion": round(div, 3),
        "peso_dep_pct": round(peso_dep, 2),
        "PH_exports_$B": round(ph_exp_eff, 3),
        "PH_imports_$B": round(ph_imp_eff, 3),
        "PH_DWL_$B": round(ph_dwl, 4),
        "CN_DWL_$B": round(cn_dwl, 4),
        "PH_rev_$B": round(ph_revenue, 4),
        "CN_rev_$B": round(cn_revenue, 4),
        "PH_net_welfare_$B": round(ph_net_welfare, 4),
        "CN_net_welfare_$B": round(cn_net_welfare, 4),
        "PH_GDP_effect_$B": round(ph_gdp_effect, 4),
        "PH_DWL_pctGDP": round(100 * ph_dwl / p.PH_GDP, 4),
        "CN_DWL_pctGDP": round(100 * cn_dwl / p.CN_GDP, 4),
        "PH_net_welfare_pctGDP": round(100 * ph_net_welfare / p.PH_GDP, 4),
        "CN_net_welfare_pctGDP": round(100 * cn_net_welfare / p.CN_GDP, 4),
        "PH_consumer_loss_$B": round(ph_consumer_loss, 4),
        "PH_producer_loss_$B": round(ph_producer_loss, 4),
        "PH_ToT_$B": round(ph_tot, 4),
    }


# ======================================================================
# 3. RUN / DISCOUNT / SENSITIVITY
# ======================================================================
def run_scenario(name: str, tariffs: Dict[str, float], p: Params,
                 el_mult: float = 1.0, div_mult: float = 1.0,
                 fx_mult: float = 1.0) -> pd.DataFrame:
    rows = [annual_step(tariffs["ph_tariff_on_cn"], tariffs["cn_tariff_on_ph"],
                        yr, p, el_mult, div_mult, fx_mult)
            for yr in range(1, 6)]
    df = pd.DataFrame(rows)
    df.insert(0, "Scenario", name)
    return df


def add_pv(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    df = df.copy()
    df["discount_factor"] = 1.0 / (1.0 + rate) ** (df["Year"] - 1)
    for col in ["PH_DWL_$B", "CN_DWL_$B",
                "PH_net_welfare_$B", "CN_net_welfare_$B"]:
        new = col.replace("$B", "PV_$B")
        df[new] = df[col] * df["discount_factor"]
    return df


def run_sensitivity(scenario_name: str, tariffs: Dict[str, float],
                    p: Params) -> pd.DataFrame:
    rows = []
    for el_name, el_mult in ELASTICITY_BANDS.items():
        for div_name, div_mult in DIVERSION_BANDS.items():
            df = run_scenario(scenario_name, tariffs, p, el_mult, div_mult)
            pv = (df["PH_DWL_$B"] /
                  (1 + p.social_discount_rate) ** (df["Year"] - 1)).sum()
            rows.append({
                "Scenario": scenario_name,
                "elasticity": el_name,
                "diversion": div_name,
                "PH_DWL_$B": round(df["PH_DWL_$B"].sum(), 2),
                "PH_DWL_PV_$B": round(pv, 2),
            })
    return pd.DataFrame(rows)


# ======================================================================
# 4. SUMMARY / DECOMPOSITION / MEMO
# ======================================================================
def build_summary(full: pd.DataFrame, p: Params) -> pd.DataFrame:
    rows = []
    for scen in SCENARIOS:
        d = full[full["Scenario"] == scen]
        ph_dwl_cum = d["PH_DWL_$B"].sum()
        cn_dwl_cum = d["CN_DWL_$B"].sum()
        ph_net_cum = d["PH_net_welfare_$B"].sum()
        cn_net_cum = d["CN_net_welfare_$B"].sum()
        rows.append({
            "Scenario": scen,
            "PH_DWL_$B": round(ph_dwl_cum, 2),
            "CN_DWL_$B": round(cn_dwl_cum, 2),
            "PH_net_welfare_$B": round(ph_net_cum, 2),
            "CN_net_welfare_$B": round(cn_net_cum, 2),
            "PH_DWL_PV_$B": round(d["PH_DWL_PV_$B"].sum(), 2),
            "CN_DWL_PV_$B": round(d["CN_DWL_PV_$B"].sum(), 2),
            "PH_GDP_effect_$B": round(d["PH_GDP_effect_$B"].sum(), 2),
        })
    s = pd.DataFrame(rows)
    s["PH_pctGDP"] = 100 * s["PH_DWL_$B"] / p.PH_GDP
    s["CN_pctGDP"] = 100 * s["CN_DWL_$B"] / p.CN_GDP
    # Decompose asymmetry into flow-size and GDP-size effects
    s["DWL_ratio"] = s["PH_DWL_$B"] / s["CN_DWL_$B"]
    s["GDP_ratio"] = p.CN_GDP / p.PH_GDP
    s["Asymmetry_Ratio"] = (s["DWL_ratio"] * s["GDP_ratio"]).round(1)
    return s


def build_welfare_decomp(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scen in SCENARIOS:
        d = full[full["Scenario"] == scen]
        rows.append({
            "Scenario": scen,
            "PH_consumer_loss": round(d["PH_consumer_loss_$B"].sum(), 2),
            "PH_producer_loss": round(d["PH_producer_loss_$B"].sum(), 2),
            "PH_ToT_effect": round(d["PH_ToT_$B"].sum(), 2),
            "PH_DWL": round(d["PH_DWL_$B"].sum(), 2),
            "PH_tariff_rev": round(d["PH_rev_$B"].sum(), 2),
            "PH_net_welfare": round(d["PH_net_welfare_$B"].sum(), 2),
        })
    return pd.DataFrame(rows)


# --- ASCII-normalize the memo (fixes GBK UnicodeEncodeError) ---
_NORMALIZE = {
    "\u2212": "-",   # unicode minus
    "\u00d7": "x",   # multiplication sign
    "\u2014": "--",  # em dash
    "\u2013": "-",   # en dash
    "\u2019": "'",   # curly apostrophe
    "\u201c": '"',
    "\u201d": '"',
}

def ascii_safe(s: str) -> str:
    for k, v in _NORMALIZE.items():
        s = s.replace(k, v)
    # Fallback: strip any remaining non-ASCII
    return s.encode("ascii", errors="replace").decode("ascii")


def build_memo(summary: pd.DataFrame, sens: pd.DataFrame,
               decomp: pd.DataFrame, mc: pd.DataFrame, p: Params) -> str:
    L = []
    L.append("=" * 100)
    L.append("PHILIPPINES-CHINA TRADE WAR: 5-YEAR COST ESTIMATES (MODEL v2.1)")
    L.append("Asymmetric DWL | revenue-aware | open-economy multiplier | MC")
    L.append("=" * 100)
    L.append("")
    L.append(f"{'Scenario':<18}{'PH DWL':>9}{'CN DWL':>9}{'PH net':>9}"
             f"{'PH PV':>8}{'PH GDP':>9}{'PH %GDP':>9}{'CN %GDP':>9}{'Asym':>8}")
    L.append("-" * 100)
    for _, r in summary.iterrows():
        L.append(f"{r['Scenario']:<18}"
                 f"{r['PH_DWL_$B']:>9.2f}{r['CN_DWL_$B']:>9.2f}"
                 f"{r['PH_net_welfare_$B']:>9.2f}"
                 f"{r['PH_DWL_PV_$B']:>8.2f}{r['PH_GDP_effect_$B']:>9.2f}"
                 f"{r['PH_pctGDP']:>8.3f}%{r['CN_pctGDP']:>8.4f}%"
                 f"{r['Asymmetry_Ratio']:>7.1f}x")
    L.append("-" * 100)
    L.append(f"  GDP: PH=${p.PH_GDP:,.0f}B | CN=${p.CN_GDP:,.0f}B | "
             f"ratio={p.CN_GDP/p.PH_GDP:.1f}x")
    L.append(f"  Open-economy multiplier = "
             f"{1/(1-p.MPC*(1-p.tax_rate)+p.MPM):.2f} "
             f"(MPC={p.MPC}, tax={p.tax_rate}, MPM={p.MPM})")
    L.append("")
    L.append("ASYMMETRY DECOMPOSITION:")
    L.append(f"  {'Scenario':<18}{'PH DWL':>9}{'CN DWL':>9}"
             f"{'DWL ratio':>11}{'GDP ratio':>11}{'Combined':>11}")
    L.append("  " + "-" * 68)
    for _, r in summary.iterrows():
        L.append(f"  {r['Scenario']:<18}{r['PH_DWL_$B']:>9.2f}"
                 f"{r['CN_DWL_$B']:>9.2f}{r['DWL_ratio']:>10.2f}x"
                 f"{r['GDP_ratio']:>10.1f}x{r['Asymmetry_Ratio']:>10.1f}x")
    L.append("  Asymmetry = (PH DWL / CN DWL) x (CN GDP / PH GDP)")
    L.append("           = flow-size effect x GDP-size effect")
    L.append("")
    L.append("WELFARE DECOMPOSITION (PH side, 5-yr cumulative, $B):")
    L.append(f"  {'Scenario':<18}{'Consumer':>10}{'Producer':>10}"
             f"{'ToT':>8}{'DWL':>9}{'TariffRev':>10}{'Net':>9}")
    L.append("  " + "-" * 74)
    for _, r in decomp.iterrows():
        L.append(f"  {r['Scenario']:<18}"
                 f"{r['PH_consumer_loss']:>10.2f}{r['PH_producer_loss']:>10.2f}"
                 f"{r['PH_ToT_effect']:>8.2f}{r['PH_DWL']:>9.2f}"
                 f"{r['PH_tariff_rev']:>10.2f}{r['PH_net_welfare']:>9.2f}")
    L.append("  Net = DWL - (tariff revenue x recycling x offset share)")
    L.append("")
    L.append("SENSITIVITY BAND - Moderate War (PH 5-yr DWL, $B):")
    mod = sens[sens["Scenario"] == "Moderate War"]
    piv = mod.pivot(index="elasticity", columns="diversion",
                    values="PH_DWL_$B").reindex(index=BAND_ORDER, columns=BAND_ORDER)
    L.append(f"  {'':<10}" + "".join(f"{c:>10}" for c in piv.columns))
    for idx, row in piv.iterrows():
        L.append(f"  {idx:<10}" + "".join(f"{v:>10.2f}" for v in row.values))
    L.append(f"  Range: ${piv.values.min():.2f}B - ${piv.values.max():.2f}B | "
             f"Central: ${piv.loc['mid','mid']:.2f}B")
    L.append("")
    L.append("MONTE CARLO (10,000 draws):")
    L.append(f"  {'Scenario':<18}{'Mean':>9}{'Median':>9}{'StdDev':>9}"
             f"{'P05':>9}{'P95':>9}")
    L.append("  " + "-" * 63)
    for _, r in mc.iterrows():
        L.append(f"  {r['Scenario']:<18}{r['mean']:>9.2f}{r['median']:>9.2f}"
                 f"{r['std']:>9.2f}{r['p05']:>9.2f}{r['p95']:>9.2f}")
    L.append("")
    L.append("KEY MODELING NOTES:")
    L.append("  - DWL is ASYMMETRIC: computed per-country from its own tariff & lost volume.")
    L.append(f"  - Tariff revenue recycled at {p.revenue_recycling_efficiency:.0%}; "
             f"offset capped at {p.revenue_offset_share:.0%} of DWL.")
    L.append("  - FX module: trade pressure + capital flight - CB intervention - interest parity.")
    L.append("  - Diversion constrained by third-country absorption capacity.")
    L.append(f"  - GVC effective-rate-of-protection multiplier = {p.gvc_erp_multiplier:.2f}.")
    L.append(f"  - Extensive-margin (firm-exit) share = {p.extensive_margin_share:.0%} of DWL.")
    L.append("  - Consumer/producer/ToT decomposition included.")
    L.append("  - MC draws: elasticity, diversion, FX, MPM, absorption capacity.")
    L.append("  - Educational model - not a forecast. CGE handoff noted in README.")
    L.append("=" * 100)
    return ascii_safe("\n".join(L))


# ======================================================================
# 5. MONTE CARLO
# ======================================================================
def run_monte_carlo(p: Params, n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = {s: [] for s in SCENARIOS}
    for _ in range(n):
        el_mult  = rng.uniform(0.7, 1.4)
        div_mult = rng.uniform(0.5, 1.5)
        fx_mult  = rng.uniform(0.7, 1.3)
        mpm      = rng.uniform(0.25, 0.45)
        abs_cap  = rng.uniform(0.6, 0.95)
        p_draw = Params(**asdict(p))
        p_draw.MPM = mpm
        p_draw.absorption_capacity = abs_cap
        for name, t in SCENARIOS.items():
            df = run_scenario(name, t, p_draw, el_mult, div_mult, fx_mult)
            results[name].append(df["PH_DWL_$B"].sum())
    rows = []
    for name, vals in results.items():
        v = np.array(vals)
        rows.append({
            "Scenario": name,
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
        })
    return pd.DataFrame(rows)


# ======================================================================
# 6. BACKTEST HOOK
# ======================================================================
def backtest_us_china(p: Params) -> Dict:
    us_imports_from_cn = 540.0
    tariff = 0.20
    lost_imports = us_imports_from_cn * 0.25
    dwl_approx = 0.5 * tariff * lost_imports
    return {
        "approx_DWL_$B": round(dwl_approx, 1),
        "literature_range_$B": (10.0, 40.0),
        "plausible": 10.0 <= dwl_approx <= 40.0,
        "note": "Rough plausibility check only.",
    }


# ======================================================================
# 7. CHARTS
# ======================================================================
def plot_charts(full: pd.DataFrame, summary: pd.DataFrame,
                sens: pd.DataFrame, mc: pd.DataFrame, p: Params):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    ax = axes[0, 0]
    d = full[full["Scenario"] == "Moderate War"]
    ax.plot(d["Year"], d["PH_DWL_$B"], "o-", label="Philippines", color="tab:blue")
    ax.plot(d["Year"], d["CN_DWL_$B"], "s--", label="China", color="tab:red")
    ax.set_title("Moderate War: Annual DWL (asymmetric)")
    ax.set_xlabel("Year"); ax.set_ylabel("USD B / year")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for scen in full["Scenario"].unique():
        d = full[full["Scenario"] == scen]
        ax.plot(d["Year"], d["peso_dep_pct"], "o-", label=scen)
    ax.set_title("FX Module: Endogenous Peso Depreciation")
    ax.set_xlabel("Year"); ax.set_ylabel("% depreciation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    x = np.arange(len(summary)); w = 0.35
    ax.bar(x - w/2, summary["PH_DWL_$B"], w, label="PH DWL", color="tab:blue")
    ax.bar(x + w/2, summary["CN_DWL_$B"], w, label="CN DWL", color="tab:red")
    ax.bar(x - w/2, summary["PH_net_welfare_$B"], w,
           label="PH net welfare", color="tab:blue", alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(summary["Scenario"], rotation=20, ha="right")
    ax.set_ylabel("USD B (5-yr cumulative)")
    ax.set_title("DWL vs Net Welfare (asymmetric)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    mod = sens[sens["Scenario"] == "Moderate War"]
    piv = mod.pivot(index="elasticity", columns="diversion",
                    values="PH_DWL_$B").reindex(index=BAND_ORDER, columns=BAND_ORDER)
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"div={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"elast={i}" for i in piv.index])
    ax.set_title("Sensitivity: Moderate War PH DWL ($B)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center")
    plt.colorbar(im, ax=ax, label="USD B")

    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Welfare Decomposition (Moderate War, $B)")
    d = full[full["Scenario"] == "Moderate War"]
    table_data = [
        ["Consumer loss", f"{d['PH_consumer_loss_$B'].sum():.2f}"],
        ["Producer loss", f"{d['PH_producer_loss_$B'].sum():.2f}"],
        ["ToT effect",    f"{d['PH_ToT_$B'].sum():.2f}"],
        ["DWL",           f"{d['PH_DWL_$B'].sum():.2f}"],
        ["Tariff revenue",f"{d['PH_rev_$B'].sum():.2f}"],
        ["Net welfare",   f"{d['PH_net_welfare_$B'].sum():.2f}"],
    ]
    tbl = ax.table(cellText=table_data, colLabels=["Channel", "USD B"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)

    ax = axes[1, 2]
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(2000):
        el_mult = rng.uniform(0.7, 1.4); div_mult = rng.uniform(0.5, 1.5)
        fx_mult = rng.uniform(0.7, 1.3); mpm = rng.uniform(0.25, 0.45)
        abs_cap = rng.uniform(0.6, 0.95)
        p_draw = Params(**asdict(p)); p_draw.MPM = mpm
        p_draw.absorption_capacity = abs_cap
        df = run_scenario("Moderate War", SCENARIOS["Moderate War"],
                          p_draw, el_mult, div_mult, fx_mult)
        draws.append(df["PH_DWL_$B"].sum())
    ax.hist(draws, bins=40, color="tab:purple", alpha=0.75)
    ax.set_title("Monte Carlo: Moderate War PH DWL ($B)")
    ax.set_xlabel("USD B"); ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ph_cn_model_v21_charts.png", dpi=150)
    plt.close(fig)
    print("Saved: ph_cn_model_v21_charts.png")


# ======================================================================
# 8. MAIN
# ======================================================================
def main():
    p = Params()

    frames = []
    for name, t in SCENARIOS.items():
        df = run_scenario(name, t, p)
        df = add_pv(df, p.social_discount_rate)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full[["Scenario"] + [c for c in full.columns if c != "Scenario"]]

    summary = build_summary(full, p)
    decomp = build_welfare_decomp(full)
    sens = pd.concat([run_sensitivity(n, t, p) for n, t in SCENARIOS.items()],
                     ignore_index=True)
    mc = run_monte_carlo(p, n=10_000)

    bt = backtest_us_china(p)
    print("\nBACKTEST (US-China 2018-19 plausibility):", bt)

    print("\n" + "=" * 100)
    print("MODEL v2.1 - MODERATE WAR, YEAR BY YEAR")
    print("=" * 100)
    mod = full[full["Scenario"] == "Moderate War"]
    print(mod[["Year", "phase_in", "elast_mult", "diversion", "peso_dep_pct",
               "PH_DWL_$B", "CN_DWL_$B", "PH_net_welfare_$B",
               "PH_GDP_effect_$B", "PH_DWL_pctGDP", "CN_DWL_pctGDP"]].to_string(index=False))

    memo = build_memo(summary, sens, decomp, mc, p)
    print("\n" + memo)
    with open("ph_cn_memo_v21.txt", "w", encoding="utf-8") as f:
        f.write(memo)

    full.to_csv("ph_cn_v21_annual.csv", index=False, encoding="utf-8")
    summary.to_csv("ph_cn_v21_summary.csv", index=False, encoding="utf-8")
    sens.to_csv("ph_cn_v21_sensitivity.csv", index=False, encoding="utf-8")
    decomp.to_csv("ph_cn_v21_welfare_decomp.csv", index=False, encoding="utf-8")
    mc.to_csv("ph_cn_v21_monte_carlo.csv", index=False, encoding="utf-8")
    print("\nSaved: ph_cn_memo_v21.txt + 5 CSVs + charts.")

    plot_charts(full, summary, sens, mc, p)


if __name__ == "__main__":
    main()

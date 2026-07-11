"""
One-way ANOVA for NEE by heatwave method.

This file is intentionally written as a simple top-level analysis script so it
can be run line by line in Spyder.

It runs one-way ANOVAs for:
- NEE_during_avg
- NEE_norm_during_avg
- NEE_during_avg - NEE_before_avg_10
- NEE_norm_during_avg - NEE_norm_before_avg_10
"""

from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from auxiliary import (
    loadAMF,
    clean_flux_by_qc_and_years,
    filter_complete_heatwaves,
    normalize_nee_by_site,
    calc_flux_multi_lag,
)


# Locate the repo root in a way that works in Spyder.
if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parent
else:
    CWD = Path.cwd().resolve()
    if CWD.name == "HeatwaveFramework":
        REPO_ROOT = CWD
    elif (CWD / "HeatwaveFramework").exists():
        REPO_ROOT = CWD / "HeatwaveFramework"
    elif CWD.parent.name == "HeatwaveFramework":
        REPO_ROOT = CWD.parent
    else:
        raise RuntimeError(
            "Could not locate HeatwaveFramework. Set the working directory to the "
            "HeatwaveFramework folder in Spyder."
        )


# File paths
DATA_PATH = REPO_ROOT / "data" / "heatwaves" / "flux_heatwaves_df.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "heatwaves"


# Load the event-level table
df = pd.read_csv(DATA_PATH)


# Calculate before-vs-during differences
df["NEE_diff_during_minus_before_10"] = df["NEE_during_avg"] - df["NEE_before_avg_10"]
df["NEE_norm_diff_during_minus_before_10"] = (
    df["NEE_norm_during_avg"] - df["NEE_norm_before_avg_10"]
)


# Positive response = increase in NEE
df["positive_NEE_response"] = (
    df["NEE_diff_during_minus_before_10"] > 0
)

# chi-square across all heatwave categories
# Contingency table
contingency = pd.crosstab(
    df["top_heatwave"],
    df["positive_NEE_response"]
)

print(contingency)

chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square = {chi2:.3f}")
print(f"Degrees of freedom = {dof}")
print(f"P-value = {p:.5f}")

# chi-square between one dimensional and multi dimensional
single = ["Day", "Night", "Overall"]
multi = ["Day-intensified",
         "Night-intensified",
         "Day-Night Spike",
         "Triad"]

df["heat_stress_complexity"] = np.where(
    df["top_heatwave"].isin(single),
    "Single",
    "Multiple"
)

contingency2 = pd.crosstab(
    df["heat_stress_complexity"],
    df["positive_NEE_response"]
)

print(contingency2)

chi2, p, dof, expected = chi2_contingency(contingency2)

print(f"Chi-square = {chi2:.3f}")
print(f"Degrees of freedom = {dof}")
print(f"P-value = {p:.5f}")

# printing the proportions
proportions = (
    contingency
    .div(contingency.sum(axis=1), axis=0)
)

print(proportions)

# doing this for post-event proportions as well
###############################################################################
# Change this if your heatwave category column has a different name
method_col = "top_heatwave"

# also doing this without day-night spike
df_no_spike = df[df[method_col] != "Day-Night Spike"].copy()

single_methods = ["Day", "Night", "Overall"]
multi_methods = ["Day-intensified", "Night-intensified", "Day-Night Spike", "Triad"]

# Average NEE columns for each period
periods = {
    "during": "NEE_during_avg",
    "after_5": "NEE_after_avg_5",
    "after_10": "NEE_after_avg_10",
    "after_15": "NEE_after_avg_15",
    "after_20": "NEE_after_avg_20",
    "after_25": "NEE_after_avg_25",
    "after_30": "NEE_after_avg_30",
}

baseline_col = "NEE_before_avg_10"

# Classify single vs multidimensional heat stress
df["heat_stress_complexity"] = np.where(
    df[method_col].isin(single_methods),
    "Single",
    np.where(df[method_col].isin(multi_methods), "Multiple", None)
)

results = []

for period_name, nee_col in periods.items():
    temp = df[[method_col, "heat_stress_complexity", baseline_col, nee_col]].dropna().copy()

    # Difference from 10-day pre-heatwave baseline
    temp["NEE_diff"] = temp[nee_col] - temp[baseline_col]

    # Positive = NEE increased relative to baseline
    temp["NEE_response"] = np.where(
        temp["NEE_diff"] > 0, "Positive",
        np.where(temp["NEE_diff"] < 0, "Negative", None)
    )

    temp = temp.dropna(subset=["NEE_response", "heat_stress_complexity"])

    contingency = pd.crosstab(
        temp["heat_stress_complexity"],
        temp["NEE_response"]
    )

    # Make sure both response columns exist
    contingency = contingency.reindex(
        index=["Single", "Multiple"],
        columns=["Negative", "Positive"],
        fill_value=0
    )

    chi2, p, dof, expected = chi2_contingency(contingency)

    proportions = contingency.div(contingency.sum(axis=1), axis=0)

    results.append({
        "period": period_name,
        "nee_col": nee_col,
        "n_single": contingency.loc["Single"].sum(),
        "n_multi": contingency.loc["Multiple"].sum(),
        "single_positive_prop": proportions.loc["Single", "Positive"],
        "multi_positive_prop": proportions.loc["Multiple", "Positive"],
        "difference_multi_minus_single": (
            proportions.loc["Multiple", "Positive"] -
            proportions.loc["Single", "Positive"]
        ),
        "chi2": chi2,
        "dof": dof,
        "p_value": p
    })

    print("\n" + "="*60)
    print(f"Period: {period_name}")
    print("\nCounts:")
    print(contingency)
    print("\nProportions:")
    print(proportions)
    print(f"\nChi-square = {chi2:.3f}, df = {dof}, p = {p:.5f}")

results_df = pd.DataFrame(results)

print("\nSummary results:")
print(results_df)

# chi-square of +/- NEE responses across event legacies between heatwave types
##############################################################################

method_col = "top_heatwave"  # change if needed
baseline_col = "NEE_before_avg_10"

periods = {
    "during": "NEE_during_avg",
    "after_5": "NEE_after_avg_5",
    "after_10": "NEE_after_avg_10",
    "after_15": "NEE_after_avg_15",
    "after_20": "NEE_after_avg_20",
    "after_25": "NEE_after_avg_25",
    "after_30": "NEE_after_avg_30",
}

results_by_type = []

for period_name, nee_col in periods.items():
    temp = df[[method_col, baseline_col, nee_col]].dropna().copy()

    temp["NEE_diff"] = temp[nee_col] - temp[baseline_col]

    temp["NEE_response"] = np.where(
        temp["NEE_diff"] > 0,
        "Positive",
        np.where(temp["NEE_diff"] < 0, "Negative", None)
    )

    temp = temp.dropna(subset=["NEE_response"])

    contingency = pd.crosstab(
        temp[method_col],
        temp["NEE_response"]
    )

    contingency = contingency.reindex(
        columns=["Negative", "Positive"],
        fill_value=0
    )

    chi2, p, dof, expected = chi2_contingency(contingency)

    proportions = contingency.div(contingency.sum(axis=1), axis=0)

    for hw_type in contingency.index:
        results_by_type.append({
            "period": period_name,
            "heatwave_type": hw_type,
            "n": contingency.loc[hw_type].sum(),
            "negative_count": contingency.loc[hw_type, "Negative"],
            "positive_count": contingency.loc[hw_type, "Positive"],
            "negative_prop": proportions.loc[hw_type, "Negative"],
            "positive_prop": proportions.loc[hw_type, "Positive"],
            "chi2_all_types": chi2,
            "dof": dof,
            "p_value_all_types": p
        })

    print("\n" + "="*60)
    print(f"Period: {period_name}")
    print("\nCounts:")
    print(contingency)
    print("\nProportions:")
    print(proportions)
    print(f"\nChi-square = {chi2:.3f}, df = {dof}, p = {p:.5f}")

results_by_type_df = pd.DataFrame(results_by_type)

print(results_by_type_df)

results_by_type_df[[
    "period",
    "heatwave_type",
    "positive_prop",
    "negative_prop",
    "n",
    "p_value_all_types"
]]

# format the chi-square resutls above
##############################################################################
# P-values from the 7 heatwave type tests
all_types = (
    results_by_type_df
    .groupby("period")["p_value_all_types"]
    .first()
)

# P-values from the single vs. multiple tests
complexity = (
    results_df
    .set_index("period")["p_value"]
)

# Create summary table
pvalue_table = pd.DataFrame({
    "During": [all_types["during"], complexity["during"]],
    "5-Day Legacy": [all_types["after_5"], complexity["after_5"]],
    "10-Day Legacy": [all_types["after_10"], complexity["after_10"]],
    "15-Day Legacy": [all_types["after_15"], complexity["after_15"]],
    "20-Day Legacy": [all_types["after_20"], complexity["after_20"]],
    "25-Day Legacy": [all_types["after_25"], complexity["after_25"]],
    "30-Day Legacy": [all_types["after_30"], complexity["after_30"]],
},
index=[
    "Across heatwave types",
    "Single vs. Multi-dimensional"
])

print(pvalue_table)

def format_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

pvalue_table = pvalue_table.map(format_p)

print(pvalue_table)

# exporting to table for poster
# Make sure p-values are formatted as strings
display_table = pvalue_table.copy()

fig, ax = plt.subplots(figsize=(9, 1.8))
ax.axis('off')

table = ax.table(
    cellText=display_table.values,
    rowLabels=display_table.index,
    colLabels=display_table.columns,
    cellLoc='center',
    rowLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.savefig(
    "ChiSquare_Legacy_Table.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()

# heatwaves of these proportions
###############################################################################
# Use the results table you already created
# results_by_type_df should have:
# period, heatwave_type, positive_prop

heatmap_df = results_by_type_df.pivot(
    index="heatwave_type",
    columns="period",
    values="positive_prop"
)

# Put columns in time order
period_order = ["during", "after_5", "after_10", "after_15", "after_20", "after_25", "after_30"]
heatmap_df = heatmap_df[period_order]

# Rename columns for poster
heatmap_df.columns = ["During", "5 d", "10 d", "15 d", "20 d", "25 d", "30 d"]

# Convert to percent
heatwave_order = [
    "Day",
    "Night",
    "Overall",
    "Day-intensified",
    "Night-intensified",
    "Day-Night Spike",
    "Triad"
]

heatwave_order = [
    "Triad",
    "Day-Night Spike",
    "Night-intensified",
    "Day-intensified",
    "Overall",
    "Day",
    "Night"
]

heatmap_percent = heatmap_percent.loc[heatwave_order]

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4.5))

sns.heatmap(
    heatmap_percent,
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    linewidths=0,          # removes cell borders
    linecolor=None,
    cbar_kws={"label": "Positive NEE response (%)"},
    annot_kws={
        "color": "black",
        "weight": "bold",
        "fontsize": 10
    },
    vmin=40,
    vmax=60,
    ax=ax
)

ax.set_xlabel("Time since heatwave")
ax.set_ylabel("Heatwave category")

plt.tight_layout()
plt.show()

# Select the columns needed for the ANOVAs
raw_nee_df = df[["top_heatwave", "NEE_during_avg"]].dropna().copy()
norm_nee_df = df[["top_heatwave", "NEE_norm_during_avg"]].dropna().copy()
raw_nee_diff_df = df[["top_heatwave", "NEE_diff_during_minus_before_10"]].dropna().copy()
norm_nee_diff_df = df[["top_heatwave", "NEE_norm_diff_during_minus_before_10"]].dropna().copy()


# Group summaries for raw NEE
raw_nee_summary = (
    raw_nee_df
    .groupby("top_heatwave")["NEE_during_avg"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .sort_values("top_heatwave")
)


# Group summaries for normalized NEE
norm_nee_summary = (
    norm_nee_df
    .groupby("top_heatwave")["NEE_norm_during_avg"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .sort_values("top_heatwave")
)


# Group summaries for raw NEE difference
raw_nee_diff_summary = (
    raw_nee_diff_df
    .groupby("top_heatwave")["NEE_diff_during_minus_before_10"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .sort_values("top_heatwave")
)


# Group summaries for normalized NEE difference
norm_nee_diff_summary = (
    norm_nee_diff_df
    .groupby("top_heatwave")["NEE_norm_diff_during_minus_before_10"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
    .sort_values("top_heatwave")
)


# Build the input groups for the one-way ANOVAs
raw_nee_groups = [
    group["NEE_during_avg"].to_numpy()
    for _, group in raw_nee_df.groupby("top_heatwave")
    if len(group) > 1
]

norm_nee_groups = [
    group["NEE_norm_during_avg"].to_numpy()
    for _, group in norm_nee_df.groupby("top_heatwave")
    if len(group) > 1
]

raw_nee_diff_groups = [
    group["NEE_diff_during_minus_before_10"].to_numpy()
    for _, group in raw_nee_diff_df.groupby("top_heatwave")
    if len(group) > 1
]

norm_nee_diff_groups = [
    group["NEE_norm_diff_during_minus_before_10"].to_numpy()
    for _, group in norm_nee_diff_df.groupby("top_heatwave")
    if len(group) > 1
]


# Run one-way ANOVA for raw NEE
raw_nee_f_stat, raw_nee_p_value = stats.f_oneway(*raw_nee_groups)


# Run one-way ANOVA for normalized NEE
norm_nee_f_stat, norm_nee_p_value = stats.f_oneway(*norm_nee_groups)


# Run one-way ANOVA for raw NEE difference
raw_nee_diff_f_stat, raw_nee_diff_p_value = stats.f_oneway(*raw_nee_diff_groups)


# Run one-way ANOVA for normalized NEE difference
norm_nee_diff_f_stat, norm_nee_diff_p_value = stats.f_oneway(*norm_nee_diff_groups)


# Collect the ANOVA results into one table
anova_results = pd.DataFrame(
    {
        "metric": [
            "NEE_during_avg",
            "NEE_norm_during_avg",
            "NEE_diff_during_minus_before_10",
            "NEE_norm_diff_during_minus_before_10",
        ],
        "grouping_variable": [
            "top_heatwave",
            "top_heatwave",
            "top_heatwave",
            "top_heatwave",
        ],
        "f_statistic": [
            raw_nee_f_stat,
            norm_nee_f_stat,
            raw_nee_diff_f_stat,
            norm_nee_diff_f_stat,
        ],
        "p_value": [
            raw_nee_p_value,
            norm_nee_p_value,
            raw_nee_diff_p_value,
            norm_nee_diff_p_value,
        ],
        "n_groups": [
            len(raw_nee_groups),
            len(norm_nee_groups),
            len(raw_nee_diff_groups),
            len(norm_nee_diff_groups),
        ],
        "n_observations": [
            len(raw_nee_df),
            len(norm_nee_df),
            len(raw_nee_diff_df),
            len(norm_nee_diff_df),
        ],
    }
)


# View results in Spyder
print("\nOne-way ANOVA results")
print(anova_results.to_string(index=False))

print("\nGroup summary: NEE_during_avg")
print(raw_nee_summary.to_string(index=False))

print("\nGroup summary: NEE_norm_during_avg")
print(norm_nee_summary.to_string(index=False))

print("\nGroup summary: NEE_diff_during_minus_before_10")
print(raw_nee_diff_summary.to_string(index=False))

print("\nGroup summary: NEE_norm_diff_during_minus_before_10")
print(norm_nee_diff_summary.to_string(index=False))

# Save outputs
raw_summary_path = OUTPUT_DIR / "anova_summary_NEE_during_avg_by_top_heatwave.csv"
norm_summary_path = OUTPUT_DIR / "anova_summary_NEE_norm_during_avg_by_top_heatwave.csv"
raw_diff_summary_path = OUTPUT_DIR / "anova_summary_NEE_diff_during_minus_before_10_by_top_heatwave.csv"
norm_diff_summary_path = OUTPUT_DIR / "anova_summary_NEE_norm_diff_during_minus_before_10_by_top_heatwave.csv"
results_path = OUTPUT_DIR / "anova_results_nee_by_top_heatwave.csv"

raw_nee_summary.to_csv(raw_summary_path, index=False)
norm_nee_summary.to_csv(norm_summary_path, index=False)
raw_nee_diff_summary.to_csv(raw_diff_summary_path, index=False)
norm_nee_diff_summary.to_csv(norm_diff_summary_path, index=False)
anova_results.to_csv(results_path, index=False)

print("\nSaved overlapping-analysis files:")
print(results_path)
print(raw_summary_path)
print(norm_summary_path)
print(raw_diff_summary_path)
print(norm_diff_summary_path)

# Metrics to reuse in the models below
analysis_metrics = [
    "NEE_during_avg",
    "NEE_norm_during_avg",
    "NEE_diff_during_minus_before_10",
    "NEE_norm_diff_during_minus_before_10",
]


# -------------------------------------------------------------------------
# Two-way ANOVA: heatwave type and IGBP
# -------------------------------------------------------------------------
two_way_heatwave_igbp_results = []

for metric in analysis_metrics:
    model_df = df[["top_heatwave", "IGBP", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(top_heatwave) + C(IGBP) + C(top_heatwave):C(IGBP)", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "two_way_heatwave_igbp"
    two_way_heatwave_igbp_results.append(anova_table)

two_way_heatwave_igbp_results = pd.concat(two_way_heatwave_igbp_results, ignore_index=True)

print("\nTwo-way ANOVA: top_heatwave and IGBP")
print(two_way_heatwave_igbp_results.to_string(index=False))

two_way_heatwave_igbp_path = OUTPUT_DIR / "two_way_anova_heatwave_igbp_nee.csv"
two_way_heatwave_igbp_results.to_csv(two_way_heatwave_igbp_path, index=False)

# -------------------------------------------------------------------------
# Two-way ANOVA: heatwave type and season
# -------------------------------------------------------------------------
two_way_heatwave_season_results = []

for metric in analysis_metrics:
    model_df = df[["top_heatwave", "Season", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(top_heatwave) + C(Season) + C(top_heatwave):C(Season)", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "two_way_heatwave_season"
    two_way_heatwave_season_results.append(anova_table)

two_way_heatwave_season_results = pd.concat(two_way_heatwave_season_results, ignore_index=True)

print("\nTwo-way ANOVA: top_heatwave and Season")
print(two_way_heatwave_season_results.to_string(index=False))

two_way_heatwave_season_path = OUTPUT_DIR / "two_way_anova_heatwave_season_nee.csv"
two_way_heatwave_season_results.to_csv(two_way_heatwave_season_path, index=False)

# -------------------------------------------------------------------------
# ANCOVA: heatwave type with precipitation and SWC as covariates
# -------------------------------------------------------------------------
ancova_heatwave_moisture_results = []

for metric in analysis_metrics:
    model_df = df[["top_heatwave", "prec_average", "swc_average", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(top_heatwave) + prec_average + swc_average", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "ancova_heatwave_moisture"
    ancova_heatwave_moisture_results.append(anova_table)

ancova_heatwave_moisture_results = pd.concat(ancova_heatwave_moisture_results, ignore_index=True)

print("\nANCOVA: top_heatwave with precipitation and SWC")
print(ancova_heatwave_moisture_results.to_string(index=False))

ancova_heatwave_moisture_path = OUTPUT_DIR / "ancova_heatwave_moisture_nee.csv"
ancova_heatwave_moisture_results.to_csv(ancova_heatwave_moisture_path, index=False)

print("\nSaved overlapping model files:")
print(two_way_heatwave_igbp_path)
print(two_way_heatwave_season_path)
print(ancova_heatwave_moisture_path)

# =============================================================================
# PRE-OVERLAP HEATWAVE ANALYSIS: TRUE MIN / MEAN / MAX EVENTS
# =============================================================================

# Build a method-level event table directly from the original pre-overlap
# heatwaves stored in heatwaves_df.csv.
method_heatwaves = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "heatwaves_df.csv")
method_heatwaves["start_dates"] = pd.to_datetime(method_heatwaves["start_dates"])
method_heatwaves["end_dates"] = pd.to_datetime(method_heatwaves["end_dates"])
method_heatwaves["duration_days"] = pd.to_timedelta(method_heatwaves["duration"]).dt.days + 1

# Add site metadata already present in the overlapping table outputs.
method_site_info = df[["Site", "IGBP"]].dropna().drop_duplicates()
method_heatwaves = pd.merge(method_heatwaves, method_site_info, on="Site", how="left")
method_heatwaves["Month"] = method_heatwaves["start_dates"].dt.month
method_heatwaves["Season"] = pd.Series(method_heatwaves["Month"]).map(
    lambda m: "Spring" if 3 <= m <= 5
    else "Summer" if 6 <= m <= 8
    else "Fall" if 9 <= m <= 11
    else "Winter"
)

# Load cleaned daily flux data and NEE QC values so we can recreate event-level
# summaries for the original Min / Mean / Max heatwaves.
method_flux_df = pd.read_csv(REPO_ROOT / "data" / "cleaned" / "AMF_DD.csv")
method_flux_df = method_flux_df.iloc[:, 1:]
method_flux_df["date"] = pd.to_datetime(method_flux_df["date"])

method_nee_qc = loadAMF(
    str(REPO_ROOT / "data" / "AMFdataDD"),
    measures=["TIMESTAMP", "NEE_VUT_REF_QC"],
)
method_nee_qc.columns = ["date", "NEE_VUT_REF_QC", "Site"]
method_nee_qc["date"] = pd.to_datetime(method_nee_qc["date"])

method_flux_df = pd.merge(method_flux_df, method_nee_qc, on=["Site", "date"], how="left")
method_flux_df.columns = [
    "date", "TA", "SW", "VPD", "P", "NEE", "RECO", "GPP", "Site", "IGBP", "NEE_VUT_REF_QC"
]

method_cleaned_df, method_year_summary, method_remove_sites = clean_flux_by_qc_and_years(
    df=method_flux_df,
    site_col="Site",
    date_col="date",
    nee_col="NEE",
    gpp_col="GPP",
    reco_col="RECO",
    qc_col="NEE_VUT_REF_QC",
    qc_threshold=0.75,
    missing_frac_threshold=0.25,
    min_years_required=5,
)
method_cleaned_df = method_cleaned_df[~method_cleaned_df.Site.isin(["CA-TPD", "MX-Tes"])]
method_cleaned_df.loc[method_cleaned_df["GPP"] < 0, "GPP"] = 0
method_cleaned_df = normalize_nee_by_site(
    df=method_cleaned_df,
    site_col="Site",
    nee_col="NEE",
    baseline_mask=None,
    method="zscore",
)

# Keep only events with complete flux coverage.
method_flux_heatwaves_df, method_flux_heatwaves_summary = filter_complete_heatwaves(
    heatwaves_df=method_heatwaves,
    flux_df=method_cleaned_df,
)

# Add before/during summaries for raw and normalized NEE.
method_flux_heatwaves_df = calc_flux_multi_lag(
    flux_name="NEE",
    heatwaves_df=method_flux_heatwaves_df,
    flux_df=method_cleaned_df,
    stats=("avg", "std"),
    before_lags=[10],
    after_lags=[],
    min_frac=0,
    site_col="Site",
    start_col="start_dates",
    end_col="end_dates",
    date_col="date",
)
method_flux_heatwaves_df = calc_flux_multi_lag(
    flux_name="NEE_norm",
    heatwaves_df=method_flux_heatwaves_df,
    flux_df=method_cleaned_df,
    stats=("avg", "std"),
    before_lags=[10],
    after_lags=[],
    min_frac=0,
    site_col="Site",
    start_col="start_dates",
    end_col="end_dates",
    date_col="date",
)

method_flux_heatwaves_df["NEE_diff_during_minus_before_10"] = (
    method_flux_heatwaves_df["NEE_during_avg"] - method_flux_heatwaves_df["NEE_before_avg_10"]
)
method_flux_heatwaves_df["NEE_norm_diff_during_minus_before_10"] = (
    method_flux_heatwaves_df["NEE_norm_during_avg"] - method_flux_heatwaves_df["NEE_norm_before_avg_10"]
)

# Attach precomputed moisture summaries for the original Min / Mean / Max events.
max_precip = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "max_precip.csv")
max_precip["Method"] = "Max"
mean_precip = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "mean_precip.csv")
mean_precip["Method"] = "Mean"
min_precip = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "min_precip.csv")
min_precip["Method"] = "Min"
method_precip = pd.concat([max_precip, mean_precip, min_precip], ignore_index=True)
method_precip["start_date"] = pd.to_datetime(method_precip["start_date"])
method_precip["end_date"] = pd.to_datetime(method_precip["end_date"])
method_precip = method_precip.rename(columns={
    "start_date": "start_dates",
    "end_date": "end_dates",
    "moisture_average": "prec_average",
    "moisture_total": "prec_total",
    "moisture_variability": "prec_variability",
})
method_precip = method_precip[["Site", "Method", "start_dates", "end_dates", "prec_average", "prec_total", "prec_variability"]]

max_swc = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "max_swc.csv")
max_swc["Method"] = "Max"
mean_swc = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "mean_swc.csv")
mean_swc["Method"] = "Mean"
min_swc = pd.read_csv(REPO_ROOT / "data" / "heatwaves" / "min_swc.csv")
min_swc["Method"] = "Min"
method_swc = pd.concat([max_swc, mean_swc, min_swc], ignore_index=True)
method_swc["start_date"] = pd.to_datetime(method_swc["start_date"])
method_swc["end_date"] = pd.to_datetime(method_swc["end_date"])
method_swc = method_swc.rename(columns={
    "start_date": "start_dates",
    "end_date": "end_dates",
    "moisture_average": "swc_average",
    "moisture_total": "swc_total",
    "moisture_variability": "swc_variability",
})
method_swc.loc[method_swc["swc_average"] == -9999, ["swc_average", "swc_total", "swc_variability"]] = pd.NA
method_swc = method_swc[["Site", "Method", "start_dates", "end_dates", "swc_average", "swc_total", "swc_variability"]]

method_flux_heatwaves_df = pd.merge(
    method_flux_heatwaves_df,
    method_precip,
    on=["Site", "Method", "start_dates", "end_dates"],
    how="left",
)
method_flux_heatwaves_df = pd.merge(
    method_flux_heatwaves_df,
    method_swc,
    on=["Site", "Method", "start_dates", "end_dates"],
    how="left",
)

# Save the method-level event table so it can be explored directly in Spyder.
method_flux_heatwaves_path = REPO_ROOT / "data" / "heatwaves" / "heatwaves_base_df.csv"
method_flux_heatwaves_df.to_csv(method_flux_heatwaves_path, index=False)
print("\nSaved pre-overlap base heatwave table:")
print(method_flux_heatwaves_path)

method_analysis_metrics = [
    "NEE_during_avg",
    "NEE_norm_during_avg",
    "NEE_diff_during_minus_before_10",
    "NEE_norm_diff_during_minus_before_10",
]

# One-way ANOVA by Method.
method_one_way_results = []
for metric in method_analysis_metrics:
    metric_df = method_flux_heatwaves_df[["Method", metric]].dropna().copy()
    metric_groups = [
        group[metric].to_numpy()
        for _, group in metric_df.groupby("Method")
        if len(group) > 1
    ]
    f_stat, p_value = stats.f_oneway(*metric_groups)
    method_one_way_results.append({
        "metric": metric,
        "grouping_variable": "Method",
        "f_statistic": f_stat,
        "p_value": p_value,
        "n_groups": len(metric_groups),
        "n_observations": len(metric_df),
    })
method_one_way_results = pd.DataFrame(method_one_way_results)
print("One-way ANOVA results by Method")
print(method_one_way_results.to_string(index=False))

# Two-way ANOVA: Method and IGBP.
two_way_method_igbp_results = []
for metric in method_analysis_metrics:
    model_df = method_flux_heatwaves_df[["Method", "IGBP", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(Method) + C(IGBP) + C(Method):C(IGBP)", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "two_way_method_igbp"
    two_way_method_igbp_results.append(anova_table)
two_way_method_igbp_results = pd.concat(two_way_method_igbp_results, ignore_index=True)
print("Two-way ANOVA: Method and IGBP")
print(two_way_method_igbp_results.to_string(index=False))

# Two-way ANOVA: Method and Season.
two_way_method_season_results = []
for metric in method_analysis_metrics:
    model_df = method_flux_heatwaves_df[["Method", "Season", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(Method) + C(Season) + C(Method):C(Season)", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "two_way_method_season"
    two_way_method_season_results.append(anova_table)
two_way_method_season_results = pd.concat(two_way_method_season_results, ignore_index=True)
print("Two-way ANOVA: Method and Season")
print(two_way_method_season_results.to_string(index=False))

# ANCOVA: Method with precipitation and SWC.
ancova_method_moisture_results = []
for metric in method_analysis_metrics:
    model_df = method_flux_heatwaves_df[["Method", "prec_average", "swc_average", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(Method) + prec_average + swc_average", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "ancova_method_moisture"
    ancova_method_moisture_results.append(anova_table)
ancova_method_moisture_results = pd.concat(ancova_method_moisture_results, ignore_index=True)
print("ANCOVA: Method with precipitation and SWC")
print(ancova_method_moisture_results.to_string(index=False))

# ANCOVA: Method with duration.
ancova_method_duration_results = []
for metric in method_analysis_metrics:
    model_df = method_flux_heatwaves_df[["Method", "duration_days", metric]].dropna().copy()
    model = ols(f"{metric} ~ C(Method) + duration_days", data=model_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
    anova_table["metric"] = metric
    anova_table["model_type"] = "ancova_method_duration"
    ancova_method_duration_results.append(anova_table)
ancova_method_duration_results = pd.concat(ancova_method_duration_results, ignore_index=True)
print("ANCOVA: Method with duration")
print(ancova_method_duration_results.to_string(index=False))


# Save pre-overlap method-level model outputs
method_one_way_path = OUTPUT_DIR / "anova_results_nee_by_method.csv"
two_way_method_igbp_path = OUTPUT_DIR / "two_way_anova_method_igbp_nee.csv"
two_way_method_season_path = OUTPUT_DIR / "two_way_anova_method_season_nee.csv"
ancova_method_moisture_path = OUTPUT_DIR / "ancova_method_moisture_nee.csv"
ancova_method_duration_path = OUTPUT_DIR / "ancova_method_duration_nee.csv"

method_one_way_results.to_csv(method_one_way_path, index=False)
two_way_method_igbp_results.to_csv(two_way_method_igbp_path, index=False)
two_way_method_season_results.to_csv(two_way_method_season_path, index=False)
ancova_method_moisture_results.to_csv(ancova_method_moisture_path, index=False)
ancova_method_duration_results.to_csv(ancova_method_duration_path, index=False)

print("\nSaved pre-overlap model files:")
print(method_one_way_path)
print(two_way_method_igbp_path)
print(two_way_method_season_path)
print(ancova_method_moisture_path)
print(ancova_method_duration_path)

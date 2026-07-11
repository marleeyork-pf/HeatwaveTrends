"""
Step 3: calculate flux summaries.

Spyder-friendly standalone script version of the canonical flux-summary
workflow. This keeps the original calculations and only cleans up imports and
paths.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats


def find_repo_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd().resolve()
    if cwd.name == "publication_repo":
        return cwd
    if (cwd / "publication_repo").exists():
        return cwd / "publication_repo"
    if cwd.parent.name == "publication_repo":
        return cwd.parent

    raise RuntimeError(
        "Could not locate publication_repo. In Spyder, set the working directory "
        "to the publication_repo folder and run the whole file."
    )


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
HEATWAVES_DIR = DATA_DIR / "heatwaves"
CLEANED_DIR = DATA_DIR / "cleaned"
AMFDATA_DD_DIR = DATA_DIR / "AMFdataDD"
BADM_DIR = DATA_DIR / "BADM"
SOIL_DIR = DATA_DIR / "soil_data"

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from auxiliary import *


def run_step3():
    # Loading in the data. For the BADM, I skip over sites that don't have the climate or MAT.
    all_heatwaves_df = pd.read_csv(HEATWAVES_DIR / "all_heatwaves_df.csv")
    all_heatwaves_df = all_heatwaves_df.iloc[:, 1:]
    
    # Load in data for daily AmeriFlux covariates and flux data
    df = pd.read_csv(CLEANED_DIR / "AMF_DD.csv")
    df = df.iloc[:, 1:]
    
    # Load in BADM data
    badm_skip = [
        str(BADM_DIR / "AMF_CA-Qc2_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-Cop_BIF_20240229.xlsx"),
        str(BADM_DIR / "AMF_US-UiD_BIF_20251017.xlsx"),
        str(BADM_DIR / "AMF_US-BMM_BIF_20221003.xlsx"),
        str(BADM_DIR / "AMF_US-NGC_BIF_20231208.xlsx"),
        str(BADM_DIR / "AMF_US-Snf_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-AR2_BIF_20231031.xlsx"),
        str(BADM_DIR / "AMF_US-SdH_BIF_20241204.xlsx"),
        str(BADM_DIR / "AMF_US-CAK_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-AR1_BIF_20231031.xlsx"),
        str(BADM_DIR / "AMF_US-BMM_BIF_20221003.xlsx"),
        str(BADM_DIR / "AMF_US-NGC_BIF_20231208.xlsx"),
        str(BADM_DIR / "AMF_CA-Qc2_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-Snf_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-SdH_BIF_20241204.xlsx"),
        str(BADM_DIR / "AMF_US-Fcr_BIF_20240401.xlsx"),
        str(BADM_DIR / "AMF_US-Sta_BIF_20250731.xlsx"),
        str(BADM_DIR / "AMF_US-AR1_BIF_20231031.xlsx"),
        str(BADM_DIR / "AMF_CA-Ca3_BIF_20241204.xlsx"),
    ]
    badm = loadBADM(
        path=str(BADM_DIR),
        skip=badm_skip,
        column="VARIABLE",
        value="DATAVALUE",
        measure=["IGBP", "CLIMATE_KOEPPEN", "MAT", "MAP", "LOCATION_LAT", "LOCATION_LONG", "LOCATION_ELEV"],
        file_type="xslx",
    )
    
    # Load in quality control flag for NEE
    NEE_qc = loadAMF(str(AMFDATA_DD_DIR), measures=["TIMESTAMP", "NEE_VUT_REF_QC"])
    NEE_qc.columns = ["date", "NEE_VUT_REF_QC", "Site"]

    # Make sure dates are all datetime variables
    df["date"] = pd.to_datetime(df["date"])
    all_heatwaves_df["start_dates"] = pd.to_datetime(all_heatwaves_df.start_dates)
    all_heatwaves_df["end_dates"] = pd.to_datetime(all_heatwaves_df.end_dates)
    NEE_qc["date"] = pd.to_datetime(NEE_qc["date"])

    # Merge the NEE quality control flag onto the dataframe
    df = pd.merge(df, NEE_qc, on=["Site", "date"])

    # Convert badm data into numeric
    badm_var = ["MAT", "MAP", "LOCATION_LAT", "LOCATION_LONG", "LOCATION_ELEV"]
    badm = badm[badm.Site.isin(all_heatwaves_df.Site.unique())].reset_index()
    badm[badm_var] = badm[badm_var].astype("float")

    # Reduce df to those sites we have heatwaves for
    df = df[df.Site.isin(all_heatwaves_df.Site.unique())]

    # Retrieve SWC data and merge onto the dataframe
    swc = loadAMF(str(AMFDATA_DD_DIR), measures=["TIMESTAMP", "SWC_F_MDS_1"])
    swc.columns = ["date", "SWC_F_MDS_1", "Site"]
    df = pd.merge(df, swc, on=["Site", "date"], how="left")

    # Replace invalid SWC values with NA or 0. Remove sites with terrible SWC
    df.loc[df.SWC_F_MDS_1 == -9999, "SWC_F_MDS_1"] = np.nan
    df.loc[df.SWC_F_MDS_1 < 0, "SWC_F_MDS_1"] = 0
    df = df.loc[df.Site != "US-xHA", :]

    # Rename dataframe columns
    df.columns = ["date", "TA", "SW", "VPD", "P", "NEE", "RECO", "GPP", "Site", "IGBP", "NEE_VUT_REF_QC", "SWC"]

    # Add inidcator for whether or not we are in a heatwave
    df = add_heatwave_indicator(df, all_heatwaves_df)

    # Read in a expanded heatwave data (each obs = 1 day in heatwave)
    heatwave_df = pd.read_csv(HEATWAVES_DIR / "heatwaves_df.csv")
    
    # Convert the dates in expanded heatwave dataset into datetime variables
    heatwave_df["start_dates"] = pd.to_datetime(heatwave_df.start_dates)
    heatwave_df["end_dates"] = pd.to_datetime(heatwave_df.end_dates)
    
    # Reducing index variable and renaming the columns
    heatwave_df = heatwave_df.iloc[:, 1:]
    heatwave_df.columns = ["Site", "top_heatwave", "start_dates", "end_dates", "duration", "QAQC_flag"]

    # Searching and cleanig the dataframe of flux data that is low quality
    # Reference the function in aux to see the exact procedure by which these were selected
    cleaned_df, year_summary, remove_sites = clean_flux_by_qc_and_years(
        df=df,
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

    drop_sites = ["CA-TPD", "MX-Tes"]
    cleaned_df = cleaned_df[~cleaned_df.Site.isin(drop_sites)]

    # Setting GPP less than 0 to 0 (not even using GPP in this analysis yet)
    cleaned_df.loc[df.GPP < 0, "GPP"] = 0

    # Filter out any heatwaves that were cut short by invalid data
    flux_heatwaves_df, flux_heatwaves_summary = filter_complete_heatwaves(
        heatwaves_df=all_heatwaves_df,
        flux_df=cleaned_df,
    )
    
    # Merge climate koeppen onto heatwaves and convert it (and IGBP) to categorical var
    all_heatwaves_df = pd.merge(all_heatwaves_df,badm[["Site","CLIMATE_KOEPPEN"]], on="Site", how="left")
    all_heatwaves_df["IGBP"] = all_heatwaves_df["IGBP"].astype("category")
    all_heatwaves_df["CLIMATE_KOEPPEN"] = all_heatwaves_df["CLIMATE_KOEPPEN"].astype("category")

    # Add an indicator column for heatwaves
    cleaned_df = cleaned_df.drop(columns=["heatwave_indicator"])
    cleaned_df = add_heatwave_indicator(cleaned_df, flux_heatwaves_df)

    # Retrieve and add soil data onto the cleaned covariates AND the heatwave dataframe
    soil = pd.read_csv(SOIL_DIR / "soil_df.csv")
    cleaned_df.Site.unique()[~pd.Series(cleaned_df.Site.unique()).isin(soil.Site.unique())]
    flux_heatwaves_df = pd.merge(flux_heatwaves_df, soil, on="Site", how="left")
    
    # Calculate the normalized NEE flux values
    cleaned_df = normalize_nee_by_site(
        df=cleaned_df,
        site_col="Site",
        nee_col="NEE",
        baseline_mask=None,
        method="zscore",
    )

    # Calculate the averages for flux and covariate values leading up to, during,
    # and after the heatwaves
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="SWC",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("avg", "std"),
        before_lags=[90, 30, 10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )
    
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="P",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("sum"),
        before_lags=[90, 30, 10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )
    
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="VPD",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("avg", "std"),
        before_lags=[90, 30, 10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )
    
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="SW",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("avg", "std"),
        before_lags=[90, 30, 10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )
    
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="NEE",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("avg", "std"),
        before_lags=[10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )
    
    flux_heatwaves_df = calc_flux_multi_lag(
        flux_name="NEE_norm",
        heatwaves_df=flux_heatwaves_df,
        flux_df=cleaned_df,
        stats=("avg", "std"),
        before_lags=[10],
        after_lags=[5, 10, 15, 20, 25, 30],
        min_frac=0,
        site_col="Site",
        start_col="start_dates",
        end_col="end_dates",
        date_col="date",
    )

    # Calculate temperature deviation based on the daily average temperature
    # for that site on that day of the year
    DOY_ta = DOY_climatology(df, "TA", smoothing_function="weighted_15")
    DOY_ta["TA_dev"] = DOY_ta["TA"] - DOY_ta["expected_TA"]

    flux_heatwaves_df["TA_mean_dev"] = [pd.NA] * len(flux_heatwaves_df)
    for idx, row in flux_heatwaves_df.iterrows():
        start = row["start_dates"]
        end = row["end_dates"]
        dates = pd.date_range(start, end)
        site = row["Site"]
        this_hw = DOY_ta[(DOY_ta.Site == site) & (DOY_ta.date.isin(dates))]
        flux_heatwaves_df.loc[idx, "TA_mean_dev"] = this_hw.TA_dev.mean()

    # Add in a start day and year of the heatwave
    flux_heatwaves_df["start_DOY"] = flux_heatwaves_df.start_dates.dt.dayofyear
    flux_heatwaves_df["start_Year"] = flux_heatwaves_df.start_dates.dt.year

    # Save the updated flux dataframe
    flux_heatwaves_df.to_csv(HEATWAVES_DIR / "flux_heatwaves_df.csv")

    print("Step 3 completed successfully.")
    print(HEATWAVES_DIR / "flux_heatwaves_df.csv")


if __name__ == "__main__":
    run_step3()

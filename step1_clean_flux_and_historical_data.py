"""
Step 1: clean flux and historical temperature data.

This is a Spyder-friendly script version intended to feel like the original
research workflow: one file you can open and run directly.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import os
from sklearn.linear_model import LinearRegression

def find_repo_root():
    """Find the publication_repo directory for normal Python and Spyder use."""
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
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
HEATWAVE_DEFINITION_DIR = REPO_ROOT / "heatwave_definition"
DATA_DIR = REPO_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"

for path in [REPO_ROOT, PREPROCESSING_DIR, HEATWAVE_DEFINITION_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from auxiliary import loadAMF, loadBADM
from auxiliary import return_best_data
from auxiliary import adjust_historical_data
from auxiliary import find_max_temperatures, find_min_temperatures


pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 100)


def run_step1():
    sb.set_theme()

    # Load in daily ameriflux data
    df = loadAMF(
        path=str(DATA_DIR / "AMFdataDD"),
        measures=[
            "TIMESTAMP",
            "TA_F",
            "SW_IN_F",
            "VPD_F",
            "P_F",
            "NEE_VUT_REF",
            "RECO_NT_VUT_REF",
            "GPP_NT_VUT_REF",
        ],
    )

    # Load in hourly ameriflux temperature
    df_hourly = loadAMF(
        path=str(DATA_DIR / "AMFdata_HH"),
        measures=["TIMESTAMP_START", "TA_F"],
    )

    # Load in IGBP from ameriflux BADM
    IGBP = loadBADM(
        path=str(DATA_DIR / "BADM"),
        skip=[""],
        column="VARIABLE",
        value="DATAVALUE",
        measure=["IGBP"],
        file_type="xslx",
    )
    
    # Merge IGBP onto the dataset
    df = pd.merge(df, IGBP, on="Site", how="left").drop_duplicates()
    df_hourly = pd.merge(df_hourly, IGBP, on="Site", how="left").drop_duplicates()

    # Remove any data that comes from a cropland site
    df = df[df["IGBP"] != "CRO"]
    df_hourly = df_hourly[df_hourly["IGBP"] != "CRO"]

    # Load in ERA and PRISM data, both maximum, minimum, and mean temperature
    ERA_max = pd.read_csv(DATA_DIR / "ERA" / "ERA_tmax_data.csv")
    ERA_min = pd.read_csv(DATA_DIR / "ERA" / "ERA_tmin_data.csv")
    ERA_mean = pd.read_csv(DATA_DIR / "ERA" / "ERA_tmean_data.csv")
    PRISM_max = pd.read_csv(DATA_DIR / "PRISM" / "extracted_daily_climate_data_tmax.csv")
    PRISM_min = pd.read_csv(DATA_DIR / "PRISM" / "extracted_daily_tmin.csv")
    PRISM_mean = pd.read_csv(DATA_DIR / "PRISM" / "extracted_daily_tmean.csv")
    AMF_mean = df[["Site", "TIMESTAMP", "TA_F"]].copy()
    AMF_mean.columns = ["Site", "date", "TA_F"]

    # Convert ERA from Kelvin to Celsius
    ERA_max["ERA_TA"] = ERA_max["t2m"] - 273.15
    ERA_min["ERA_TA"] = ERA_min["t2m"] - 273.15
    ERA_mean["ERA_TA"] = ERA_mean["t2m"] - 273.15

    # Convert the date variable to a datetime
    ERA_max["date"] = pd.to_datetime(ERA_max.valid_time)
    ERA_min["date"] = pd.to_datetime(ERA_min.valid_time)
    ERA_mean["date"] = pd.to_datetime(ERA_mean.valid_time)
    PRISM_max["date"] = pd.to_datetime(PRISM_max.date)
    PRISM_min["date"] = pd.to_datetime(PRISM_min.date)
    PRISM_mean["date"] = pd.to_datetime(PRISM_mean.date)

    # Reduce ERA datafraems to columns of interest
    ERA_max = ERA_max[["Site", "date", "ERA_TA"]]
    ERA_min = ERA_min[["Site", "date", "ERA_TA"]]
    ERA_mean = ERA_mean[["Site", "date", "ERA_TA"]]

    # Reduce ERA and PRISM data to sites in AmeriFlux data
    included_sites = df.Site.unique()
    included_sites = np.insert(included_sites, 0, "date")
    ERA_max = ERA_max[ERA_max["Site"].isin(included_sites)]
    ERA_min = ERA_min[ERA_min["Site"].isin(included_sites)]
    ERA_mean = ERA_mean[ERA_mean["Site"].isin(included_sites)]
    PRISM_included_sites = included_sites[pd.Series(included_sites).isin(PRISM_mean.columns)]
    PRISM_max = PRISM_max[PRISM_included_sites]
    PRISM_min = PRISM_min[PRISM_included_sites]
    PRISM_mean = PRISM_mean[PRISM_included_sites]

    # Search for any missing values in the PRISM and ERA data
    search_value = -9999
    missing_max = []
    missing_min = []
    missing_avg = []
    for col in PRISM_max.columns:
        if PRISM_max[col].astype(str).str.contains(str(search_value)).any():
            missing_max.append(col)
    for col in PRISM_min.columns:
        if PRISM_min[col].astype(str).str.contains(str(search_value)).any():
            missing_min.append(col)
    for col in PRISM_mean.columns:
        if PRISM_mean[col].astype(str).str.contains(str(search_value)).any():
            missing_avg.append(col)

    # Remove those columns with missing data
    PRISM_max = PRISM_max.drop(columns=missing_max)
    PRISM_min = PRISM_min.drop(columns=missing_min)
    PRISM_mean = PRISM_mean.drop(columns=missing_avg)

    # Melt the columns (each site) into one columns
    PRISM_max = pd.melt(PRISM_max, id_vars="date", var_name="Site", value_name="PRISM_TA")
    PRISM_min = pd.melt(PRISM_min, id_vars="date", var_name="Site", value_name="PRISM_TA")
    PRISM_mean = pd.melt(PRISM_mean, id_vars="date", var_name="Site", value_name="PRISM_TA")

    # Remove sites that we dont have sufficient data for
    removing_sites = [
        "US-CAK", "CA-Ca1", "US-xHE", "US-xDJ", "US-ICt", "US-Rpf", "US-xNW",
        "US-ICh", "US-Hn2", "US-EML", "US-BZS", "US-NGC", "US-Cop", "CA-SCC",
        "CA-NS2", "US-SP1", "US-Ho1", "US-Me2",
    ]
    ERA_max = ERA_max[~ERA_max.Site.isin(removing_sites)]
    ERA_min = ERA_min[~ERA_min.Site.isin(removing_sites)]
    ERA_mean = ERA_mean[~ERA_mean.Site.isin(removing_sites)]
    PRISM_max = PRISM_max[~PRISM_max.Site.isin(removing_sites)]
    PRISM_min = PRISM_min[~PRISM_min.Site.isin(removing_sites)]
    PRISM_mean = PRISM_mean[~PRISM_mean.Site.isin(removing_sites)]

    # Reduce AmeriFlux data to sites we have historical data for
    df_hourly = df_hourly[df_hourly.Site.isin(ERA_max.Site.unique())]
    df = df[df.Site.isin(ERA_max.Site.unique())]
    
    # Rename Ameriflux columns
    df.columns = [
        "date", "TA_F", "SW_IN_F", "VPD_F", "P_F", "NEE_VUT_REF",
        "RECO_NT_VUT_REF", "GPP_NT_VUT_REF", "Site", "IGBP",
    ]

    # Investigate available dates for ERA and PRISM data
    start_date = max([ERA_mean.date.min(), PRISM_mean.date.min()])
    end_date = min([ERA_mean.date.max(), PRISM_mean.date.max()])

    # Decide between ERA or PRISM data for each site and return one historical
    # dataframe that has the best option for each site
    historical_data_max, _ = return_best_data(
        AMF_data_all=df_hourly[["Site", "TIMESTAMP_START", "TA_F"]],
        ERA_data_all=ERA_max[["Site", "date", "ERA_TA"]],
        PRISM_data_all=PRISM_max[["Site", "date", "PRISM_TA"]],
        temperature_type="max",
        start_date=start_date,
        end_date=end_date,
    )
    historical_data_min, _ = return_best_data(
        AMF_data_all=df_hourly[["Site", "TIMESTAMP_START", "TA_F"]],
        ERA_data_all=ERA_min[["Site", "date", "ERA_TA"]],
        PRISM_data_all=PRISM_min[["Site", "date", "PRISM_TA"]],
        temperature_type="min",
        start_date=start_date,
        end_date=end_date,
    )
    historical_data_mean, _ = return_best_data(
        AMF_data_all=AMF_mean[["Site", "date", "TA_F"]],
        ERA_data_all=ERA_mean[["Site", "date", "ERA_TA"]],
        PRISM_data_all=PRISM_mean[["Site", "date", "PRISM_TA"]],
        temperature_type="average",
        start_date=start_date,
        end_date=end_date,
    )

    # Save these dataframes as csv
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    historical_data_max.to_csv(CLEANED_DIR / "historical_data_max.csv")
    historical_data_min.to_csv(CLEANED_DIR / "historical_data_min.csv")
    historical_data_mean.to_csv(CLEANED_DIR / "historical_data_mean.csv")
    df.to_csv(CLEANED_DIR / "AMF_DD.csv")
    df_hourly.to_csv(CLEANED_DIR / "AMF_HH.csv")

    # Reload daily and half hourly temperature
    AMF = pd.read_csv(CLEANED_DIR / "AMF_DD.csv")
    AMF["date"] = pd.to_datetime(AMF["date"])
    AMF_HH = pd.read_csv(CLEANED_DIR / "AMF_HH.csv")
    AMF_HH["TIMESTAMP_START"] = pd.to_datetime(AMF_HH["TIMESTAMP_START"])

    # Find minimum daily temperatures using half hourly data
    AMF_min = AMF_HH.groupby("Site").apply(
        lambda g: find_min_temperatures(g.TIMESTAMP_START, g.TA_F)
    ).reset_index()
    AMF_min = AMF_min[["Site", "date", "min_temperature"]]

    # Find maximum daily temperatures using half hourly data
    AMF_max = AMF_HH.groupby("Site").apply(
        lambda g: find_max_temperatures(g.TIMESTAMP_START, g.TA_F)
    ).reset_index()
    AMF_max = AMF_max[["Site", "date", "max_temperature"]]

    # Reload historical data
    hist_max = pd.read_csv(CLEANED_DIR / "historical_data_max.csv")
    hist_min = pd.read_csv(CLEANED_DIR / "historical_data_min.csv")
    hist_avg = pd.read_csv(CLEANED_DIR / "historical_data_mean.csv")
    AMF = AMF.iloc[:, 1:]
    AMF = AMF[["Site", "date", "TA_F"]]
    hist_max = hist_max.iloc[:, 1:]
    hist_min = hist_min.iloc[:, 1:]
    hist_avg = hist_avg.iloc[:, 1:]
    hist_max.columns = ["Site", "date", "hist_max", "Source_max"]
    hist_min.columns = ["Site", "date", "hist_min", "Source_min"]
    hist_avg.columns = ["Site", "date", "hist_mean", "Source_mean"]
    hist_max["date"] = pd.to_datetime(hist_max["date"])
    hist_min["date"] = pd.to_datetime(hist_min["date"])
    hist_avg["date"] = pd.to_datetime(hist_avg["date"])

    # Merge historical minimum, maximum, and average temperatures into one dataframe
    historical_data = pd.merge(hist_max, hist_min, on=["Site", "date"], how="inner")
    historical_data = pd.merge(historical_data, hist_avg, on=["Site", "date"], how="inner")
    AMF_data = pd.merge(AMF, AMF_min, on=["Site", "date"], how="inner")
    AMF_data = pd.merge(AMF_data, AMF_max, on=["Site", "date"], how="inner")
    AMF_data = AMF_data.dropna()

    # Calculate bias adjustments for PRISM/ERA data based on tower data
    adjustment_dict = adjust_historical_data(
        historical_data=historical_data,
        AMF_data=AMF_data,
        n=365 * 5,
        r2=0.9,
    )
    hist_data_adj = adjustment_dict["adjustments"]
    
    # Retrieve and save the adjusted data
    hist_data_adj.to_csv(CLEANED_DIR / "historical_data_adjusted.csv")

    print("Step 1 completed successfully.")
    print(CLEANED_DIR / "AMF_DD.csv")
    print(CLEANED_DIR / "AMF_HH.csv")
    print(CLEANED_DIR / "historical_data_max.csv")
    print(CLEANED_DIR / "historical_data_min.csv")
    print(CLEANED_DIR / "historical_data_mean.csv")
    print(CLEANED_DIR / "historical_data_adjusted.csv")


if __name__ == "__main__":
    run_step1()

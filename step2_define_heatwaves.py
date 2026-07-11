"""
Step 2: define heatwaves.

Spyder-friendly standalone script version of the canonical heatwave workflow.
This keeps the original heatwave logic and only cleans up imports and paths.
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd


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
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
HEATWAVE_DEFINITION_DIR = REPO_ROOT / "heatwave_definition"
DATA_DIR = REPO_ROOT / "data"
HEATWAVES_DIR = DATA_DIR / "heatwaves"
CLEANED_DIR = DATA_DIR / "cleaned"

for path in [REPO_ROOT, PREPROCESSING_DIR, HEATWAVE_DEFINITION_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from auxiliary import loadAMF, loadBADM, find_shared_variables
from auxiliary import fit_heatwaves, calculate_moisture
from auxiliary import find_heatwave_overlap
from auxiliary import avg_QAQC_check, minmax_QAQC_check, remove_invalid_heatwaves


pd.set_option("display.max_columns", None)


def run_step2():
    HEATWAVES_DIR.mkdir(parents=True, exist_ok=True)

    # Load in the adjusted historical data
    historical_data = pd.read_csv(CLEANED_DIR / "historical_data_adjusted.csv")
    historical_data = historical_data.iloc[:, 1:]
    
    # Convert the date to a datetime variable
    historical_data["date"] = pd.to_datetime(historical_data["date"])

    # Load in the daily ameriflux data
    AMF_DD = pd.read_csv(CLEANED_DIR / "AMF_DD.csv")
    AMF_DD = AMF_DD.iloc[:, 1:]
    AMF_DD["date"] = pd.to_datetime(AMF_DD["date"])

    # Load in hourly ameriflux data
    AMF_HH = pd.read_csv(CLEANED_DIR / "AMF_HH.csv")
    AMF_HH = AMF_HH.iloc[:, 1:]
    AMF_HH["TIMESTAMP_START"] = pd.to_datetime(AMF_HH["TIMESTAMP_START"])

    # Retrieve daily temperature specifically
    ta = loadAMF(
        path=str(DATA_DIR / "AMFdataDD"),
        skip=[""],
        measures=["TIMESTAMP", "TA_F", "TA_F_QC"],
    )
    
    # Retrieve hourly temperature specifically
    ta_HH = loadAMF(
        path=str(DATA_DIR / "AMFdata_HH"),
        skip=[""],
        measures=["TIMESTAMP_START", "TA_F", "TA_F_QC"],
    )

    # Determine which sites have SWC data available
    shared_swc = find_shared_variables(
        path=str(DATA_DIR / "AMFdataDD"),
        measures=["SWC_F_MDS_1"],
    )
    
    # Load in SWC data, ignoring any sites without swc data available
    sites_wo_swc = list(
        shared_swc["site_presence"][shared_swc["site_presence"].SWC_F_MDS_1 == 0].Site
    )
    swc_data = loadAMF(
        path=str(DATA_DIR / "AMFdataDD"),
        skip=sites_wo_swc,
        measures=["TIMESTAMP", "SWC_F_MDS_1"],
    )
    
    # Load in precipitation data for those sites with swc data available as well
    prec_data = loadAMF(
        path=str(DATA_DIR / "AMFdataDD"),
        skip=sites_wo_swc,
        measures=["TIMESTAMP", "P_F"],
    )

    # Load in the IGBP
    IGBP = loadBADM(
        path=str(DATA_DIR / "BADM"),
        skip=[""],
        column="VARIABLE",
        value="DATAVALUE",
        measure=["IGBP"],
        file_type="xslx",
    )

    ###############################################################################
    # MAX
    ###############################################################################
    # Calculate heatwaves at each site based on maximum temperature
    heatwaves = {}
    for site in swc_data.Site.unique():
        print(site)
        flux_data = AMF_HH[AMF_HH["Site"] == site]
        historical_site_data = historical_data[historical_data["Site"] == site][["date", "hist_max_adj"]]
        site_precip = prec_data[prec_data["Site"] == site].copy()
        site_swc = swc_data[swc_data["Site"] == site].copy()

        site_heatwaves = fit_heatwaves(
            flux_dates=flux_data.TIMESTAMP_START,
            flux_temperature=flux_data.TA_F,
            historical_dates=historical_site_data["date"],
            historical_temperature=historical_site_data["hist_max_adj"],
            quantile_threshold=.95,
            window_length=15,
            threshold_comparison="greater",
            min_heatwave_length=3,
            tolerance=1,
            gap_days_window=8,
            site=site,
            method="max",
        )

        if site_heatwaves is None:
            continue

        # Calculate the moisture averages for each of the heatwaves
        site_heatwave_precip = calculate_moisture(
            timeseries_dates=site_precip.TIMESTAMP,
            timeseries_moisture=site_precip.P_F,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )
        site_heatwave_swc = calculate_moisture(
            timeseries_dates=site_swc.TIMESTAMP,
            timeseries_moisture=site_swc.SWC_F_MDS_1,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )

        heatwaves[site] = site_heatwaves
        heatwaves[site]["precip"] = site_heatwave_precip
        heatwaves[site]["swc"] = site_heatwave_swc

    # Remove heatwaves with low quality temperature data
    heatwaves_qaqc = pd.DataFrame(columns=["start_dates", "end_dates", "duration", "magnitude", "QAQC_flag"])
    for site in list(heatwaves.keys()):
        print(site)
        site_heatwave_dictionary = heatwaves[site]
        dates = ta_HH[ta_HH["Site"] == site].TIMESTAMP_START
        TA = ta_HH[ta_HH["Site"] == site].TA_F
        TA_QAQC = ta_HH[ta_HH["Site"] == site].TA_F_QC
        site_check = minmax_QAQC_check(
            site_heatwave_dictionary, dates, TA, TA_QAQC, heatwave_threshold=.5, method="max"
        )
        if site_check is None:
            continue
        site_check["Site"] = [site] * site_check.shape[0]
        heatwaves_qaqc = pd.concat([heatwaves_qaqc, site_check])

    invalid_heatwaves = heatwaves_qaqc[heatwaves_qaqc["QAQC_flag"] == 1]
    invalid_heatwaves = invalid_heatwaves[["Site", "start_dates", "end_dates"]]
    invalid_heatwaves.columns = ["Site", "start_date", "end_date"]
    heatwaves = remove_invalid_heatwaves(heatwaves_dictionary=heatwaves, invalid_heatwaves=invalid_heatwaves)

    all_precip = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "Site", "Duration"])
    for site in heatwaves.keys():
        site_data = heatwaves[site]["precip"]
        site_data["Site"] = [site] * len(site_data)
        site_data["Duration"] = heatwaves[site]["summary"].duration
        all_precip = pd.concat([all_precip, site_data])

    all_swc = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "Site"])
    for site in heatwaves.keys():
        site_data = heatwaves[site]["swc"]
        site_data["Site"] = [site] * len(site_data)
        all_swc = pd.concat([all_swc, site_data])
    all_swc = all_swc.dropna()

    # Save maximum heatwave dataframes
    with open(HEATWAVES_DIR / "heatwaves_max.pkl", "wb") as f:
        pickle.dump(heatwaves, f)
    all_precip.to_csv(HEATWAVES_DIR / "max_precip.csv")
    all_swc.to_csv(HEATWAVES_DIR / "max_swc.csv")
    invalid_heatwaves.to_csv(HEATWAVES_DIR / "invalid_heatwaves_max.csv")

    ###############################################################################
    # MEAN
    ###############################################################################
    # Calculate heatwaves for each site based on the average temperature calculations
    heatwaves_mean = {}
    for site in swc_data.Site.unique():
        historical_site_data = historical_data[historical_data["Site"] == site]
        flux_data = AMF_DD[AMF_DD["Site"] == site].copy()
        site_precip = prec_data[prec_data["Site"] == site].copy()
        site_swc = swc_data[swc_data["Site"] == site].copy()

        site_heatwaves = fit_heatwaves(
            flux_dates=flux_data.date,
            flux_temperature=flux_data.TA_F,
            historical_dates=historical_site_data.date,
            historical_temperature=historical_site_data.hist_mean_adj,
            quantile_threshold=.97,
            window_length=15,
            threshold_comparison="greater",
            min_heatwave_length=3,
            tolerance=1,
            gap_days_window=8,
            site=site,
            method="mean",
        )

        if site_heatwaves is None:
            continue

        site_heatwave_precip = calculate_moisture(
            timeseries_dates=site_precip.TIMESTAMP,
            timeseries_moisture=site_precip.P_F,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )
        site_heatwave_swc = calculate_moisture(
            timeseries_dates=site_swc.TIMESTAMP,
            timeseries_moisture=site_swc.SWC_F_MDS_1,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )

        heatwaves_mean[site] = site_heatwaves
        heatwaves_mean[site]["precip"] = site_heatwave_precip
        heatwaves_mean[site]["swc"] = site_heatwave_swc

    # Remove heatwaves with low quality temperature data
    heatwave_QAQC = pd.DataFrame(columns=["Site", "start_date", "end_date", "QAQC_percentage", "heatwave_invalidity"])
    for site in list(heatwaves_mean.keys()):
        site_heatwave_dictionary = heatwaves_mean[site]
        dates = ta[ta["Site"] == site].TIMESTAMP
        TA_QAQC = ta[ta["Site"] == site].TA_F_QC
        site_QAQC = avg_QAQC_check(
            site_heatwave_dictionary, dates, TA_QAQC, QAQC_threshold=.5, heatwave_threshold=.75
        )
        if site_QAQC is None:
            continue
        site_QAQC["Site"] = [site] * site_QAQC.shape[0]
        heatwave_QAQC = pd.concat([heatwave_QAQC, site_QAQC])

    invalid_heatwaves_mean = heatwave_QAQC[heatwave_QAQC["heatwave_invalidity"] == 1]
    invalid_heatwaves_mean = invalid_heatwaves_mean[["Site", "start_date", "end_date"]]
    heatwaves_mean = remove_invalid_heatwaves(
        heatwaves_dictionary=heatwaves_mean, invalid_heatwaves=invalid_heatwaves_mean
    )

    all_precip_mean = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "moisture_variability", "Site", "Duration"])
    for site in heatwaves_mean.keys():
        site_data = heatwaves_mean[site]["precip"]
        site_data["Site"] = [site] * len(site_data)
        site_data["Duration"] = heatwaves_mean[site]["summary"].duration
        all_precip_mean = pd.concat([all_precip_mean, site_data])

    all_swc_mean = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "moisture_variability", "Site"])
    for site in heatwaves_mean.keys():
        site_data = heatwaves_mean[site]["swc"]
        site_data["Site"] = [site] * len(site_data)
        all_swc_mean = pd.concat([all_swc_mean, site_data])
    all_swc_mean = all_swc_mean.dropna()

    # Save the heatwaves
    with open(HEATWAVES_DIR / "heatwaves_mean.pkl", "wb") as f:
        pickle.dump(heatwaves_mean, f)
    all_precip_mean.to_csv(HEATWAVES_DIR / "mean_precip.csv")
    all_swc_mean.to_csv(HEATWAVES_DIR / "mean_swc.csv")
    invalid_heatwaves_mean.to_csv(HEATWAVES_DIR / "invalid_heatwaves_mean.csv")

    ###############################################################################
    # MIN
    ###############################################################################
    # Calculate heatwaves for each site based on minimum temperatures
    heatwaves_min = {}
    for site in swc_data.Site.unique():
        flux_data = AMF_HH[AMF_HH["Site"] == site]
        historical_site_data = historical_data[historical_data["Site"] == site]
        site_precip = prec_data[prec_data["Site"] == site]
        site_swc = swc_data[swc_data["Site"] == site]

        print(f"I'm currently on site {site}.")
        site_heatwaves = fit_heatwaves(
            flux_dates=flux_data.TIMESTAMP_START,
            flux_temperature=flux_data.TA_F,
            historical_dates=historical_site_data.date,
            historical_temperature=historical_site_data.hist_min_adj,
            quantile_threshold=.97,
            min_heatwave_length=3,
            tolerance=1,
            gap_days_window=8,
            site=site,
            method="min",
        )

        if site_heatwaves is None:
            continue

        # Calculate moisture conditions during the heatwave
        site_heatwave_precip = calculate_moisture(
            timeseries_dates=site_precip.TIMESTAMP,
            timeseries_moisture=site_precip.P_F,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )
        site_heatwave_swc = calculate_moisture(
            timeseries_dates=site_swc.TIMESTAMP,
            timeseries_moisture=site_swc.SWC_F_MDS_1,
            start_dates=site_heatwaves["start_dates"],
            end_dates=site_heatwaves["end_dates"],
        )

        heatwaves_min[site] = site_heatwaves
        heatwaves_min[site]["precip"] = site_heatwave_precip
        heatwaves_min[site]["swc"] = site_heatwave_swc

    # Remove heatwaves with low qualit temperature data
    heatwaves_qaqc = pd.DataFrame(columns=["start_dates", "end_dates", "duration", "magnitude", "QAQC_flag"])
    for site in list(heatwaves_min.keys()):
        print(site)
        site_heatwave_dictionary = heatwaves_min[site]
        dates = ta_HH[ta_HH["Site"] == site].TIMESTAMP_START
        TA = ta_HH[ta_HH["Site"] == site].TA_F
        TA_QAQC = ta_HH[ta_HH["Site"] == site].TA_F_QC
        site_check = minmax_QAQC_check(
            site_heatwave_dictionary, dates, TA, TA_QAQC, heatwave_threshold=.75, method="min"
        )
        if site_check is None:
            continue
        site_check["Site"] = [site] * site_check.shape[0]
        heatwaves_qaqc = pd.concat([heatwaves_qaqc, site_check])

    invalid_heatwaves_min = heatwaves_qaqc[heatwaves_qaqc["QAQC_flag"] == 1]
    invalid_heatwaves_min = invalid_heatwaves_min[["Site", "start_dates", "end_dates"]]
    invalid_heatwaves_min.columns = ["Site", "start_date", "end_date"]
    heatwaves_min = remove_invalid_heatwaves(
        heatwaves_dictionary=heatwaves_min, invalid_heatwaves=invalid_heatwaves_min
    )
    
    # Calculate average moisture conditions during the heatwave
    all_precip_min = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "Site", "Duration"])
    for site in heatwaves_min.keys():
        site_data = heatwaves_min[site]["precip"]
        site_data["Site"] = [site] * len(site_data)
        site_data["Duration"] = heatwaves_min[site]["summary"].duration
        all_precip_min = pd.concat([all_precip_min, site_data])

    all_swc_min = pd.DataFrame(columns=["start_date", "end_date", "moisture_average", "moisture_total", "Site"])
    for site in heatwaves_min.keys():
        site_data = heatwaves_min[site]["swc"]
        site_data["Site"] = [site] * len(site_data)
        all_swc_min = pd.concat([all_swc_min, site_data])
    all_swc_min = all_swc_min.dropna()

    # Save the heatwave
    with open(HEATWAVES_DIR / "heatwaves_min.pkl", "wb") as f:
        pickle.dump(heatwaves_min, f)
    all_precip_min.to_csv(HEATWAVES_DIR / "min_precip.csv")
    all_swc_min.to_csv(HEATWAVES_DIR / "min_swc.csv")
    invalid_heatwaves_min.to_csv(HEATWAVES_DIR / "invalid_heatwaves_min.csv")

    ###############################################################################
    # UNPACK
    ###############################################################################
    
    max_sites = list(heatwaves.keys())
    mean_sites = list(heatwaves_mean.keys())
    min_sites = list(heatwaves_min.keys())

    heatwaves_df = pd.DataFrame(columns=["Site", "Method", "start_dates", "end_dates", "duration", "QAQC_flag"])
    for site in max_sites:
        site_df = heatwaves[site]["summary"]
        site_df["Method"] = "Max"
        heatwaves_df = pd.concat([heatwaves_df, site_df])

    for site in mean_sites:
        site_df = heatwaves_mean[site]["summary"]
        site_df["Site"] = [site] * len(heatwaves_mean[site]["summary"])
        site_df["Method"] = "Mean"
        heatwaves_df = pd.concat([heatwaves_df, site_df])

    for site in min_sites:
        site_df = heatwaves_min[site]["summary"]
        site_df["Method"] = "Min"
        site_df["Site"] = site
        site_df["QAQC_flag"] = np.nan
        heatwaves_df = pd.concat([heatwaves_df, site_df])

    heatwaves_df.to_csv(HEATWAVES_DIR / "heatwaves_df.csv")

    ###############################################################################
    # COMBINE
    ###############################################################################
    # Find sites that have all minimum, maximum, and mean heatwaves calculated
    max_sites = list(heatwaves.keys())
    min_sites = list(heatwaves_min.keys())
    mean_sites = list(heatwaves_mean.keys())
    all_sites = list(set(max_sites) & set(min_sites) & set(mean_sites) & set(historical_data.Site.unique()))

    # Reduce min, max, and mean heatwave dataframes to those sites that have all
    # 3 forms of heatwaves defined
    reduced_heatwaves = {k: heatwaves[k] for k in all_sites if k in heatwaves}
    reduced_heatwaves_min = {k: heatwaves_min[k] for k in all_sites if k in heatwaves_min}
    reduced_heatwaves_mean = {k: heatwaves_mean[k] for k in all_sites if k in heatwaves_mean}

    # Define overlapping heatwaves
    all_heatwaves = find_heatwave_overlap(
        min_heatwaves=reduced_heatwaves_min,
        max_heatwaves=reduced_heatwaves,
        avg_heatwaves=reduced_heatwaves_mean,
    )

    # Save this as a pkl file
    with open(HEATWAVES_DIR / "all_heatwaves.pkl", "wb") as f:
        pickle.dump(all_heatwaves, f)

    # Create an expanded version of the heatwave dataframe
    # Expanding heatwaves so that 1 row = 1 day of a heatwave
    all_heatwaves_df = pd.DataFrame(columns=["Site", "start_dates", "end_dates", "top_heatwave"])
    for site in list(all_heatwaves.keys()):
        this_site_heatwaves = all_heatwaves[site]["heatwave_type"]
        this_site_heatwaves["Site"] = [site] * this_site_heatwaves.shape[0]
        all_heatwaves_df = pd.concat([all_heatwaves_df, this_site_heatwaves])

    # Merge IGBP and month onto this expanded dataframe
    all_heatwaves_df = pd.merge(all_heatwaves_df, IGBP, on="Site", how="left").drop_duplicates()
    all_heatwaves_df["Month"] = all_heatwaves_df["start_dates"].dt.month

    # Calculate the season of the dataframe from Month (this is a rough interpretation)
    season = []
    for i in range(all_heatwaves_df.shape[0]):
        month = all_heatwaves_df.iloc[i]["Month"]
        if 3 <= month <= 5:
            season.append("Spring")
        elif 6 <= month <= 8:
            season.append("Summer")
        elif 9 <= month <= 11:
            season.append("Fall")
        else:
            season.append("Winter")
    all_heatwaves_df["Season"] = season

    # Also expand the moisture during heatwaves
    # One row = moisture for one day of a heatwave
    all_moisture = pd.DataFrame(columns=[
        "Site", "start_dates", "end_dates",
        "prec_average", "prec_total", "prec_variability",
        "swc_average", "swc_total", "swc_variability",
    ])
    
    for site in all_heatwaves.keys():
        print(site)
        site_swc = swc_data[swc_data["Site"] == site]
        site_prec = prec_data[prec_data["Site"] == site]
        site_heatwaves = all_heatwaves_df[all_heatwaves_df["Site"] == site]

        heatwave_swc = calculate_moisture(
            timeseries_dates=site_swc.TIMESTAMP,
            timeseries_moisture=site_swc.SWC_F_MDS_1,
            start_dates=site_heatwaves.start_dates,
            end_dates=site_heatwaves.end_dates,
        )
        heatwave_swc.columns = ["start_dates", "end_dates", "swc_average", "swc_total", "swc_variability"]
        mask = heatwave_swc["swc_total"] < 0
        heatwave_swc.loc[mask, ["swc_average", "swc_total", "swc_variability"]] = np.nan

        heatwave_prec = calculate_moisture(
            timeseries_dates=site_prec.TIMESTAMP,
            timeseries_moisture=site_prec.P_F,
            start_dates=site_heatwaves.start_dates,
            end_dates=site_heatwaves.end_dates,
        )
        heatwave_prec.columns = ["start_dates", "end_dates", "prec_average", "prec_total", "prec_variability"]
        mask = heatwave_prec["prec_total"] < 0
        heatwave_prec.loc[mask, ["prec_average", "prec_total", "prec_variability"]] = np.nan

        site_heatwave_moisture = pd.merge(
            heatwave_prec, heatwave_swc, on=["start_dates", "end_dates"], how="inner"
        )
        site_heatwave_moisture["Site"] = [site] * len(site_heatwave_moisture)
        all_moisture = pd.concat([all_moisture, site_heatwave_moisture])

    # Merge heatwave and moisture expansions
    all_heatwaves_df = pd.merge(
        all_heatwaves_df, all_moisture, on=["Site", "start_dates", "end_dates"], how="left"
    )
    
    # Calculate and add the duration of heatwaves
    all_heatwaves_df["duration"] = all_heatwaves_df.end_dates - all_heatwaves_df.start_dates
    all_heatwaves_df["duration"] = all_heatwaves_df["duration"].dt.days + 1
    all_heatwaves_df = all_heatwaves_df[all_heatwaves_df["duration"] > 2]
    all_heatwaves_df.to_csv(HEATWAVES_DIR / "all_heatwaves_df.csv")

    # Save all dataframes to csv or pkl
    print("Step 2 completed successfully.")
    print(HEATWAVES_DIR / "heatwaves_max.pkl")
    print(HEATWAVES_DIR / "heatwaves_mean.pkl")
    print(HEATWAVES_DIR / "heatwaves_min.pkl")
    print(HEATWAVES_DIR / "heatwaves_df.csv")
    print(HEATWAVES_DIR / "all_heatwaves.pkl")
    print(HEATWAVES_DIR / "all_heatwaves_df.csv")


if __name__ == "__main__":
    run_step2()

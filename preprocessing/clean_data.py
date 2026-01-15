'''
This script will load in and provide the cleaned data for all performed analysis.
All analysis should be performed using this cleaned data.
'''
import os
os.chdir("/Users/marleeyork/Documents/project2/preprocessing")
from load_data import *
from heatwave_QAQC import *
from PRISM_ERA_QAQC import *
from adjusting_data import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
pd.set_option('display.max_columns',300)
pd.set_option('display.max_rows',100)

# Loading in the AmeriFlux data across all sites with soil water content
# This automatically loads selected for columns, defined in loadAMF
df = loadAMF(path='/Users/marleeyork/Documents/project2/data/AMFdataDD',
                 measures=['TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF'])

df_hourly = loadAMF(path='/Users/marleeyork/Documents/project2/data/AMFdata_HH',
                 measures=['TIMESTAMP_START','TA_F'])

# Load the IGBP data and merge to df
# site_data = pd.read_csv("/Users/marleeyork/Documents/project2/data/site_list.csv",encoding='latin1')
# IGBP = site_data[['Site ID','Vegetation Abbreviation (IGBP)']]
# IGBP.columns = ['Site','IGBP']
# df = pd.merge(df,IGBP,on="Site",how="inner").drop_duplicates()

# Loading IGBP for the long list of sites
IGBP = loadBADM(path="/Users/marleeyork/Documents/project2/data/BADM",skip=[''],
                column='VARIABLE',value='DATAVALUE',measure=['IGBP'],file_type='xslx')
df = pd.merge(df,IGBP,on='Site',how='left').drop_duplicates()
df_hourly = pd.merge(df_hourly,IGBP,on='Site',how='left').drop_duplicates()

# Drop any croplands from the AmeriFlux data
df = df[df['IGBP']!='CRO']
df_hourly = df_hourly[df_hourly['IGBP']!='CRO']

###############################################################################
##                        Historical Data QAQC                               ##
###############################################################################
# Load in the data
ERA_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmax_data.csv")
ERA_min = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmin_data.csv")
ERA_mean = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmean_data.csv")
PRISM_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_climate_data_tmax.csv")
PRISM_min = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_tmin.csv")
PRISM_mean = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_tmean.csv")
AMF_mean = df[['Site','TIMESTAMP','TA_F']]
AMF_mean.columns = ['Site','date','TA_F']


# Transform ERA temperature from Kelvin to C
ERA_max['ERA_TA'] = ERA_max["t2m"] - 273.15
ERA_min['ERA_TA'] = ERA_min["t2m"] - 273.15
ERA_mean['ERA_TA'] = ERA_mean["t2m"] - 273.15

# Transform dates into datetime variables
ERA_max['date'] = pd.to_datetime(ERA_max.valid_time)
ERA_min['date'] = pd.to_datetime(ERA_min.valid_time)
ERA_mean['date'] = pd.to_datetime(ERA_mean.valid_time)
PRISM_max['date'] = pd.to_datetime(PRISM_max.date)
PRISM_min['date'] = pd.to_datetime(PRISM_min.date)
PRISM_mean['date'] = pd.to_datetime(PRISM_mean.date)

# Reduce down to columns of interest
ERA_max = ERA_max[['Site','date','ERA_TA']]
ERA_min = ERA_min[['Site','date','ERA_TA']]
ERA_mean = ERA_mean[['Site','date','ERA_TA']]

# Drop any sites that aren't in df
included_sites = df.Site.unique()
included_sites = np.insert(included_sites,0,'date')
ERA_max = ERA_max[ERA_max['Site'].isin(included_sites)]
ERA_min = ERA_min[ERA_min['Site'].isin(included_sites)]
ERA_mean = ERA_mean[ERA_mean['Site'].isin(included_sites)]
PRISM_included_sites = included_sites[pd.Series(included_sites).isin(PRISM_mean.columns)]
PRISM_max = PRISM_max[PRISM_included_sites]
PRISM_min = PRISM_min[PRISM_included_sites]
PRISM_mean = PRISM_mean[PRISM_included_sites]

# Findig which sites have missing values in PRISM data
search_value = -9999
missing_max = []
missing_min = []
missing_avg = []
for col in PRISM_max.columns:
    # Check if the search_value exists in the current column
    if PRISM_max[col].astype(str).str.contains(str(search_value)).any():
        missing_max.append(col)

for col in PRISM_min.columns:
    if PRISM_min[col].astype(str).str.contains(str(search_value)).any():
        missing_min.append(col)

for col in PRISM_mean.columns:
    if PRISM_mean[col].astype(str).str.contains(str(search_value)).any():
        missing_avg.append(col)
        
PRISM_max = PRISM_max.drop(columns=missing_max)
PRISM_min = PRISM_min.drop(columns=missing_min)
PRISM_mean = PRISM_mean.drop(columns=missing_avg)

# Pivot longer
PRISM_max = pd.melt(PRISM_max,id_vars='date',var_name='Site',value_name='PRISM_TA')
PRISM_min = pd.melt(PRISM_min,id_vars='date',var_name='Site',value_name='PRISM_TA')
PRISM_mean = pd.melt(PRISM_mean,id_vars='date',var_name='Site',value_name='PRISM_TA')

# Based on low correlation investigations (done in QAQC.py), we now drop 
# certain sites
removing_sites = ['US-CAK',"CA-Ca1","US-xHE","US-xDJ","US-ICt","US-Rpf","US-xNW",
                  "US-ICh","US-Hn2","US-EML","US-BZS","US-NGC","US-Cop","CA-SCC",
                  "CA-NS2","US-SP1","US-Ho1","US-Me2"]

ERA_max = ERA_max[~ERA_max.Site.isin(removing_sites)]
ERA_min = ERA_min[~ERA_min.Site.isin(removing_sites)]
ERA_mean = ERA_mean[~ERA_mean.Site.isin(removing_sites)]
PRISM_max = PRISM_max[~PRISM_max.Site.isin(removing_sites)]
PRISM_min = PRISM_min[~PRISM_min.Site.isin(removing_sites)]
PRISM_mean = PRISM_mean[~PRISM_mean.Site.isin(removing_sites)]

# Restrict AMF sites to those we have long-term data for
df_hourly = df_hourly[df_hourly.Site.isin(ERA_max.Site.unique())]
df = df[df.Site.isin(ERA_max.Site.unique())]
df.columns = ['date','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF',
             'GPP_NT_VUT_REF','Site','IGBP']

# Quick check that all of the datasets have the correct number of sites
len(df_hourly.Site.unique())
len(df.Site.unique())
len(ERA_max.Site.unique())
len(ERA_min.Site.unique())
len(ERA_mean.Site.unique())
len(PRISM_max.Site.unique())
len(PRISM_min.Site.unique())
len(PRISM_mean.Site.unique())

# Get start and end dates for historical data based on shared dates between PRISM and ERA
start_date = max([ERA_mean.date.min(),PRISM_mean.date.min()])
end_date = min([ERA_mean.date.max(),PRISM_mean.date.max()])

# This is how long historical data should be for all PRISM or ERA data selected at a site
len(pd.date_range(start_date,end_date))

# Determine which site is better (ERA or PRISM) for each site
historical_data_max, data_source_max = return_best_data(AMF_data_all = df_hourly[['Site','TIMESTAMP_START','TA_F']],
                                                        ERA_data_all = ERA_max[['Site','date','ERA_TA']],
                                                        PRISM_data_all = PRISM_max[['Site','date','PRISM_TA']],
                                                        temperature_type = 'max',
                                                        start_date=start_date,
                                                        end_date=end_date)
historical_data_min, data_source_min = return_best_data(AMF_data_all = df_hourly[['Site','TIMESTAMP_START','TA_F']],
                                                        ERA_data_all = ERA_min[['Site','date','ERA_TA']],
                                                        PRISM_data_all = PRISM_min[['Site','date','PRISM_TA']],
                                                        temperature_type = 'min',
                                                        start_date=start_date,
                                                        end_date=end_date)
historical_data_mean, data_source_mean = return_best_data(AMF_data_all = df[['Site','date','TA_F']],
                                                        ERA_data_all = ERA_mean[['Site','date','ERA_TA']],
                                                        PRISM_data_all = PRISM_mean[['Site','date','PRISM_TA']],
                                                        temperature_type = 'average',
                                                        start_date=start_date,
                                                        end_date=end_date)

# Now we save this data for future heatwave definition use
os.chdir("/Users/marleeyork/Documents/project2/data/cleaned/")
historical_data_max.to_csv("historical_data_max.csv")
historical_data_min.to_csv("historical_data_min.csv")
historical_data_mean.to_csv("historical_data_mean.csv")
df.to_csv("AMF_DD.csv")
df_hourly.to_csv("AMF_HH.csv")


###############################################################################
##                        Bad Data Edits                                     ##
###############################################################################

# Now we are going to adjust the PRISM/ERA historical data based on regressions
# fit to historical data by AmeriFlux data. Not all sites will have enough data or
# low enough variance to adjust the data. This is removing bias so that heatwaves
# are based on more accurate historical data. 
# Load in daily data
AMF = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/AMF_DD.csv")
AMF.date = pd.to_datetime(AMF.date)
AMF_HH = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/AMF_HH.csv")
AMF_HH.TIMESTAMP_START = pd.to_datetime(AMF_HH.TIMESTAMP_START)

# Calculate daily minimum temperatures
AMF_min = AMF_HH.groupby("Site").apply(lambda g: find_min_temperatures(g.TIMESTAMP_START, g.TA_F)).reset_index()
AMF_min = AMF_min[['Site','date','min_temperature']]

# Calculate daily maximum temperatures
AMF_max = AMF_HH.groupby("Site").apply(lambda g: find_max_temperatures(g.TIMESTAMP_START, g.TA_F)).reset_index()
AMF_max = AMF_max[['Site','date','max_temperature']]

# Load in the historical data for mean, max, and min temperatures
hist_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_max.csv")
hist_min = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_min.csv")
hist_avg = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_mean.csv")
AMF = AMF.iloc[:,1:]
AMF = AMF[['Site','date','TA_F']]
hist_max = hist_max.iloc[:,1:]
hist_min = hist_min.iloc[:,1:]
hist_avg = hist_avg.iloc[:,1:]
hist_max.columns = ['Site','date','hist_max','Source_max']
hist_min.columns = ['Site','date','hist_min','Source_min']
hist_avg.columns = ['Site','date','hist_mean','Source_mean']
hist_max.date = pd.to_datetime(hist_max.date)
hist_min.date = pd.to_datetime(hist_min.date)
hist_avg.date = pd.to_datetime(hist_avg.date)

# Merge all these dataframes!
# Historical data has no missing data
historical_data = pd.merge(hist_max, hist_min, on=['Site','date'], how='inner')
historical_data = pd.merge(historical_data, hist_avg, on=['Site','date'], how='inner')
AMF_data = pd.merge(AMF, AMF_min, on=['Site','date'], how='inner')
AMF_data = pd.merge(AMF_data, AMF_max, on=['Site','date'], how='inner')
AMF_data = AMF_data.dropna()

# Get the adjusted data
results = find_historical_bias(AMF_data)
validity_df = can_we_correct(results,n=365*5,r2=.9)
adjusted_data = make_adjustment(historical_data,results,validity_df)

# Getting entire dictionary
adjustment_dict = adjust_historical_data(historical_data=historical_data,
                                         AMF_data=AMF_data,
                                         n=365*5, 
                                         r2=.9)

validity_df = adjustment_dict["validity"]
reg_results = adjustment_dict["regressions"]
hist_data_adj = adjustment_dict["adjustments"]

# We can visually plot these differences now
AMF_data = pd.merge(AMF_data, historical_data,on=['Site','date'],how='inner')
for site in hist_data_adj.Site.unique():
    print(site)
    # Grabbing the site data over the AMF dates
    site_data = AMF_data[AMF_data.Site==site]
    site_data_adj = hist_data_adj[hist_data_adj.Site==site]
    site_data = pd.merge(site_data, site_data_adj, on=['Site','date'],how='left')
    
    # Plotting
    fig, ax = plt.subplots(3,1,figsize=(6,10))
    ax[0].scatter(site_data.TA_F, site_data.hist_mean, color="red", s=.5)
    ax[0].scatter(site_data.TA_F, site_data.hist_mean_adj, color="blue", s=.5)
    ax[0].plot([-30,30],[-30,30],color="black")
    ax[0].set_ylabel("Historical Data (PRISM/ERA)")
    ax[0].set_title(f"{site} Mean Temperature")
    ax[1].scatter(site_data.max_temperature, site_data.hist_max, color="red", s=.5)
    ax[1].scatter(site_data.max_temperature, site_data.hist_max_adj, color="blue", s=.5)
    ax[1].plot([-30,30],[-30,30],color="black")
    ax[1].set_ylabel("Historical Data (PRISM/ERA)")
    ax[1].set_title(f"{site} Max Temperature")
    ax[2].scatter(site_data.min_temperature, site_data.hist_min, color="red", s=.5, label="Original Data")
    ax[2].scatter(site_data.min_temperature, site_data.hist_min_adj, color="blue", s=.5, label="Adjusted Data")
    ax[2].plot([-30,30],[-30,30],color="black")
    ax[2].set_ylabel("Historical Data (PRISM/ERA)")
    ax[2].set_xlabel("AmeriFlux Temperature")
    ax[2].set_title(f"{site} Min Temperature")
    ax[2].legend()  
    plt.tight_layout()
    plt.show()
    
    input("Press [enter] to continue...")

# Save the historical data to a csv
hist_data_adj.to_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_adjusted.csv")

###############################################################################
##                         Loading in SWC data                               ##
###############################################################################

# First, find the sites that actually have SWC measures
swc_measures = ["SWC_F_MDS_1","SWC_F_MDS_2","SWC_F_MDS_3","SWC_F_MDS_4","SWC_F_MDS_5",
                "SWC_F_MDS_1_QC","SWC_F_MDS_2_QC","SWC_F_MDS_3_QC","SWC_F_MDS_4_QC",
                "SWC_F_MDS_5_QC"]
shared_swc = find_shared_variables('/Users/marleeyork/Documents/project2/data/cleaned/AMF_DD.csv',swc_measures)

site_presence = shared_swc['site_presence']
site_presence[site_presence['Site'].isin(['US-Ho1','US-Syv','US-Kon'])]

###############################################################################
##                        Bad Data Edits                                     ##
###############################################################################

# Drop any rows with values that are -9999 (new shape is (190298,11))
mask = df.apply(lambda col: col == -9999).any(axis=1)
df = df[~mask]

'''
# If GPP is negative, set it to 0
df.loc[df['GPP_NT_VUT_REF']<0,'GPP_NT_VUT_REF'] = 0

# Now we will do site specific filtering
# Initializing a drop list that we will fill with indices to remove
drop_list = []

# SRG days before 05/02/2018 (drop_list.len == 70)
drop_list.extend(df[(df['Site'] == 'US-SRG') & (df['TIMESTAMP'] < '2008-05-02')].index)

# SRM days after 2017-05-01 (drop_list.len == 2871)
drop_list.extend(df[(df['Site'] == 'US-SRM') & (df['TIMESTAMP'] > '2017-05-01')].index)

# LP1 days after 2016-12-01 (drop_list.len == 3997)
drop_list.extend(df[(df['Site'] == 'CA-LP1') & (df['TIMESTAMP'] > '2016-12-01')].index)

# Me2 days after 2021-10-01 (drop_list.len == 4295)
drop_list.extend(df[(df['Site'] == 'US-Me2') & (df['TIMESTAMP'] > '2021-10-01')].index)

# BZS days prior to 2015-01-01 (drop_list.len == 4582)
drop_list.extend(df[(df['Site'] == 'US-BZS') & (df['TIMESTAMP'] < '2015-01-01')].index)

# Syv days prior to 2012-01-01 (drop_list.len == 6926)
drop_list.extend(df[(df['Site'] == 'US-Syv') & (df['TIMESTAMP'] < '2012-01-01')].index)

# KFS days prior to 2009-01-01 (drop_list.len == 7292)
drop_list.extend(df[(df['Site'] == 'US-KFS') & (df['TIMESTAMP'] < '2009-01-01')].index)

# Ton days after to 2020-05-01 (drop_list.len == 8997)
drop_list.extend(df[(df['Site'] == 'US-Ton') & (df['TIMESTAMP'] > "2020-05-01")].index)

# Whs days before 2016
drop_list.extend(df[(df['Site']=='US-Whs') & (df['TIMESTAMP'] > "2016-01-01")].index)

# Ho1 days before 2008
drop_list.extend(df[(df['Site']=='US-Ho1') & (df['TIMESTAMP'] < "2008-01-01")].index)

# These are corrections made to cropland (CRO) sites, which have been removed
# Mo1 days before 2016-01-01 (drop_list.len == 2978)
# drop_list.extend(df[(df['Site'] == 'US-Mo1') & (df['TIMESTAMP'] < '2016-01-01')].index)

# Drop the drop list! (df.shape == (159306,10))
df = df.drop(index=drop_list,errors='ignore')

# Now we need to create a date column in the hourly data that has the timestamp removed
df_hourly['TIMESTAMP'] = df_hourly['TIMESTAMP_START'].dt.strftime("%-Y-%-m-%d")
df_hourly['TIMESTAMP'] = pd.to_datetime(df_hourly['TIMESTAMP'])

# Initialize new dataframe
df_HH = pd.DataFrame(columns=['TIMESTAMP_START','TA_F','Site'])
# Loop through each site
for site in df_hourly.Site.unique():
    # Find all the dates in the daily data for that site
    dates = df[df['Site']==site]['TIMESTAMP']
    # Isolate these dates in the half hourly data
    site_keep = df_hourly[(df_hourly['Site']==site) & (df_hourly['TIMESTAMP'].isin(dates))]
    # Add these days to the dataframe
    df_HH = pd.concat([df_HH,site_keep[['TIMESTAMP_START','TA_F','Site']]])
    
# USE df_HH FOR ALL HOURLY ANALYSIS
'''

###############################################################################
##                        Replacing SWC Edits                                ##
###############################################################################
'''
The following code selects sites that have bad SWC_F_MDS_1 data and replaces it
with another depth of SWC. A categorical variables "SWC_depth" is added that indicates
the depth of SWC value we are using for a given site. So far, this has only been
useful for site US-Ho1.

At the end of the day, the data for Syv and Kon came out even worse doing this,
so they are going to be removed. If we need to do this with other sites in the future,
then we can go ahead and use this code framework to reproduce integrating different
SWC variables.
'''
'''
# Check what SWC variables these three files have
swc_measures = ["SWC_F_MDS_1","SWC_F_MDS_2","SWC_F_MDS_3","SWC_F_MDS_4","SWC_F_MDS_5",
                "SWC_F_MDS_1_QC","SWC_F_MDS_2_QC","SWC_F_MDS_3_QC","SWC_F_MDS_4_QC",
                "SWC_F_MDS_5_QC"]

shared_swc = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdataDD',swc_measures)
site_presence = shared_swc['site_presence']
site_presence[site_presence['Site'].isin(['US-Ho1','US-Syv','US-Kon'])]

# Find filepaths for for US-Ho1, US-Syv, and US-Kon
Ho1_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Ho1_FLUXNET_SUBSET_DD_1996-2023_3-6.csv"
Kon_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Kon_FLUXNET_SUBSET_DD_2004-2019_5-7.csv"
Syv_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Syv_FLUXNET_SUBSET_DD_2001-2023_4-6.csv"

# Load in the dataset with all SWC variables of interest (starting with SWC_F_MDS_2 for now)
Ho1_SWC = loadAMFFile(this_file=Ho1_path, measures=["TIMESTAMP","SWC_F_MDS_2"])
Kon_SWC = loadAMFFile(this_file=Kon_path,measures=["TIMESTAMP","SWC_F_MDS_2"])
Syv_SWC = loadAMFFile(this_file=Syv_path,measures=["TIMESTAMP","SWC_F_MDS_3"])

# Add site label to each
Ho1_SWC['Site'] = ['US-Ho1'] * Ho1_SWC.shape[0]
Kon_SWC['Site'] = ['US-Kon'] * Kon_SWC.shape[0]
Syv_SWC['Site'] = ['US-Syv'] * Syv_SWC.shape[0]

# Concatenate these into one dataframe
swc2_data = pd.concat([Ho1_SWC,Kon_SWC])
swc3_data = Syv_SWC

# Create two other dataframes for sites with the other two depths of soil water data
df_swc = df[df['Site'].isin(['US-Ho1','US-Kon'])]
df_swc_Syv = df[df['Site'].isin(['US-Syv'])]

# Merge the two dataframes together
df_swc = pd.merge(df_swc,swc2_data,on=['Site','TIMESTAMP'],how='inner')
df_swc_Syv = pd.merge(df_swc_Syv,swc3_data,on=['Site','TIMESTAMP'],how='inner')

# Adding a categorical variable that is soil water depth we are going to use for that site
df_swc['SWC_depth'] = ['2'] * df_swc.shape[0]
df_swc_Syv['SWC_depth'] = ['3'] * df_swc_Syv.shape[0]

# Remove these 3 sites from the overall dataframe
df = df[~df['Site'].isin(['US-Ho1','US-Kon','US-Syv'])]

# Add a soil water depth identifier
df['SWC_depth'] = ['1'] * df.shape[0]

# Remove SWC_F_MDS_1 from the 3 problem sites df
df_swc = df_swc.drop(columns=['SWC_F_MDS_1'])
df_swc_Syv = df_swc_Syv.drop(columns=['SWC_F_MDS_1'])

# Rename the SWC variables to a neutral name
df = df.rename(columns={'SWC_F_MDS_1':'SWC'})
df_swc = df_swc.rename(columns={'SWC_F_MDS_2': 'SWC'})
df_swc_Syv = df_swc_Syv.rename(columns={'SWC_F_MDS_3':'SWC'})

# Check that all the columns align
df_swc.columns.isin(df.columns)
df.columns.isin(df_swc.columns)
df_swc.columns.isin(df_swc_Syv.columns)
df.columns.isin(df_swc_Syv.columns)
df_swc_Syv.columns.isin(df.columns)
df_swc_Syv.columns.isin(df_swc.columns)

# Concatenate the two dataframes
df = pd.concat([df,df_swc,df_swc_Syv])

# Removing Syv and Kon since these other SWC values did not help
df = df[~df['Site'].isin(['US-Kon','US-Syv'])]
'''
'''
In this script, I will do QAQC including:
    - inspecting site variable values and removing missing areas
    - correcting negative values that shouldn't be there
    - comparing temperature values in PRISM and AmeriFlux data for consistent
    
Currently, I am compring quantiles between PRISM and AMF for max temperatures. 
They align at most the sites, except two, which I am investigating month specific 
misalignment. Next I need to do this for minimum and average temperatures, but 
first I will need to acquire the PRISM for these measrements.

Also, we need PRISM data for Canada too.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
pd.set_option("display.max_rows", 150)

###############################################################################
##                                  QAQC                                     ##
###############################################################################
'''
The following code will plot timeseries of important variables to investigate
any issues in the data, or sites that need to be dropped.
'''
vars_to_plot = [
    "TA_F",
    "SW_IN_F",
    "VPD_F",
    "P_F",
    "GPP_NT_VUT_REF"
]

for site in df.Site.unique():
    flux_data = df[(df['Site']==site)]
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))
    ax = ax.flatten()

    for a, var in zip(ax, vars_to_plot):
        a.scatter(flux_data.TIMESTAMP, flux_data[var], s=0.5)
        a.set_title(f"{site} {var}", fontsize=7)
        a.set_xlabel("Date", fontsize=6)
        a.tick_params(axis='x', rotation=45, labelsize=6)
        a.tick_params(axis='y', labelsize=6)
        #a.axvline(x=pd.to_datetime("2020-05-01"),color="red")


    plt.tight_layout()
    plt.show()
    
    input("Press [Enter] to continue...")
    
###############################################################################
##                          HEATWAVE CHECK                                   ##
###############################################################################
# We are going to plot the heatwaves at each site
for site in heatwave_df.Site.unique():
    site_EHF = heatwave_df[(heatwave_df.Site==site) & (heatwave_df.Method=='EHF')]
    site_max = heatwave_df[(heatwave_df.Site==site) & (heatwave_df.Method=='Max')]
    site_min = heatwave_df[(heatwave_df.Site==site) & (heatwave_df.Method=='Min')]
    site_df = df[df.Site==site]
    
    fig, ax = plt.subplots(3,1)
    ax[0].scatter(site_df.date,site_df.TA_F,s=.5)
    for row in site_EHF[['start_dates','end_dates']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        ax[0].scatter(site_df[site_df.date.isin(date_range)].date,site_df[site_df.date.isin(date_range)].TA_F,c='red',s=.5)
    ax[0].set_title(f"Average Temperature Heatwaves for {site}")
    ax[1].scatter(site_df.date,site_df.TA_F,s=.5)
    for row in site_max[['start_dates','end_dates']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        ax[1].scatter(site_df[site_df.date.isin(date_range)].date,site_df[site_df.date.isin(date_range)].TA_F,c='red',s=.5)
    ax[1].set_title(f"Max Temperature Heatwaves for {site}")
    ax[2].scatter(site_df.date,site_df.TA_F,s=.5)
    for row in site_min[['start_dates','end_dates']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        ax[2].scatter(site_df[site_df.date.isin(date_range)].date,site_df[site_df.date.isin(date_range)].TA_F,c='red',s=.5)
    ax[2].set_title(f"Min Temperature Heatwaves for {site}")
    plt.tight_layout()
    plt.show()
    input("Press [enter] to continue...")
    
###############################################################################
##                         TEMPERATURE QC CHECK                              ##
###############################################################################
'''
The following code will investigate the quality of temperature data, especially
that of the highest 95th percentile for each day.
'''
# This confirms that the QAQC for daily temperature is available at all our sites
shared_tempQAQC = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdataDD',measures=['TA_F_QC'])
print(shared_tempQAQC['available_variables'])

# This confirms that the QAQC for hourly temperature is available at all our sites
shared_tempQAQC_hourly = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdata_HH',measures=['TA_F_QC'])
print(shared_tempQAQC_hourly['available_variables'])

# Download the daily temperature and QAQC variables
ta = loadAMF(path = "/Users/marleeyork/Documents/project2/data/AMFdataDD",measures=['TIMESTAMP','TA_F','TA_F_QC'])

# Loading in the heatwaves so that I may see if any of them are defined by these low QAQC days
# This will load in 3 dictionaries: heatwaves, heatwaves_EHF, and heatwaves_min
os.chdir("/Users/marleeyork/Documents/project2/heatwave_definition/")
from testing_heatwaves import *

# Remove any observations not in df
df_obs = df[['Site','TIMESTAMP']]
ta = pd.merge(df_obs,ta,on=['Site','TIMESTAMP'],how='inner')

# Looking at the observations with a quality control flag
QAQC_counts = ta.groupby('Site')['TA_F_QC'].apply(lambda x: (x < .5).sum())
print(QAQC_counts)

QAQC_counts_hourly = ta_H.groupby(['Site'])['TA_F_QC'].apply(lambda x: (x < .5).sum())
print(QAQC_counts_hourly)

# Adding a month variable to see if there a certain time of the year that is an issue
ta['Month'] = ta.TIMESTAMP.dt.month

# Grouping again, but this time by site and month
QAQC_counts = ta.groupby(['Site','Month'])['TA_F_QC'].apply(lambda x: (x < .5).sum())
with pd.option_context('display.max_rows', None):
    print(QAQC_counts)


# Create label of quality control
TA_QAQC = []
for value in ta.TA_F_QC:
    if (value < .5):
        TA_QAQC.append(1)
    else:
        TA_QAQC.append(0)
        
ta['TA_flag'] = TA_QAQC

# Plotting temperature timeseries with daily QAQC flag to see if they are continuous
colors = {'0': 'blue', '1':'red'}

for site in ta.Site.unique():
    flux_data = ta[(ta['Site']==site)]
    flux_data['TA_flag'] = flux_data['TA_flag'].astype("str")
    fig, ax = plt.subplots()
    
    plt.scatter(flux_data.TIMESTAMP,flux_data.TA_F,c=flux_data.TA_flag.map(colors),s=.5)
    plt.title(site)

    plt.tight_layout()
    plt.show()
    
    input("Press [Enter] to continue...")

# Comparing low QAQC temperature days with PRISM temperature
# Merge PRISM data with daily temperature data
tmean_long = pd.melt(historical_tmean,
                     id_vars='date',
                     var_name = 'Site',
                     value_name = 'Tmean_PRISM'
                     )

tmean_long.date = pd.to_datetime(tmean_long.date)
tmean_long.columns = ['TIMESTAMP','Site','Tmean_PRISM']

# Merge AmeriFlux data with PRISM data
ta = pd.merge(ta,tmean_long,on=['Site','TIMESTAMP'],how='inner')

# Unpack heatwave days from heatwaves_EHF
# Since this QAQC is looking at average daily temperature, I am only starting
# with the EHF heatwaves
heatwave_indicator_EHF = pd.DataFrame(columns=['date','heatwave_indicator','Site'])
for site in ta.Site.unique():
    site_heatwaves = heatwaves_EHF[site]['indicator']
    site_heatwaves['Site'] = [site] * site_heatwaves.shape[0]
    heatwave_indicator_EHF = pd.concat([heatwave_indicator_EHF,site_heatwaves])
heatwave_indicator_EHF.columns = ['TIMESTAMP','heatwave_indicator','Site']

# Merge heatwave indicator onto the temperature QAQC data
ta = pd.merge(ta,heatwave_indicator_EHF,on=['Site','TIMESTAMP'])

# Plotting QAQC flagged days by site
# This is excluding any Canada and Alaska sites for now
flagged_ta = ta[ta['TA_flag']==1]
flagged_ta = flagged_ta[flagged_ta['Tmean_PRISM']!=-9999]

site_cat = flagged_ta['Site'].unique()
colors = plt.cm.tab10(range(len(site_cat))) 
color_map = dict(zip(site_cat, colors))

fig, ax = plt.subplots()
for site in site_cat:
    subset = flagged_ta[flagged_ta['Site'] == site]
    plt.scatter(subset['TA_F'], subset['Tmean_PRISM'], color=color_map[site], label=site,s=.7,alpha=.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("AMF Daily Avg Temperature")
plt.ylabel("PRISM Daily Avg Temperature")
plt.show()

# Plotting QAQC flagged days by whether or not they are in a heatwave
heatwave_cat = flagged_ta['heatwave_indicator'].unique()
colors_heatwave = plt.cm.tab10(range(len(heatwave_cat)))
color_heatwave_map = dict(zip(heatwave_cat,colors_heatwave))

fig, ax = plt.subplots()
for presence in heatwave_cat:
    subset = flagged_ta[flagged_ta['heatwave_indicator'] == presence]
    plt.scatter(subset['TA_F'], subset['Tmean_PRISM'], color = color_heatwave_map[presence], label=presence, s=.7, alpha=.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("AMF Daily Avg Temperature")
plt.ylabel("PRISM Daily Avg Temperature")
plt.show()

# Finding those heatwaves that are invalid due to too much low quality temperature data
heatwave_QAQC = pd.DataFrame(columns=['start_date','end_date','QAQC_percentage','heatwave_invalidity','Site'])
for site in heatwaves_EHF.keys():
    print(site)
    site_qaqc = avg_QAQC_check(site_heatwave_dictionary = heatwaves_EHF[site],
                               dates = ta[ta['Site']==site].TIMESTAMP,
                               TA_QAQC = ta[ta['Site']==site].TA_F_QC,
                               QAQC_threshold = .5,
                               heatwave_threshold = .75
                               )
    site_qaqc['Site'] = [site] * site_qaqc.shape[0]
    heatwave_QAQC = pd.concat([heatwave_QAQC,site_qaqc])
    
# For the average heatwaves, how many are considered invalid based on this threshold
print(sum(heatwave_QAQC.heatwave_invalidity)) 

# Which sites are these invalid heatwaves coming from   
print(heatwave_QAQC.groupby('Site').heatwave_invalidity.sum())    

# What percentage of each sites heatwaves are invalid
# None of these are too bad, except maybe CA-SCC
print(heatwave_QAQC.groupby('Site').heatwave_invalidity.sum() / heatwave_QAQC.groupby('Site').heatwave_invalidity.count())

# Ultimately, we will want to remove these heatwaves from the batches!

# Running QAQC heatwave check to explore heatwaves that do not pass quality checks
###############################################################################
##             MAXIMUM, MINIMUM, and AVERAGE TEMPERATURE CHECKS              ##
###############################################################################

# Comparing max temperature PRISM to AmeriFlux
# Right now, I have this available across all 33 sites!

# Load in the data
ERA_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmax_data.csv")
ERA_min = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmin_data.csv")
ERA_mean = pd.read_csv("/Users/marleeyork/Documents/project2/data/ERA/ERA_tmean_data.csv")
PRISM_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_climate_data_tmax.csv")
PRISM_min = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_tmin.csv")
PRISM_mean = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_tmean.csv")

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
included_PRISM_sites = included_sites[pd.Series(included_sites).isin(PRISM_max.columns)]
PRISM_max = PRISM_max[included_PRISM_sites]
PRISM_min = PRISM_min[included_PRISM_sites]
PRISM_mean = PRISM_mean[included_PRISM_sites]

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
        
print(f"Sites that we don't have max PRISM data include: {missing_max}")
print(f"Sites that we don't have min PRISM data include: {missing_min}")
print(f"Sites that we don't have mean PRISM data include: {missing_avg}")

# These columns are completely missing all PRISM temperature, interesting
# As of now, its the Canada sites we are missing data for (PRISM doesn't go to Canada)
# and two US sites that are in Alaska
PRISM_max[missing_max]
PRISM_mean[missing_avg]

# Starting off with the ones that we do know
PRISM_max = PRISM_max.drop(columns=missing_max)
PRISM_min = PRISM_min.drop(columns=missing_min)
PRISM_mean = PRISM_mean.drop(columns=missing_avg)

# Pivot PRISM data longer
PRISM_max = pd.melt(PRISM_max,id_vars=['date'],var_name='Site',value_name='PRISM_TA')
PRISM_min = pd.melt(PRISM_min,id_vars=['date'],var_name='Site',value_name='PRISM_TA')
PRISM_mean = pd.melt(PRISM_mean,id_vars=['date'],var_name='Site',value_name='PRISM_TA')

# Now I am going to calculate the daily maximum and minimum temperatures for each of my sites
# using my function as done in my heatwave definition
AMF_mean = df[['Site','TIMESTAMP','TA_F']]
AMF_mean.columns = ['Site','date','TA_F']

AMF_max = pd.DataFrame(columns=['Site','date','max_temperature'])
for site in df.Site.unique():
    this_site = df_hourly[df_hourly['Site']==site]
    this_site_temp = find_max_temperatures(this_site.TIMESTAMP_START,this_site.TA_F)
    this_site_temp['Site'] = [site] * this_site_temp.shape[0]
    # concatenate with entire dataframe
    AMF_max = pd.concat([AMF_max,this_site_temp])

AMF_min = pd.DataFrame(columns=['Site','date','min_temperature'])
for site in df.Site.unique():
    this_site = df_hourly[df_hourly['Site']==site]
    this_site_temp = find_min_temperatures(this_site.TIMESTAMP_START,this_site.TA_F)
    this_site_temp['Site'] = [site] * this_site_temp.shape[0]
    # concatenate with entire dataframe
    AMF_min = pd.concat([AMF_min,this_site_temp])

# Merging ERA and PRISM data with AMF_max
# ERA_max.shape = (1713056, 6)
# PRISM_max.shape = (675315, 3)
# AMF_max.shape = (447814, 3)
ERA_max = pd.merge(AMF_max, ERA_max, on=['Site','date'],how="left")
ERA_min = pd.merge(AMF_min, ERA_min, on=['Site','date'],how="left")
ERA_mean = pd.merge(AMF_mean, ERA_mean, on=['Site','date'],how="left")
PRISM_max = pd.merge(AMF_max, PRISM_max, on=['Site','date'],how="left")
PRISM_min = pd.merge(AMF_min,PRISM_min, on=['Site','date'],how="left")
PRISM_mean = pd.merge(AMF_mean,PRISM_mean, on=['Site','date'],how="left")

# Rename columns
ERA_max.columns = ['Site','date','max_temperature','hist_TA']
ERA_min.columns = ['Site','date','min_temperature','hist_TA']
ERA_mean.columns = ['Site','date','TA_F','hist_TA']

PRISM_max.columns = ['Site','date','max_temperature','hist_TA']
PRISM_min.columns = ['Site','date','min_temperature','hist_TA']
PRISM_mean.columns = ['Site','date','TA_F','hist_TA']

# Drop those sites that we don't have for ERA and PRISM
mask = ~PRISM_max.hist_TA.isna()
PRISM_max = PRISM_max[mask]
mask = ~PRISM_min.hist_TA.isna()
PRISM_min = PRISM_min[mask]
mask = ~PRISM_mean.hist_TA.isna()
PRISM_mean = PRISM_mean[mask]

mask = ~ERA_max.hist_TA.isna()
ERA_max = ERA_max[mask]
mask = ~ERA_min.hist_TA.isna()
ERA_min = ERA_min[mask]
mask = ~ERA_mean.hist_TA.isna()
ERA_mean = ERA_mean[mask]

# Lets look at the overall correlation of ERA and PRISM with AmeriFlux data now
fig, ax = plt.subplots(3,2,figsize=(8,12))
ax = ax.flatten()
ax[0].scatter(ERA_max.max_temperature,ERA_max.hist_TA, s=.5,alpha=.1)
ax[0].plot([-30, 50], [-30, 50], '--',c='red')
ax[0].set_xlabel("AMF Max Temperature")
ax[0].set_ylabel("ERA Max Temperature")
ax[0].set_title("ERA")
ax[1].scatter(PRISM_max.max_temperature,PRISM_max.hist_TA,s=.5,alpha=.1)
ax[1].plot([-30, 50], [-30, 50], '--',c='red')
ax[1].set_xlabel("AMF Max Temperature")
ax[1].set_ylabel("PRISM Max Temperature")
ax[1].set_title("PRISM")
ax[2].scatter(ERA_min.min_temperature,ERA_min.hist_TA, s=.5,alpha=.1)
ax[2].plot([-30, 50], [-30, 50], '--',c='red')
ax[2].set_xlim(-30,50)
ax[2].set_ylim(-30,50)
ax[2].set_xlabel("AMF Min Temperature")
ax[2].set_ylabel("ERA Min Temperature")
ax[3].scatter(PRISM_min.min_temperature,PRISM_min.hist_TA,s=.5,alpha=.1)
ax[3].plot([-30, 50], [-30, 50], '--',c='red')
ax[3].set_xlim(-30,50)
ax[3].set_ylim(-30,50)
ax[3].set_xlabel("AMF Min Temperature")
ax[3].set_ylabel("PRISM Min Temperature")
ax[4].scatter(ERA_mean.TA_F,ERA_mean.hist_TA, s=.5,alpha=.1)
ax[4].plot([-30, 50], [-30, 50], '--',c='red')
ax[4].set_xlabel("AMF Mean Temperature")
ax[4].set_ylabel("ERA Mean Temperature")
ax[5].scatter(PRISM_mean.TA_F,PRISM_mean.hist_TA,s=.5,alpha=.1)
ax[5].plot([-30, 50], [-30, 50], '--',c='red')
ax[5].set_xlabel("AMF Mean Temperature")
ax[5].set_ylabel("PRISM Mean Temperature")
plt.tight_layout()
plt.show()

# Checking the overall correlation coefficient
# Correlation is 93%, which is pretty good
np.corrcoef(ERA_max.max_temperature,ERA_max.hist_TA)
np.corrcoef(PRISM_max.max_temperature,PRISM_max.hist_TA)

np.corrcoef(ERA_min.min_temperature,ERA_min.hist_TA)
np.corrcoef(PRISM_min.min_temperature,PRISM_min.hist_TA)

valid = ERA_mean[["TA_F", "hist_TA"]].dropna()
np.corrcoef(valid["TA_F"], valid["hist_TA"])
np.corrcoef(PRISM_mean.TA_F,PRISM_mean.hist_TA)

# Checking site by site correlation to look for any anomalies
# Ultimately, MSE indicates closeness between the data, but low correlation
# can indicate sites that may have really poor data.
corr_df = pd.DataFrame(columns=['Site','ERA_max','PRISM_max',
                                'ERA_min','PRISM_min','ERA_mean','PRISM_mean'])
for site in ERA_max.Site.unique():
    # Isolate all the max, min, and mean ERA and PRISM data for those sites
    site_ERA_max = ERA_max[ERA_max['Site']==site]
    site_ERA_min = ERA_min[ERA_min['Site']==site]
    site_ERA_mean = ERA_mean[ERA_mean['Site']==site]
    site_PRISM_max = PRISM_max[PRISM_max['Site']==site]
    site_PRISM_min = PRISM_min[PRISM_min['Site']==site]
    site_PRISM_mean = PRISM_mean[PRISM_mean['Site']==site]
    
    # Calculate the correlations with AmeriFlux data for each
    correlations = [site,
                    round(np.corrcoef(site_ERA_max.max_temperature,site_ERA_max.hist_TA)[0][1],2),
                    round(np.corrcoef(site_PRISM_max.max_temperature,site_PRISM_max.hist_TA)[0][1],2),
                    round(np.corrcoef(site_ERA_min.min_temperature,site_ERA_min.hist_TA)[0][1],2),
                    round(np.corrcoef(site_PRISM_min.min_temperature,site_PRISM_min.hist_TA)[0][1],2),
                    round(np.corrcoef(site_ERA_mean.TA_F,site_ERA_mean.hist_TA)[0][1],2),
                    round(np.corrcoef(site_PRISM_mean.TA_F,site_PRISM_mean.hist_TA)[0][1],2)]
    
    # Add to the dataframe of site correlations
    corr_df.loc[len(corr_df)] = correlations
    
# If a site has less than .9 correlation, it is flagged and we are going to check
# the ameriflux and reanalysis data closer.
flag_sites = corr_df[(corr_df.drop(columns="Site") < 0.9).any(axis=1)].Site  
corr_df[corr_df['Site'].isin(flag_sites)]

# Plotting these sites to assess difference in temperature
for site in further_investigation:
    # Isolate the data
    site_ERA_max = ERA_max[ERA_max['Site']==site]
    site_ERA_min = ERA_min[ERA_min['Site']==site]
    site_ERA_mean = ERA_mean[ERA_mean['Site']==site]
    site_PRISM_max = PRISM_max[PRISM_max['Site']==site]
    site_PRISM_min = PRISM_min[PRISM_min['Site']==site]
    site_PRISM_mean = PRISM_mean[PRISM_mean['Site']==site]
    
    # Plot!
    fig, ax = plt.subplots(3,1,figsize=(12,8))
    ax = ax.flatten()

    ax[0].scatter(site_ERA_max.date,site_ERA_max.max_temperature,c='black',s=.05,label="AMF")
    ax[0].scatter(site_ERA_max.date,site_ERA_max.ERA_TA,c="red",s=.05,label="ERA")
    ax[0].scatter(site_PRISM_max.date,site_PRISM_max.PRISM_TA,c="blue",s=.05,label="PRISM")
    ax[0].set_title(f"{site} max temperature")
    
    ax[1].scatter(site_ERA_min.date,site_ERA_min.min_temperature,c="black",s=.05,label="AMF")
    ax[1].scatter(site_ERA_min.date,site_ERA_min.ERA_TA,c="red",s=.05,label="ERA")
    ax[1].scatter(site_PRISM_min.date,site_PRISM_min.PRISM_TA,c="blue",s=.05,label="PRISM")
    ax[1].set_title(f"{site} min temperature")
    
    ax[2].scatter(site_ERA_mean.date,site_ERA_mean.TA_F,c="black",s=.05,label="AMF")
    ax[2].scatter(site_ERA_mean.date,site_ERA_mean.ERA_TA,c="red",s=.05,label="ERA")
    ax[2].scatter(site_PRISM_mean.date,site_PRISM_mean.PRISM_TA,c="blue",s=.05,label="PRISM")
    ax[2].set_title(f"{site} mean temperature")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    input("Press [enter] to continue...")
    
# After investigation of the low correlation sites, these are the ones to be removed.
removing_sites = ['US-CAK',"CA-Ca1","US-xHE","US-xDJ","US-ICt","US-Rpf","US-xNW",
                  "US-ICh","US-Hn2","US-EML","US-BZS","US-NGC","US-Cop","CA-SCC",
                  "CA-NS2","US-SP1","US-Ho1"]   
further_investigation = set(flag_sites) - set(removing_sites)

# Clean these sites out of each data set
ERA_max = ERA_max[~ERA_max.Site.isin(removing_sites)]
ERA_min = ERA_min[~ERA_min.Site.isin(removing_sites)]
ERA_mean = ERA_mean[~ERA_mean.Site.isin(removing_sites)]
PRISM_max = PRISM_max[~PRISM_max.Site.isin(removing_sites)]
PRISM_min = PRISM_min[~PRISM_min.Site.isin(removing_sites)]
PRISM_mean = PRISM_mean[~PRISM_mean.Site.isin(removing_sites)]

# Finding whether the ERA or PRISM data is better
preference_max = []
preference_min = []
preference_mean = []
for i in range(len(corr_df)):
    site_preference_max = 'PRISM' if (corr_df.PRISM_max[i] >= corr_df.ERA_max[i]) else 'ERA'
    site_preference_min = 'PRISM' if (corr_df.PRISM_min[i] >= corr_df.ERA_min[i]) else 'ERA'
    site_preference_mean = 'PRISM' if (corr_df.PRISM_mean[i] >= corr_df.ERA_mean[i]) else 'ERA'
    preference_max.append(site_preference_max)
    preference_min.append(site_preference_min)
    preference_mean.append(site_preference_mean)
corr_df['preference_max'] = preference_max
corr_df['preference_min'] = preference_min
corr_df['preference_mean'] = preference_mean

# Since the ERA data is preferred everytime, we are going to move forward with that
# Finding the 95th percentile for each
for site in ERA_max.Site.unique():
    this_site = ERA_max[ERA_max['Site']==site]
    # Calculate the 95th quantile overall
    AMF_95 = np.quantile(this_site.max_temperature,.95)
    ERA_95 = np.quantile(this_site.ERA_max,.95)
    print(f"95th quantiles at site {site} are {round(AMF_95,2)} and {round(ERA_95,2)}.")
    
# Find correlation between daily 95th quantile values
# All of these look good except for US-CAK, CA-Ca1, and US-CS2
daily_quantiles = pd.DataFrame(columns=['Site','month_day','AMF_quantiles','ERA_quantiles','PRISM_quantiles'])
daily_95_corr = pd.DataFrame(columns=['Site','ERA_corr','PRISM_corr'])
for site in ERA_max.Site.unique():
    this_site = ERA_max[ERA_max['Site']==site]
    this_site_PRISM = PRISM_max[PRISM_max['Site']==site]
    # Calculate window quantiles for each
    AMF_daily_95 = moving_window_quantile(this_site.date,this_site.max_temperature,.95, 15)
    AMF_daily_95.columns = ["month_day","AMF_quantiles"]
    ERA_daily_95 = moving_window_quantile(this_site.date,this_site.ERA_max,.95,15)
    ERA_daily_95.columns = ["month_day","ERA_quantiles"]
    # If the site exists for PRISM, then also calculate the quantiles
    if len(this_site_PRISM) > 0:
        PRISM_daily_95 = moving_window_quantile(this_site_PRISM.date,this_site_PRISM.PRISM_max,.95,15)
        PRISM_daily_95.columns = ['month_day','PRISM_quantiles']
        PRISM_corr = round(np.corrcoef(AMF_daily_95.AMF_quantiles,PRISM_daily_95.PRISM_quantiles)[0][1])
    else:
        PRISM_daily_95 = pd.DataFrame({'month_day':AMF_daily_95.month_day,
                                       'PRISM_quantiles':[np.nan]*len(AMF_daily_95)})
        PRISM_corr = np.nan
    # Merge these together
    window_quantiles = pd.merge(AMF_daily_95,ERA_daily_95,on='month_day',how='inner')
    window_quantiles = pd.merge(window_quantiles,PRISM_daily_95,on='month_day',how='left')
    # Add a site columns
    window_quantiles['Site'] = [site] * window_quantiles.shape[0]
    # Stack it onto the dataframe across all sites
    daily_quantiles = pd.concat([daily_quantiles,window_quantiles])
    # Calculate the correlations for each
    ERA_corr = round(np.corrcoef(AMF_daily_95.AMF_quantiles,ERA_daily_95.ERA_quantiles)[0][1],2)
    # Save correlations into a dataframe
    daily_95_corr.loc[len(daily_95_corr)] = [site,ERA_corr,PRISM_corr]
print(daily_95_corr)
print(daily_95_corr[daily_95_corr['ERA_corr'] < .93])
    
# Plotting the AMF and PRISM 95th moving quantiles
fig, ax = plt.subplots()
sb.lmplot(x='AMF_quantiles', y='PRISM_quantiles',hue='Site',data=daily_quantiles)
plt.plot([0, 40], [0,40], '--',c='black')
plt.show()

fig, ax = plt.subplots()
sb.lmplot(x='AMF_quantiles',y='ERA_quantiles',hue='Site',data=daily_quantiles)
plt.plot([-20, 40], [-20,40], '--',c='black')
plt.tight_layout()
plt.show()

# Lets look at the squared difference of daily 95th quantiles for each site
daily_quantiles['MSE_ERA_95'] = (daily_quantiles.ERA_quantiles - daily_quantiles.AMF_quantiles)**2
daily_quantiles['MSE_PRISM_95'] = (daily_quantiles.PRISM_quantiles - daily_quantiles.AMF_quantiles)**2
daily_95_MSE = daily_quantiles.groupby('Site')[["MSE_ERA_95","MSE_PRISM_95"]].sum() / 365
daily_95_MSE = pd.DataFrame(daily_95_MSE).reset_index()
daily_95_MSE.loc[daily_95_MSE['MSE_PRISM_95']==0,'MSE_PRISM_95'] = np.nan

# Lets also look at the squared difference of daily temperatures
ERA_max['MSE_ERA'] = (ERA_max.ERA_max - ERA_max.max_temperature) ** 2
PRISM_max['MSE_PRISM'] = (PRISM_max.PRISM_max - PRISM_max.max_temperature) ** 2
daily_MSE = pd.merge(ERA_max, PRISM_max, on=['Site','date','max_temperature'], how='left')
daily_MSE.groupby('Site')[['MSE_ERA','MSE_PRISM']].sum()

# Now lets get the sites where the PRISM data performs better than ERA
PRISM_is_better95 = daily_95_MSE[daily_95_MSE['MSE_PRISM_95'] < daily_95_MSE['MSE_ERA_95']].Site.unique()
PRISM_is_better = daily_MSE[daily_MSE['MSE_PRISM'] < daily_MSE['MSE_ERA']].Site.unique()

# Checking the above to see if its a certain month or season we should be worried about
# It is consistently these October/November 95th quantile temperatures that
# are very different.
for site in PRISM_is_better95.Site.unique():
    site_quantiles = daily_quantiles[daily_quantiles['Site']==site]
    site_quantiles.month_day = pd.to_datetime(site_quantiles.month_day,format='%m-%d')
    fig, ax = plt.subplots()
    plt.scatter(site_quantiles.month_day, site_quantiles.AMF_quantiles,c='black',s=.5,label="AMF")
    plt.scatter(site_quantiles.month_day, site_quantiles.ERA_quantiles,c='red',s=.5,label="ERA")
    plt.scatter(site_quantiles.month_day, site_quantiles.PRISM_quantiles,c='blue',s=.5,label="PRISM")
    plt.legend()
    plt.title(f"Historical Daily 95th Quantile for {site}")
    plt.show()
    
    input("Press enter to continue: ")

Mo2_quantiles = daily_quantiles[daily_quantiles['Site']=='US-Mo2']
Mo2_quantiles.month_day = pd.to_datetime(Mo2_quantiles.month_day,format='%m-%d')
fig, ax = plt.subplots()
plt.scatter(Mo2_quantiles.month_day, Mo2_quantiles.AMF_quantiles,c='red',s=.5)
plt.scatter(Mo2_quantiles.month_day, Mo2_quantiles.PRISM_quantiles,c='blue',s=.5)
plt.title("Historical Daily 95th Quantile for US-Mo2")
plt.show()

# Creating site list variable
# IGBP is available across all sites
find_shared_variables_longfile(path="/Users/marleeyork/Documents/project2/data/BADM",
                               measures=["IGBP"],column='VARIABLE',value='DATAVALUE',file_type='xslx')

###############################################################################
##                         INVALID HEATWAVES CHECK                           ##
###############################################################################
os.chdir("/Users/marleeyork/Documents/project2/data/heatwaves/")
invalid_heatwaves_min = pd.read_csv("invalid_heatwaves_min.csv")
invalid_heatwaves_max = pd.read_csv("invalid_heatwaves_max.csv")
invalid_heatwaves_mean = pd.read_csv("invalid_heatwaves_mean.csv")

with open("heatwaves_max.pkl", "rb") as file:
    heatwaves = pickle.load(file)
    
with open("heatwaves_min.pkl", "rb") as file:
    heatwaves_min = pickle.load(file)

with open("heatwaves_mean.pkl", "rb") as file:
    heatwaves_mean = pickle.load(file)

# Calculate percentages of invalid heatwaves at each site
invalid_percentages_min = pd.DataFrame(columns=['Site','invalid_perc'])
for site in heatwaves_min.keys():
    # Pull the number overall and invalid
    num_heatwaves = len(heatwaves_min[site]['start_dates'])
    num_invalid = len(invalid_heatwaves_min[invalid_heatwaves_min.Site==site])
    # Calculate percentage of invalid heatwaves at site
    if num_heatwaves > 0:
        invalid_perc = num_invalid / num_heatwaves
    else:
        invalid_perc = 0
    # Add to dataframe
    invalid_percentages_min.loc[len(invalid_percentages_min)] = [site,invalid_perc]

invalid_percentages_mean = pd.DataFrame(columns=['Site','invalid_perc'])
for site in heatwaves_mean.keys():
    # Pull the number overall and invalid
    num_heatwaves = len(heatwaves_mean[site]['start_dates'])
    num_invalid = len(invalid_heatwaves_mean[invalid_heatwaves_mean.Site==site])
    # Calculate percentage of invalid heatwaves at site
    if num_heatwaves > 0:
        invalid_perc = num_invalid / num_heatwaves
    else:
        invalid_perc = 0
    # Add to dataframe
    invalid_percentages_mean.loc[len(invalid_percentages_mean)] = [site,invalid_perc]

invalid_percentages = pd.DataFrame(columns=['Site','invalid_perc'])
for site in heatwaves.keys():
    # Pull the number overall and invalid
    num_heatwaves = len(heatwaves[site]['start_dates'])
    num_invalid = len(invalid_heatwaves_max[invalid_heatwaves_max.Site==site])
    # Calculate percentage of invalid heatwaves at site
    if num_heatwaves > 0:
        invalid_perc = num_invalid / num_heatwaves
    else:
        invalid_perc = 0
    # Add to dataframe
    invalid_percentages.loc[len(invalid_percentages)] = [site,invalid_perc]

# Sites with a high amount of invalid heatwaves
invalid_sites_min = invalid_percentages_min[invalid_percentages_min.invalid_perc >= .25].Site
invalid_sites_mean = invalid_percentages_mean[invalid_percentages_mean.invalid_perc >= .25].Site
invalid_sites_max = invalid_percentages[invalid_percentages.invalid_perc >= .25].Site

# Find the unique set of sites that have some type of invalidity
invalid_sites = set(list(invalid_sites_min) + list(invalid_sites_max) + list(invalid_sites_mean))

# Investigating specific sites that are causing issues
struggle_sites = ['MX-Tes','US-ONA','US-xSJ','US-Wkg','US-SRM','US-xBN','US-xSB',
                  'US-Prr','US-xJE','US-xGR']

# Plot the valid and invalid heatwaves for each site for min, max, and mean heatwaves
for site in struggle_sites:
    # Isolate data for that site
    site_df = df[df.Site==site]
    site_heatwaves_max = heatwaves[site]
    site_heatwaves_min = heatwaves_min[site]
    site_heatwaves_mean = heatwaves_mean[site]
    site_invalid_max = invalid_heatwaves_max[invalid_heatwaves_max.Site==site]
    site_invalid_min = invalid_heatwaves_min[invalid_heatwaves_min.Site==site]
    site_invalid_mean = invalid_heatwaves_mean[invalid_heatwaves_mean.Site==site]
    
    # Isolate the start and end dates for the heatwaves
    max_start = site_heatwaves_max['start_dates']
    max_end = site_heatwaves_max['end_dates']
    min_start = site_heatwaves_min['start_dates']
    min_end = site_heatwaves_min['end_dates']
    mean_start = site_heatwaves_mean['start_dates']
    mean_end = site_heatwaves_mean['end_dates']
    
    # Plot the site daily temperature for each
    fig, ax = plt.subplots(3,1)
    ax[0].scatter(site_df.date,site_df.TA_F,s=.5,c='lightgrey')
    ax[0].set_title(f"Max heatwaves for site {site}")
    ax[1].scatter(site_df.date,site_df.TA_F,s=.5,c='lightgrey')
    ax[1].set_title(f"Mean heatwaves for site {site}")
    ax[2].scatter(site_df.date,site_df.TA_F,s=.5,c='lightgrey')
    ax[2].set_title(f"Min heatwaves for site {site}")
    
    # Plot the valid maximum heatwaves
    for start, end in zip(max_start,max_end):
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[0].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='red',label='Valid heatwave')
    
    # Plot the invalid maximum heatwaves
    for row in site_invalid_max[['start_date','end_date']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[0].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='blue',label='Invalid heatwave')
    
    # Plot the valid mean heatwaves
    for start, end in zip(mean_start,mean_end):
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[1].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='red')
    
    # Plot the invalid mean heatwaves
    for row in site_invalid_mean[['start_date','end_date']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[1].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='blue')
    
    # Plot the valid minimum heatwaves
    for start, end in zip(min_start,min_end):
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[2].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='red')
    
    # Plot the invalid minimum heatwaves
    for row in site_invalid_min[['start_date','end_date']].itertuples(index=False):
        start, end = row
        date_range = pd.date_range(start,end)
        heatwave_df = site_df[site_df.date.isin(date_range)]
        ax[2].scatter(heatwave_df.date,heatwave_df.TA_F,s=.5,c='blue')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    input("Press [enter] to continue...")

# Plotting the historical versus AMF data for each of the struggle sites
for site in heatwaves.keys():
    # Isolate historical data
    site_hist_max = historical_data_max[historical_data_max.Site==site]
    site_hist_min = historical_data_min[historical_data_min.Site==site]
    site_hist_mean = historical_data_mean[historical_data_mean.Site==site]
    
    # Isolate AMF data
    site_AMF_mean = df[df.Site==site][['TIMESTAMP','Site','TA_F']]
    site_AMF_mean.columns = ['date','Site','TA_F']
    site_AMF_HH = df_hourly[df_hourly.Site==site]
    site_AMF_max = find_max_temperatures(site_AMF_HH.TIMESTAMP_START,site_AMF_HH.TA_F)
    site_AMF_min = find_min_temperatures(site_AMF_HH.TIMESTAMP_START,site_AMF_HH.TA_F)
    
    # Merge the data together
    site_max = pd.merge(site_AMF_max,site_hist_max,on='date',how='left')
    site_min = pd.merge(site_AMF_min,site_hist_min,on='date',how='left')
    site_mean = pd.merge(site_AMF_mean,site_hist_mean,on='date',how='left')
    
    # Now we plot everything
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].scatter(site_max.max_temperature,site_max.hist_TA,s=.5)
    ax[0].set_title(f"Max for site {site}")
    ax[0].plot([0, 30], [0, 30], 'r--', label="1:1 line")

    ax[1].scatter(site_min.min_temperature,site_min.hist_TA,s=.5)
    ax[1].set_title(f"Min for site {site}")
    ax[1].plot([0, 30], [0, 30], 'r--', label="1:1 line")

    ax[2].scatter(site_mean.TA_F,site_mean.hist_TA,s=.5)
    ax[2].set_title(f"Mean for site {site}")
    ax[2].plot([0, 30], [0, 30], 'r--', label="1:1 line")

    # --- Adding regressions safely (drop NaNs) ---
    def safe_regression(x, y, axis):
        # Drop NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask].reshape(-1,1)
        y_clean = y[mask]
        if len(x_clean) > 1:  # need at least 2 points
            model = LinearRegression().fit(x_clean, y_clean)
            y_pred = model.predict(x_clean)
            axis.plot(x_clean.flatten(), y_pred, 'b-', label="Regression fit")
            axis.legend(fontsize=8)

    safe_regression(site_max.max_temperature.values, site_max.hist_TA.values, ax[0])
    safe_regression(site_min.min_temperature.values, site_min.hist_TA.values, ax[1])
    safe_regression(site_mean.TA_F.values, site_mean.hist_TA.values, ax[2])
    
    plt.tight_layout()
    plt.show()
    
    test_line_difference(site_max.max_temperature.values,
                     site_max.hist_TA.values,
                     label=f"{site} max")

    test_line_difference(site_min.min_temperature.values,
                     site_min.hist_TA.values,
                     label=f"{site} min")

    test_line_difference(site_mean.TA_F.values,
                     site_mean.hist_TA.values,
                     label=f"{site} mean")
    
    input("Press [enter] to continue...")


struggling_sites = ['US-EA6','US-EA5','US-xSR','US-CGG','US-xYE','CA-TPD']

# I am saving the final sites
final_sites = pd.DataFrame({"Site":heatwaves.keys()})
final_sites.to_csv("final_site_list.csv")


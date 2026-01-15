'''
This script includes functions that check the quality of heatwaves based on 
the AmeriFlux QAQC temperature flags.
'''
import pandas as pd
import os
os.chdir(path="/Users/marleeyork/Documents/project2/heatwave_definition")
from define_heatwaves import *

def avg_QAQC_check(site_heatwave_dictionary, dates, TA_QAQC, QAQC_threshold,
                   heatwave_threshold):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data.
    
    Parameters
    ----------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : TYPE
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : TYPE
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : TYPE
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : TYPE
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    # If the site didn't have any valid data, then we skip it
    if pd.isna(site_heatwave_dictionary['start_dates']).all():
        return 
    # Otherwise, we check for invalid heatwave flags
    start_dates = site_heatwave_dictionary['start_dates']
    end_dates = site_heatwave_dictionary['end_dates']
    ta_qaqc = pd.DataFrame({'dates':dates,'QAQC':TA_QAQC})
    heatwave_qaqc = pd.DataFrame(columns=['start_date','end_date','QAQC_percentage','heatwave_invalidity'])
    # Loop through each heatwave
    for start, end in zip(start_dates,end_dates):
        # Create a range of dates between the start and end
        date_range = pd.date_range(start=start, end=end)
        # Find the QAQC values in these dates
        heatwave_QAQC_values = ta_qaqc[ta_qaqc['dates'].isin(date_range)]
        # Determine if they are flagged as being below the threshold
        QAQC_flag = []
        for quality in heatwave_QAQC_values.QAQC:
            if (quality < QAQC_threshold):
                QAQC_flag.append(1)
            else:
                QAQC_flag.append(0)
        
        if len(QAQC_flag) == 0:
            continue
        else:
            # Find percentage of flagged days
            QAQC_percentage = sum(QAQC_flag) / len(QAQC_flag)
            invalidity_flag = 0 if (QAQC_percentage < heatwave_threshold) else 1
            # Add start date, end date, percentage of bad data days, and heatwave validity to dataframe
            this_site = pd.DataFrame({'start_date':[start],
                                  'end_date':[end],
                                  'QAQC_percentage':[QAQC_percentage],
                                  'heatwave_invalidity':[invalidity_flag]})
            # Concatenate with QAQC of other heatwaves
            heatwave_qaqc = pd.concat([heatwave_qaqc,this_site])
    return heatwave_qaqc

def minmax_QAQC_check(site_heatwave_dictionary, dates, TA, TA_QAQC, heatwave_threshold,method='max'):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data. This is for those heatwaves defined by
    hourly data, like the min/max approaches.
    
    I don't think I actually need to use this, but TBD. Could still do an hourly thing
    where if the max/min temperature is bad, it defaults to the next temperature.
    Our min/max temperatures were really well correlated with PRISM data though.
    
    Parameters
    ----------
    site_heatwave_dictionary : DICTIONARY
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : datetime vector
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : float vector
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : float decimal [0,1]
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : float decimal [0,1]
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    method : string ['max' or 'min']
        DESCRIPTION. Specifies whether this is being used on heatwaves defined by
        maximum or minimum daily temperatures
    
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    # If the site doesn't have valid data, then skip it
    if pd.isna(site_heatwave_dictionary['start_dates']).all():
        return 
    
    # Initialize list to hold flags for heatwaves surpassing the valid amount of downscaled data
    flag = []
    # Get find max hourly temperature
    hourly_TA = pd.DataFrame({'dates':dates,'TA':TA,'TA_QAQC':TA_QAQC})
    hourly_TA['dates_dt'] = pd.to_datetime(hourly_TA['dates'].dt.date)
    
    # Loop through each heatwave
    data = site_heatwave_dictionary['summary']
    for i in range(data.shape[0]):
    
        # Get the ith heatwave — this was the major bug
        this_heatwave = data.iloc[i]

        # Build the list of dates in the heatwave
        this_heatwave_dates = pd.date_range(
            this_heatwave.start_dates, 
            this_heatwave.end_dates
            )

        # Extract hourly records in those dates
        this_heatwave_hourly = hourly_TA[
            hourly_TA['dates_dt'].isin(this_heatwave_dates)
            ]
    
        downscaled = []

        for date in this_heatwave_dates:

            # Subset hourly data for the date
            daily_hourly = this_heatwave_hourly[
                this_heatwave_hourly['dates_dt'] == date
                ]

            if len(daily_hourly) == 0:
                # Handle empty day
                downscaled.append(1)   # or whatever logic makes sense
                continue
        
            # Identify the hour with the max/min temperature
            if method == 'max':
                idx = daily_hourly['TA'].idxmax()
            else:
                idx = daily_hourly['TA'].idxmin()
        
            qaqc = daily_hourly.loc[idx, 'TA_QAQC']

            # Mark whether gap filled based on QC coding
            downscaled.append(1 if qaqc == 2 else 0)

        heatwave_percentage = sum(downscaled) / len(downscaled)
        fail = 1 if heatwave_percentage >= heatwave_threshold else 0

        flag.append(fail)

    # Merge this onto the heatwave summary
    data['QAQC_flag'] = flag
    # If it is bad, then flag it as bad
    heatwave_qaqc = data
    
    return heatwave_qaqc



def remove_invalid_heatwaves(heatwaves_dictionary, invalid_heatwaves):
    '''
    Parameters
    ----------
    heatwave_dictionary : Dictionary
        DESCRIPTION. The heatwave dictionary for all site.
    invalid_heatwaves : DataFrame
        DESCRIPTION. DataFrame of invalid heatwaves with columns=['Site','start_dates','end_dates']

    Returns
    -------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Heatwave dictionary returned with the invalid heatwaves removed.

    '''
    # Remove these invalid heatwaves from the list of heatwaves
    for site in invalid_heatwaves.Site.unique():
        print(f"Cleaning up site {site}...")
        # Isolate invalid heatwaves for that site
        site_invalid = invalid_heatwaves[invalid_heatwaves['Site']==site]
        # Loop through the 
        for i in range(site_invalid.shape[0]):
            invalid_heatwave = site_invalid.iloc[i]
            print(f"Removing invalid heatwave {invalid_heatwave}.")
            # Drop the invalid from the start date
            heatwaves_dictionary[site]['start_dates'] = [
                d for d in heatwaves_dictionary[site]['start_dates']
                if d != invalid_heatwave.start_date
                ]
            # Drop the invalid from the end date
            heatwaves_dictionary[site]['end_dates'] = [
                d for d in heatwaves_dictionary[site]['end_dates']
                if d != invalid_heatwave.end_date
                ]
            # Drop the invalid from the summary
            data = heatwaves_dictionary[site]['summary']
            heatwaves_dictionary[site]['summary'] = data[
                data['start_dates'] != invalid_heatwave.start_date
                ].reset_index(drop=True)
            # Change the indicator values to 0 at these invalid heatwaves
            data = heatwaves_dictionary[site]['indicator']
            mask = (data['date'] >= invalid_heatwave.start_date) & \
                   (data['date'] <= invalid_heatwave.end_date)
            data.loc[mask, 'avg_indicator'] = 0
            heatwaves_dictionary[site]['indicator'] = data
            # Remove these same dates from periods
            data = heatwaves_dictionary[site]['periods']
            mask = (data['date'] >= invalid_heatwave.start_date) & \
                   (data['date'] <= invalid_heatwave.end_date)
            heatwaves_dictionary[site]['periods'] = heatwaves_dictionary[site]['periods'][~mask]
            # Remove the invalid heatwave dates from precip
            data = heatwaves_dictionary[site]['precip']
            mask = (data['start_date'] == invalid_heatwave.start_date) & \
                   (data['end_date'] == invalid_heatwave.end_date)
            heatwaves_dictionary[site]['precip'] = heatwaves_dictionary[site]['precip'][~mask]
            # Remove the invalid heatwave dates from swc
            data = heatwaves_dictionary[site]['swc']
            mask = (data['start_date'] == invalid_heatwave.start_date) & \
                   (data['end_date'] == invalid_heatwave.end_date)
            heatwaves_dictionary[site]['swc'] = heatwaves_dictionary[site]['swc'][~mask]
    
    return heatwaves_dictionary


















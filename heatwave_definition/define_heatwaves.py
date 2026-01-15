# This script will explore the different approaches for the definition of heatwaves

# Importing packages
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import pyplot as plt

# FUNCTIONS ###################################################################

def find_max_temperatures(date_vector, temperature_vector):
    '''
    Name: find_max_temperatures()
    Summary: These functions aggregates subdaily values of temperature (or any other
             variable) and calculates the maximum/minimum for each day

    Input: date_vector ~ Datetime stamp with at a subdaily level (e.g., hourly or 30 minute)
           temperature_vector ~ Temperatures associated with date_vector

    Output: (min/max)_temperatures ~ Pandas dataframe with column 'date' specifying
            daily dates and max_temperature specifying the summary statistic
            for that day
    '''
    # Create dataframe of timestamp and subdaily temperature
    temp_df = pd.DataFrame({'timestamp': date_vector,
                            'temperature': temperature_vector})
    
    # Perform daily aggregation, calculate max, reogranize into new dataframe
    max_temperatures = (
    temp_df
    .groupby(temp_df['timestamp'].dt.date)['temperature']
    .max()
    .reset_index()
    .rename(columns={'timestamp': 'date', 'temperature': 'max_temperature'})
    )   
    
    # Reset the date to a datetime variable
    max_temperatures['date'] = pd.to_datetime(max_temperatures['date'])
    
    return max_temperatures


def find_min_temperatures(date_vector, temperature_vector):
    '''
    Name: find_min_temperatures()
    Summary: These functions aggregates subdaily values of temperature (or any other
             variable) and calculates the maximum/minimum for each day

    Input: date_vector ~ Datetime stamp with at a subdaily level (e.g., hourly or 30 minute)
           temperature_vector ~ Temperatures associated with date_vector

    Output: (min/max)_temperatures ~ Pandas dataframe with column 'date' specifying
            daily dates and min_temperature specifying the summary statistic
            for that day
    '''
    # Create a dataframe of timestamp and subdaily temperature
    temp_df = pd.DataFrame({'timestamp': date_vector,
                            'temperature': temperature_vector})
    
    # Perform daily aggregation, calculate max, reorganize into new dataframe
    min_temperatures = (
    temp_df
    .groupby(temp_df['timestamp'].dt.date)['temperature']
    .min()
    .reset_index()
    .rename(columns={'timestamp': 'date', 'temperature': 'min_temperature'})
    )   
    
    # Reset the date to a datetime variable
    min_temperatures['date'] = pd.to_datetime(min_temperatures['date'])

    
    return min_temperatures


def moving_window_quantile(dates, measure, measure_quantile, window_length):
    '''
    Name: moving_window_quantiles()
    Summary: For a given length of moving window, calculate a quantile for
             across all historical values of a given measure (e.g., temperature, 
             max temperatures, etc) for each day of the year. Each day's quantile
             will be based on a surrounding window, such that the day is the centre
             of the window being calculated.

    Input: window_length ~ an odd number of days you want the window length to be
           dates ~ daily dates of historical data as datetime variable
           measure ~ daily measure associated with each day of dates
           quantile ~ quantile you want to calculate over the window (e.g., 90th)

    Output: window_quantiles ~ dataframe with 'day' as a datetime variable specifying
            month and day, and 'quantile' specifying the quantile of interest for 
            that day over the surrounding window.
    
    '''
    # Make sure dates are in datetime format
    dates = pd.to_datetime(dates)
    
    # Create dataframe of dates and measure
    measure_df = pd.DataFrame({'date': dates,
                               'month_day': dates.dt.strftime('%m-%d'),
                               'measure': measure})
    
    # Create range of all month and days throughout one year
    window_centre = pd.date_range(start = '1800-01-01', end = '1800-12-31', freq = 'D')
    
    # Determine the length of days backwards and forwards we need to look
    # so that our day of interest is at the centre of our window
    half_window_length = int((window_length - 1) / 2)
    
    # Subtract and add half_window_length from date to determine start and end
    # points of the window.
    window_start = window_centre - pd.to_timedelta(half_window_length, unit='d')
    window_end = window_centre + pd.to_timedelta(half_window_length, unit = 'd')
    
    # Calculate the quantiles in each window
    # Loop through the centre, start, and end of the windows
    window_quantile = []
    for centre, start, end in zip(window_centre, window_start, window_end):
        # Create a range of M-D inside the window
        window_range = pd.date_range(start=start, end=end, freq='D')
        
        # Isolate the month and day for the window range
        window_range = window_range.strftime('%m-%d')
        
        # Create indicator for days from 'dates' that fall into the window range
        window_mask = measure_df.month_day.isin(window_range)
        
        # Calculate the quantiles for the measure over all days that fall in the window
        window_quantile.append(measure_df[window_mask].measure.quantile(measure_quantile))
        
    # Create dataframe of the month-day and the associated quantile
    window_quantiles = pd.DataFrame({'month_day': window_centre.strftime('%m-%d'),
                                    'quantiles': window_quantile})
    
    return window_quantiles


def define_hotdays(timeseries_dates, timeseries_temperature, threshold_month_day, threshold, comparison = "greater"):
    '''
    Name: define_hotdays()
    Summary: Returns an indicator (0/1) whether each day is a hot day or not based on
             whether is greater than, less than, or equal to some daily threshold (e.g.,
             90th quantile of maximum temperature)    

    Input: timeseries_dates ~ dates associated with the timeseries we want to define as hot or not
           timeseries_temperature ~ temperatures associated with each day of the timeseries
           threshold ~ the daily threshold value, in order by month-day (01-01, ..., 12-31)
           comparison ~ value "greater", "less", or "equal" to threshold

    Output: hotdays ~ indicator vector (0 or 1) of length timeseries that defines
                       each day is 1 = hot day, or 0 = not hot day
    '''
    
    # Create separate timeseries and threshold dataframes
    timeseries_df = pd.DataFrame({'date':timeseries_dates,
                                  'month_day': timeseries_dates.dt.strftime("%m-%d"),
                                  'temperature':timeseries_temperature})
    threshold_df = pd.DataFrame({'month_day':threshold_month_day,
                                 'threshold':threshold})
    
    # Merge by month so we have the timeseries with its corresponding threshold
    data = pd.merge(timeseries_df, threshold_df, on="month_day", how="left")
    
    # Create a T/F mask for whether the value for a given day is hot
    if (comparison == 'greater'):
        threshold_mask = data.temperature > data.threshold
    elif (comparison == "lesser"):
        threshold_mask = data.temperature < data.threshold
    elif (comparison == "equal"):
        threshold_mask = data.temperature == data.threshold
    else:
        print("Not a valid comparison entry... enter greater, lesser, or equal")
        
    # Create a hotdays vector that is 1/0 corresponding to T/F in a dataframe
    hotdays = pd.DataFrame({'date':timeseries_dates,
                            'hotday_indicator':threshold_mask.astype(int).values})
    
    return hotdays


def define_EHF_hotdays(timeseries_dates, timeseries_temperature, 
                       historical_dates, historical_temperature,quantile_threshold):
    '''
    Name: define_EHF_hotdays()
    Summary: Returns a dataframe of heatwave dates, 0/1 indicator of EHF hotday,
             and EHF score

    Input: timeseries_dates ~ daily dates associated with average temperatures
           timeseries_temperature ~ daily average temperatures over timeseries of interest
           historical_dates ~ vector of dates over historical/climatological data
           historical_average_temperature ~ this must be DAILY AVERAGES, not min/max

    Output: EHF_df ~ dataframe including the EHF score and an indicator vector 
                        (0 or 1) of length timeseries that defines
                       each day is 1 = hot day, or 0 = not hot day based on EHF
    '''
    # Calculate the 3 day moving average for historical data
    historical_DMT = []
    for i in range(2,len(historical_temperature)):
        DMT = historical_temperature[(i-2):(i+1)].sum() / 3
        historical_DMT.append(DMT)
    # Organize into dataframe
    historical_DMT_df = pd.DataFrame({"date":historical_dates[2:],
                                      "DMT":historical_DMT})
    # Add a month day column
    historical_DMT_df['month_day'] = historical_DMT_df.date.dt.strftime('%m-%d')
    # Group by month day and calculate 95th of DMT
    T95_by_day = (
        historical_DMT_df
        .groupby("month_day")["DMT"]
        .quantile(quantile_threshold)
        .reset_index(name="T95")
    )
    # Create a dataframe of timeseries data
    timeseries_df = pd.DataFrame({'date':timeseries_dates,
                                  'temperature':timeseries_temperature})
    # Add a month_day column
    timeseries_df['month_day'] = timeseries_df.date.dt.strftime('%m-%d')
    # Merge timeseries data with the DMT 95th quantiles
    timeseries_df = pd.merge(timeseries_df,T95_by_day,on="month_day",how="left")

    # Initialize lists to store all the indices
    three_day_mean_list = []
    thirty_day_mean_list = []
    EHI_accl_list = []
    EHI_sig_list = []   # keep building these similarly
    EHF_hotdays = []

    n = len(timeseries_temperature)

    for i in range(n):
        # 3-day mean (needs indices i-2, i-1, i)
        if i < 2:
            three_day_mean = np.nan
        else:
            three_day_mean = timeseries_temperature[(i-2):(i+1)].mean()
        three_day_mean_list.append(three_day_mean)

        # 30-day mean (needs i-29 ... i)
        if i < 29:
            thirty_day_mean = np.nan
        else:
            thirty_day_mean = timeseries_temperature[(i-29):(i+1)].mean()
        thirty_day_mean_list.append(thirty_day_mean)

        # EHI_accl only when both means exist
        if np.isnan(three_day_mean) or np.isnan(thirty_day_mean):
            EHI_accl_list.append(np.nan)
        else:
            EHI_accl_list.append(three_day_mean - thirty_day_mean)


    # Add calculated mean indices to the timeseries dataframe
    timeseries_df.loc[:,'EHI_accl'] = EHI_accl_list
    timeseries_df.loc[:,'thirty_day_mean'] = thirty_day_mean_list
    timeseries_df.loc[:,'three_day_mean'] = three_day_mean_list
    # Calculate the universal score
    timeseries_df.loc[:,'EHI_sig'] = timeseries_df.three_day_mean - timeseries_df.T95
    # Calculate for max(0,EHI_accl)
    timeseries_df["EHI_accl"] = [max(0, x) for x in timeseries_df["EHI_accl"]]
    # Use the prior to find the extreme heat factor
    timeseries_df.loc[:,'EHF'] = timeseries_df.EHI_sig * timeseries_df.EHI_accl
    # Create an indicator for EHF   
    hotday_indicator = [1 if x > 0 else 0 for x in timeseries_df.EHF]
    # Organize into a dataframe   
    print("We made it to the dataframe")
    print("Timeseries dates: " +str(len(timeseries_dates[29:])))
    print("Timeseries_df.EHF: " +str(len(timeseries_df.EHF[29:])))
    print("hotday_indicator: " +str(len(hotday_indicator[29:])))
    EHF_df = pd.DataFrame({'date':timeseries_dates[29:],
                           'EHF_score':timeseries_df.EHF.values[29:],
                           'hotday_indicator':hotday_indicator[29:]}).reset_index(drop=True)
    print("We made it past the dataframe.")
    
    return EHF_df

def find_heatwaves(hotdays, dates, minimum_length, tolerance, gap_day_window):
    """
    Name: find_heatwaves()
    Summary: Identify heatwaves, allowing up to `tolerance` non-hot days per `gap_day_window`.
    Heatwave continues as long as hotdays accumulate before tolerance runs out.

    Input: dates ~ dates across the timeseries of interest
           hotdays ~ binary (0/1) indicator of hot day, received from define_hotdays()
           minimum_length ~ minimum number of contiguous hotdays to be considered heatwave
           gap_days ~ number of gap days allowed per gap_day_window
           gap_day_window ~ number of days that gap_days can fall in 

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves

    """
    active = hotdays == 1

    start_dates, end_dates = [], []
    in_heatwave = False
    start_time = None
    tolerance_left = tolerance
    hotday_count = 0
    total_days_in_window = 0  # total days (hot + not) within current gap window

    for i in range(len(active)):
        if active.iloc[i]:
            if not in_heatwave:
                # Start new heatwave
                in_heatwave = True
                start_time = dates.iloc[i]
                hotday_count = 1
                total_days_in_window = 1
                tolerance_left = tolerance
            else:
                # Continue existing heatwave
                hotday_count += 1
                total_days_in_window += 1

                # Reset tolerance if window reached
                if total_days_in_window >= gap_day_window:
                    tolerance_left = tolerance
                    total_days_in_window = 0  # start a new window
        else:
            if in_heatwave:
                total_days_in_window += 1
                if tolerance_left > 0:
                    # Allow a cool day, but don't add to hotday_count
                    tolerance_left -= 1
                else:
                    # No tolerance left → heatwave ends
                    if hotday_count >= minimum_length:
                        end_time = dates.iloc[i - 1]
                        start_dates.append(start_time)
                        end_dates.append(end_time)
                    # Reset
                    in_heatwave = False
                    hotday_count = 0
                    tolerance_left = tolerance
                    total_days_in_window = 0

    # Handle if heatwave goes till end
    if in_heatwave and hotday_count >= minimum_length:
        end_time = dates.iloc[-1] if hasattr(dates, "iloc") else dates.iloc[-1]
        start_dates.append(start_time)
        end_dates.append(end_time)
        
    # Trim off any non-hotdays around the end dates
    trimmed_end_dates = []
    for start, end in zip(start_dates, end_dates):
       mask = (dates >= start) & (dates <= end)
       segment = active[mask]

       # Walk backward to find last hotday
       for j in range(len(segment) - 1, -1, -1):
           if segment.iloc[j] if hasattr(segment, "iloc") else segment[j]:
               if hasattr(dates, "iloc"):
                   trimmed_end_dates.append(dates[mask].iloc[j])
               else:
                   trimmed_end_dates.append(np.array(dates)[mask][j])
               break
       else:
           trimmed_end_dates.append(end)

    return start_dates, trimmed_end_dates

# The following are examples of the heatwave definition with gap days options
# start_dates_new, end_dates_new = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=3, gap_days=1, gap_day_window=8)
# start_dates_new2, end_dates_new2 = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=5, gap_days=1, gap_day_window=5)

def build_date_range(start_dates, end_dates,frequency):
    '''
    Name: build_date_range()
    Summary: Takes the start and end dates of heatwaves and provides all heatwave dates

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates

    Output: date_range ~ list of list of all dates for each heatwave
    '''
    
    date_range = []
    for start, end in zip(start_dates, end_dates):
        date_range.append(pd.date_range(start,end,freq=frequency))
    return date_range
    

def find_daily_quantiles(historical_dates, historical_temperatures, my_quantile):
    '''
    Name: find_daily_quantiles()
    Summary: Finds quantiles for a given day of the year over historical data

    Input: historical_dates ~ vecotr of dates over historical data
           historical_temperatures ~ vector of temperatures over the historical data
           my_quantile ~ percentile you want to calculate

    Output: historical_quantiles ~ dataframe with month-day column and the historical quantile
    Note: This is being used for heatwave definition, so variables are temp based but can be used in other contexts

    '''
    # Make sure dates are in datetime format
    historical_dates = pd.to_datetime(historical_dates)
    
    # Isolate month and day for each date
    month_day = historical_dates.dt.strftime('%m-%d')
    # create a dataframe of month-day and temp
    daily_max_temp = pd.DataFrame({'month_day':month_day,
                                   'max_temp':historical_temperatures})
    
    # group by month-day and calculate quantile
    quantile_temperatures = (
    daily_max_temp
    .groupby('month_day')['max_temp']
    .quantile(my_quantile)
    .reset_index()
    .rename(columns={'max_temp': 'quantile_temperature'})
    )
    
    return quantile_temperatures


def describe_heatwaves(start_dates, end_dates, timeseries_dates, timeseries_temperature,
                       historical_dates, historical_temperatures, method):
    '''
    Name: describe_heatwaves()
    Summary: Gives some indices and information about the heatwaves 

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           timeseries_dates ~ vector of dates from timeseries of temp
           timeseries_temperature ~ vector of max temperature for each day of timeseries

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves
            duration ~ length of days heatwave lasted
            magnitude ~ heatwave magnitude index defined by Marengo (2025)
    '''
    
    heatwave_df = pd.DataFrame()
    heatwave_df['start_dates'] = start_dates
    heatwave_df['end_dates'] = end_dates
    heatwave_df['duration'] = pd.Series(end_dates) - pd.Series(start_dates) + pd.Timedelta(days=1)
    
    # Now calculating the Marengo 2025 heatwave magnitude index
    # 30 year 25th and 75th percentile temperature, and maximum daily temp
    quantile_temp_25 = find_daily_quantiles(historical_dates, historical_temperatures, .25)
    quantile_temp_75 = find_daily_quantiles(historical_dates, historical_temperatures, .75)
    # Find date ranges of the heatwaves
    heatwave_dates = build_date_range(start_dates, end_dates, frequency='D')
        
    return heatwave_df

def get_heatwave_indicator(start_dates, end_dates, daily_dates):
    '''
    Name: get_heatwave_indicator()
    Summary: This takes the start and end dates of heatwaves and returns an vector
             with a 0/1 indicator of a heatwave day (DIFFERENT THAN HOTDAYS)

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           all_dates ~ vector of all timeseries dates

    Output: heatwave_days ~ dataframe with date column and 0/1 indicator of whether 
            a day is part of a heatwave defined by find_heatwaves()
    
    '''
    # Initialize a list with all zeroes
    heatwave_days = pd.DataFrame({'date':daily_dates,
                                  'heatwave_indicator':[0]*len(daily_dates)})
    
    # For each start and end dates, create a range of dates in between
    for start, end in zip(start_dates, end_dates):
        date_range = pd.date_range(start=start,end=end,freq='D')
        heatwave_days.loc[heatwave_days['date'].isin(date_range),'heatwave_indicator'] = 1
     
    return heatwave_days
    
# Example of the above
# heatwaves = get_heatwave_indicator(start_dates_new, end_dates_new, max_temperatures_Whs.date)

# Plotting this example
# plt.figure()
# plt.scatter(max_temperatures_Whs.date, max_temperatures_Whs.max_temperature, s = .5)
# plt.scatter(max_temperatures_Whs.date[heatwaves['heatwave_indicator']==1],max_temperatures_Whs.max_temperature[heatwaves['heatwave_indicator']==1],c="red",s=.5)
# plt.show()

def fit_heatwaves(flux_dates, flux_temperature, 
                  historical_dates, historical_temperature,
                  quantile_threshold = .95,
                  window_length = 15,
                  threshold_comparison = 'greater',
                  min_heatwave_length = 3,
                  tolerance = 1,
                  gap_days_window = 8,
                  site = "Example",
                  method = "max"
                  ):
    '''
     Name: fit_heatwaves()
     Summary: This wraps defining, summarizing, and plotting heatwaves into one.  

     Input:    method ~ "max" for maximum temperature quantile approach,
                        "min" for minimum temperature quantile approach,
                        "mean" for mean approach (use daily temperature for flux temp)
            
            

     Output:   heatwaves ~ a dictionary including...
               start_dates, end_dates ~ all start and end dates of each heatwave
               summary ~ the summary returned by the describe_heatwaves() function
               indicator ~ 0/1 indicator of heatwave inclusion
               periods ~ dates and max temperature for heatwave days only
               plot ~ plot of max temperatures with heatwave days in red
    '''
    # Define heatwaves using the max approach
    if (historical_temperature.isna().sum() > 0):
        return
        
    elif (method == 'max'):
        # Find maximum temperature for each flux day
        daily_max_temperatures = find_max_temperatures(flux_dates,flux_temperature)
        # Find moving quantile window for flux data temperature
        # Default parameters are a window of 15 days and 90th quantile
        max_temperature_quantiles = moving_window_quantile(
            dates = historical_dates,
            measure = historical_temperature,
            measure_quantile = quantile_threshold,
            window_length = window_length
            )
        # Define hotdays with the maximum temperature quantiles determined
        # prior as the threshold
        print(f"Finding max temperature hot days for site {site}.")
        hotdays = define_hotdays(
            timeseries_dates = daily_max_temperatures.date,
            timeseries_temperature = daily_max_temperatures.max_temperature,
            threshold_month_day = max_temperature_quantiles.month_day,
            threshold = max_temperature_quantiles.quantiles,
            comparison = threshold_comparison
            )
        
        # Determine the start and end dates of the heatwaves
        # Default leniency is 1 gap day per every 8 days of heatwave, with min 
        # number of 3 consecutive hotdays for a heatwave
        start_dates, end_dates = find_heatwaves(
            dates = hotdays.date, 
            hotdays = hotdays.hotday_indicator,
            minimum_length = min_heatwave_length, 
            tolerance = tolerance, 
            gap_day_window = gap_days_window
            )
        
        # Get the summary of the heatwaves
        summary = describe_heatwaves(
            start_dates = start_dates, 
            end_dates = end_dates, 
            timeseries_dates = daily_max_temperatures.date, 
            timeseries_temperature = daily_max_temperatures.max_temperature,
            historical_dates = historical_dates,
            historical_temperatures = historical_temperature,
            method = method
            )
        
        # Get the vector indicator of hot days
        indicator = get_heatwave_indicator(start_dates = start_dates, 
                                           end_dates = end_dates, 
                                           daily_dates = daily_max_temperatures.date
                                           )
        # Get the max temperatures associated with each heatwave
        periods = pd.DataFrame({"date": daily_max_temperatures.date[indicator.heatwave_indicator == 1],
                                "max_temperature": daily_max_temperatures.max_temperature[indicator.heatwave_indicator == 1]})
        # Provide a plot of a heatwave
        fig, ax = plt.subplots()
        ax.scatter(daily_max_temperatures.date,daily_max_temperatures.max_temperature, s=.5, c='lightgrey')
        ax.scatter(periods.date,periods.max_temperature,s=.5,c="red")
        ax.set_title(f"Max: {site}")
        heatwave_plot = fig
    
    elif (method == "min"):
        print(f"Finding Min hotdays for site {site}.")
        # Find maximum temperature for each flux day
        daily_min_temperatures = find_min_temperatures(flux_dates,flux_temperature)
        # Find moving quantile window for flux data temperature
        # Default parameters are a window of 15 days and 95th quantile
        min_temperature_quantiles = moving_window_quantile(
            dates = historical_dates,
            measure = historical_temperature,
            measure_quantile = quantile_threshold,
            window_length = window_length
            )
        # Define hotdays with the maximum temperature quantiles determined
        # prior as the threshold
        hotdays = define_hotdays(
            timeseries_dates = daily_min_temperatures.date,
            timeseries_temperature = daily_min_temperatures.min_temperature,
            threshold_month_day = min_temperature_quantiles.month_day,
            threshold = min_temperature_quantiles.quantiles,
            comparison = threshold_comparison
            )
        
        # Determine the start and end dates of the heatwaves
        # Default leniency is 1 gap day per every 8 days of heatwave, with min 
        # number of 3 consecutive hotdays for a heatwave
        start_dates, end_dates = find_heatwaves(
             dates = hotdays.date, 
             hotdays = hotdays.hotday_indicator,
             minimum_length = min_heatwave_length, 
             tolerance = tolerance, 
             gap_day_window = gap_days_window
             )
        
        # Get the summary of the heatwaves
        summary = describe_heatwaves(
            start_dates = start_dates, 
            end_dates = end_dates, 
            timeseries_dates = daily_min_temperatures.date, 
            timeseries_temperature = daily_min_temperatures.min_temperature,
            historical_dates = historical_dates,
            historical_temperatures = historical_temperature,
            method = method
            )
        
        # Get the vector indicator of hot days
        indicator = get_heatwave_indicator(start_dates = start_dates, 
                                           end_dates = end_dates, 
                                           daily_dates = daily_min_temperatures.date
                                           )
        # Get the max temperatures associated with each heatwave
        periods = pd.DataFrame({"date": daily_min_temperatures.date[indicator.heatwave_indicator == 1],
                                "min_temperature": daily_min_temperatures.min_temperature[indicator.heatwave_indicator == 1]})
        # Provide a plot of a heatwave
        fig, ax = plt.subplots()
        ax.scatter(daily_min_temperatures.date,daily_min_temperatures.min_temperature, s=.5, c='lightgrey')
        ax.scatter(periods.date,periods.min_temperature,s=.5,c="red")
        ax.set_title(f"Min: {site}")
        heatwave_plot = fig
        
    # Define heatwaves using the EHF approach
    elif (method == "mean"):
        print(f"Finding mean hotdays for site {site}.")
        # Find maximum temperature for each flux day
        daily_mean_temperatures = pd.DataFrame({
            "date": flux_dates,
            "mean_temperature": flux_temperature
            })
        # Find moving quantile window for flux data temperature
        # Default parameters are a window of 15 days and 90th quantile
        mean_temperature_quantiles = moving_window_quantile(
            dates = historical_dates,
            measure = historical_temperature,
            measure_quantile = quantile_threshold,
            window_length = window_length
            )
        # Define hotdays with the maximum temperature quantiles determined
        # prior as the threshold
        hotdays = define_hotdays(
            timeseries_dates = daily_mean_temperatures.date,
            timeseries_temperature = daily_mean_temperatures.mean_temperature,
            threshold_month_day = mean_temperature_quantiles.month_day,
            threshold = mean_temperature_quantiles.quantiles,
            comparison = threshold_comparison
            )
        
        # Determine the start and end dates of the heatwaves
        # Default leniency is 1 gap day per every 8 days of heatwave, with min 
        # number of 3 consecutive hotdays for a heatwave
        start_dates, end_dates = find_heatwaves(
             dates = hotdays.date, 
             hotdays = hotdays.hotday_indicator,
             minimum_length = min_heatwave_length, 
             tolerance = tolerance, 
             gap_day_window = gap_days_window
             )
        
        # Get the summary of the heatwaves
        summary = describe_heatwaves(
            start_dates = start_dates, 
            end_dates = end_dates, 
            timeseries_dates = daily_mean_temperatures.date, 
            timeseries_temperature = daily_mean_temperatures.mean_temperature,
            historical_dates = historical_dates,
            historical_temperatures = historical_temperature,
            method = method
            )
        
        # Get the vector indicator of hot days
        indicator = get_heatwave_indicator(start_dates = start_dates, 
                                           end_dates = end_dates, 
                                           daily_dates = daily_mean_temperatures.date
                                           )
        # Get the max temperatures associated with each heatwave
        periods = pd.DataFrame({"date": daily_mean_temperatures.date[indicator.heatwave_indicator == 1],
                                "mean_temperature": daily_mean_temperatures.mean_temperature[indicator.heatwave_indicator == 1]})
        
        # Merge historical temperature quantiles onto  daily mean temperatures
        daily_mean_temperatures['month_day'] = daily_mean_temperatures.date.dt.strftime('%m-%d')
        daily_mean_temperatures = pd.merge(daily_mean_temperatures,mean_temperature_quantiles,on='month_day',how='left')
        
        # Provide a plot of a heatwave
        fig, ax = plt.subplots()
        ax.scatter(daily_mean_temperatures.date,daily_mean_temperatures.mean_temperature, s=.5, c='lightgrey')
        ax.scatter(periods.date,periods.mean_temperature,s=.5,c="red")
        ax.plot(daily_mean_temperatures.date,daily_mean_temperatures.quantiles,c='black',linewidth=1)
        ax.set_title(f"Mean: {site}")
        heatwave_plot = fig
    
    # Create a dictionary and store all of these inside it!
    heatwaves = {
        "start_dates":start_dates,
        "end_dates":end_dates,
        "summary":summary,
        "indicator":indicator,
        "periods":periods,
        "plot":heatwave_plot
        }
    
    return heatwaves

def calculate_moisture(timeseries_dates,timeseries_moisture,start_dates,end_dates):
    '''
    Name: calculate_moisture()
    Summary: This provides the average moisture conditions during a given heatwave
    evet.

    Input: timeseries_dates ~ daily dates over the moisture timeseries
           timeserues_moisture ~ daily moisture conditions of interest over the timeseries
           start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           

    Output: moisture_averages ~ dataframe of start date, end date, and average
            moisture conditions over that time period
    '''
    moisture_averages = []
    moisture_totals = []
    moisture_variability = []
    # Loop through each heatwave
    for start, end in zip(start_dates, end_dates):
        # Create date range for the heatwave
        date_range = pd.date_range(start=start,end=end,freq='D')
        # Find moisture conditions during that period
        moisture = timeseries_moisture[timeseries_dates.isin(date_range)]
        # If we have moisture for that dataset
        if len(moisture) > 0:
            # Calculate average moisture
            average = np.nansum(moisture) / np.sum(~np.isnan(moisture))
            total = np.sum(~np.isnan(moisture))
            variability = np.nanstd(moisture)
            # Add to list of heatwave moisture averages
        else:
            average = pd.NA
            total = pd.NA
            moisture = pd.NA
            variability = pd.NA
        moisture_averages.append(average)
        moisture_totals.append(total)
        moisture_variability.append(variability)
    
    heatwave_moisture = pd.DataFrame({'start_date':list(start_dates),
                                      'end_date':list(end_dates),
                                      'moisture_average':list(moisture_averages),
                                      'moisture_total':list(moisture_totals),
                                      'moisture_variability':list(moisture_variability)
                                      })
    
    return heatwave_moisture


print("Heatwave defining functions all loaded.")
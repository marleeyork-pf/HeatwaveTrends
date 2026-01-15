'''
This will be used to identify and categorize joint heatwaves.
Heatwaves can fall into the following criteria:
    1. Min --> nighttime extreme heatwave
    2. Max --> daytime extreme heatwave
    3. Average --> overall heatwave
    4. Min and average --> nocturnally intensified heatwave
    5. Max and average --> diurnally intensified heatwave
    6. Min and max --> day-night spike heatwave
    7. Min, max, and average --> triad heatwave
'''
import pandas as pd

# Provided 3 dictionaries for the heawaves, return a category of what each 
# day belongs to 

def find_heatwave_overlap(min_heatwaves, max_heatwaves, avg_heatwaves):
    
    final_heatwaves = {}
    
    # Check that each of the heatwaves has the same sites
    sites_min = len(min_heatwaves.keys())
    sites_max = len(max_heatwaves.keys())
    sites_avg = len(avg_heatwaves.keys())
    
    if (sites_min == sites_max == sites_avg):
        # For each site
        for site in list(min_heatwaves.keys()):
            print(f"Working on site {site}...")
            # Create subdictioary for the site
            final_heatwaves[site] = {}
            
            # Pull the indicator for each type of heatwave
            min_indicator = min_heatwaves[site]['indicator'].iloc[:, 0:2] 
            max_indicator = max_heatwaves[site]['indicator'][['date','heatwave_indicator']]
            avg_indicator = avg_heatwaves[site]['indicator'][['date','heatwave_indicator']]
            
            # Rename the columns for the type of indicator it is
            min_indicator.columns = ['date','min_indicator']
            max_indicator.columns = ['date','max_indicator']
            avg_indicator.columns = ['date','avg_indicator']
            
            # Merge all of these indicators into one dataframe
            indicator_df = pd.merge(min_indicator,max_indicator,on='date',how='inner')
            indicator_df = pd.merge(indicator_df,avg_indicator,on='date',how='inner')
            
            # Get the heatwave category
            heatwave_categories = get_heatwave_category(indicator_df)
            indicator_df['heatwave_categories'] = heatwave_categories
            
            # Daily heatwave indicator
            final_heatwaves[site]['category_indicator'] = indicator_df
            
            # Finding consecutive time periods from the category indictor
            in_heatwave = []
            for i in range(indicator_df.shape[0]):
                if (indicator_df.loc[i,'heatwave_categories'] == 'None'):
                    in_heatwave.append(0)
                else:
                    in_heatwave.append(1)
            indicator_df['in_heatwave'] = in_heatwave
            
            # Find start and end dates of consecutive days
            start_dates, end_dates = find_consecutive_runs(indicator_df,'date','in_heatwave')
            final_heatwaves[site]['start_dates'] = start_dates
            final_heatwaves[site]['end_dates'] = end_dates
            
            # Find the heatwave type with the highest composition 
            # This currently does not consider if there are multiple with the same percentage of one heatwave type
            top_heatwave = calc_top_heatwave(start_dates,end_dates,indicator_df.date,indicator_df.heatwave_categories)
            final_heatwaves[site]['heatwave_type'] = top_heatwave
            final_heatwaves[site]['heatwave_type_counts'] = top_heatwave.top_heatwave.value_counts()
        
    else:
        print("Heatwaves are fit to different sites, check your data.")
    
    
    return final_heatwaves


def get_heatwave_category(indicator_df):
    '''
    This function categorizes a type of heatwave for each day based on overlapping criteria for 
    minimum, maximum, and average heatwaves.
    
    Parameters
    ----------
    indicator_df : TYPE
        DESCRIPTION. A dataframe that includes at least 3 columns: min_indicator, 
        max_indicator, avg_indicator. These are the indicator columns for each 
        heatwave type made in fit_heatwaves.

    Returns
    -------
    heatwave_categories : TYPE
        DESCRIPTION. A categorical list for the type of overlapping heatwave each 
        day is, including None, Night, Day, Overall, Night-intensified, 
        Day-intensified, Day-Night Spike, and Triad

    '''
    # Set the indicators to unique values that sum to unique values also 
    indicator_df.loc[indicator_df['min_indicator']==1,'min_indicator'] = 3
    indicator_df.loc[indicator_df['max_indicator']==1,'max_indicator'] = 4
    indicator_df.loc[indicator_df['avg_indicator']==1,'avg_indicator'] = 5
    
    # Create a column that is the sum of the 3
    indicator_df['indicator_sum'] = indicator_df['min_indicator'] + indicator_df['max_indicator'] + indicator_df['avg_indicator']
    
    # Assign heatwave type based on the unique sum values
    heatwave_categories = []
    for i in range(indicator_df.shape[0]):
        this_sum = indicator_df.loc[i,'indicator_sum']
        if (this_sum == 0): heatwave_categories.append('None')
        elif (this_sum == 3): heatwave_categories.append('Night')
        elif (this_sum == 4): heatwave_categories.append('Day')
        elif (this_sum == 5): heatwave_categories.append('Overall')
        elif (this_sum == 7): heatwave_categories.append('Day-Night Spike')
        elif (this_sum == 8): heatwave_categories.append('Night-intensified')
        elif (this_sum == 9): heatwave_categories.append('Day-intensified')
        elif (this_sum == 12): heatwave_categories.append('Triad')
        else: print("Something went real wrong here.")
    
    # Return the heatwave categories as a list!
    return heatwave_categories

def find_consecutive_runs(df, date_col, flag_col):
    '''
    This function takes returns the start and end date of all consecutive heatwave days.
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION. 
    date_col : TYPE
        DESCRIPTION. Date column of dataframe
    flag_col : TYPE
        DESCRIPTION. Whether or not its a heatwave day column.

    Returns
    -------
    runs : TYPE
        DESCRIPTION. 

    '''
    # Ensure sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    start_dates = []
    end_dates = []
    in_run = False
    
    for i in range(len(df)):
        flag = df.loc[i, flag_col]
        date = df.loc[i, date_col]
        
        if flag == 1 and not in_run:
            # Start of a new run
            start_date = date
            in_run = True
        
        # End of a run if next value is 0 or we reached the end
        is_last = (i == len(df) - 1)
        next_is_zero = (not is_last) and (df.loc[i+1, flag_col] == 0)
        
        if in_run and (is_last or next_is_zero):
            end_date = date
            start_dates.append(start_date)
            end_dates.append(end_date)
            in_run = False
    
    return start_dates, end_dates


def calc_top_heatwave(start_dates, end_dates, dates, heatwave_categories):

    # --- priority ranking: lower = higher priority ---
    priority = {
        'Triad': 1,
        'Day-Night Spike': 2,
        'Day-intensified': 3,
        'Night-intensified': 4,
        'Overall': 5,
        'Day': 6,
        'Night': 7
    }

    heatwave_df = pd.DataFrame({
        'dates': dates,
        'heatwave_category': heatwave_categories
    })

    heatwave_type = []

    for start, end in zip(start_dates, end_dates):
        date_range = pd.date_range(start, end)

        this_heatwave = heatwave_df[heatwave_df['dates'].isin(date_range)]

        pct = (
            this_heatwave
                .groupby('heatwave_category')
                .size()
                .div(len(this_heatwave))
                .mul(100)
                .reset_index(name='percent')
        )

        # Add priority column
        pct['priority'] = pct['heatwave_category'].map(priority)

        # Sort by:
        # 1️⃣ highest percent
        # 2️⃣ lowest priority number
        pct = pct.sort_values(
            by=['percent', 'priority'],
            ascending=[False, True]
        )

        # Select the top category after sorting
        top_heatwave = pct.iloc[0]['heatwave_category']
        heatwave_type.append(top_heatwave)

    return pd.DataFrame({
        'start_dates': start_dates,
        'end_dates': end_dates,
        'top_heatwave': heatwave_type
    })





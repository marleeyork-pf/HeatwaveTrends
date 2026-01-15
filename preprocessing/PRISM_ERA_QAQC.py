'''
This script provides a function that determines whether ERA or PRISM data 
should be used for each site.
'''
import pandas as pd
import os
os.chdir("/Users/marleeyork/Documents/project2/heatwave_definition")
from define_heatwaves import *
import warnings
import numpy as np
import statsmodels.api as sm

warnings.simplefilter(action="ignore", category=FutureWarning)



def ERA_or_PRISM(site_AMF_data,site_ERA_data,site_PRISM_data,site):
    """
    Parameters
    ----------
    site_AMF_data : dataframe
        DESCRIPTION. Dataframe of daily AmeriFlux temperatures (max, min, or avg) 
        calculated at a given site. Should contain 3 columns: Site, date, and AMF_TA.
    site_ERA_data : dataframe
        DESCRIPTION. Dataframe of all ERA data at a given site. 
        This should have 3 columns: Site, date, and ERA_TA.
    site_PRISM_data : dataframe
        DESCRIPTION. Dataframe of all PRISM data at a given site.
        This should have 3 columns: Site, date, and PRISM_TA.
    site: str
        DESCRIPTION. Current site of focus

    Returns
    -------
    preference : str
        DESCRIPTION. Which historical dataset should be used for that site, either
        "PRISM" or "ERA".
    """
    # If the site is not in the PRISM data, its automatically ERA data
    if site not in site_PRISM_data.Site.unique():
        preference = "ERA"
    else:
        # Merge the dataframes on the AMF dates
        site_data = pd.merge(site_AMF_data,site_ERA_data,on=['Site','date'],how='left')
        site_data = pd.merge(site_data,site_PRISM_data,on=['Site','date'],how='left')
            
        # Calculate the squared error for ERA and PRISM data from AMF
        site_data['ERA_MSE'] = (site_data.ERA_TA - site_data.AMF_TA)**2
        site_data['PRISM_MSE'] = (site_data.PRISM_TA - site_data.AMF_TA)**2
        
        # Compare the summed square error of the PRISM and ERA data
        # Determine which dataset has lower summed square error
        preference = 'PRISM' if (site_data.PRISM_MSE.mean() < site_data.ERA_MSE.mean()) else 'ERA'
    
    return preference

def retrieve_correct_data(site, site_ERA_data, site_PRISM_data, preference):
    """
    Parameters
    ----------
    site : str
        DESCRIPTION. Current site of focus
    site_ERA_data : dataframe
        DESCRIPTION. Dataframe of all ERA data at a given site. 
        This should have 3 columns: Site, date, and ERA_TA.
    site_PRISM_data : dataframe
        DESCRIPTION. Dataframe of all PRISM data at a given site.
        This should have 3 columns: Site, date, and PRISM_TA.
    preference : str
        DESCRIPTION. Whichever dataset was determined to be better suited for the site.
        This can be "ERA" or "PRISM".

    Returns
    -------
    correct_site_data : datafrmae
        DESCRIPTION. Dataframe containing all the historical data from the 
        appropriate source, PRISM or ERA, for a given site.
    """
    
    # Select the correct data source for the site based on the provided preference
    if (preference =='ERA'):
        correct_site_data = site_ERA_data.copy()
        correct_site_data['Source'] = ["ERA"] * len(correct_site_data)
    elif (preference == 'PRISM'):
        correct_site_data = site_PRISM_data.copy()
        correct_site_data['Source'] = ["PRISM"] * len(correct_site_data)
    else:
        print("Something has gone wrong in retrieving the correct data, neither PRISM nor ERA was received.")
    
    return correct_site_data

def return_best_data(AMF_data_all,ERA_data_all,PRISM_data_all,temperature_type,
                     start_date,end_date):
    """
    Parameters
    ----------
    AMF_data_all : TYPE
        DESCRIPTION. Dataframe of all hourly AmeriFlux data across sites if temperature
        type is maximum or minimum, or daily data for if temperature type is average.
        This should have three columns: Site, TIMESTAMP_START, and TA_F
    ERA_data_all : TYPE
        DESCRIPTION. Dataframe of all ERA data across sites. 
        This should have 3 columns: Site, date, and ERA_TA.
    PRISM_data_all : TYPE
        DESCRIPTION. Dataframe of all PRISM data across sites.
        This should have 3 columns: Site, date, and PRISM_TA.
    temperature_type : str
        DESCRIPTION. Either maximum ('max'), minimum ('min'), 
        or average ('average') temperature.

    Returns
    -------
    None.

    """
    # Initialize dataframe to store which historical dataset should be used for
    # each site.
    data_source = pd.DataFrame(columns=['Site','Preference'])
    
    # Initialize dataframe to store the historical data itself
    historical_data = pd.DataFrame(columns=['Site','date','hist_TA','Source'])
    
    # Calculate the maximum, minimum, or average of the Ameriflux data, depending
    # on the temperature type
    for site in AMF_data_all.Site.unique():
        print(f"Finding best data for site {site}.")
        site_AMF_data = AMF_data_all[AMF_data_all['Site']==site]
        site_ERA_data = ERA_data_all[ERA_data_all['Site']==site]
        site_PRISM_data = PRISM_data_all[PRISM_data_all['Site']==site]
        
        # Limit data years to those years specified
        site_ERA_data = site_ERA_data[(site_ERA_data.date >= start_date) & (site_ERA_data.date <= end_date)]
        site_PRISM_data = site_PRISM_data[(site_PRISM_data.date >= start_date) & (site_PRISM_data.date <= end_date)]
        
        if (temperature_type == "max"):
            site_AMF_data = find_max_temperatures(AMF_data_all.TIMESTAMP_START,AMF_data_all.TA_F)
            site_AMF_data['Site'] = [site] * len(site_AMF_data)
            site_AMF_data.columns = ['date','AMF_TA','Site']
        elif (temperature_type == "min"):
            site_AMF_data = find_min_temperatures(AMF_data_all.TIMESTAMP_START,AMF_data_all.TA_F)
            site_AMF_data['Site'] = [site] * len(site_AMF_data)
            site_AMF_data.columns = ['date','AMF_TA','Site']
        elif (temperature_type == "average"):
            site_AMF_data.columns = ['Site','date','AMF_TA']
        else:
            print("Incorrect input for temperature type. Choose max, min, or average.")
    
        # Find out which has a lower MSE: ERA or PRISM data and store into dataframe
        preference = ERA_or_PRISM(site_AMF_data,site_ERA_data,site_PRISM_data,site)
        data_source.loc[len(data_source)] = [site,preference]
        
        # Based on the site preference, pull the data for that site
        site_historical_data = retrieve_correct_data(site,site_ERA_data,site_PRISM_data,preference)
        site_historical_data.columns = ['Site','date','hist_TA','Source']
        
        # Concatenate with the other sites
        historical_data = pd.concat([historical_data,site_historical_data])
    
    
    return historical_data, data_source

def test_line_difference(x, y, label=""):
    """Test whether slope=1 and intercept=0 using statsmodels."""
    
    # Drop NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        print(f"{label}: Not enough data for statistical test")
        return

    # Add constant for intercept
    X = sm.add_constant(x_clean)

    # Fit regression
    model = sm.OLS(y_clean, X).fit()

    # Extract coefficients
    intercept = model.params[0]
    slope = model.params[1]

    # Hypothesis tests
    slope_test = model.t_test("x1 = 1")
    intercept_test = model.t_test("const = 0")

    print(f"\n--- {label} ---")
    print(f"Slope estimate:     {slope:.3f}")
    print(f"Intercept estimate: {intercept:.3f}")
    print(f"P-value (slope ≠ 1):     {slope_test.pvalue:.4f}")
    print(f"P-value (intercept ≠ 0): {intercept_test.pvalue:.4f}")










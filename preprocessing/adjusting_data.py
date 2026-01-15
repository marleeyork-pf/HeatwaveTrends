"""
This script includes functions to  adjust the PRISM and ERA data based on the 
AmeriFlux data for each site.
"""
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fit_sklearn(group: pd.DataFrame, x, y):
    """
    Description: This fits a regression to historical data by Ameriflux data
    and returns a dataframe a results across all sites.
    
    Parameters
    ----------
    group : pd.DataFrame
        DESCRIPTION. The grouped by dataframe that we are going to fit a regression to
        
    x : str
        DESCRIPTION. String column name for AMF data
    
    y : str
        DESCRIPTION. String column name for historical data

    Returns
    -------
    TYPE
        DESCRIPTION. Regression output

    """
    g = group[[x,y]].dropna()
    y = g[y].to_numpy()
    X = g[[x]].to_numpy()
    reg = LinearRegression().fit(X, y)
    
    r2 = reg.score(X, y)
    out = {
        "n": len(y),
        "r2": float(r2),
        "intercept": float(reg.intercept_),
        "coef_x": float(reg.coef_[0])
        }
    
    return pd.Series(out)

def find_historical_bias(final_data):
    """
    Description: This calculates the bias regressions for historical mean, min,
    and max across all sites.
    
    Parameters
    ----------
    final_data : TYPE
        DESCRIPTION. This is the dataframe including historical and AMF data for 
        each site. 

    Returns
    -------
    site_historical_fit : TYPE
        DESCRIPTION. This is a dictionary with each site as a key and information
        about regression fit for historical data (PRISM or ERA) to AmeriFlux data

    """
    # Fit mean regression
    mean_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="TA_F", y="hist_mean"))
        .reset_index()
        )
    mean_results.columns = ['Site','n_mean','r2_mean','intercept_mean','coef_x_mean']
    
    # Fit the min regression
    min_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="min_temperature", y="hist_min"))
        .reset_index()
        )
    min_results.columns = ['Site','n_min','r2_min','intercept_min','coef_x_min']
    
    # Fit the max regression
    max_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="max_temperature", y="hist_max"))
        .reset_index()
        )
    max_results.columns = ['Site','n_max','r2_max','intercept_max','coef_x_max']
    
    # Merge all the regression results together
    results = pd.merge(mean_results, min_results, on='Site', how='left')
    results = pd.merge(results, max_results, on='Site', how='left')
    
    return results

def can_we_correct(results: pd.DataFrame, n, r2):
    """
    Description: We only want to adjust sites with 5 years of data and an R^2 > .9.
    This function will tell us which data can be adjusted for each of these sites.
    
    Parameters
    ----------
    results : pd.DataFrame
        DESCRIPTION. DF of regression outputs for PRISM/ERA vs AMF for min, max, and mean temp
        
    n : int
        DESCRIPTION. The minimum days worth of data a site needs to be adjusted
    
    r2 : float
        DESCRIPTION. The minimum R^2 a site regression needs to have to adjust data

    Returns
    -------
    validity_df : pd.DataFrame
        DESCRIPTION. Includes each site and whether we can use the min, max, and mean data

    """
    # Initialize dataframe
    validity_df = pd.DataFrame()
    validity_df['Site'] = results.Site
    
    # Finding those sites with n count and R^2 wanted
    validity_df['Max'] = np.where((results['n_max'] > n) & (results['r2_max'] > r2),1,0)
    validity_df['Min'] = np.where((results['n_min'] > n) & (results['r2_min'] > r2),1,0)
    validity_df['Mean'] = np.where((results['n_mean'] > n) & (results['r2_mean'] > r2),1,0)
    
    return validity_df

def make_adjustment(historical_data: pd.DataFrame, results: pd.DataFrame, validity_df: pd.DataFrame):
    """
    Description: This adjusts the data by those bias regressions and returns the 
    adjusted historical data for 1990-2023
    
    Parameters
    ----------
    historical_data : pd.DataFrame
        DESCRIPTION. The complete (1990-2025) historical timeseries of min, max, and mean temperatures
    results : pd.DataFrame
        DESCRIPTION. Dataframe from find_historical_bias of the adjustment regressions
    validity_df : pd.DataFrame
        DESCRIPTION. Dataframe from can_we_correct of whether or not we can adjust site data

    Returns
    -------
    adjusted_data : TYPE
        DESCRIPTION. Dataframe of sites and adjusted data

    """
    # Initialize dataframe for adjusted data
    adjusted_data = pd.DataFrame(columns=['Site','date','hist_max_adj','hist_min_adj','hist_mean_adj'])
    # Loop through each site
    for site in validity_df.Site:
        print(site)
        # Isolate site data
        site_data = historical_data[historical_data.Site==site]
        site_adjusted = site_data[['Site','date']]
        # Check if we can adjust the max data
        if (validity_df[validity_df.Site==site].Max.iloc[0] == 1):
            # Find the regression 
            intercept = results[results.Site==site].intercept_max.iloc[0]
            slope = results[results.Site==site].coef_x_max.iloc[0]
            # Use this regression to adjust the data
            site_adjusted["hist_max_adj"] = (site_data.hist_max / slope) - intercept
        else:
            # Return NA
            site_adjusted["hist_max_adj"] = [np.nan]*len(site_data)
        
        # Check if we can adjust the min data
        if (validity_df[validity_df.Site==site].Min.iloc[0]==1):
            # Find the regression
            intercept = results[results.Site==site].intercept_min.iloc[0]
            slope = results[results.Site==site].coef_x_min.iloc[0]
            # Use this regression to adjust the data
            site_adjusted["hist_min_adj"] = (site_data.hist_min / slope) - intercept
        else:
            # Return NA
            site_adjusted["hist_min_adj"] = [np.nan]*len(site_data)
            
        # Finally, checking if we can ajust mean data
        if (validity_df[validity_df.Site==site].Mean.iloc[0]==1):
            # Find the regression
            intercept = results[results.Site==site].intercept_mean.iloc[0]
            slope = results[results.Site==site].coef_x_mean.iloc[0]
            # Us this regression to adjust the data
            site_adjusted["hist_mean_adj"] = (site_data.hist_mean / slope) - intercept
        else:
            # Return NA
            site_adjusted["hist_mean_adj"] = [np.nan]*len(site_data)
        
        # Concatenate the site adjusted to all other
        adjusted_data = pd.concat([adjusted_data, site_adjusted])

    return adjusted_data

def adjust_historical_data(historical_data: pd.DataFrame, AMF_data: pd.DataFrame, n: int, r2: float):
    """
    Description: This performs final adjustment and returns the bias corrections,
    validity, and the adjusted data itself.
    
    Parameters
    ----------
    historical_data : pd.DataFrame
        DESCRIPTION. 34 years of historical PRISM or ERA data including column names
        ["Site","date","hist_max","hist_min","hist_mean"]
    AMF_data : pd.DataFrame
        DESCRIPTION. AmeriFlux data for temperature variables including columns
        ["Site","date","TA_F","max_temperature","min_temperature"]
    n : int
        DESCRIPTION. Minimum number of AMF days we want for a proper adjusting regression
    r2 : float
        DESCRIPTION. Minimum R2 we want for a proper adjusting regression

    Returns
    -------
    adjustment_dict : Dictionary
        DESCRIPTION. Includes dataframe of which sites we can validly adjust,
        dataframe of the fittest regressions between AMF and historical data, 
        and dataframe of adjusted historical data
    """
    
    # Merge historical and AMF datasets
    reg_data = pd.merge(AMF_data, historical_data, on=['Site','date'],how='left').dropna()
    
    # Use historical data over AMF dates to calculate regression
    reg_results = find_historical_bias(reg_data)
    
    # Find those sites which we can adjust the historical data for based on the regressions
    validity_df = can_we_correct(reg_results, n=365*5, r2=.9)
    
    # Now we actually adjust the data
    adjusted_data = make_adjustment(historical_data, reg_results, validity_df)
    
    # Store this into a dictionary
    adjustment_dict = {
        "validity":validity_df,
        "regressions":reg_results,
        "adjustments":adjusted_data
        }
    
    
    return adjustment_dict

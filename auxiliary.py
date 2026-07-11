'''
Ths script includes all auxiliary functions for the entire project.

'''
#  Loading in all necessary packages
import pandas as pd
from datetime import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import warnings
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression

warnings.simplefilter(action="ignore", category=FutureWarning)

###############################################################################
##                            Loading Data                                   ##
###############################################################################

def loadAMF(path, skip=None, measures=None):
    '''
    Name: loadAMF()
    Summary: Reads a directory of AmeriFlux files into one dataframe

    Input: path ~ filepath for AmeriFlux data directory
           measures ~ list of variables we want to pull from each site, not
           not including site itself (that will be done for you)

    # Output: AMF_data ~ merged dataframe of all site data with a site identifier added
    '''
    
    if skip is None:
        skip = ['']
    
    if measures is None:
        measures = [
            'TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F',
            'NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF'
        ]
    
    # Check if we were given an actual filepath
    try:
        os.scandir(path)
    except FileNotFoundError:
        print("Thats not a valid filepath, check for error.")
        return
    except NotADirectoryError:
        print("Path for a single file was input, please provide a directory.")
        return

    print('Directory name passed in')

    # Add site to the columns
    my_columns = measures + ['Site']
    # Find all the files within our directory
    paths = [f.path for f in os.scandir(path) if (f.is_file() and f.name != '.DS_Store')]
    # Initialize dataframe with columns of interest
    AMF_data = pd.DataFrame(columns=my_columns)
    # Loop through each file and use loadAMFFile to read in the data
    for this_file in paths:
        if this_file in skip:
            continue
        # Concat onto our cross-site dataframe
        data = loadAMFFile(this_file, measures)      # pass ORIGINAL measures
        AMF_data = pd.concat([AMF_data, data], ignore_index=True)

    return AMF_data

def loadAMFFile(this_file, measures):
    '''
    Name: loadAMFFile()
    Summary: Reads a singular AmeriFlux file into a dataframe

    Input: this_file ~ filepath for AMF data saved as csv
           measures ~ list of variables that will be included in data

    Output: file_df ~ AMF data organized into dataframe with additional site column
    '''
    # Pull site from the filename
    filename = os.path.basename(this_file)
    site = filename[4:10]
    print("Loading site... " + site)
    
    # Determine the resolution (HH or DD) of the file
    resolution = filename[26]
    file_df = pd.read_csv(this_file)
    
    # Format the date based on the resolution
    if resolution == "H":
        file_df.TIMESTAMP_START = pd.to_datetime(file_df.TIMESTAMP_START, format='%Y%m%d%H%M')
    elif resolution == "D":
        file_df.TIMESTAMP = pd.to_datetime(file_df.TIMESTAMP, format='%Y%m%d')
    # Add site to the columns
    required_cols = measures + ["Site"]
    # Check if the file has all the measures we want
    try:
        # If not, isolate the measures of interest
        file_df = file_df[measures]
    except KeyError:
        # If it is, return empty dataframe and print missing measure
        print(f"Site {site} was not loaded because it is missing some measures.")
        missing = set(measures) - set(file_df.columns)
        print(f"Missing columns at {site}: {missing}")
        return pd.DataFrame(columns=required_cols)
    # Add the site column
    file_df["Site"] = site

    # Force correct column order for all sites
    return file_df.reindex(columns=required_cols)

def find_shared_variables(path,measures):
    
    '''
    Name: find_shared_variables()
    Summary: This returns information on variables shared across all AmeriFlux
            files from a given directory. The main purpose is to identify moisture
            variables that can be studied when looking at many sites.

    Input: path ~ filepath for directory of AmeriFlux datasets
           measures ~ the variables you want if the files have

    Output: variable_info ~ dictionary that includes the following information
                site_presence ~ 0/1 indicator of presence of each variable at each site
                total_presence ~ how many sites each variable is collected at
                available_variables ~ variables available at all sites
                unavailable_variables ~ variables not available at all sites
    '''
    try:
        os.scandir(path) 
    except FileNotFoundError:
        print("Thats not a valid filepath, check for error.")

    except NotADirectoryError:
        # it wasn't a directory that was passed in
        # but this doesn't yet test if the file exists, fix that!
        print("Path for a single file was input, please provide a directory.")
        
    else:
        # it was a directory that was passed in, so let's make use of it
        print('Directory name passed in')
        
        # If given a successful directory...
        # Add site to the column list
        measures.insert(0,'Site')
        my_columns = measures
        # Pull all the filepaths within the directory
        paths = [f.path for f in os.scandir(path) if (f.is_file() and f.name != '.DS_Store')]
        
        merged_data = pd.DataFrame(columns=my_columns)
        # Loop through each file in the path
        for this_file in paths:
            # Retrieve an AmeriFlux dataframe
            data =  check_for_variable(this_file,measures)
            # Concatenate the data
            merged_data = pd.concat([merged_data,data],ignore_index=True)
    
    # Sum how many sites each variable is available at
    presence_counts = merged_data.sum()
    presence_counts = presence_counts.drop('Site')
    # Print some review statements about site counts
    
    # Store all this information into a dictionary so that it is readily available
    variable_info = {}
    variable_info['site_presence'] = merged_data
    variable_info['total_presence'] = presence_counts
    
    # Make a list of variables that are available at all sites 
    available_variables = []
    unavailable_variables = []
    # Print some information on variable availability
    for i in range(len(presence_counts)):
        if (presence_counts[i] == len(merged_data)):
            available_variables.append(presence_counts.index[i])
        else:
            unavailable_variables.append(presence_counts.index[i])
            
    variable_info['available_variables'] = available_variables
    variable_info['unavailable_variables'] = unavailable_variables
    
    # Print some statements about what variables you can use
    print('You have ' + str(len(available_variables)) + ' variables shared across all sites.')
    
    #if (len(available_variables) > 0):
        #print('Variables available at all sites: ' + ', '.join(available_variables)))
    
    return variable_info
    

def check_for_variable(this_file,measures):
    '''
    Name: check_for_variable()
    Summary: Checks which variables of a list exist in a file

    Input: this_file ~ the path of the file you are checking
           measures ~ the variables you want to check if the file has

    Output: measure_df ~ a pandas dataframe with the site and presence/absence
            for each variable
    
    '''
    
    # Pull the site name
    filename = os.path.basename(this_file)
    site = filename[4:10]
    # Open the file
    file = open(this_file,'r')
    # Read read the first line
    first_line = file.readline()
    # Extract all the columns
    file_columns = first_line.strip('\n').split(',')
    # Close the file
    file.close()
    # Create a dataframe with the measures as the column names
    measure_df = pd.DataFrame(columns=measures)
    # For each measure we want to test, presence in the column names = 1, and 
    # absence = 0
    presence = []
    for measure in measure_df.columns:
        if (measure == 'Site'):
            presence.append(site)
        elif (measure in file_columns):
            presence.append(1)
        else:
            presence.append(0)
    
    # Add the presence absence as a row in the dataframe
    measure_df.loc[0,:] = presence
    
    return measure_df


def find_shared_variables_longfile(path,measures,column,value,file_type='xslx'):
    
    '''
    Name: find_shared_variables_longfile()
    Summary: This returns information on variables shared across all AmeriFlux
            files from a given directory. The main purpose is to identify metadata
            variables that can be studied when looking at many sites.

    Input: path ~ filepath for directory of AmeriFlux BADM datasets (metadata)
           measures ~ the variables you want if the files have
           column ~ the column with the list of variables we want to investigate
           value ~ the column name with the values associated with above
           file_type ~ what kind of file it is (AMF BADM is xlsx)

    Output: variable_info ~ dictionary that includes the following information
                site_presence ~ 0/1 indicator of presence of each variable at each site
                total_presence ~ how many sites each variable is collected at
                available_variables ~ variables available at all sites
                unavailable_variables ~ variables not available at all sites
    '''
    try:
        os.scandir(path) 
    except FileNotFoundError:
        print("Thats not a valid filepath, check for error.")

    except NotADirectoryError:
        # it wasn't a directory that was passed in
        # but this doesn't yet test if the file exists, fix that!
        print("Path for a single file was input, please provide a directory.")
        
    else:
        # it was a directory that was passed in, so let's make use of it
        print('Directory name passed in')
        
        # If given a successful directory...
        # Add site to the column list
        measures.insert(0,'Site')
        my_columns = measures
        # Pull all the filepaths within the directory
        paths = [
            f.path for f in os.scandir(path)
            if f.is_file()
            and f.name.endswith('.xlsx')
            and not f.name.startswith('~$')
            and f.name != '.DS_Store'
            ]

        
        merged_data = pd.DataFrame(columns=my_columns)
        # Loop through each file in the path
        for this_file in paths:
            # Retrieve an AmeriFlux dataframe
            data = check_for_variable_longfile(this_file,column,value,measures,file_type)
            # Concatenate the data
            merged_data = pd.concat([merged_data.reset_index(drop=True),
                         data.reset_index(drop=True)])
    
    # Sum how many sites each variable is available at
    presence_counts = merged_data.sum()
    presence_counts = presence_counts.drop('Site')
    # Print some review statements about site counts
    
    # Store all this information into a dictionary so that it is readily available
    variable_info = {}
    variable_info['site_presence'] = merged_data
    variable_info['total_presence'] = presence_counts
    
    # Make a list of variables that are available at all sites 
    available_variables = []
    unavailable_variables = []
    # Print some information on variable availability
    for i in range(len(presence_counts)):
        if (presence_counts[i] == len(merged_data)):
            available_variables.append(presence_counts.index[i])
        else:
            unavailable_variables.append(presence_counts.index[i])
            
    variable_info['available_variables'] = available_variables
    variable_info['unavailable_variables'] = unavailable_variables
    
    # Print some statements about what variables you can use
    print('You have ' + str(len(available_variables)) + ' variables shared across all sites.')
    
    #if (len(available_variables) > 0):
        #print('Variables available at all sites: ' + ', '.join(available_variables)))
    
    return variable_info
    


def check_for_variable_longfile(this_file,column,value,measures,file_type='xslx'):
    """
    Name: check_for_variable_longfile()
    Summary: This checks if a variable is present in the long file raw form of the
            BADM data.
    Input: this_file ~ the pathname for the file
           column ~ the column that the measure names are listed in
           value ~ the column name with the values associated with above
           measures ~ what values we want to check for
           file_type ~ what kind of file it is (AMF BADM is xlsx)

    Output: measure_df ~ dataframe with the presence or absence of each measure
                        in the BADM dataset
    """
    
    if (file_type == 'xslx'):
        # Pull the site name
        filename = os.path.basename(this_file)
        print(filename)
        site = filename[4:10]
        print(f"Pulling data for site {site}")
        # Create a dataframe with the measures as the column names
        measure_df = pd.DataFrame(columns=measures)
        # Read in excel file
        df = pd.read_excel(this_file)
        # Isolate variable column and its values
        df = df[[column,value]]
        # Pivot to wide so that the variables are the columns
        wide = df.set_index(column).T
        # Reset the index
        wide.reset_index(drop=True, inplace=True)
        # Isolate the columns
        file_columns = wide.columns
         
        # Loop through measures and check if they are present in the columns
        presence = []
        for measure in measures:
            if (measure == "Site"):
                presence.append(site)
            elif (measure in file_columns):
                presence.append(1)
            else:
                presence.append(0)
         
        # Add the presence absence as a row in the dataframe
        measure_df.loc[0] = presence
         
    else:
        print("Function doesn't accomodate that kind of file yet.")
    
    return measure_df

print("All data loading functions are loaded!")


def loadBADM(path,skip,column,value,measure,file_type='xslx'):
    """
    Name: loadBADM()
    Summary: This loads a directory of BADM data and organizes into a dataframe
    Input: path ~ the pathname for the directory
           skip ~ any files we want to leave out
           column ~ the column that the measure names are listed in
           value ~ the column name with the values associated with above
           measures ~ the measures we want to oull
           file_type ~ what kind of file it is (AMF BADM is xlsx)

    Output: BADM_data ~ dataframe of all sites and the BADM measures as columns
    """
    
    if skip is None:
        skip = ['']
    
    if measure is None:
        measure = ['IGBP']
    
    # Check if we were given an actual filepath
    try:
        os.scandir(path)
    except FileNotFoundError:
        print("Thats not a valid filepath, check for error.")
        return
    except NotADirectoryError:
        print("Path for a single file was input, please provide a directory.")
        return

    print('Directory name passed in')

    # Add site to the columns
    my_columns = ['Site'] + measure
    # Find all the files within our directory
    paths = [
        f.path for f in os.scandir(path)
        if (f.is_file() and f.name != '.DS_Store' and not f.name.startswith('~$'))
    ]

    # Initialize dataframe with columns of interest
    BADM_data = pd.DataFrame(columns=my_columns)
    # Loop through each file and use loadAMFFile to read in the data
    for this_file in paths:
        if this_file in skip:
            continue
        # Concat onto our cross-site dataframe
        data = loadBADMFile(this_file,column,value,measure,file_type)
        data = data.reset_index(drop=True)
        data = data.loc[:, ~data.columns.duplicated()]
        BADM_data = pd.concat([BADM_data, data], ignore_index=True)

    return BADM_data

def loadBADMFile(this_file,column,value,measures,file_type):
    """
    Name: loadBADMFile()
    Summary: This loads a file of BADM data and pulls out specific measures from it
    Input: this_file ~ the pathname for the file
           column ~ the column that the measure names are listed in
           value ~ the column name with the values associated with above
           measures ~ the measures we want to pull
           file_type ~ what kind of file it is (AMF BADM is xlsx)

    Output: BADM_data ~ dataframe of all sites and the BADM measures as columns
    """
    if (file_type == 'xslx'):
        # Pull the site name
        filename = os.path.basename(this_file)
        site = filename[4:10]
        print(filename)
        print(f"Pulling data for site {site}.")
        # New columns
        new_columns = ['Site'] + measures
        # Read in excel file
        df = pd.read_excel(this_file)
        # Isolate variable column and its values
        df = df[[column,value]]
        # Pivot to wide so that the variables are the columns
        wide = df.set_index(column).T
        # Reset the index
        wide.reset_index(drop=True, inplace=True)
        # Isolate the columns
        file_columns = wide.columns
        # Add set column
        wide['Site'] = site
        # Isolate columns we want
        measure_df = wide[new_columns]
        
    return measure_df

###############################################################################
##                            PRISM/ERA QAQC                                 ##
###############################################################################

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

###############################################################################
##                            AmeriFlux QAQC                                 ##
###############################################################################
    
def plot_flux_QAQC_to_pdf(site, date, NEE, GPP, Reco, NEE_QC, Temp, heatwave,
                          output_pdf="flux_QAQC_all_sites.pdf"):
    """
    Create one multi-page PDF with flux QA/QC plots for each site.

    Parameters
    ----------
    site : array-like
        Site identifier for each observation.
    date : array-like
        Dates for each observation.
    NEE : array-like
        Net ecosystem exchange values.
    GPP : array-like
        Gross primary productivity values.
    Reco : array-like
        Ecosystem respiration values.
    NEE_QC : array-like
        NEE quality control values.
    Temp : array-like
        Temperature values.
    heatwave : array-like
        Heatwave indicator (1 = heatwave point, otherwise not).
    output_pdf : str
        Output PDF filename.
    """

    df = pd.DataFrame({
        "site": site,
        "date": pd.to_datetime(date),
        "NEE": NEE,
        "GPP": GPP,
        "Reco": Reco,
        "NEE_QC": NEE_QC,
        "Temp": Temp,
        "heatwave": heatwave
    })

    df = df.sort_values(["site", "date"])

    def plot_with_heatwave_outline(ax, x, y, bad_qc_mask, heatwave_mask,
                                   x_label="", y_label="", title="",
                                   add_hline_zero=False, add_vline_zero=False):
        # Base colors: red if QC < 0.75, black otherwise
        colors = ["red" if bad else "lightgrey" for bad in bad_qc_mask]

        # Plot all points
        ax.scatter(x, y, c=colors, s=10, alpha=0.7, linewidths=0)

        # Overlay heatwave points with bold black outline
        hw = heatwave_mask.fillna(False)
        ax.scatter(
            x[hw], y[hw],
            facecolors="none",
            edgecolors="black",
            s=42,
            linewidths=1.4
        )

        if add_hline_zero:
            ax.axhline(0, linestyle="--", linewidth=1, color="gray")
        if add_vline_zero:
            ax.axvline(0, linestyle="--", linewidth=1, color="gray")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    with PdfPages(output_pdf) as pdf:
        for s in sorted(df["site"].dropna().unique()):
            d = df[df["site"] == s].copy()

            bad_qc = d["NEE_QC"] < 0.75
            heatwave_mask = d["heatwave"] == 1

            fig, ax = plt.subplots(4, 1, figsize=(12, 14))
            fig.suptitle(f"Flux QA/QC: {s}", fontsize=14)

            # 1. NEE over time
            plot_with_heatwave_outline(
                ax=ax[0],
                x=d["date"],
                y=d["NEE"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="NEE",
                title="NEE over time",
                add_hline_zero=True
            )

            # 2. GPP over time
            plot_with_heatwave_outline(
                ax=ax[1],
                x=d["date"],
                y=d["GPP"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="GPP",
                title="GPP over time",
                add_hline_zero=True
            )

            # 3. Reco over time
            plot_with_heatwave_outline(
                ax=ax[2],
                x=d["date"],
                y=d["Reco"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="Reco",
                title="Reco over time",
                add_hline_zero=False
            )

            # 4. GPP vs temperature
            plot_with_heatwave_outline(
                ax=ax[3],
                x=d["Temp"],
                y=d["GPP"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                x_label="Temperature",
                y_label="GPP",
                title="GPP vs Temperature",
                add_hline_zero=False,
                add_vline_zero=True
            )

            # Legend on first panel only
            legend_handles = [
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='black', markeredgecolor='black',
                           markersize=5, label='QC ≥ 0.75'),
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='red', markeredgecolor='red',
                           markersize=5, label='QC < 0.75'),
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='none', markeredgecolor='black',
                           markeredgewidth=1.4, markersize=7, label='Heatwave = 1')
            ]
            ax[0].legend(handles=legend_handles, loc="best")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved QA/QC plots to {output_pdf}")
    

# This function will do the following to clean the data:
    # Remove all NEE, GPP, and Reco valies where qaqc is less than .75
    # Remove any year where >15% of data is missing
    # Provide a list of sites that no longer have 5 years of data after cleaning

def clean_flux_by_qc_and_years(
    df,
    site_col="Site",
    date_col="date",
    nee_col="NEE",
    gpp_col="GPP",
    reco_col="Reco",
    qc_col="NEE_VUT_REF_QC",
    qc_threshold=0.75,
    missing_frac_threshold=0.15,
    min_years_required=5):
    """
    Clean daily flux data by:
      1) replacing NEE, GPP, and Reco with NA where QC < qc_threshold
      2) treating -9999 as missing
      3) removing site-years where > missing_frac_threshold of records are missing
      4) identifying sites with < min_years_required remaining years

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Original dataframe with bad-QC fluxes set to NA and bad site-years removed.
    year_summary : pandas.DataFrame
        Site-year summary with missingness fractions and keep/remove flag.
    sites_lt_5_years : list
        List of sites with fewer than min_years_required remaining years.
    """

    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Extract year
    df["year"] = df[date_col].dt.year

    # Treat sentinel values as missing
    flux_cols = [nee_col, gpp_col, reco_col]
    cols_to_clean = flux_cols + [qc_col]
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-9999, np.nan)

    # Set fluxes to NA where QC < threshold or QC is missing
    bad_qc = df[qc_col].isna() | (df[qc_col] < qc_threshold)
    df.loc[bad_qc, flux_cols] = np.nan

    # Summarize missingness by site-year
    year_summary = (
        df.groupby([site_col, "year"], dropna=False)
          .agg(
              n_records=(date_col, "size"),
              nee_missing=(nee_col, lambda x: x.isna().sum()),
              gpp_missing=(gpp_col, lambda x: x.isna().sum()),
              reco_missing=(reco_col, lambda x: x.isna().sum())
          )
          .reset_index()
    )

    # Fractions missing
    year_summary["nee_missing_frac"] = year_summary["nee_missing"] / year_summary["n_records"]
    year_summary["gpp_missing_frac"] = year_summary["gpp_missing"] / year_summary["n_records"]
    year_summary["reco_missing_frac"] = year_summary["reco_missing"] / year_summary["n_records"]

    # Remove year if any flux variable exceeds threshold
    year_summary["remove_year"] = (
        (year_summary["nee_missing_frac"] > missing_frac_threshold) |
        (year_summary["gpp_missing_frac"] > missing_frac_threshold) |
        (year_summary["reco_missing_frac"] > missing_frac_threshold)
    )
    year_summary["keep_year"] = ~year_summary["remove_year"]

    # Keep only good site-years
    years_to_keep = year_summary.loc[year_summary["keep_year"], [site_col, "year"]]

    cleaned_df = df.merge(
        years_to_keep.assign(_keep_merge=1),
        on=[site_col, "year"],
        how="inner"
    ).drop(columns="_keep_merge")

    # Count remaining years per site
    remaining_years = (
        cleaned_df[[site_col, "year"]]
        .drop_duplicates()
        .groupby(site_col)
        .size()
        .reset_index(name="n_years_remaining")
    )

    sites_lt_5_years = remaining_years.loc[
        remaining_years["n_years_remaining"] < min_years_required,
        site_col
    ].tolist()

    return cleaned_df, year_summary, sites_lt_5_years


def filter_complete_heatwaves(
    heatwaves_df,
    flux_df,
    site_col="Site",
    start_col="start_dates",
    end_col="end_dates",
    date_col="date",
    flux_cols=("NEE", "RECO", "GPP")):
    """
    Filter heatwaves to keep only those where ALL days in the heatwave
    have available flux data.

    Parameters
    ----------
    heatwaves_df : pd.DataFrame
        One row per heatwave (with start and end dates)
    flux_df : pd.DataFrame
        Daily flux data (must include Site, date, and flux_cols)
    site_col : str
        Site column name
    start_col : str
        Heatwave start date column
    end_col : str
        Heatwave end date column
    date_col : str
        Date column in flux_df
    flux_cols : tuple
        Flux columns that must be non-missing

    Returns
    -------
    clean_heatwaves : pd.DataFrame
        Only heatwaves with complete flux coverage
    summary : pd.DataFrame
        Heatwaves with completeness diagnostics
    """

    # -----------------------
    # Copy + standardize dates
    # -----------------------
    hw = heatwaves_df.copy()
    flux = flux_df.copy()

    hw[start_col] = pd.to_datetime(hw[start_col])
    hw[end_col] = pd.to_datetime(hw[end_col])
    flux[date_col] = pd.to_datetime(flux[date_col])

    # -----------------------
    # Define valid flux days
    # -----------------------
    flux_valid = flux.dropna(subset=[site_col, date_col, *flux_cols])

    valid_days = set(zip(flux_valid[site_col], flux_valid[date_col]))

    # -----------------------
    # Check completeness per heatwave
    # -----------------------
    def check_heatwave(row):
        dates = pd.date_range(row[start_col], row[end_col], freq="D")
        pairs = [(row[site_col], d) for d in dates]

        n_total = len(pairs)
        n_available = sum(pair in valid_days for pair in pairs)

        return pd.Series({
            "n_heatwave_days": n_total,
            "n_flux_days_available": n_available,
            "all_flux_days_available": n_total == n_available
        })

    summary_cols = hw.apply(check_heatwave, axis=1)
    summary = pd.concat([hw, summary_cols], axis=1)

    # -----------------------
    # Filter to complete heatwaves
    # -----------------------
    clean_heatwaves = summary.loc[summary["all_flux_days_available"]].copy()

    return clean_heatwaves[hw.columns], summary

def normalize_nee_by_site(
    df,
    site_col="Site",
    nee_col="NEE",
    baseline_mask=None,
    method="zscore"
):
    """
    Normalize NEE within each site before heatwave summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing site and NEE columns.
    site_col : str
        Column name for site ID.
    nee_col : str
        Column name for raw NEE values.
    baseline_mask : pandas.Series or None
        Boolean mask indicating which rows to use to compute site baseline stats.
        If None, uses all rows within each site.
    method : str
        "center" = NEE - site_mean
        "zscore" = (NEE - site_mean) / site_sd

    Returns
    -------
    pandas.DataFrame
        Copy of df with:
        - site_mean
        - site_sd
        - NEE_norm
    """
    df = df.copy()

    if baseline_mask is None:
        baseline_df = df.copy()
    else:
        baseline_df = df.loc[baseline_mask].copy()

    # Compute site-level baseline stats
    site_stats = (
        baseline_df.groupby(site_col)[nee_col]
        .agg(site_mean="mean", site_sd="std")
        .reset_index()
    )

    # Join stats back to full dataframe
    df = df.merge(site_stats, on=site_col, how="left")

    if method == "center":
        df["NEE_norm"] = df[nee_col] - df["site_mean"]

    elif method == "zscore":
        # avoid divide-by-zero
        df["NEE_norm"] = np.where(
            df["site_sd"].notna() & (df["site_sd"] != 0),
            (df[nee_col] - df["site_mean"]) / df["site_sd"],
            np.nan
        )
    else:
        raise ValueError("method must be 'center' or 'zscore'")

    return df

###############################################################################
##                       Adjusting flux data                                 ##
###############################################################################
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

###############################################################################
##                             Heatwave QAQC                                 ##
###############################################################################

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


def plot_flux_QAQC(site, date, NEE, GPP, Reco, NEE_QC, Temp):
    # Build dataframe
    df = pd.DataFrame({
        "site": site,
        "date": pd.to_datetime(date),
        "NEE": NEE,
        "GPP": GPP,
        "Reco": Reco,
        "NEE_QC": NEE_QC,
        "Temp": Temp
    })
    
    # Sort values
    df = df.sort_values(["site", "date"])
    
    # Loop through sites
    for s in df["site"].dropna().unique():
        d = df[df["site"] == s].copy()
        
        fig, ax = plt.subplots(4, 1, figsize=(12, 14))
        fig.suptitle(f"Flux QA/QC: {s}", fontsize=14)
        
        # 1. NEE over time, QC < 0.75 in red
        nee_bad = d["NEE_QC"] < 0.75
        ax[0].scatter(d.loc[~nee_bad, "date"], d.loc[~nee_bad, "NEE"], s=8, label="QC ≥ 0.75", color="grey")
        ax[0].scatter(d.loc[nee_bad, "date"], d.loc[nee_bad, "NEE"], s=8, color="red", label="QC < 0.75")
        ax[0].axhline(0, linestyle="--", linewidth=1)
        ax[0].set_ylabel("NEE")
        ax[0].set_title("NEE over time")
        ax[0].legend()
        
        # 2. GPP over time, GPP < 0 in bright blue
        gpp_neg = d["GPP"] < 0
        ax[1].scatter(d.loc[~gpp_neg, "date"], d.loc[~gpp_neg, "GPP"], s=8, label="GPP ≥ 0", color="grey")
        ax[1].scatter(d.loc[gpp_neg, "date"], d.loc[gpp_neg, "GPP"], s=8, color="deepskyblue", label="GPP < 0")
        ax[1].axhline(0, linestyle="--", linewidth=1)
        ax[1].set_ylabel("GPP")
        ax[1].set_title("GPP over time")
        ax[1].legend()
        
        # 3. Reco over time
        ax[2].scatter(d["date"], d["Reco"], s=8,color="grey")
        ax[2].axhline(0, linestyle="--", linewidth=1)
        ax[2].set_ylabel("Reco")
        ax[2].set_title("Reco over time")
        
        # 4. GPP by temperature, Temp < 0 in bright blue
        temp_freezing = d["Temp"] < 0
        ax[3].scatter(d.loc[~temp_freezing, "Temp"], d.loc[~temp_freezing, "GPP"], s=8, label="Temp ≥ 0°C",color="grey")
        ax[3].scatter(d.loc[temp_freezing, "Temp"], d.loc[temp_freezing, "GPP"], s=8, color="deepskyblue", label="Temp < 0°C")
        ax[3].axvline(0, linestyle="--", linewidth=1)
        ax[3].axhline(0, linestyle="--", linewidth=1)
        ax[3].set_xlabel("Temperature")
        ax[3].set_ylabel("GPP")
        ax[3].set_title("GPP vs Temperature")
        ax[3].legend()
        
        plt.tight_layout()
        plt.show()
        

###############################################################################
##                        Defining Heatwaves                                 ##
###############################################################################

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

###############################################################################
##                        Find heatwave overlap                              ##
###############################################################################

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
            print(f"Working on site {site}")
            # Create subdictioary for the site
            final_heatwaves[site] = {}
            
            # Pull the indicator for each type of heatwave
            min_indicator = min_heatwaves[site]['indicator'].iloc[:,0:2]
            max_indicator = max_heatwaves[site]['indicator'].iloc[:,0:2]
            avg_indicator = avg_heatwaves[site]['indicator'].iloc[:,0:2]
            
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
            
            # Find the 3d composition of each heatwave
            heatwaves3d = get_3dheatwave(top_heatwave, indicator_df)
            
            # Merge the cateogrical and 3d heatwave types
            top_heatwave = pd.merge(top_heatwave,heatwaves3d,on=['start_dates','end_dates'],how='left')
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


def get_3dheatwave(top_heatwave, indicator_df):
    
    heatwave_perc = pd.DataFrame(columns=['start_dates','end_dates',
                                                 'Max_perc','Min_perc','Mean_perc'])
    for i in range(0,len(top_heatwave)):
        # Isolate a heatwave
        hw = top_heatwave.iloc[i]
        # Find the range of dates it covers
        hw_date_range = pd.date_range(start=hw.start_dates,end=hw.end_dates)
        # Pull the indicator data for those dates
        hw_indicator = indicator_df[indicator_df.date.isin(hw_date_range)]
        # Find percentage of each that is greater indicator that is not 0
        max_percent = len(hw_indicator[hw_indicator.max_indicator > 0]) / len(hw_indicator)
        min_percent = len(hw_indicator[hw_indicator.min_indicator > 0]) / len(hw_indicator)
        avg_percent = len(hw_indicator[hw_indicator.avg_indicator > 0]) / len(hw_indicator)
        # Organize results into a dataframe
        new_hw = [hw.start_dates,hw.end_dates,max_percent,min_percent,avg_percent]
        new_hw = pd.DataFrame([new_hw], columns=heatwave_perc.columns)
        # Concat with overall dataframe
        heatwave_perc = pd.concat([heatwave_perc,new_hw])
    return heatwave_perc

###############################################################################
##                          Heatwave Indices                                 ##
###############################################################################

def cumulative_exceedence(heatwaves_dictionary, AMF_TA, historical_data, TA_name, method):
    '''
    This function integrates over a heatwave and returns the summed and average 
    deviation from th 95th percentile of historical temperature for that day. This
    can be applied to mean, minimum, or maximum temperatures.
    
    Parameters
    ----------
    heatwaves_dictionary : TYPE
        DESCRIPTION. Dictionary of all heatwaves.
    
    
    historical_dates : TYPE
        DESCRIPTION. list of all heatwaves
    
    historical_temperatures : TYPE
        DESCRIPTION. list of all dates corresponding to historical data
        
    method : str
        DESCRIPTION. Either 'max','min', or 'mean', based on the temperature measure
        we are interested in.
    
    Returns
    -------
    heatwave_categories : TYPE
        DESCRIPTION. A categorical list for the type of overlapping heatwave each 
        day is, including None, Night, Day, Overall, Night-intensified, 
        Day-intensified, Day-Night Spike, and Triad

    '''
    # Initialize dataframe for all heatwaves
    heatwave_indices = pd.DataFrame(columns=['Site','start_dates','end_dates',
                                             'cumulative_exceedence','average_exceedence'])
    
    for site in heatwaves_dictionary.keys():
        print(f"Integrating over site {site}")
        if (method=='max'):
            site_hourly_temp = AMF_TA[AMF_TA.Site==site].TA_F
            site_hourly_date = AMF_TA[AMF_TA.Site==site].TIMESTAMP_START
            site_temperature = find_max_temperatures(site_hourly_date,site_hourly_temp)
            site_temperature.columns = ['date','TA_F']
        elif (method=='min'):
            site_hourly_temp = AMF_TA[AMF_TA.Site==site].TA_F
            site_hourly_date = AMF_TA[AMF_TA.Site==site].TIMESTAMP_START
            site_temperature = find_min_temperatures(site_hourly_date,site_hourly_temp)
            site_temperature.columns = ['date','TA_F']
        else:
            site_temperature = AMF_TA[AMF_TA.Site==site]
            
        # Pull out the dictionary for that site
        site_dictionary = heatwaves_dictionary[site]
        site_historical = historical_data[historical_data['Site']==site]
        historical_temperature = site_historical[TA_name]
        historical_dates = pd.to_datetime(site_historical['date'])
        site_heatwaves = pd.DataFrame({'start_dates':site_dictionary['start_dates'],
                                       'end_dates':site_dictionary['end_dates']})
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
            .quantile(.95)
            .reset_index(name="T95")
        )
        # Add a month_day column to site temperature (not historical)
        site_temperature['month_day'] = site_temperature.date.dt.strftime('%m-%d')
        # Merge timeseries data with the DMT 95th quantiles
        site_temperature = pd.merge(site_temperature,T95_by_day,on="month_day",how="left")
        # Calculate difference between threshold and observed temperature
        site_temperature['difference'] = site_temperature.TA_F - site_temperature.T95

        # Now loop through each heatwave
        cumulative_exceedence = []
        average_exceedence = []
        for i in range(len(site_heatwaves)):
            # Find date range of heatwaves
            start = site_heatwaves['start_dates'][i]
            end = site_heatwaves['end_dates'][i]
            date_range = pd.date_range(start,end)
            # Find the corresponding temperatures for these dates
            heatwave_difference = site_temperature[site_temperature['date'].isin(date_range)].difference
            # Replace negative values with zero
            heatwave_difference[heatwave_difference<0] = 0
            cumulative_exceedence.append(sum(heatwave_difference))
            average_exceedence.append(sum(heatwave_difference) / len(heatwave_difference))
        site_heatwaves['cumulative_exceedence'] = cumulative_exceedence
        site_heatwaves['average_exceedence'] = average_exceedence
        site_heatwaves['Site'] = [site]*len(site_heatwaves)
    
        # Concat the site heatwaves with all heatwaves
        heatwave_indices = pd.concat([heatwave_indices,site_heatwaves])
    
    return heatwave_indices

###############################################################################
##                        Calculating Summaries                              ##
###############################################################################

def calc_flux_avg(flux_name, heatwaves_df, flux_df, before_lag, after_lag):
    """
    Parameters
    ----------
    flux_name : STR
        DESCRIPTION. String name of the flux you want to calculate for
    heatwaves_df : pd.DataFrame
        DESCRIPTION. Includes all heatwaves including start_date, end_date
    flux_df : pd.DataFrame
        DESCRIPTION. Includes columns including date and "flux_name"
    before_lag : INT
        DESCRIPTION. Number of days prior to heatwave you want in before period
    after_lag : INT
        DESCRIPTION. Number of days after heatwave you want in after period

    Returns
    -------
    heatwaves_df : TYPE
        DESCRIPTION. Same dataframe with added columns for average flux over
        before, during, and after heatwave periods

    """
    
    # Find the dates of before and after heatwave periods we want to calculate for
    heatwaves_df['before_hw'] = heatwaves_df.start_dates - timedelta(days=before_lag)
    heatwaves_df['after_hw'] = heatwaves_df.end_dates + timedelta(days=after_lag)
    
    # Initialize list to store the period averages we calculate
    before_flux_avg = []
    during_flux_avg = []
    after_flux_avg = []
    before_flux_std = []
    during_flux_std = []
    after_flux_std = []
    
    # Now loop through each heatwave and calculate the period averages
    for k in range(0,len(heatwaves_df)):
        
        # Isolate start and end dates for each heatwave
        site = heatwaves_df.iloc[k].Site
        before = heatwaves_df.iloc[k].before_hw
        start = heatwaves_df.iloc[k].start_dates
        end = heatwaves_df.iloc[k].end_dates
        after = heatwaves_df.iloc[k].after_hw
        
        # Create a date range for each period
        before_dates = pd.date_range(before, start)
        during_dates = pd.date_range(start, end)
        after_dates = pd.date_range(end, after)
        
        # Calculate average and append to list
        before_flux_avg.append(flux_df[(flux_df.date.isin(before_dates)) & (flux_df.Site==site)][flux_name].mean())
        during_flux_avg.append(flux_df[(flux_df.date.isin(during_dates)) & (flux_df.Site==site)][flux_name].mean())
        after_flux_avg.append(flux_df[(flux_df.date.isin(after_dates)) & (flux_df.Site==site)][flux_name].mean())
        
        # Calculate std and append to list
        before_flux_std.append(flux_df[(flux_df.date.isin(before_dates)) & (flux_df.Site==site)][flux_name].std())
        during_flux_std.append(flux_df[(flux_df.date.isin(during_dates)) & (flux_df.Site==site)][flux_name].std())
        after_flux_std.append(flux_df[(flux_df.date.isin(after_dates)) & (flux_df.Site==site)][flux_name].std())
        
    # Assign these to the dataframe and return!
    heatwaves_df[flux_name + "_before_avg"] = before_flux_avg
    heatwaves_df[flux_name + "_during_avg"] = during_flux_avg
    heatwaves_df[flux_name + "_after_avg"] = after_flux_avg
    heatwaves_df[flux_name + "_before_std"] = before_flux_std
    heatwaves_df[flux_name + "_during_std"] = during_flux_std
    heatwaves_df[flux_name + "_after_std"] = after_flux_std
    
    return heatwaves_df

def calc_flux_multi_lag(
    flux_name,
    heatwaves_df,
    flux_df,
    stats=("avg", "std","sum"),
    before_lags=None,
    after_lags=None,
    min_frac=0.7,
    site_col="Site",
    start_col="start_dates",
    end_col="end_dates",
    date_col="date"):
    """
    Calculate selected summary statistics of a flux before, during, and after heatwaves
    for multiple lag windows.
    
    Average and std will be calculated if above 70% of data is available for that lag.
    Sum will be not be calculated if there is any missing data in the lag.

    Parameters
    ----------
    flux_name : str
        Column in flux_df to summarize.
    heatwaves_df : pd.DataFrame
        One row per heatwave.
    flux_df : pd.DataFrame
        Daily flux dataframe.
    stats : str or iterable of str
        Which statistics to calculate. Options: "sum", "avg", "std".
        Example: stats="sum" or stats=("sum", "avg")
    before_lags : list or None
        Lag lengths before the heatwave.
    after_lags : list or None
        Lag lengths after the heatwave.
    min_frac : float
        Minimum fraction of non-missing days required for avg/std.
    site_col, start_col, end_col, date_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        heatwaves_df with added columns.
    """

    if before_lags is None:
        before_lags = []
    if after_lags is None:
        after_lags = []

    if isinstance(stats, str):
        stats = [stats]
    stats = [s.lower() for s in stats]

    valid_stats = {"sum", "avg", "std"}
    invalid_stats = set(stats) - valid_stats
    if invalid_stats:
        raise ValueError(f"Invalid stats: {invalid_stats}. Choose from {valid_stats}.")

    heatwaves_df = heatwaves_df.copy()
    flux_df = flux_df.copy()

    heatwaves_df[start_col] = pd.to_datetime(heatwaves_df[start_col])
    heatwaves_df[end_col] = pd.to_datetime(heatwaves_df[end_col])
    flux_df[date_col] = pd.to_datetime(flux_df[date_col])

    def calc_requested_stats(vals, expected_len):
        """
        Rules:
        - sum: require 100% non-missing coverage
        - avg/std: require >= min_frac non-missing coverage
        """
        n_available = vals.notna().sum()
        out = {}

        if "sum" in stats:
            out["sum"] = vals.sum() if n_available == expected_len else pd.NA

        if "avg" in stats:
            out["avg"] = vals.mean() if expected_len > 0 and (n_available / expected_len) >= min_frac else pd.NA

        if "std" in stats:
            out["std"] = vals.std() if expected_len > 0 and (n_available / expected_len) >= min_frac else pd.NA

        return out

    def get_window_vals(site, start_date, end_date):
        window_dates = pd.date_range(start_date, end_date)
        vals = flux_df[
            (flux_df[site_col] == site) &
            (flux_df[date_col].isin(window_dates))
        ][flux_name]
        return vals, len(window_dates)

    # ----- DURING -----
    during_results = {stat: [] for stat in stats}

    for k in range(len(heatwaves_df)):
        row = heatwaves_df.iloc[k]
        vals, expected_len = get_window_vals(row[site_col], row[start_col], row[end_col])
        out = calc_requested_stats(vals, expected_len)

        for stat in stats:
            during_results[stat].append(out[stat])

    for stat in stats:
        heatwaves_df[f"{flux_name}_during_{stat}"] = during_results[stat]

    # ----- BEFORE LAGS -----
    for lag in before_lags:
        lag_results = {stat: [] for stat in stats}

        for k in range(len(heatwaves_df)):
            row = heatwaves_df.iloc[k]
            before_start = row[start_col] - timedelta(days=lag)
            before_end = row[start_col] - timedelta(days=1)

            vals, expected_len = get_window_vals(row[site_col], before_start, before_end)
            out = calc_requested_stats(vals, expected_len)

            for stat in stats:
                lag_results[stat].append(out[stat])

        for stat in stats:
            heatwaves_df[f"{flux_name}_before_{stat}_{lag}"] = lag_results[stat]

    # ----- AFTER LAGS -----
    for lag in after_lags:
        lag_results = {stat: [] for stat in stats}

        for k in range(len(heatwaves_df)):
            row = heatwaves_df.iloc[k]
            after_start = row[end_col] + timedelta(days=1)
            after_end = row[end_col] + timedelta(days=lag)

            vals, expected_len = get_window_vals(row[site_col], after_start, after_end)
            out = calc_requested_stats(vals, expected_len)

            for stat in stats:
                lag_results[stat].append(out[stat])

        for stat in stats:
            heatwaves_df[f"{flux_name}_after_{stat}_{lag}"] = lag_results[stat]

    return heatwaves_df


def DOY_climatology(df, var_name, smoothing_function="weighted_15"):
    
    '''
    EDITS: NEED TO REMOVE HEATWAVE DAYS FROM THIS AND SMOOTH THE VALUES BEFORE RETURNING
    
    Description: This function calculates an expected value for each DOY at each site
    for some variable. It will be used in cumulative deviation calculations. It
    excludes any heatwave days when calculating the expected value
    
    Parameters
    ----------
    df: pd.DataFrame
        DESCRIPTION. includes site, date, and daily values of variable var_name
    var_name : string
        DESCRIPTION. variable name in df that we are calculated expected value for.
        Options include weighted_15, weighted_20, weighted_25, fourier
    smoothing_function: str
        DESCRIPTION. smoothing function you want to use for calculating expected flux value

    Returns
    -------
    new_df : TYPE
        DESCRIPTION. Same dataframe with added column with DOY climatology-based 
        expected value for var_name
    '''
    
    # Convert date to day of year format
    df["DOY"] = df.date.dt.dayofyear
    
    # Check if we have a heatwave indicator, if not tell us to add it
    if ("heatwave_indicator" not in df.columns):
        print("Please add a heatwave indictor to this df.")
        print("Use: df = add_heatwave_indicator(df,all_heatwaves_df)")
    else:
        print("You have the heatwave indicator column!")
    
    # Remove any colums with heatwaves
    nohw_df = df[df.heatwave_indicator==0]
    
    # Extract columns of df
    col_list = df.columns.tolist()
    col_list.append("DOY_"+var_name)
    
    # Create dataframe to store new df
    new_df = pd.DataFrame(columns=col_list)
    
    # Loop through each site
    for site in df.Site.unique():
        site_df_nohw = nohw_df[nohw_df.Site==site]
        site_df = df[df.Site==site]
        
        # Calculate mean flux for each DOY
        expected_value = site_df_nohw.groupby('DOY')[var_name].mean().reset_index()
        expected_value.columns = ["DOY", "DOY_"+var_name]
        
        # Now we want to smooth this
        if (smoothing_function == "fourier"):
            expected_value["expected_"+var_name] = fourier_smooth_fft(expected_value["DOY_"+var_name], n_harmonics=3)
        elif (smoothing_function == "weighted_15"):
            expected_value["expected_"+var_name] = rolling_weighted_mean(pd.Series(expected_value["DOY_"+var_name]),window=15)
        elif (smoothing_function == "weighted_20"):
            expected_value["expected_"+var_name] = rolling_weighted_mean(pd.Series(expected_value["DOY_"+var_name]),window=20)
        elif (smoothing_function == "weighted_25"):
            expected_value["expected_"+var_name] = rolling_weighted_mean(pd.Series(expected_value["DOY_"+var_name]),window=25)
        else:
            print("You didn't provide a valid smoothing function. Try weighted_15 or fourier.")
        
        # Merge with site df
        site_df = pd.merge(site_df,expected_value,on="DOY",how="left")
        
        # Concat with other sites
        new_df = pd.concat([new_df,site_df])
        
    return new_df


def fourier_smooth_fft(y, n_harmonics=3):
    """
    This is the fourier smoothing in order to get expected flux for deviation
    calculations.
    Keep only the lowest n_harmonics seasonal harmonics (plus the mean).
    n_harmonics=1 keeps annual cycle only,
    2 keeps annual + semiannual, etc.
    """
    y = np.asarray(y, dtype=float)
    N = y.size

    # FFT
    Y = np.fft.rfft(y)

    # Zero out high frequencies (keep k=0..n_harmonics)
    Y_filtered = np.zeros_like(Y)
    Y_filtered[:n_harmonics + 1] = Y[:n_harmonics + 1]

    # Inverse FFT to get smoothed signal
    y_smooth = np.fft.irfft(Y_filtered, n=N)
    return y_smooth

def rolling_weighted_mean(series, window):
    """
    Centered triangular-weight rolling mean.
    """
    # Create symmetric triangular weights
    half = window // 2
    weights = np.arange(1, half + 2)
    if window % 2 == 0:
        weights = np.concatenate([weights, weights[::-1]])
    else:
        weights = np.concatenate([weights, weights[-2::-1]])

    weights = weights / weights.sum()

    return series.rolling(window, center=True, min_periods=1) \
                 .apply(lambda x: np.dot(x, weights[:len(x)]), raw=True)


def add_heatwave_indicator(df,heatwaves):
    """
    This function takes the daily flux dataframe and dataframe of all heatwaves
    (with start and end dates) and returns the flux dataframe with an indicator
    column of whether or not it is in a heatwave that day.
    """
    # Create a copy of the df
    new_df = df
    
    # Reduce columns
    heatwaves = heatwaves[["Site","start_dates","end_dates","top_heatwave"]]
    
    # Create column holding date range of heatwaves
    heatwaves["date"] = heatwaves.apply(
        lambda row: pd.date_range(row["start_dates"], row["end_dates"]),
        axis=1
        )
    
    # Explode the heatwave column
    heatwaves_expanded = heatwaves.explode("date").reset_index(drop=True)
    
    # Create indicator column for the heatwaves
    heatwaves_expanded["heatwave_indicator"] = [1] * len(heatwaves_expanded)
    
    # Left merge with the df
    new_df = pd.merge(new_df,heatwaves_expanded,on=['Site','date'],how="left")

    # Replacing missing heatwave indicators with 0
    new_df.loc[new_df.heatwave_indicator.isna(),"heatwave_indicator"] = 0
    
    return new_df

def symmetric_dev_calc(observed,expected):
    '''
    This function takes the observed and expected (calculated from DOY climatology)
    in order to calculate the symmetric percent change or the daily flux "deviance".
    
    '''
    symmetric_percent_change = 2*(observed - expected) / (abs(observed) + abs(expected))
    return symmetric_percent_change


def ttest_by_cat(df,testing_var,grouping_var):
    
    results = []

    for hw, g in all_heatwaves_df.groupby(grouping_var):
        
        vals = pd.to_numeric(g[testing_var], errors="coerce").dropna()
        
        mean_val = vals.mean()
        t_stat, p_val = stats.ttest_1samp(vals, popmean=0)
        
        results.append({
            "top_heatwave": hw,
            "mean": mean_val,
            "t_stat": t_stat,
            "p_value": p_val,
            "n": len(vals)
        })

    ttest_df = pd.DataFrame(results)
    ttest_df
    
    return ttest_df

def compute_metrics(df, expected_len, col, min_frac_mean=0.7):
    """
    Returns (sum_val, mean_val) with rules:
    - sum only if 100% data present
    - mean if >= min_frac_mean data present
    
    Using this to calculate the cumulative and mean deviation prior to and during 
    a heatwave.
    """
    n_available = df[col].notna().sum()
    
    # SUM: require full coverage
    if n_available == expected_len:
        sum_val = df[col].sum()
    else:
        sum_val = pd.NA
    
    # MEAN: require >= threshold
    if expected_len > 0 and (n_available / expected_len) >= min_frac_mean:
        mean_val = df[col].mean()
    else:
        mean_val = pd.NA
    
    return sum_val, mean_val

###############################################################################
##                             RF Regressions                                ##
###############################################################################

def fit_RFregression(
    X,
    y,
    cat_cols=None,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_pred=True,
    plot_importance=True):
    """
    Fit a LightGBM regressor and return model outputs + diagnostics.

    Returns a dictionary containing:
    - fitted model
    - predictions
    - RMSE and R2
    - split importance
    - gain importance
    - permutation importance
    - combined importance table
    """

    if cat_cols is None:
        cat_cols = []

    # Fit model
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )

    model.fit(
        X,
        y,
        categorical_feature=cat_cols,
        eval_set=[(X, y)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    )

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("RMSE:", rmse)
    print("R2:", r2)

    # Split importance
    split_importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="split_importance"
    ).sort_values(ascending=False)

    split_importance_percent = (
        split_importance / split_importance.sum()
    ).rename("split_importance_percent")

    # Gain importance
    gain_importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
        name="gain_importance"
    ).sort_values(ascending=False)

    gain_importance_percent = (
        gain_importance / gain_importance.sum()
    ).rename("gain_importance_percent")

    # Permutation importance
    perm_result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    permutation_importance_mean = pd.Series(
        perm_result.importances_mean,
        index=X.columns,
        name="permutation_importance_mean"
    ).sort_values(ascending=False)

    permutation_importance_std = pd.Series(
        perm_result.importances_std,
        index=X.columns,
        name="permutation_importance_std"
    )

    # Combined table
    importance_table = pd.concat(
        [
            split_importance,
            split_importance_percent,
            gain_importance,
            gain_importance_percent,
            permutation_importance_mean,
            permutation_importance_std
        ],
        axis=1
    )

    # Observed vs predicted plot
    if plot_pred:
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))

        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, s=5, alpha=0.5, c="black")
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", c="red")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title("Observed vs Predicted")
        plt.tight_layout()
        plt.show()

    # Importance plots
    if plot_importance:
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))

        # Split importance
        split_importance.plot.bar(ax=axes[0])
        axes[0].set_title("Split Importance")
        axes[0].set_ylabel("Split Count")
        axes[0].tick_params(axis="x", rotation=90)

        # Gain importance
        gain_importance.plot.bar(ax=axes[1])
        axes[1].set_title("Gain Importance")
        axes[1].set_ylabel("Total Gain")
        axes[1].tick_params(axis="x", rotation=90)

        # Permutation importance
        permutation_importance_mean.plot.bar(ax=axes[2])
        axes[2].set_title("Permutation Importance")
        axes[2].set_ylabel("Mean Importance")
        axes[2].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.show()

    results = {
        "model": model,
        "predictions": y_pred,
        "rmse": rmse,
        "r2": r2,
        "split_importance": split_importance,
        "split_importance_percent": split_importance_percent,
        "gain_importance": gain_importance,
        "gain_importance_percent": gain_importance_percent,
        "permutation_importance_mean": permutation_importance_mean,
        "permutation_importance_std": permutation_importance_std,
        "importance_table": importance_table
    }

    return results

def fit_RFregression_split(
    X,
    y,
    site_col,
    cat_cols=None,
    test_size=0.2,
    val_size=0.2,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_pred=True,
    plot_importance=True):
    """
    Fit a LightGBM regressor with site-balanced train/validation/test splits.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor dataframe. Must include `site_col`.
    y : pd.Series or array-like
        Response variable.
    site_col : str
        Column in X identifying site membership.
    cat_cols : list, optional
        List of categorical columns for LightGBM.
    test_size : float, default 0.2
        Fraction of each site's rows assigned to test.
    val_size : float, default 0.2
        Fraction of each site's remaining non-test rows assigned to validation.
        So with test_size=0.2 and val_size=0.2, the final split is:
        train = 64%, val = 16%, test = 20% within each site.

    Returns
    -------
    dict
        Dictionary with fitted model, split data, predictions, metrics,
        and importance tables.
    """

    if cat_cols is None:
        cat_cols = []

    if site_col not in X.columns:
        raise ValueError(f"`site_col`='{site_col}' not found in X.columns")

    X = X.copy()
    y = pd.Series(y, index=X.index, name="y")

    # Check split sizes
    if not (0 < test_size < 1):
        raise ValueError("`test_size` must be between 0 and 1.")
    if not (0 < val_size < 1):
        raise ValueError("`val_size` must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    train_idx = []
    val_idx = []
    test_idx = []

    # Split within each site
    for site, site_index in X.groupby(site_col).groups.items():
        site_index = np.array(list(site_index))
        n_site = len(site_index)

        if n_site < 3:
            raise ValueError(
                f"Site '{site}' has only {n_site} row(s). "
                "Need at least 3 rows per site for train/val/test splitting."
            )

        shuffled = rng.permutation(site_index)

        n_test = max(1, int(round(n_site * test_size)))
        remaining_after_test = n_site - n_test

        n_val = max(1, int(round(remaining_after_test * val_size)))

        # Make sure at least 1 observation remains for training
        if remaining_after_test - n_val < 1:
            n_val = remaining_after_test - 1

        if n_val < 1:
            raise ValueError(
                f"Site '{site}' does not have enough rows after test split "
                "to allocate validation and training data."
            )

        site_test = shuffled[:n_test]
        site_val = shuffled[n_test:n_test + n_val]
        site_train = shuffled[n_test + n_val:]

        if len(site_train) < 1:
            raise ValueError(
                f"Site '{site}' ended up with no training rows. "
                "Try decreasing test_size or val_size."
            )

        train_idx.extend(site_train)
        val_idx.extend(site_val)
        test_idx.extend(site_test)

    # Convert to pandas indices
    train_idx = pd.Index(train_idx)
    val_idx = pd.Index(val_idx)
    test_idx = pd.Index(test_idx)

    # Create split datasets
    X_train = X.loc[train_idx].copy()
    X_val = X.loc[val_idx].copy()
    X_test = X.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_val = y.loc[val_idx].copy()
    y_test = y.loc[test_idx].copy()

    # Fit model
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    )

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Train RMSE:", train_rmse)
    print("Validation RMSE:", val_rmse)
    print("Test RMSE:", test_rmse)
    print("Train R2:", train_r2)
    print("Validation R2:", val_r2)
    print("Test R2:", test_r2)

    print("\nRows per split:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    print("\nSite counts per split:")
    print(f"Train sites: {X_train[site_col].nunique()}")
    print(f"Validation sites: {X_val[site_col].nunique()}")
    print(f"Test sites: {X_test[site_col].nunique()}")

    # Split importance
    split_importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="split_importance"
    ).sort_values(ascending=False)

    split_importance_percent = (
        split_importance / split_importance.sum()
    ).rename("split_importance_percent")

    # Gain importance
    gain_importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
        name="gain_importance"
    ).sort_values(ascending=False)

    gain_importance_percent = (
        gain_importance / gain_importance.sum()
    ).rename("gain_importance_percent")

    # Permutation importance on test set
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    permutation_importance_mean = pd.Series(
        perm_result.importances_mean,
        index=X.columns,
        name="permutation_importance_mean"
    ).sort_values(ascending=False)

    permutation_importance_std = pd.Series(
        perm_result.importances_std,
        index=X.columns,
        name="permutation_importance_std"
    )

    # Combined table
    importance_table = pd.concat(
        [
            split_importance,
            split_importance_percent,
            gain_importance,
            gain_importance_percent,
            permutation_importance_mean,
            permutation_importance_std
        ],
        axis=1
    )

    # Optional: summary table of site split counts
    split_summary = pd.concat(
        [
            X_train[site_col].value_counts().rename("train_n"),
            X_val[site_col].value_counts().rename("val_n"),
            X_test[site_col].value_counts().rename("test_n")
        ],
        axis=1
    ).fillna(0).astype(int)

    # Observed vs predicted plot (test set)
    if plot_pred:
        min_val_plot = min(np.min(y_test), np.min(y_test_pred))
        max_val_plot = max(np.max(y_test), np.max(y_test_pred))

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_test_pred, s=8, alpha=0.5, c="black")
        plt.plot(
            [min_val_plot, max_val_plot],
            [min_val_plot, max_val_plot],
            linestyle="dashed",
            c="red"
        )
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title("Observed vs Predicted (Test Set)")
        plt.tight_layout()
        plt.show()

    # Importance plots
    if plot_importance:
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))

        split_importance.plot.bar(ax=axes[0])
        axes[0].set_title("Split Importance")
        axes[0].set_ylabel("Split Count")
        axes[0].tick_params(axis="x", rotation=90)

        gain_importance.plot.bar(ax=axes[1])
        axes[1].set_title("Gain Importance")
        axes[1].set_ylabel("Total Gain")
        axes[1].tick_params(axis="x", rotation=90)

        permutation_importance_mean.plot.bar(ax=axes[2])
        axes[2].set_title("Permutation Importance (Test Set)")
        axes[2].set_ylabel("Mean Importance")
        axes[2].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.show()

    results = {
        "model": model,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_predictions": y_train_pred,
        "val_predictions": y_val_pred,
        "test_predictions": y_test_pred,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "split_importance": split_importance,
        "split_importance_percent": split_importance_percent,
        "gain_importance": gain_importance,
        "gain_importance_percent": gain_importance_percent,
        "permutation_importance_mean": permutation_importance_mean,
        "permutation_importance_std": permutation_importance_std,
        "importance_table": importance_table,
        "split_summary_by_site": split_summary,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }

    return results

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fit_RFregression_cv(
    X,
    y,
    site_col,
    cat_cols=None,
    n_splits=5,
    val_size=0.2,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_importance=True,
    include_site_as_predictor=False
):
    """
    Fit a LightGBM regressor using site-balanced cross-validation.

    Outer loop:
        site-balanced K-fold CV where each fold contains rows from all sites.

    Inner step within each outer training fold:
        site-balanced validation split for early stopping.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor dataframe. Must include `site_col` so splits can be made.
    y : pd.Series or array-like
        Response variable.
    site_col : str
        Column identifying site membership.
    cat_cols : list, optional
        List of categorical columns for LightGBM.
    include_site_as_predictor : bool, default False
        If False, `site_col` is used only for splitting and is dropped before modeling.
        If True, `site_col` is retained as a predictor.
    """

    if cat_cols is None:
        cat_cols = []

    if site_col not in X.columns:
        raise ValueError(f"`site_col`='{site_col}' not found in X.columns")

    if n_splits < 2:
        raise ValueError("`n_splits` must be at least 2")

    if not (0 < val_size < 1):
        raise ValueError("`val_size` must be between 0 and 1")

    X = X.copy()
    y = pd.Series(y, index=X.index, name="y")

    # remove unused categorical levels if present
    if pd.api.types.is_categorical_dtype(X[site_col]):
        X[site_col] = X[site_col].cat.remove_unused_categories()

    rng = np.random.default_rng(random_state)

    # Build list of predictor columns
    # Always keep site_col for splitting, but control whether it's used for modeling
    predictor_cols = X.columns.tolist()

    if not include_site_as_predictor:
        predictor_cols = [col for col in predictor_cols if col != site_col]

    # Keep only categorical columns that are actually in predictors
    model_cat_cols = [col for col in cat_cols if col in predictor_cols]

    # ---------------------------------------------------
    # Build outer site-balanced folds
    # ---------------------------------------------------
    fold_indices = {i: [] for i in range(n_splits)}

    for site, idx in X.groupby(site_col, observed=True).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)

        if len(idx) < n_splits:
            raise ValueError(
                f"Site '{site}' has only {len(idx)} rows, fewer than n_splits={n_splits}. "
                "Reduce n_splits or remove very small sites."
            )

        split_site = np.array_split(idx, n_splits)

        for fold_num in range(n_splits):
            fold_indices[fold_num].extend(split_site[fold_num])

    fold_results = []
    permutation_tables = []

    all_test_obs = []
    all_test_pred = []

    # ---------------------------------------------------
    # Outer CV loop
    # ---------------------------------------------------
    for fold_num in range(n_splits):
        test_idx = pd.Index(fold_indices[fold_num])
        train_outer_idx = pd.Index(
            np.concatenate([fold_indices[i] for i in range(n_splits) if i != fold_num])
        )

        X_train_outer = X.loc[train_outer_idx].copy()
        X_test_full = X.loc[test_idx].copy()

        y_train_outer = y.loc[train_outer_idx].copy()
        y_test = y.loc[test_idx].copy()

        # ---------------------------------------------------
        # Inner validation split within outer training set
        # ---------------------------------------------------
        inner_train_idx = []
        inner_val_idx = []

        inner_rng = np.random.default_rng(random_state + fold_num)

        for site, idx in X_train_outer.groupby(site_col, observed=True).groups.items():
            idx = np.array(list(idx))
            inner_rng.shuffle(idx)

            n_site = len(idx)
            n_val = max(1, int(round(n_site * val_size)))

            if n_site - n_val < 1:
                n_val = n_site - 1

            if n_val < 1:
                raise ValueError(
                    f"Site '{site}' in fold {fold_num} does not have enough rows "
                    "for inner train/validation split."
                )

            site_val = idx[:n_val]
            site_train = idx[n_val:]

            inner_train_idx.extend(site_train)
            inner_val_idx.extend(site_val)

        inner_train_idx = pd.Index(inner_train_idx)
        inner_val_idx = pd.Index(inner_val_idx)

        X_train_full = X.loc[inner_train_idx].copy()
        X_val_full = X.loc[inner_val_idx].copy()

        y_train = y.loc[inner_train_idx].copy()
        y_val = y.loc[inner_val_idx].copy()

        # Drop site column if not using it as predictor
        X_train = X_train_full[predictor_cols].copy()
        X_val = X_val_full[predictor_cols].copy()
        X_test = X_test_full[predictor_cols].copy()

        # ---------------------------------------------------
        # Fit model
        # ---------------------------------------------------
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state + fold_num
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=model_cat_cols,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
        )

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Permutation importance on outer test fold
        perm_result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state + fold_num,
            n_jobs=-1
        )

        perm_table = pd.DataFrame({
            "feature": X_test.columns,
            "fold": fold_num,
            "permutation_importance_mean": perm_result.importances_mean,
            "permutation_importance_std": perm_result.importances_std
        })

        permutation_tables.append(perm_table)

        fold_results.append({
            "fold": fold_num,
            "model": model,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "test_r2": test_r2,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test)
        })

        all_test_obs.append(pd.Series(y_test.values, index=y_test.index))
        all_test_pred.append(pd.Series(y_test_pred, index=y_test.index))

        print(f"\nFold {fold_num + 1}/{n_splits}")
        print("Train RMSE:", train_rmse)
        print("Validation RMSE:", val_rmse)
        print("Test RMSE:", test_rmse)
        print("Train R2:", train_r2)
        print("Validation R2:", val_r2)
        print("Test R2:", test_r2)

    # ---------------------------------------------------
    # Summaries across folds
    # ---------------------------------------------------
    metrics_df = pd.DataFrame(fold_results)

    permutation_df = pd.concat(permutation_tables, axis=0, ignore_index=True)

    permutation_summary = (
        permutation_df.groupby("feature")[["permutation_importance_mean", "permutation_importance_std"]]
        .agg({
            "permutation_importance_mean": ["mean", "std"],
            "permutation_importance_std": ["mean"]
        })
    )

    permutation_summary.columns = [
        "perm_mean_across_folds",
        "perm_sd_across_folds",
        "perm_inner_sd_mean"
    ]

    permutation_summary = permutation_summary.sort_values(
        "perm_mean_across_folds", ascending=False
    )

    y_obs_cv = pd.concat(all_test_obs).sort_index()
    y_pred_cv = pd.concat(all_test_pred).sort_index()

    overall_cv_rmse = np.sqrt(mean_squared_error(y_obs_cv, y_pred_cv))
    overall_cv_r2 = r2_score(y_obs_cv, y_pred_cv)

    print("\nCross-validation summary")
    print("Mean test RMSE:", metrics_df["test_rmse"].mean())
    print("SD test RMSE:", metrics_df["test_rmse"].std())
    print("Mean test R2:", metrics_df["test_r2"].mean())
    print("SD test R2:", metrics_df["test_r2"].std())
    print("Overall CV RMSE:", overall_cv_rmse)
    print("Overall CV R2:", overall_cv_r2)

    if plot_importance:
        plt.figure(figsize=(12, 6))
        permutation_summary["perm_mean_across_folds"].plot.bar()
        plt.title("Permutation Importance Across CV Folds")
        plt.ylabel("Mean Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    results = {
        "fold_metrics": metrics_df,
        "permutation_importance_by_fold": permutation_df,
        "permutation_importance_summary": permutation_summary,
        "cv_observed": y_obs_cv,
        "cv_predicted": y_pred_cv,
        "overall_cv_rmse": overall_cv_rmse,
        "overall_cv_r2": overall_cv_r2,
        "models_by_fold": [fr["model"] for fr in fold_results],
        "predictor_cols_used": predictor_cols,
        "categorical_cols_used": model_cat_cols
    }

    return results

'''
def plot_importance(results):

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    results["split_importance"].sort_values(ascending=False).plot.bar(ax=axes[0])
    axes[0].set_title("Split Importance")

    results["gain_importance"].sort_values(ascending=False).plot.bar(ax=axes[1])
    axes[1].set_title("Gain Importance")

    results["permutation_importance_mean"].sort_values(ascending=False).plot.bar(ax=axes[2])
    axes[2].set_title("Permutation Importance")

    for ax in axes:
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()
'''

###############################################################################
##                             Visualizations                                ##
###############################################################################

# Boxplot of one column across the categories of another column
def boxplot_by_category(df, value_col, category_col, title, figsize=(10,6)):
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=category_col, y=value_col)
    sns.despine()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Boxplot of multiple columns alongside eachother for comparison
def multi_boxplots(df, value_cols, category_col, title, figsize=(12,6),showfliers=True):
    # reshape to long format
    long_df = df.melt(id_vars=category_col, value_vars=value_cols,
                      var_name="variable", value_name="value")

    plt.figure(figsize=figsize)
    sns.boxplot(data=long_df, x="variable", y="value",hue=category_col,showfliers=showfliers)
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Boxplots of multiple columns and grouped by a category, organized by all columns
# of one group next to eachother
def multi_boxplots_grouped(df, value_cols, category_col, title, figsize=(12,6),showfliers=True):
    # reshape to long format
    long_df = df.melt(
        id_vars=category_col,
        value_vars=value_cols,
        var_name="variable",
        value_name="value"
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=long_df,
        x=category_col,   # categories on x-axis
        y="value",
        hue="variable",    # variables grouped within each category
        showfliers=showfliers
    )
    sns.despine()
    plt.title(title)
    plt.tight_layout()
    plt.show()






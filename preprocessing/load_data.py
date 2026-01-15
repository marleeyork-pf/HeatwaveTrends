# This script includes all the functions to load in your data
# Loading packages
import pandas as pd
from datetime import datetime
import os

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
        
        

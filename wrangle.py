import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import host, user, password

###################### Acquire Zillow Data ######################

def get_db_url(url):
    url = f'mysql+pymysql://{user}:{password}@{host}/{url}'
    return url


def new_zillow_data():
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = """
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017

    LEFT JOIN propertylandusetype USING(propertylandusetypeid)

    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")"""

    
    df = pd.read_sql(sql_query, get_db_url('zillow'))


    # renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})
    return df



def acquire_zillow():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df

##################### clean Zillow Data ######################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
    
    #for exploration
def prepare_zillow_exploration(df):
    #Clean
    #not scaled
    #is split for exploring in "TRAIN"
    return "needs work"



def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])
    
    # drop taxamount
    df = df.drop(columns = 'taxamount')

    # converting column datatypes
    df.fips = df.fips.astype(int)
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object) 
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='most_frequent')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])    
    
    train[['year_built']] = train[['year_built']].astype(int)
    validate[['year_built']] = train[['year_built']].astype(int)
    test[['year_built']] = train[['year_built']].astype(int)

    train[['year_built']] = train[['year_built']].astype(object)
    validate[['year_built']] = train[['year_built']].astype(object)
    test[['year_built']] = train[['year_built']].astype(object)

    return train, validate, test 

def wrangle_zillow():
    # For modeling
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test
    
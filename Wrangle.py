import env
import os
import pandas as pd
from sklearn.model_selection import train_test_split

query = '''select bedroomcnt,bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid from properties_2017
WHERE propertylandusetypeid = 261;'''

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

url = get_connection('zillow')

prop17 = pd.read_sql(query,url)


def cache_dataframe(df, file_name):
    '''
    Cache a Pandas DataFrame as a CSV file in the current working directory

    Parameters:
    df (pandas.DataFrame): The DataFrame to cache
    file_name (str): The name of the CSV file

    Returns:
    None
    '''

    df.to_csv(file_name, index=False)

cache_dataframe(prop17,'prop17.csv')

def clean_prop17(prop17):
    '''This will clean the prop17/ data by converting capable cols from floats to int
    and drop the propertylandusetypeid col and all na values, still leaving 99.4% of data'''
    prop17 = prop17.drop(columns='propertylandusetypeid')
    prop17 = prop17.dropna()
    prop17['yearbuilt'] = prop17.yearbuilt.astype(int)
    prop17['calculatedfinishedsquarefeet'] = prop17.yearbuilt.astype(int)
    prop17['taxvaluedollarcnt'] = prop17.taxvaluedollarcnt.astype(int)
    prop17['fips'] = prop17.fips.astype(int)
    return prop17







def wrangle_zillow():
    '''combines the acquisition and prepartion of prop17 data'''
    clean_prop17(prop17)

    def split_dataframe(df, train_size=0.6, val_size=0.2, test_size=0.2, random_state=None):
        '''
        Split a Pandas DataFrame into train, validation, and test sets

        Parameters:
        df (pandas.DataFrame): The DataFrame to split
        train_size (float): The proportion of the data to use for training (default=0.6)
        val_size (float): The proportion of the data to use for validation (default=0.2)
        test_size (float): The proportion of the data to use for testing (default=0.2)
        random_state (int): The random seed to use for the train/test split (default=None)

        Returns:
        train_df (pandas.DataFrame): The training set
        val_df (pandas.DataFrame): The validation set
        test_df (pandas.DataFrame): The test set
        '''

        # Split the data into train and test sets
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        # Calculate the proportion of the original data to allocate to validation
        val_prop = val_size / (train_size + val_size)

        # Split the remaining data into train and validation sets
        train, val = train_test_split(train, test_size=val_prop, random_state=random_state)

        return train, val, test
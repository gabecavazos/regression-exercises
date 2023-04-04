import pandas as pd


def prep_titanic(df):
    '''
    clean titanic will take in a single pandas dataframe
    and will proceed to drop redundant columns
    and nonuseful information
    in addition to addressing null values
    and encoding categorical variables
    '''
    #drop out any redundant, excessively empty, or bad columns
    df = df.drop(columns=['passenger_id','embarked','deck','class'])
    # impute average age and most common embark_town:
    df['age'] = df['age'].fillna(df.age.mean())
    df['embark_town'] = df['embark_town'].fillna('Southampton')
    # encode categorical values:
    df = pd.concat(
    [df, pd.get_dummies(df[['sex', 'embark_town']],
                        drop_first=True)], axis=1)
    return df


def prep_iris(df):
    '''Prep iris will clean that pandas df up'''
    # drop species id
    iris.drop(columns=['species_id'], inplace=True)
    # change species name to species
    iris.rename(columns={'species_name': 'species'}, inplace=True)
    # get encoded categorical values for species
    iris = pd.concat(
        [iris, pd.get_dummies(iris['species'])], axis=1)

    return df

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test.
    Test is 20% of the original dataset, validate is .30*.80= 24% of the
    original dataset, and train is .70*.80= 56% of the original dataset.
    The function returns, in this order, train, validate and test dataframes.
    '''
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=seed,
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3,
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

def dem_dummies(df):
    # Get all categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Create dummy variables for each categorical column
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)

    return df
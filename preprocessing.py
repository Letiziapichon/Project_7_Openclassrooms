import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    def __init__(self, encode_columns_bool=True):
        self.app_test = pd.read_csv('data/application_test.csv')
        self.app_train = pd.read_csv('data/application_train.csv')
        self.bureau = pd.read_csv('data/bureau.csv')
        self.previous = pd.read_csv('data/previous_application.csv')
        self.encode_columns_bool = encode_columns_bool
    
    def convert_types(self, df, print_info = False):
        """Convert all columns format in a specific type of the format"""
        # Iterate through each column
        for c in df:

            # Convert ids and booleans to integers
            if ('SK_ID' in c):
                df[c] = df[c].fillna(0).astype(np.int32)

            # Convert objects to category
            elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
                df[c] = df[c].astype('category')

            # Booleans mapped to integers
            elif list(df[c].unique()) == [1, 0]:
                df[c] = df[c].astype(bool)

            # Float64 to float32
            elif df[c].dtype == float:
                df[c] = df[c].astype(np.float32)

            # Int64 to int32
            elif df[c].dtype == int:
                df[c] = df[c].astype(np.int32)


        return df
    
    def agg_numeric(self, df, parent_var, df_name):
        """
        Groups and aggregates the numeric values in a child dataframe
        by the parent variable.

        Parameters
        --------
            df (dataframe): 
                the child dataframe to calculate the statistics on
            parent_var (string): 
                the parent variable used for grouping and aggregating
            df_name (string): 
                the variable used to rename the columns

        Return
        --------
            agg (dataframe): 
                a dataframe with the statistics aggregated by the `parent_var` for 
                all numeric columns. Each observation of the parent variable will have 
                one row in the dataframe with the parent variable as the index. 
                The columns are also renamed using the `df_name`. Columns with all duplicate
                values are removed. 

        """
        # Remove id variables other than grouping variable
        for col in df:
            if col != parent_var and 'SK_ID' in col:
                df = df.drop(columns = col)

        # Only want the numeric variables
        parent_ids = df[parent_var].copy()
        numeric_df = df.select_dtypes('number').copy()
        numeric_df[parent_var] = parent_ids

        # Group by the specified variable and calculate the statistics
        agg = numeric_df.groupby(parent_var).agg(['mean'])

        # Need to create new column names
        columns = []

        # Iterate through the variables names
        for var in agg.columns.levels[0]:
            if var != parent_var:
                # Iterate through the stat names
                for stat in agg.columns.levels[1]:
                    # Make a new column name for the variable and stat
                    columns.append('%s_%s_%s' % (df_name, var, stat))

        agg.columns = columns

        # Remove the columns with all redundant values
        _, idx = np.unique(agg, axis = 1, return_index=True)
        agg = agg.iloc[:, idx]

        return agg

    def agg_categorical(self, df, parent_var, df_name):
        """
        Aggregates the categorical features in a child dataframe
        for each observation of the parent variable.

        Parameters
        --------
        df : dataframe 
            The dataframe to calculate the value counts for.

        parent_var : string
            The variable by which to group and aggregate the dataframe. For each unique
            value of this variable, the final dataframe will have one row

        df_name : string
            Variable added to the front of column names to keep track of columns


        Return
        --------
        categorical : dataframe
            A dataframe with aggregated statistics for each observation of the parent_var
            The columns are also renamed and columns with duplicate values are removed.

    """
        # Select the categorical columns
        categorical = pd.get_dummies(df.select_dtypes('category'))

        # Make sure to put the identifying id on the column
        categorical[parent_var] = df[parent_var]

        # Groupby the group var and calculate the sum and mean
        categorical = categorical.groupby(parent_var).agg(['mean'])

        column_names = []

        # Iterate through the columns in level 0
        for var in categorical.columns.levels[0]:
            # Iterate through the stats in level 1
            for stat in ['mean']:
                # Make a new column name
                column_names.append('%s_%s_%s' % (df_name, var, stat))

        categorical.columns = column_names

        # Remove duplicate columns by values
        _, idx = np.unique(categorical, axis = 1, return_index = True)
        categorical = categorical.iloc[:, idx]

        return categorical
        
    def encode_columns(self):
        """Apply label encoder on category columns with 2 categories"""
        le = LabelEncoder()
        le_count = 0

        # Iterate through the columns
        for col in self.app_train:
            if self.app_train[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(self.app_train[col].unique())) <= 2:
                    # Train on the training data
                    le.fit(self.app_train[col])
                    # Transform both training and testing data
                    self.app_train[col] = le.transform(self.app_train[col])
                    self.app_test[col] = le.transform(self.app_test[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1
        
    
    def cleaning_app(self):
        """Clean app_train and app_test dataframes."""
        percent_missing = self.app_train.isnull().sum() * 100 / len(self.app_train)
        percent_missing = percent_missing.reset_index().rename(columns={0:'perc_null'})

        columns_missing = percent_missing[percent_missing['perc_null'] > 50]['index'].to_list()
        columns_missing.remove('EXT_SOURCE_1')
        
        self.app_train.drop(columns=columns_missing, inplace=True)
        self.app_test.drop(columns=columns_missing, inplace=True)
        
        #self.app_train['AGE_BIN'] = pd.cut(self.app_train['DAYS_BIRTH'], bins = np.linspace(20, 70, num = 11))
        #self.app_test['AGE_BIN'] = pd.cut(self.app_test['DAYS_BIRTH'], bins = np.linspace(20, 70, num = 11))
        
        self.app_train.drop(columns=['CODE_GENDER'], inplace=True)
        self.app_test.drop(columns=['CODE_GENDER'], inplace=True)
        
        education  = {
            'Lower secondary': 0,
            'Secondary / secondary special': 1,
            'Higher education': 2,
            'Incomplete higher' : 3,
            'Academic degree': 4
        }

        self.app_train['NAME_EDUCATION_TYPE'] = self.app_train['NAME_EDUCATION_TYPE'].apply(lambda x: education[x])
        self.app_test['NAME_EDUCATION_TYPE'] = self.app_test['NAME_EDUCATION_TYPE'].apply(lambda x: education[x])
        
        if self.encode_columns_bool:
            self.encode_columns()
        
            self.app_train = pd.get_dummies(self.app_train)
            self.app_test = pd.get_dummies(self.app_test)
        
            target = self.app_train['TARGET']
            self.app_train, self.app_test = self.app_train.align(self.app_test, join = 'inner', axis = 1)

            self.app_train['TARGET'] = target
        
        self.app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
        self.app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
        
        self.app_train[self.app_train.select_dtypes(['int64']).columns] = self.app_train.select_dtypes(['int64']).astype('int32')
        self.app_train[self.app_train.select_dtypes(['float64']).columns] = self.app_train.select_dtypes(['float64']).astype('float32')
    
    def fe_app(self):
        """Features engineering on app_train and app_test datafarmes."""
        for df in [self.app_train, self.app_test]:
            df['AGE'] = df['DAYS_BIRTH'] / -365
            df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
            df['CREDIT_ANNUITY_RATION'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
            df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
            df['PAYMENT_RATE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
            df['PERCENT_GOODS_NOT_PAYED'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
            df['EXT_SOURCE_MEAN'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3'] ) / 3
            df['EXT_SOURCE_MUL'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3'] 
            df['EXT_SOURCE_MAX'] = [max(ele1,ele2,ele3) for ele1, ele2, ele3 in zip(df['EXT_SOURCE_1'], df['EXT_SOURCE_2'], df['EXT_SOURCE_3'])]
            df['EXT_SOURCE_MIN'] = [min(ele1,ele2,ele3) for ele1, ele2, ele3 in zip(df['EXT_SOURCE_1'], df['EXT_SOURCE_2'], df['EXT_SOURCE_3'])]
            df['EXT_SOURCE_VAR'] = [np.var([ele1,ele2,ele3]) for ele1, ele2, ele3 in zip(df['EXT_SOURCE_1'], df['EXT_SOURCE_2'], df['EXT_SOURCE_3'])]
            df['WEIGHTED_EXT_SOURCE'] =  df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
        
            df.drop(columns=[
                'DAYS_EMPLOYED',
                'AMT_ANNUITY',
                'DAYS_BIRTH'
            ], inplace=True)

    def fe_bureau(self):
        """Features engineering on bureau datafarme."""
        self.bureau = pd.concat([self.bureau, pd.get_dummies(self.bureau['CREDIT_ACTIVE'])], axis=1)
        
        credit_active = self.bureau.groupby(['SK_ID_CURR'])[['Active', 'Bad debt', 'Closed', 'Sold']].sum().reset_index()
        credit_sum = self.bureau.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index()
        
        self.app_train = self.app_train.merge(credit_active, on='SK_ID_CURR', how='left')
        self.app_train = self.app_train.merge(credit_sum, on='SK_ID_CURR', how='left')
        self.app_test = self.app_test.merge(credit_active, on='SK_ID_CURR', how='left')
        self.app_test = self.app_test.merge(credit_sum, on='SK_ID_CURR', how='left')

    def fe_previous(self):
        """Features engineering on previous application datafarme."""
        previous = self.convert_types(self.previous, print_info=True)
        previous_agg = self.agg_numeric(previous, 'SK_ID_CURR', 'previous')
        previous_counts = self.agg_categorical(previous, 'SK_ID_CURR', 'previous')
        
        self.app_train = self.app_train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
        self.app_train = self.app_train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

        self.app_test = self.app_test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
        self.app_test = self.app_test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
        
        self.app_train.drop(columns=[
                'previous_RATE_INTEREST_PRIVILEGED_mean',
                'previous_RATE_INTEREST_PRIMARY_mean'
            ], inplace=True)
        self.app_test.drop(columns=[
                'previous_RATE_INTEREST_PRIVILEGED_mean',
                'previous_RATE_INTEREST_PRIMARY_mean'
            ], inplace=True)
    
    def run(self):
        self.cleaning_app()

        self.fe_app()
        self.fe_bureau()
        self.fe_previous()
        
        return self.app_train, self.app_test
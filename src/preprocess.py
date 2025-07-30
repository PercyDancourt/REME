import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        """
        Inicializa el preprocesador. Puedes agregar parámetros como columnas clave si los necesitas.
        """
        pass

    def date_to_datetime(self, df):
        """
        Convert a specified column in a DataFrame to datetime format.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        pd.DataFrame: The DataFrame with the specified column converted to datetime.
        """
        df = df.copy() # Avoid modifying the original DataFrame
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    def temp_features(self, df):
        """
        Extracts temporal features from a DataFrame with a 'Date' column.

        Parameters:
        df (pd.DataFrame): The DataFrame containing a 'Date' column of datetime type.

        Returns:
        pd.DataFrame: The DataFrame with additional temporal features.
        """
        df = df.copy()  # Avoid modifying the original DataFrame

        # Basic date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday  # 0 = lunes

        # Additional features
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

        # cyclical features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

        return df

    def encode_categorical(self, df):
        """
        Encode categorical columns in a DataFrame using label encoder.

        Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns.

        Returns:
        pd.DataFrame: The DataFrame with categorical columns encoded.
        """
        df = df.copy()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        encoders = {}
        encoded_cols = []

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            encoded_cols.append(col)

        return df, encoders, encoded_cols
    
    def sort_by_date_and_id(self, df, date_col='Date', id_cols=['Store ID', 'Product ID']):
        """
        Sort a DataFrame by a specified date column and an ID column.

        Parameters:
        df (pd.DataFrame): The DataFrame to sort.
        date_column (str): The name of the date column to sort by.
        id_column (str): The name of the ID column to sort by.

        Returns:
        pd.DataFrame: The sorted DataFrame.
        """
        df = df.copy()  # Avoid modifying the original DataFrame
        return df.sort_values(by=id_cols + [date_col])
    
    def create_lags(self, df, lags=[1, 7, 14, 30], target_col = 'Units Sold'):
        """
        Create lag features for a specified target column in a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        lags (list): A list of integers representing the lag periods to create.
        target_col (str): The name of the target column for which to create lags.

        Returns:
        pd.DataFrame: The DataFrame with lag features added.
        """
        df = df.copy()  # Avoid modifying the original DataFrame
        df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])  # Ensure the DataFrame is sorted by Store ID, Product ID, and Date
        group_cols = ['Store ID', 'Product ID'] # we need to create lags for each store and product, each series will be shifted independently
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        return df
    
    def add_rolling_means(self, df, window_sizes=[7, 14, 30], target_col='Units Sold'):
        """
        Add rolling mean features to a DataFrame for specified window sizes.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        window_sizes (list): A list of integers representing the window sizes for rolling means.

        Returns:
        pd.DataFrame: The DataFrame with rolling mean features added.
        """
        df = df.copy()  # Avoid modifying the original DataFrame
        df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])  # Ensure the DataFrame is sorted by Store ID, Product ID, and Date
        group_cols = ['Store ID', 'Product ID']  # we need to create rolling means for each store and product, each series will be smoothed independently
        for window in window_sizes:
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        return df
    
    def clean_dataframe(self, df):
        """
        Clean a DataFrame by dropping rows with NaN values.

        Parameters:
        df (pd.DataFrame): The DataFrame to clean.

        Returns:
        pd.DataFrame: The cleaned DataFrame with NaN values dropped.
        """
        df = df.copy()  # Avoid modifying the original DataFrame
        return df.dropna()
    
    def interaction_features(self, df):
        """
        Create interaction features from numeric columns in a DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame. Must include the columns:
            'Price', 'Discount', 'Competitor Pricing', 'Inventory Level',
            'Units Ordered', and 'Units Sold'.

        Returns:
        pd.DataFrame: The DataFrame with new interaction features added:
            - price_discount
            - price_competitiveness
            - inventory_gap
            - inventory_turnover
        """
        df = df.copy()  # Avoid modifying the original DataFrame

        # Combinations
        df['price_discount'] = df['Price'] * df['Discount']
        df['price_competitiveness'] = df['Price'] / (df['Competitor Pricing'] + 1e-5)
        df['inventory_gap'] = df['Inventory Level'] - df['Units Ordered']
        return df
    
    def common_preprocess_data(self, df):
        """
        Preprocess the DataFrame by applying a series of transformations:
        - Convert 'Date' to datetime format
        - Extract temporal features
        - Encode categorical variables
        - Sort by date and ID
        - Create lag features
        - Add rolling means
        - Clean the DataFrame by dropping NaN values
        - Create interaction features

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        df = self.date_to_datetime(df)
        df = self.temp_features(df)
        df = self.encode_categorical(df)[0]
        df = self.sort_by_date_and_id(df)
        df = self.create_lags(df)
        df = self.add_rolling_means(df)
        df = self.clean_dataframe(df)
        df = self.interaction_features(df)
        
        return df

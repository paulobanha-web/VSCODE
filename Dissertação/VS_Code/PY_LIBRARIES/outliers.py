################################################################################################################################################
'''Python Libraries'''
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import zscore, norm
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from IPython.display import display
################################################################################################################################################
#import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy.stats import zscore, norm
################################################################################################################################################
def detect_univariate_outliers_zscore_df_pk(df: pd.DataFrame, primary_key_cols: list,
                                            numerical_columns_list: list = None, # Made optional
                                            threshold: float = 3.0,
                                            exclude_cols: list = None) -> pd.DataFrame:
    """
    Detects univariate outliers in a Pandas DataFrame using the z-score method for specified numerical features.
    If `numerical_columns_list` is not provided, the function automatically identifies all numerical columns
    in the DataFrame (excluding primary keys and explicitly excluded columns).
    It returns a table with the primary key, the outlier column, and the outlier value.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        primary_key_cols (list): A list of column names that form the primary key.
        numerical_columns_list (list, optional): A list of column names explicitly classified as numerical
                                               for which to detect outliers. If None (default), the function
                                               will automatically detect numerical columns in the DataFrame.
        threshold (float, optional): The z-score threshold for numerical outlier detection.
            Defaults to 3.0.
        exclude_cols (list, optional): A list of column names to exclude from
            outlier detection (even if present in `numerical_columns_list` or automatically detected).
            Defaults to None (no columns excluded).

    Returns:
        pandas.DataFrame: A DataFrame containing the primary key, outlier column,
            and outlier value for each outlier, sorted by outlier column and
            then by primary key.
            Returns an empty DataFrame if no outliers are found or no numerical columns
            are selected for detection after exclusion.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if not isinstance(primary_key_cols, list) or not all(isinstance(col, str) for col in primary_key_cols):
        raise TypeError("Input 'primary_key_cols' must be a list of strings.")
    if not all(col in df.columns for col in primary_key_cols):
        raise ValueError("One or more primary key columns not found in the DataFrame.")
    
    # Validate numerical_columns_list if it's provided (not None)
    if numerical_columns_list is not None and (not isinstance(numerical_columns_list, list) or not all(isinstance(col, str) for col in numerical_columns_list)):
        raise TypeError("Input 'numerical_columns_list' must be a list of strings or None.")
    
    if exclude_cols is None:
        exclude_cols = []
    
    outliers = []

    # Determine the set of numerical columns to analyze
    if numerical_columns_list is None:
        # Automatically detect numerical columns if the list is not provided
        print("`numerical_columns_list` not provided. Automatically detecting numerical columns...")
        # Get all columns that are not primary keys and not explicitly excluded
        candidate_cols_for_auto_detection = [
            col for col in df.columns if col not in primary_key_cols and col not in exclude_cols
        ]
        # Select only truly numerical dtypes from the candidates
        target_numerical_cols = df[candidate_cols_for_auto_detection].select_dtypes(include=np.number).columns.tolist()
    else:
        # Use the provided list, filtering out primary keys and excluded columns
        target_numerical_cols = [
            col for col in numerical_columns_list
            if col not in primary_key_cols and col not in exclude_cols and col in df.columns # Ensure column exists
        ]

    final_numerical_cols = target_numerical_cols # This is the list we will iterate over

    print("Starting univariate outlier detection (numerical features only)...")

    # Outlier detection for numerical features (using z-score)
    if final_numerical_cols:
        print("Detecting outliers in numerical columns:")
        for column in final_numerical_cols:
            # Drop NaNs for z-score calculation to avoid warnings/errors
            col_data_numeric = df[column].dropna()
            
            if col_data_numeric.empty or col_data_numeric.std() == 0:
                print(f"  - Skipping numerical column '{column}' due to insufficient data or constant values.")
                continue

            z_scores = np.abs((col_data_numeric - col_data_numeric.mean()) / col_data_numeric.std())
            
            # Find the indices of outliers in the original DataFrame
            outlier_indices = z_scores[z_scores > threshold].index
            
            if not outlier_indices.empty:
                # Use .loc with the original DataFrame for safety and performance
                column_outliers = df.loc[outlier_indices, primary_key_cols + [column]]
                for _, row in column_outliers.iterrows():
                    outliers.append(list(row[primary_key_cols]) + [column, row[column]])
                print(f"  - Column '{column}': Found {len(outlier_indices)} Z-score outliers.")
            else:
                print(f"  - Column '{column}': No Z-score outliers found.")
    else:
        print("No numerical columns to analyze based on selection/auto-detection or all were excluded.")

    # Create a DataFrame from the outliers list
    outlier_df_columns = primary_key_cols + ['Outlier_Column', 'Outlier_Value']
    outlier_df = pd.DataFrame(outliers, columns=outlier_df_columns)

    if outlier_df.empty:
        print("No outliers found based on the specified criteria.")
        return pd.DataFrame(columns=outlier_df_columns) # Return empty DataFrame with correct columns

    # Sort by outlier column and then by primary key
    outlier_df = outlier_df.sort_values(by=['Outlier_Column'] + primary_key_cols)

    # Format numerical values in primary key columns if they are numeric
    for pk_col in primary_key_cols:
        if pd.api.types.is_numeric_dtype(df[pk_col]) and pk_col in outlier_df.columns:
            # Ensure the column exists in outlier_df and check if conversion to int is safe (no NaNs after drop)
            # Only convert to int if all values are non-null floats that can be represented as int
            if outlier_df[pk_col].apply(lambda x: pd.notna(x) and x == int(x) if pd.api.types.is_float_dtype(outlier_df[pk_col]) else True).all():
                outlier_df[pk_col] = outlier_df[pk_col].astype(int)

    outlier_df = outlier_df.drop_duplicates().reset_index(drop=True)

    return outlier_df
##########################################################################################################################################
#import pandas as pd
#import numpy as np
##########################################################################################################################################
def remove_outliers_interactively(df: pd.DataFrame, outlier_df: pd.DataFrame, primary_key_cols: list) -> pd.DataFrame:
    """
    Interactively allows the user to remove (set to NaN) outlier values from the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame from which outliers were detected.
        outlier_df (pd.DataFrame): The DataFrame returned by a detection function (e.g.,
                                   detect_univariate_outliers_zscore_df_pk), containing
                                   'Outlier_Column', 'Outlier_Value', and primary key columns.
        primary_key_cols (list): The list of column names that form the primary key
                                 in both df and outlier_df.

    Returns:
        pd.DataFrame: A new DataFrame with the selected outlier values set to NaN.
                      Returns a copy of the original DataFrame if no outliers are removed or
                      if outlier_df is empty.
    """
    if df.empty or outlier_df.empty:
        print("No outliers to remove or original DataFrame is empty. Returning a copy of the original DataFrame.")
        return df.copy() # Return a copy to ensure original is not modified unexpectedly elsewhere

    df_modified = df.copy() # Work on a copy to avoid modifying the original directly

    print("\n--- Detected Outliers for Review ---")
    # Add a temporary 'Option' column to outlier_df for user selection
    outlier_df_display = outlier_df.copy()
    outlier_df_display['Option'] = range(1, len(outlier_df_display) + 1)
    
    # Reorder columns for display
    display_cols = ['Option'] + primary_key_cols + ['Outlier_Column', 'Outlier_Value']
    
    # Use print for standard console output if IPython.display.display is not available
    # If running in Jupyter/IPython, you can use 'display(outlier_df_display[display_cols])'
    print(outlier_df_display[display_cols].to_string(index=False))


    while True:
        print("\nWhat would you like to do with these outliers?")
        print("1. Remove ALL detected outliers (set their values to NaN)")
        print("2. Remove specific outliers by number (e.g., '1 3 5')")
        print("3. Do NOTHING (keep all outliers as is)")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip().lower()

        if choice == '1':
            print("Removing all detected outliers...")
            for index, row in outlier_df.iterrows():
                pk_values = tuple(row[col] for col in primary_key_cols)
                outlier_column = row['Outlier_Column']
                
                # Construct a boolean mask for finding the exact row using primary keys
                mask = pd.Series([True] * len(df_modified), index=df_modified.index)
                for pk_col, pk_val in zip(primary_key_cols, pk_values):
                    mask &= (df_modified[pk_col] == pk_val)
                
                row_index = df_modified[mask].index
                
                if not row_index.empty:
                    df_modified.loc[row_index, outlier_column] = np.nan
            print(f"All {len(outlier_df)} outliers have been set to NaN.")
            break

        elif choice == '2':
            selected_options_str = input("Enter the numbers of the outliers to remove, separated by spaces (e.g., '1 3 5'): ").strip()
            try:
                selected_options = [int(num) for num in selected_options_str.split()]
                if not selected_options:
                    print("No options entered. Please try again.")
                    continue

                removed_count = 0
                for option_num in selected_options:
                    if 1 <= option_num <= len(outlier_df_display):
                        # Get the outlier details from the display DataFrame using the 'Option'
                        outlier_row_to_remove = outlier_df_display[outlier_df_display['Option'] == option_num].iloc[0]
                        
                        pk_values = tuple(outlier_row_to_remove[col] for col in primary_key_cols)
                        outlier_column = outlier_row_to_remove['Outlier_Column']

                        # Construct a boolean mask for finding the exact row using primary keys
                        mask = pd.Series([True] * len(df_modified), index=df_modified.index)
                        for pk_col, pk_val in zip(primary_key_cols, pk_values):
                            mask &= (df_modified[pk_col] == pk_val)
                        
                        row_index = df_modified[mask].index

                        if not row_index.empty:
                            df_modified.loc[row_index, outlier_column] = np.nan
                            removed_count += 1
                        else:
                            print(f"Warning: Could not find original row for outlier option {option_num} (PK: {pk_values}). Skipping.")
                    else:
                        print(f"Option {option_num} is out of valid range. Skipping.")
                print(f"Successfully set {removed_count} selected outliers to NaN.")
                break # Exit loop after removing specific outliers
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")
            except IndexError:
                print("Invalid option number. Please select from the displayed options.")
        
        elif choice == '3':
            print("No outliers removed. Returning the original DataFrame (unmodified).")
            return df.copy() # Return the unmodified original DataFrame
        
        else:
            print("Invalid choice. Please enter '1', '2', or '3'.")

    print("\nOutlier removal process complete.")
    return df_modified


##########################################################################################################################################
#import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy.stats import zscore, norm
################################################################################################################################################


def detect_univariate_outliers_interactiveUNI(df: pd.DataFrame, primary_key_cols: list,
                                              numerical_columns_list: list = None,
                                              categorical_columns_list: list = None,
                                              exclude_cols: list = None,
                                              **kwargs) -> tuple:
    """
    Detects univariate outliers in a DataFrame interactively using different methods.
    User selects the method first, then enters the threshold, and finally selects the column to analyze.
    Includes options for visualization (box plots, histogram/density plots, count plots) and
    detection methods (Z-score, IQR, MAD for numerical; Frequency-based for categorical).
    When outliers are found, all columns of the outlier rows (identified by primary key) are displayed.

    Note on Method Applicability and Column Handling:
    - If `numerical_columns_list` is provided (and `categorical_columns_list` is None),
      only numerical features will be considered, and only numerical methods will be offered.
    - If `categorical_columns_list` is provided (and `numerical_columns_list` is None),
      only categorical features will be considered, and only categorical methods will be offered.
    - If BOTH lists are provided, both numerical and categorical features from the provided lists
      will be considered, and all relevant methods will be offered.
    - If NEITHER list is provided (both are None), columns will be automatically detected
      based on their actual data types, and all relevant methods will be offered.
    - Runtime type checks are performed on the chosen column before calculations to prevent errors if
      a user explicitly listed a column with a type incompatible with the chosen method (e.g., string in numerical_columns_list).

    Args:
        df (pd.DataFrame): The input DataFrame.
        primary_key_cols (list): A list of column names representing the primary key.
        numerical_columns_list (list, optional): A list of column names explicitly classified as numerical.
                                               If None (default), numerical columns will be automatically detected.
        categorical_columns_list (list, optional): A list of column names explicitly classified as categorical.
                                                  If None (default), categorical columns will be automatically detected.
        exclude_cols (list, optional): A list of column names to exclude from outlier detection.
                                       Defaults to None (no columns excluded).
        **kwargs: Retained for potential future extensions, but method-specific thresholds are now interactive inputs.

    Returns:
        tuple: A tuple containing:
               - df_outliers (pd.DataFrame): DataFrame of outlier rows (all columns) for the selected column.
               - outlier_indices (list): List of outlier indices for the selected column.
               - chosen_method (str): The name of the method chosen by the user ('zscore', 'iqr', 'mad', 'frequency').
               - chosen_column (str): The name of the column chosen by the user.
               Returns (pd.DataFrame(), [], None, None) if no relevant features are found or the process is interrupted.
    """
    df_copy = df.copy()

    if exclude_cols is None:
        exclude_cols = []
    
    # Validate primary key columns
    if not all(col in df_copy.columns for col in primary_key_cols):
        raise ValueError("One or more primary key columns not found in the DataFrame.")

    # Filter out primary key and excluded columns from all potential features initially
    initial_candidate_features = [col for col in df_copy.columns if col not in primary_key_cols and col not in exclude_cols]

    numerical_features_to_analyze = []
    categorical_features_to_analyze = []

    numerical_list_was_provided = numerical_columns_list is not None
    categorical_list_was_provided = categorical_columns_list is not None

    # --- Column Determination Logic (Strict Mode based on user input) ---
    if numerical_list_was_provided and not categorical_list_was_provided:
        # Scenario 1: Only numerical_columns_list provided. Strictly use it.
        numerical_features_to_analyze = [col for col in numerical_columns_list if col in initial_candidate_features]
        if numerical_features_to_analyze:
            print(f"Using provided numerical columns (and ignoring other data types): {', '.join(numerical_features_to_analyze)}")
        else:
            print("Provided `numerical_columns_list` resulted in no valid features after filtering by existence/exclusions.")

    elif categorical_list_was_provided and not numerical_list_was_provided:
        # Scenario 2: Only categorical_columns_list provided. Strictly use it.
        categorical_features_to_analyze = [col for col in categorical_columns_list if col in initial_candidate_features]
        if categorical_features_to_analyze:
            print(f"Using provided categorical columns (and ignoring other data types): {', '.join(categorical_features_to_analyze)}")
        else:
            print("Provided `categorical_columns_list` resulted in no valid features after filtering by existence/exclusions.")

    elif numerical_list_was_provided and categorical_list_was_provided:
        # Scenario 3: Both lists provided. Use both. No auto-detection for other types.
        numerical_features_to_analyze = [col for col in numerical_columns_list if col in initial_candidate_features]
        categorical_features_to_analyze = [col for col in categorical_columns_list if col in initial_candidate_features]
        if numerical_features_to_analyze:
            print(f"Using provided numerical columns: {', '.join(numerical_features_to_analyze)}")
        if categorical_features_to_analyze:
            print(f"Using provided categorical columns: {', '.join(categorical_features_to_analyze)}")
        if not numerical_features_to_analyze and not categorical_features_to_analyze:
            print("Neither provided `numerical_columns_list` nor `categorical_columns_list` yielded valid features after filtering.")

    else: # not numerical_list_was_provided and not categorical_list_was_provided
        # Scenario 4: Neither list provided. Auto-detect both.
        numerical_features_to_analyze = df_copy[initial_candidate_features].select_dtypes(include=np.number).columns.tolist()
        categorical_features_to_analyze = df_copy[initial_candidate_features].select_dtypes(exclude=np.number).columns.tolist()
        if numerical_features_to_analyze:
            print(f"Automatically detected numerical columns: {', '.join(numerical_features_to_analyze)}")
        else:
            print("No numerical columns auto-detected.")
        if categorical_features_to_analyze:
            print(f"Automatically detected categorical columns: {', '.join(categorical_features_to_analyze)}")
        else:
            print("No categorical columns auto-detected.")

    # Ensure uniqueness (though with this revised logic, less likely to have duplicates within each list)
    numerical_features_to_analyze = list(set(numerical_features_to_analyze))
    categorical_features_to_analyze = list(set(categorical_features_to_analyze))

    # Final check before proceeding with method selection
    if not numerical_features_to_analyze and not categorical_features_to_analyze:
        print("No numerical or categorical features available for outlier detection after all filtering and detection processes. Exiting.")
        return pd.DataFrame(), [], None, None

    # --- Present Method Options to the User ---
    methods = {
        1: 'Z-score (Numerical)',
        2: 'IQR (Interquartile Range - Numerical)',
        3: 'MAD (Median Absolute Deviation - Numerical)',
        4: 'Frequency-based (Categorical)'
    }

    print("\nSelect an outlier detection method:")
    displayed_methods = {}
    method_counter = 1
    
    # Display methods based strictly on what was determined to be available
    if numerical_features_to_analyze:
        displayed_methods[method_counter] = methods[1] # Z-score
        print(f"{method_counter}: {methods[1]}")
        method_counter += 1
        displayed_methods[method_counter] = methods[2] # IQR
        print(f"{method_counter}: {methods[2]}")
        method_counter += 1
        displayed_methods[method_counter] = methods[3] # MAD
        print(f"{method_counter}: {methods[3]}")
        method_counter += 1

    if categorical_features_to_analyze:
        displayed_methods[method_counter] = methods[4] # Frequency-based
        print(f"{method_counter}: {methods[4]}")
        method_counter += 1
    
    if not displayed_methods: # Should ideally be caught by the initial empty check, but robust
        print("No suitable outlier detection methods can be offered as no numerical or categorical columns are available for analysis. Exiting.")
        return pd.DataFrame(), [], None, None

    chosen_method_name = None
    method_type = None
    while True:
        try:
            method_choice = int(input("Enter the number of your chosen method: "))
            if method_choice in displayed_methods:
                chosen_method_name = displayed_methods[method_choice]
                if 'Numerical' in chosen_method_name:
                    method_type = 'numerical'
                elif 'Categorical' in chosen_method_name:
                    method_type = 'categorical'
                break
            else:
                print("Invalid choice. Please enter a number from the list of displayed methods.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Interactive Threshold Input ---
    threshold_value = None
    if 'Z-score' in chosen_method_name:
        default_threshold = 3.0
        print(f"\nRecommended Z-score thresholds are typically 2.0, 2.5, or 3.0 (default).")
        while True:
            try:
                threshold_input = input(f"Enter Z-score threshold (e.g., {default_threshold}, press Enter for default): ")
                threshold_value = float(threshold_input) if threshold_input else default_threshold
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        zscore_threshold = threshold_value # Assign to specific variable for method logic

    elif 'IQR' in chosen_method_name:
        default_multiplier = 1.5
        print(f"\nRecommended IQR multipliers are typically 1.5 (default) for mild outliers, or 3.0 for extreme outliers.")
        while True:
            try:
                threshold_input = input(f"Enter IQR multiplier (e.g., {default_multiplier}, press Enter for default): ")
                threshold_value = float(threshold_input) if threshold_input else default_multiplier
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        iqr_multiplier = threshold_value # Assign to specific variable for method logic

    elif 'MAD' in chosen_method_name:
        default_multiplier = 3.0
        print(f"\nRecommended MAD multipliers are typically 2.5, 3.0 (default), or 3.5.")
        while True:
            try:
                threshold_input = input(f"Enter MAD multiplier (e.g., {default_multiplier}, press Enter for default): ")
                threshold_value = float(threshold_input) if threshold_input else default_multiplier
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        mad_multiplier = threshold_value # Assign to specific variable for method logic

    elif 'Frequency-based' in chosen_method_name:
        default_freq = 0.05 # 5%
        print(f"\nRecommended frequency thresholds (as a proportion) are typically 0.01 (1%), 0.05 (5% default), or 0.10 (10%).")
        while True:
            try:
                threshold_input = input(f"Enter frequency threshold (e.g., {default_freq}, press Enter for default): ")
                threshold_value = float(threshold_input) if threshold_input else default_freq
                if 0 < threshold_value <= 1: # Ensure it's a valid proportion
                    break
                else:
                    print("Invalid input. Please enter a value between 0 and 1 (exclusive of 0).")
            except ValueError:
                print("Invalid input. Please enter a number.")
        categorical_freq_threshold = threshold_value # Assign to specific variable for method logic

    # --- Present Column Options to the User ---
    available_columns_for_selection = []
    if method_type == 'numerical':
        available_columns_for_selection = numerical_features_to_analyze
    elif method_type == 'categorical':
        available_columns_for_selection = categorical_features_to_analyze

    # This check ensures that a suitable list of columns exists for the chosen method type
    if not available_columns_for_selection:
        print(f"Error: You selected a {method_type} method ('{chosen_method_name}'), but no suitable {method_type} columns are available for analysis after filtering. This should not happen if the method was displayed correctly. Exiting.")
        return pd.DataFrame(), [], None, None

    print(f"\nSelect a {method_type} column for outlier detection:")
    column_options = {i + 1: col for i, col in enumerate(available_columns_for_selection)}
    for number, col_name in column_options.items():
        print(f"{number}: {col_name}")

    chosen_column = None
    while True:
        try:
            col_choice = int(input("Enter the number of your chosen column: "))
            if col_choice in column_options:
                chosen_column = column_options[col_choice]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    outlier_indices = []
    chosen_method = None

    print(f"\nAnalyzing column: '{chosen_column}' using {chosen_method_name.upper()} with chosen threshold {threshold_value}...")

    # --- Runtime Type Check (Crucial for User-Provided Lists) ---
    if method_type == 'numerical' and not pd.api.types.is_numeric_dtype(df_copy[chosen_column]):
        print(f"Error: You selected a numerical method for column '{chosen_column}', but it is not a numerical data type ({df_copy[chosen_column].dtype}). Please select a compatible column or method. Exiting.")
        return pd.DataFrame(), [], None, None
    elif method_type == 'categorical' and pd.api.types.is_numeric_dtype(df_copy[chosen_column]):
        # This allows numerical columns (like 0/1 for has_children) to be used with categorical method
        # but warns if it's explicitly numerical and user wants it categorical.
        if pd.api.types.is_float_dtype(df_copy[chosen_column]):
            print(f"Warning: Column '{chosen_column}' is a floating-point numerical data type ({df_copy[chosen_column].dtype}), which is usually not ideal for frequency-based categorical outlier detection. Proceeding anyway based on your selection.")

    # --- Visualize and Apply Outlier Detection ---
    if method_type == 'numerical':
        print(f"Visualizing '{chosen_column}':")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.boxplot(y=df_copy[chosen_column])
        plt.title(f'Box Plot of {chosen_column}')

        plt.subplot(1, 2, 2)
        sns.histplot(df_copy[chosen_column], kde=True)
        plt.title(f'Histogram and Density Plot of {chosen_column}')

        plt.tight_layout()
        plt.show()

        # Apply the chosen numerical outlier detection method
        if 'Z-score' in chosen_method_name:
            print(f"Applying {chosen_method_name} with threshold={zscore_threshold}...")
            col_data_dropna = df_copy[chosen_column].dropna()
            if col_data_dropna.empty or col_data_dropna.std() == 0:
                print(f"Skipping '{chosen_column}': insufficient data or constant values for Z-score.")
                return pd.DataFrame(), [], 'zscore_failed', chosen_column
            
            z_scores = np.abs(zscore(col_data_dropna))
            outlier_indices = col_data_dropna.index[z_scores > zscore_threshold].tolist()
            chosen_method = 'zscore'

        elif 'IQR' in chosen_method_name:
            print(f"Applying {chosen_method_name} with multiplier={iqr_multiplier}...")
            Q1 = df_copy[chosen_column].quantile(0.25)
            Q3 = df_copy[chosen_column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            outlier_indices = df_copy[(df_copy[chosen_column] < lower_bound) | (df_copy[chosen_column] > upper_bound)].index.tolist()
            chosen_method = 'iqr'

        elif 'MAD' in chosen_method_name:
            print(f"Applying {chosen_method_name} with multiplier={mad_multiplier}...")
            col_data_dropna = df_copy[chosen_column].dropna()
            if col_data_dropna.empty:
                print(f"Skipping '{chosen_column}': insufficient data for MAD.")
                return pd.DataFrame(), [], 'mad_failed', chosen_column

            median = col_data_dropna.median()
            mad = np.median(np.abs(col_data_dropna - median))

            if mad == 0:
                print(f"MAD is 0 for column '{chosen_column}'. Cannot detect outliers using MAD (data is constant around median).")
                return pd.DataFrame(), [], 'mad_failed', chosen_column

            modified_z_scores = 0.6745 * (col_data_dropna - median) / mad
            outlier_indices = col_data_dropna.index[np.abs(modified_z_scores) > mad_multiplier].tolist()
            chosen_method = 'mad'

    elif method_type == 'categorical':
        print(f"Visualizing '{chosen_column}' (Value Counts):")
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df_copy[chosen_column], order=df_copy[chosen_column].value_counts().index, palette='viridis')
        plt.title(f'Frequency Distribution of {chosen_column}')
        plt.xlabel('Count')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.show()

        # Apply the chosen categorical outlier detection method
        print(f"Applying {chosen_method_name} with frequency threshold={categorical_freq_threshold*100:.2f}%...")
        col_data_categorical = df_copy[chosen_column].dropna()
        if col_data_categorical.empty:
            print(f"Skipping '{chosen_column}': insufficient data for frequency-based detection.")
            return pd.DataFrame(), [], 'frequency_failed', chosen_column

        freq = col_data_categorical.value_counts(normalize=True)
        rare_categories = freq[freq < categorical_freq_threshold].index.tolist()

        if rare_categories:
            outlier_indices = df_copy.loc[df_copy[chosen_column].isin(rare_categories)].index.tolist()
            print(f"Rare categories identified as outliers: {rare_categories}")
        else:
            print(f"No rare categories found below threshold {categorical_freq_threshold*100:.2f}%.")
            
        chosen_method = 'frequency'

    # Get the original rows from the input DataFrame based on outlier indices
    df_outliers = df.loc[outlier_indices]

    # Display results for the selected column, showing all columns of the outlier rows
    if not df_outliers.empty:
        print(f"\nDetected Univariate Outliers in '{chosen_column}' using {chosen_method_name.upper()}:")
        print(df_outliers.to_string()) 
        print("Outlier Indices:", outlier_indices)
    else:
        print(f"\nNo univariate outliers detected in '{chosen_column}' using {chosen_method_name.upper()}.")

    return df_outliers, outlier_indices, chosen_method, chosen_column

##########################################################################################################################################
#import pandas as pd
#from IPython.display import display
#import time
#import numpy as np
##########################################################################################################################################

def remove_multivariate_outliers_interactive(df, detected_outliers_tuple, primary_key_cols, numerical_features, categorical_features, message="What would you like to do with these outliers?\n1: Remove them\n2: Impute them\n3: Do nothing\nEnter choice (1/2/3): ", show_outlier_df=True):
    """
    Interactively handles multivariate outliers from a DataFrame based on detected outliers.
    Allows user to choose between removing, imputing (with various methods, including group-based), or doing nothing.

    Args:
        df (pd.DataFrame): The original DataFrame.
        detected_outliers_tuple (tuple): A tuple containing (outlier_df, outlier_records, method_used)
                                         from a multivariate outlier detection function.
        primary_key_cols (list): List of column names forming the primary key, used for group-based imputation.
        numerical_features (list): List of numerical feature names used in detection.
        categorical_features (list): List of categorical feature names used in detection.
        message (str, optional): The initial prompt message to display to the user.
        show_outlier_df (bool, optional): Whether to display the detected outlier DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the confirmed outliers handled (removed or imputed).
                      Returns a copy of the original df if no outliers or action cancelled/do nothing.
    """
    # --- Input Validation and Setup ---
    if not isinstance(detected_outliers_tuple, tuple) or len(detected_outliers_tuple) != 3:
        print("Invalid detected_outliers_tuple format. Expected (outlier_df, outlier_records, method_used).")
        return df.copy()

    outlier_df, outlier_records, method_used = detected_outliers_tuple

    if outlier_df.empty:
        method_name = method_used.upper() if method_used else "N/A"
        print(f"No outliers were detected using {method_name}. No action needed.")
        return df.copy()

    df_processed = df.copy() # Work on a copy to avoid modifying original df directly

    num_outliers = len(outlier_df)
    method_name = method_used.upper() if method_used else "N/A"
    print(f"\n{num_outliers} outliers detected using {method_name}.")

    if show_outlier_df:
        print("\nDetected Outlier Rows:")
        try:
            display(outlier_df)
            time.sleep(0.1) # Small delay to allow display to render
        except Exception as e:
            print(f"Could not display outlier DataFrame: {e}")
            print("Proceeding without displaying outlier DataFrame.")

    outlier_row_indices = outlier_df.index.tolist()
    print(f"\nThese rows (original DataFrame indices) were detected as outliers: {outlier_row_indices}")

    # --- Main Loop for User Action Choice ---
    while True:
        user_choice_str = input(message).strip()

        if user_choice_str == '1': # Remove
            print("User chose to remove outliers.")
            df_processed = df_processed.drop(index=outlier_row_indices)
            print(f"{num_outliers} outlier rows removed from the DataFrame.")
            return df_processed

        elif user_choice_str == '2': # Impute
            print("User chose to impute outliers.")
            
            # Initialize chosen imputation methods (store descriptive names)
            numerical_impute_method_name = "Skipped"
            categorical_impute_method_name = "Skipped"
            custom_fill_value_input = None # To store the user's input for 'fill_value'

            # Determine if numerical/categorical features are present in the outliers
            present_numerical_outliers = any(f in df.columns and f in numerical_features for f in outlier_df.columns)
            present_categorical_outliers = any(f in df.columns and f in categorical_features for f in outlier_df.columns)

            # --- Prompt for Numerical Feature Imputation ---
            if present_numerical_outliers:
                num_method_map = {'1': 'Mean', '2': 'Median', '3': 'Winsorize', '4': 'Custom Fill Value', '5': 'Skipped'}
                while True:
                    num_impute_message = (
                        "\nChoose imputation method for NUMERICAL Features (present in outliers):\n"
                        "  1: Mean (Group-based, falls back to global mean if group is sparse)\n"
                        "  2: Median (Group-based, falls back to global median if group is sparse)\n"
                        "  3: Winsorize (Caps values at global 1st and 99th percentile)\n"
                        "  4: Custom Fill Value\n"
                        "  5: Skip Numerical Imputation\n"
                        "Enter choice (1/2/3/4/5): "
                    )
                    num_choice = input(num_impute_message).strip()
                    if num_choice in num_method_map:
                        numerical_impute_method_name = num_method_map[num_choice]
                        if num_choice == '4': # If custom fill value chosen for numerical
                            custom_fill_value_input = input("Enter custom value for numerical features (e.g., 0, -1, 'NaN'): ").strip()
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            else:
                print("\nNo numerical features found in outliers for imputation. Skipping numerical imputation.")

            # --- Prompt for Categorical Feature Imputation ---
            if present_categorical_outliers:
                cat_method_map = {'1': 'Mode', '2': 'Custom Fill Value', '3': 'Skipped'}
                while True:
                    cat_impute_message = (
                        "\nChoose imputation method for CATEGORICAL Features (present in outliers):\n"
                        "  1: Mode (Group-based, falls back to global mode if group is sparse)\n"
                        "  2: Custom Fill Value\n"
                        "  3: Skip Categorical Imputation\n"
                        "Enter choice (1/2/3): "
                    )
                    cat_choice = input(cat_impute_message).strip()
                    if cat_choice in cat_method_map:
                        categorical_impute_method_name = cat_method_map[cat_choice]
                        if cat_choice == '2' and custom_fill_value_input is None: # If custom fill chosen and not already asked for numerical
                            custom_fill_value_input = input("Enter custom value for categorical features (e.g., 'Unknown', 'N/A'): ").strip()
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
            else:
                print("\nNo categorical features found in outliers for imputation. Skipping categorical imputation.")

            # --- Final Confirmation Before Applying Imputation ---
            if numerical_impute_method_name == "Skipped" and categorical_impute_method_name == "Skipped":
                print("No imputation method selected for any feature type. Returning original DataFrame.")
                return df.copy()

            confirm_apply = input(f"\nConfirm application:\n  Numerical: {numerical_impute_method_name}\n  Categorical: {categorical_impute_method_name}\nDo you want to proceed? (yes/no): ").strip().lower()
            if confirm_apply != 'yes':
                print("Imputation cancelled. Returning the original DataFrame.")
                return df.copy()

            # --- Apply Imputation ---
            imputation_applied_count = 0 # Counter for individual cell imputations
            
            # Pre-process custom_fill_value_input if it exists
            processed_custom_fill_value = None
            if custom_fill_value_input is not None:
                try:
                    # Attempt numerical conversion first
                    if '.' in custom_fill_value_input:
                        processed_custom_fill_value = float(custom_fill_value_input)
                    else:
                        processed_custom_fill_value = int(custom_fill_value_input)
                except ValueError:
                    # If not numerical, keep as string
                    processed_custom_fill_value = custom_fill_value_input

            # Iterate through each outlier record
            for original_idx in outlier_row_indices:
                # Get the primary key values for the current outlier record
                pk_values = df.loc[original_idx, primary_key_cols].to_dict()

                # Filter the ORIGINAL df to get data for the current primary key group,
                # EXCLUDING the current outlier record itself.
                group_mask = pd.Series(True, index=df.index)
                for pk_col, pk_val in pk_values.items():
                    group_mask = group_mask & (df[pk_col] == pk_val)
                group_data_exclude_outlier = df.loc[group_mask & (df.index != original_idx)]

                # Iterate through all features considered during outlier detection for this record
                for feature in numerical_features + categorical_features:
                    if feature not in df_processed.columns: # Skip if feature somehow doesn't exist
                        continue

                    current_val_in_outlier = df_processed.loc[original_idx, feature]
                    value_to_impute = None # Initialize value to impute for current cell

                    # Skip imputation if the current value is already NaN and we are not doing a custom fill
                    if pd.isna(current_val_in_outlier) and (numerical_impute_method_name != 'Custom Fill Value' and categorical_impute_method_name != 'Custom Fill Value'):
                        continue

                    is_numerical = feature in numerical_features
                    is_categorical = feature in categorical_features
                    applied_for_this_cell = False # Flag to know if this specific cell was imputed

                    # --- Numerical Feature Imputation Logic ---
                    if is_numerical and numerical_impute_method_name != "Skipped":
                        if numerical_impute_method_name == 'Mean':
                            group_feature_data = group_data_exclude_outlier[feature].dropna()
                            if len(group_feature_data) >= 2:
                                value_to_impute = group_feature_data.mean()
                                if pd.isna(value_to_impute):
                                    value_to_impute = df[feature].mean()
                            else:
                                value_to_impute = df[feature].mean()
                        
                        elif numerical_impute_method_name == 'Median':
                            group_feature_data = group_data_exclude_outlier[feature].dropna()
                            if len(group_feature_data) >= 2:
                                value_to_impute = group_feature_data.median()
                                if pd.isna(value_to_impute):
                                    value_to_impute = df[feature].median()
                            else:
                                value_to_impute = df[feature].median()
                        
                        elif numerical_impute_method_name == 'Winsorize':
                            lower_bound = df[feature].quantile(0.01)
                            upper_bound = df[feature].quantile(0.99)
                            if pd.notna(current_val_in_outlier):
                                if current_val_in_outlier < lower_bound:
                                    value_to_impute = lower_bound
                                elif current_val_in_outlier > upper_bound:
                                    value_to_impute = upper_bound
                                else:
                                    value_to_impute = current_val_in_outlier # No change
                            # If current_val_in_outlier is NaN, value_to_impute remains None, no change.
                        
                        elif numerical_impute_method_name == 'Custom Fill Value':
                            value_to_impute = processed_custom_fill_value

                        if value_to_impute is not None and (pd.isna(current_val_in_outlier) or current_val_in_outlier != value_to_impute):
                            df_processed.loc[original_idx, feature] = value_to_impute
                            applied_for_this_cell = True

                    # --- Categorical Feature Imputation Logic ---
                    elif is_categorical and categorical_impute_method_name != "Skipped":
                        if categorical_impute_method_name == 'Mode':
                            group_feature_data = group_data_exclude_outlier[feature].dropna()
                            mode_val_series = group_feature_data.mode()
                            if not mode_val_series.empty:
                                value_to_impute = mode_val_series.iloc[0]
                            else:
                                global_mode_val_series = df[feature].mode()
                                if not global_mode_val_series.empty:
                                    value_to_impute = global_mode_val_series.iloc[0]
                                else:
                                    print(f"Warning: No mode found for '{feature}' (global or group). Cannot impute at index {original_idx}.")
                                    value_to_impute = None # Cannot impute if no mode
                            
                            if value_to_impute is not None and (pd.isna(current_val_in_outlier) or current_val_in_outlier != value_to_impute):
                                df_processed.loc[original_idx, feature] = value_to_impute
                                applied_for_this_cell = True

                        elif categorical_impute_method_name == 'Custom Fill Value':
                            value_to_impute = processed_custom_fill_value
                            if value_to_impute is not None and (pd.isna(current_val_in_outlier) or current_val_in_outlier != value_to_impute):
                                df_processed.loc[original_idx, feature] = str(value_to_impute) # Ensure categorical is stored as string
                                applied_for_this_cell = True
                    
                    if applied_for_this_cell:
                        imputation_applied_count += 1

            # --- Final Summary Message ---
            if imputation_applied_count > 0:
                print("\nOutlier imputation completed:")
                if present_numerical_outliers:
                    print(f"  Numerical Features Imputed using: {numerical_impute_method_name}")
                if present_categorical_outliers:
                    print(f"  Categorical Features Imputed using: {categorical_impute_method_name}")
                print(f"Total individual values imputed across {num_outliers} outlier records: {imputation_applied_count}")
                return df_processed
            else:
                print("No values were imputed with the selected methods (perhaps all were already NaN or values were within bounds for Winsorize). Returning the original DataFrame.")
                return df.copy()

        elif user_choice_str == '3': # Do nothing
            print("User chose to do nothing. Returning the original DataFrame.")
            return df.copy()

        else:
            print("Invalid input. Please enter '1', '2', or '3'.")

##########################################################################################################################################
#import pandas as pd
#import numpy as np
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.ensemble import IsolationForest
#from sklearn.covariance import EllipticEnvelope
#from sklearn.svm import OneClassSVM
#from sklearn.cluster import DBSCAN
#from scipy.spatial.distance import mahalanobis
#from scipy.stats import chi2
#from sklearn.preprocessing import StandardScaler, OrdinalEncoder
#from sklearn.impute import SimpleImputer
#from datetime import datetime # Needed for type checking datetime objects
##########################################################################################################################################

def detect_multivariate_outliers_interactiveMULTI(
    df: pd.DataFrame,
    primary_key_cols: list,
    numerical_features: list = None,
    categorical_features: list = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """
    Detects multivariate outliers in a DataFrame interactively using different methods,
    combining numerical and optionally categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        primary_key_cols (list): A list of column names representing the primary key.
        numerical_features (list, optional): A pre-defined list of numerical column names to use
                                            for outlier detection. If None, all numerical columns
                                            (float64, int64) excluding primary keys will be used.
                                            Defaults to None.
        categorical_features (list, optional): A pre-defined list of categorical column names to use
                                                for outlier detection. These will be OrdinalEncoded.
                                                If None, no categorical features will be used.
                                                Defaults to None.
        **kwargs: Additional keyword arguments for the chosen outlier detection method (overrides defaults).

    Returns:
        tuple: A tuple containing:
                - df_outliers (pd.DataFrame): A DataFrame containing the detected outlier rows (original data).
                - outlier_records (pd.DataFrame): A DataFrame suitable for interactive removal,
                                                    with 'Outlier_Column', 'Outlier_Value' and PKs.
                                                    'Outlier_Column' will be "Multivariate_Features",
                                                    and 'Outlier_Value' will contain a dictionary
                                                    of original values of all features used for detection,
                                                    along with a 'reason' if applicable.
                - chosen_method (str | None): The name of the method chosen by the user, or None if detection failed.
    """
    df_copy = df.copy()

    all_features_for_detection = []
    numerical_cols_for_detection = []
    categorical_cols_for_detection = []

    # 1. Determine numerical features to use
    if numerical_features is not None:
        numerical_cols_for_detection = [col for col in numerical_features
                                        if col in df_copy.columns and col not in primary_key_cols]
        if numerical_cols_for_detection:
            print(f"Using provided numerical features: {', '.join(numerical_cols_for_detection)}")
    else:
        numerical_cols_for_detection = [col for col in df_copy.columns
                                        if pd.api.types.is_numeric_dtype(df_copy[col])
                                        and col not in primary_key_cols]
        if numerical_cols_for_detection:
            print(f"Automatically detected numerical features: {', '.join(numerical_cols_for_detection)}")

    # 2. Determine categorical features to use and preprocess them
    encoded_categorical_df = pd.DataFrame(index=df_copy.index)
    encoded_categorical_mappings = {}

    if categorical_features is not None:
        categorical_cols_for_detection = [col for col in categorical_features
                                          if col in df_copy.columns and col not in primary_key_cols]
        if categorical_cols_for_detection:
            print(f"Using provided categorical features: {', '.join(categorical_cols_for_detection)}")

            temp_cat_df = df_copy[categorical_cols_for_detection].copy()

            # Handle datetime objects by converting to string before encoding
            for col in categorical_cols_for_detection:
                if temp_cat_df[col].dropna().apply(lambda x: isinstance(x, (pd.Timestamp, datetime))).any():
                    temp_cat_df[col] = temp_cat_df[col].astype(str)
                    print(f"Converted datetime-like values in categorical column '{col}' to string for encoding.")

            # Impute missing values in categorical features BEFORE encoding
            cat_imputer = SimpleImputer(strategy='most_frequent')
            temp_cat_df_imputed = pd.DataFrame(
                cat_imputer.fit_transform(temp_cat_df),
                columns=temp_cat_df.columns,
                index=temp_cat_df.index
            )

            # Ordinal encode categorical features
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # Use -1 for unknown
            encoded_data = ordinal_encoder.fit_transform(temp_cat_df_imputed)
            encoded_categorical_df = pd.DataFrame(
                encoded_data,
                columns=[f"{col}_encoded" for col in categorical_cols_for_detection],
                index=df_copy.index
            )
            # Store mappings
            for i, col in enumerate(categorical_cols_for_detection):
                encoded_categorical_mappings[col] = dict(zip(ordinal_encoder.categories_[i], range(len(ordinal_encoder.categories_[i]))))
        else:
            print("No valid categorical features found in the provided list.")

    # Combine numerical features and encoded categorical features
    all_features_for_detection = numerical_cols_for_detection + [col for col in encoded_categorical_df.columns]

    if not all_features_for_detection:
        print("No numerical or categorical features available for multivariate outlier detection after exclusions.")
        return pd.DataFrame(), pd.DataFrame(), None

    # Prepare combined data for detection (original numerical + encoded categorical)
    # Ensure all columns are aligned by index
    X_numerical_temp = df_copy[numerical_cols_for_detection] if numerical_cols_for_detection else pd.DataFrame(index=df_copy.index)

    X_combined_detection = pd.concat([X_numerical_temp, encoded_categorical_df], axis=1)

    # Handle missing values by dropping rows with NaNs in the combined detection features
    X_combined_detection_cleaned = X_combined_detection.dropna()
    cleaned_indices = X_combined_detection_cleaned.index

    if X_combined_detection_cleaned.empty:
        print("No complete cases found after handling missing values in combined numerical and categorical features. Cannot perform outlier detection.")
        return pd.DataFrame(), pd.DataFrame(), None

    print(f"\nUsing a total of {len(all_features_for_detection)} features for detection.")
    print(f"Features used for detection (original names and encoded categorical names): {', '.join(all_features_for_detection)}")

    # Present method options to the user
    methods = {
        1: 'Local Outlier Factor (LOF)',
        2: 'Isolation Forest',
        3: 'Elliptic Envelope',
        4: 'One-Class SVM',
        5: 'DBSCAN',
        6: 'Mahalanobis Distance'
    }

    print("\nSelect an outlier detection method:")
    for number, method_name in methods.items():
        print(f"{number}: {method_name}")

    chosen_method_name = None
    while True:
        try:
            choice = int(input("Enter the number of your chosen method: "))
            if choice in methods:
                chosen_method_name = methods[choice]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Define default kwargs for each method
    default_kwargs = {
        'Local Outlier Factor (LOF)': {'n_neighbors': 20, 'contamination': 'auto'},
        'Isolation Forest': {'n_estimators': 100, 'contamination': 'auto', 'random_state': 42},
        'Elliptic Envelope': {'contamination': 0.1, 'random_state': 42},
        'One-Class SVM': {'nu': 0.1, 'kernel': 'rbf'},
        'DBSCAN': {'eps': 0.5, 'min_samples': 5},
        'Mahalanobis Distance': {'alpha': 0.01} # Default significance level
    }

    # Merge default kwargs with user-provided kwargs
    method_kwargs = default_kwargs.get(chosen_method_name, {})
    method_kwargs.update(kwargs)

    outlier_labels = None
    chosen_method_short = None # This will store the short method name ('lof', 'isolation_forest', etc.)

    print(f"\nUsing {chosen_method_name} for outlier detection with parameters: {method_kwargs}")

    # Standardize data before detection (important for distance-based methods)
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_combined_detection_cleaned)
    X_scaled_df = pd.DataFrame(X_scaled_array, index=cleaned_indices, columns=X_combined_detection_cleaned.columns)

    if 'Local Outlier Factor' in chosen_method_name:
        lof = LocalOutlierFactor(**method_kwargs)
        outlier_labels = lof.fit_predict(X_scaled_df)
        chosen_method_short = 'lof'

    elif 'Isolation Forest' in chosen_method_name:
        iso_forest = IsolationForest(**method_kwargs)
        outlier_labels = iso_forest.fit_predict(X_scaled_df)
        chosen_method_short = 'isolation_forest'

    elif 'Elliptic Envelope' in chosen_method_name:
        # EllipticEnvelope specifically requires at least n_samples >= n_features + 1
        # and non-singular covariance.
        if X_scaled_df.shape[0] < X_scaled_df.shape[1] + 1:
            print(f"Warning: Elliptic Envelope requires at least {X_scaled_df.shape[1] + 1} samples, but only {X_scaled_df.shape[0]} are available.")
            print("Consider using a different method or providing more data.")
            return pd.DataFrame(), pd.DataFrame(), None

        try:
            elliptic_envelope = EllipticEnvelope(**method_kwargs)
            outlier_labels = elliptic_envelope.fit_predict(X_scaled_df)
            chosen_method_short = 'elliptic_envelope'
        except ValueError as e:
            print(f"Error applying Elliptic Envelope: {e}. This often happens if the data is degenerate (e.g., highly correlated features, insufficient distinct points).")
            print("Consider removing highly correlated features or using a different detection method like LOF or Isolation Forest.")
            return pd.DataFrame(), pd.DataFrame(), None

    elif 'One-Class SVM' in chosen_method_name:
        one_class_svm = OneClassSVM(**method_kwargs)
        one_class_svm.fit(X_scaled_df)
        outlier_labels = one_class_svm.predict(X_scaled_df)
        chosen_method_short = 'one_class_svm'

    elif 'DBSCAN' in chosen_method_name:
        dbscan = DBSCAN(**method_kwargs)
        outlier_labels = dbscan.fit_predict(X_scaled_df)
        chosen_method_short = 'dbscan'

    elif 'Mahalanobis Distance' in chosen_method_name:
        cov_matrix = X_scaled_df.cov()
        if cov_matrix.empty or np.linalg.det(cov_matrix) == 0:
            print("Cannot calculate Mahalanobis distance: Covariance matrix is singular or empty on scaled data.")
            print("This usually means some features are perfectly correlated or have zero variance.")
            return pd.DataFrame(), pd.DataFrame(), None

        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("Cannot calculate Mahalanobis distance: Covariance matrix is not invertible (likely singular).")
            print("Consider removing highly correlated or constant features.")
            return pd.DataFrame(), pd.DataFrame(), None

        mean_vector = X_scaled_df.mean()
        mahalanobis_distances = []
        for index, row in X_scaled_df.iterrows():
            mahalanobis_distances.append(mahalanobis(row, mean_vector, inv_cov_matrix))

        mahalanobis_distances = pd.Series(mahalanobis_distances, index=cleaned_indices)

        degrees_of_freedom = X_scaled_df.shape[1]
        alpha = method_kwargs.get('alpha', 0.01)
        threshold = chi2.ppf(1 - alpha, degrees_of_freedom)
        print(f"Mahalanobis Distance Threshold (alpha={alpha}): {threshold:.4f}")

        outlier_indices_threshold = mahalanobis_distances[mahalanobis_distances > threshold].index
        outlier_labels = np.ones(len(X_scaled_df))
        outlier_labels[X_scaled_df.index.isin(outlier_indices_threshold)] = -1

        chosen_method_short = 'mahalanobis'

    df_outliers = pd.DataFrame()
    outlier_records = pd.DataFrame()

    if outlier_labels is not None:
        outlier_indices_final = cleaned_indices[outlier_labels == -1]
        if not outlier_indices_final.empty:
            df_outliers = df.loc[outlier_indices_final].copy()

            records_list = []
            # Calculate overall feature statistics once for efficiency
            # This is used for providing context (mean, median, IQR) for numerical features
            numerical_stats = {}
            for num_col in numerical_cols_for_detection:
                data_series = df[num_col].dropna()
                if not data_series.empty:
                    q1 = data_series.quantile(0.25)
                    q3 = data_series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr) # Using 1.5 as standard multiplier for example
                    upper_bound = q3 + (1.5 * iqr)
                    numerical_stats[num_col] = {
                        'mean': data_series.mean(),
                        'median': data_series.median(),
                        'std': data_series.std(),
                        'iqr_lower': lower_bound,
                        'iqr_upper': upper_bound
                    }


            for original_idx in outlier_indices_final:
                record = {}
                for pk_col in primary_key_cols:
                    record[pk_col] = df.loc[original_idx, pk_col]

                record['Outlier_Column'] = "Multivariate_Features"

                # Get original values for all features used in detection, with context
                outlier_feature_details = {}
                for num_col in numerical_cols_for_detection:
                    value = df.loc[original_idx, num_col]
                    reason = "No specific deviation detected (multivariate outlier)" # Default reason
                    
                    if num_col in numerical_stats:
                        stats = numerical_stats[num_col]
                        # Check against Z-score (simple deviation from mean)
                        if stats['std'] > 0:
                            z_score = (value - stats['mean']) / stats['std']
                            if z_score > 3: # Example threshold for "high"
                                reason = f"Value ({value:.2f}) is significantly high (Z-score: {z_score:.2f})"
                            elif z_score < -3: # Example threshold for "low"
                                reason = f"Value ({value:.2f}) is significantly low (Z-score: {z_score:.2f})"
                        
                        # Check against IQR bounds
                        if value < stats['iqr_lower']:
                            reason = f"Value ({value:.2f}) is below IQR lower bound ({stats['iqr_lower']:.2f})"
                        elif value > stats['iqr_upper']:
                            reason = f"Value ({value:.2f}) is above IQR upper bound ({stats['iqr_upper']:.2f})"

                    outlier_feature_details[num_col] = {'value': value, 'reason': reason}

                for cat_col in categorical_cols_for_detection:
                    value = df.loc[original_idx, cat_col]
                    # For categorical, a "reason" is harder to assign individually in multivariate context
                    # unless it's based on rarity. For now, just state it's part of the outlier combination.
                    reason = f"Categorical value '{value}' contributes to multivariate outlier pattern"
                    outlier_feature_details[cat_col] = {'value': value, 'reason': reason}

                record['Outlier_Value'] = outlier_feature_details # Store as a dictionary of dictionaries
                records_list.append(record)

            outlier_records = pd.DataFrame(records_list)

            print(f"\nDetected {len(outlier_indices_final)} outliers using {chosen_method_name}.")
        else:
            print(f"No outliers detected using {chosen_method_name}.")
    else:
        print("Outlier detection could not be performed due to method failure or no method selected.")

    return df_outliers, outlier_records, chosen_method_short

################################################################################################################################################
'''Python Libraries'''
################################################################################################################################################
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer # Import KNNImputer for K-Nearest Neighbors imputation
################################################################################################################################################
def fill_missing_values_interactive_multi(df, 
                                    numerical_columns_list=None,
                                    categorical_columns_list=None,
                                    primary_key_cols=None, 
                                    exclude_columns=None,
                                    default_strategy='mean', # Default for auto-suggestion
                                    verbose=True):
    """
    Handles missing values interactively, suggesting and allowing user input for imputation strategy per column.
    Strategies include 'mean', 'median', 'mode', 'fill_value', and 'knn' (K-Nearest Neighbors).

    For numerical columns with 'mean' or 'median' strategies and composite primary keys, 
    the grouping for imputation uses only the first column of the primary key for broader context.
    'knn' imputation is applied globally across selected numerical features.
    Imputed numerical values are rounded and converted to integer type if the original column was integer-like.

    Args:
        df: Pandas DataFrame.
        numerical_columns_list (list, optional): List of columns explicitly numerical. Auto-detected if None.
        categorical_columns_list (list, optional): List of columns explicitly categorical. Auto-detected if None.
        primary_key_cols: List of columns forming the primary key. These are excluded from imputation.
        exclude_columns: List of additional column names to exclude from filling.
        default_strategy: Default imputation strategy to suggest ('mean', 'median', 'mode').
                          This is a general suggestion; 'knn' is specifically suggested for numericals.
        verbose: If True, prints information about missing values.

    Returns:
        Pandas DataFrame with imputed missing values.

    Notes on KNN Imputation:
    - If 'knn' is selected for any numerical column, KNNImputer will be applied to ALL numerical columns
      identified for imputation (i.e., those not excluded). It operates globally by finding neighbors
      in the feature space of these columns. It does not perform group-wise imputation.
    """

    df_processed = df.copy() 

    if exclude_columns is None:
        exclude_columns = []
    if primary_key_cols is None:
        primary_key_cols = []
    
    # Validate primary key columns
    if not all(col in df_processed.columns for col in primary_key_cols):
        raise ValueError("One or more primary key columns not found in the DataFrame.")

    # Store original dtypes to decide on rounding later
    original_dtypes = {col: df_processed[col].dtype for col in df_processed.columns}

    # Combine primary keys and explicitly excluded columns for complete exclusion from imputation
    all_excluded_cols_from_imputation = list(set(primary_key_cols + exclude_columns))

    # Initial candidate features for auto-detection or user-provided lists
    initial_candidate_features = [
        col for col in df_processed.columns 
        if col not in all_excluded_cols_from_imputation
    ]

    # --- Column Type Determination Logic (from previous functions) ---
    numerical_features_to_impute = []
    categorical_features_to_impute = []

    numerical_list_was_provided = numerical_columns_list is not None
    categorical_list_was_provided = categorical_columns_list is not None

    if numerical_list_was_provided and not categorical_list_was_provided:
        numerical_features_to_impute = [col for col in numerical_columns_list if col in initial_candidate_features]
    elif categorical_list_was_provided and not numerical_list_was_provided:
        categorical_features_to_impute = [col for col in categorical_columns_list if col in initial_candidate_features]
    elif numerical_list_was_provided and categorical_list_was_provided:
        numerical_features_to_impute = [col for col in numerical_columns_list if col in initial_candidate_features]
        categorical_features_to_impute = [col for col in categorical_columns_list if col in initial_candidate_features]
    else: # Auto-detect both
        numerical_features_to_impute = df_processed[initial_candidate_features].select_dtypes(include=np.number).columns.tolist()
        categorical_features_to_impute = df_processed[initial_candidate_features].select_dtypes(exclude=np.number).columns.tolist()
    
    numerical_features_to_impute = sorted(list(set(numerical_features_to_impute)))
    categorical_features_to_impute = sorted(list(set(categorical_features_to_impute)))

    # Identify columns with actual missing values that need imputation
    columns_to_impute_num = [col for col in numerical_features_to_impute if df_processed[col].isnull().any()]
    columns_to_impute_cat = [col for col in categorical_features_to_impute if df_processed[col].isnull().any()]
    missing_cols_overall = columns_to_impute_num + columns_to_impute_cat

    if verbose and missing_cols_overall:
        print("Columns with missing values identified for imputation:")
        if columns_to_impute_num: print(f"  Numerical: {', '.join(columns_to_impute_num)}")
        if columns_to_impute_cat: print(f"  Categorical: {', '.join(columns_to_impute_cat)}")
    elif verbose:
        print("No missing values found in the identified columns. Returning original DataFrame.")
        return df_processed 

    # --- Phase 1: Get User Strategies for all columns ---
    chosen_strategies = {}
    numerical_cols_for_knn = [] # Keep track of numerical columns for which KNN was chosen

    if verbose: print("\n--- Strategy Selection ---")
    
    # Suggest strategies and get user input for each column
    for col in missing_cols_overall:
        is_numeric = col in numerical_features_to_impute
        
        # Determine proposed strategy
        proposed_strategy = default_strategy
        if is_numeric:
            if df_processed[col].skew() > 1: # Highly skewed data often better with median
                proposed_strategy = 'median'
            else:
                proposed_strategy = 'mean'
            # Suggest KNN as a potential advanced option
            strategy_options = f"({proposed_strategy}/mean/median/knn/fill_value)"
        else: # Categorical
            proposed_strategy = 'mode'
            strategy_options = f"({proposed_strategy}/mode/fill_value)"
        
        while True:
            strategy_input = input(f"Enter strategy for '{col}' {strategy_options} (default: {proposed_strategy}, press Enter for default): ").lower().strip()
            chosen_strategy = strategy_input if strategy_input else proposed_strategy
            
            # Validate chosen strategy
            valid_for_numeric = ['mean', 'median', 'knn', 'fill_value']
            valid_for_categorical = ['mode', 'fill_value']

            if is_numeric and chosen_strategy in valid_for_numeric:
                break
            elif not is_numeric and chosen_strategy in valid_for_categorical:
                break
            else:
                print(f"Invalid strategy '{chosen_strategy}' for column '{col}' of type {'numeric' if is_numeric else 'categorical'}. Please choose from {strategy_options}.")
        
        chosen_strategies[col] = chosen_strategy
        if chosen_strategy == 'knn':
            numerical_cols_for_knn.append(col)
        
        if verbose: print(f"  '{col}' will be filled using '{chosen_strategy}'.")

    # --- Phase 2: Perform KNN Imputation (if selected) ---
    if numerical_cols_for_knn:
        if verbose: print("\n--- Executing KNN Imputation for selected numerical columns ---")
        
        # Prompt for k value only once
        k_val = 5 # Default K for KNN
        while True:
            try:
                k_input = input(f"Enter the number of neighbors (k) for KNN (default: {k_val}, press Enter for default): ").strip()
                k_val = int(k_input) if k_input else k_val
                if k_val > 0:
                    break
                else:
                    print("Invalid input. k must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        if verbose: print(f"Using k={k_val} for KNN imputation.")

        # KNNImputer needs all numerical features (not just the ones chosen for 'knn' imputation)
        # to find neighbors accurately. It should operate on `numerical_features_to_impute`.
        if not numerical_features_to_impute:
            if verbose: print("No numerical features available for KNN imputation. Skipping KNN.")
        else:
            if verbose: print(f"Applying KNNImputer to numerical columns: {', '.join(numerical_features_to_impute)}.")
            
            # Create a temporary df for KNN, converting to float as KNNImputer expects it
            df_numerical_for_knn = df_processed[numerical_features_to_impute].astype(float)
            
            imputer = KNNImputer(n_neighbors=k_val)
            # Fit and transform only on the numerical features identified for imputation
            imputed_data = imputer.fit_transform(df_numerical_for_knn)
            
            # Update the original df_processed with the imputed numerical data
            df_processed[numerical_features_to_impute] = imputed_data

            # Apply rounding/type conversion immediately after KNN for all affected numerical columns
            for col in numerical_features_to_impute: # Apply to all columns KNN touched
                if 'int' in str(original_dtypes[col]):
                    if verbose: print(f"  Rounding and converting '{col}' to integer type after KNN imputation.")
                    df_processed[col] = df_processed[col].round().astype(pd.Int64Dtype())
            if verbose: print("KNN Imputation complete.")
    else:
        if verbose: print("No numerical columns selected for KNN imputation. Skipping KNN phase.")

    # --- Phase 3: Perform other imputations (mean/median/mode/fill_value) ---
    if verbose: print("\n--- Executing other Imputation Strategies ---")
    
    # Keep track of columns already filled by KNN to avoid re-imputing
    columns_imputed_by_knn = set(numerical_features_to_impute) if numerical_cols_for_knn else set()

    for col in missing_cols_overall:
        if col in columns_imputed_by_knn and col in numerical_cols_for_knn:
            if verbose: print(f"  '{col}' already filled by KNN. Skipping other strategies.")
            continue # Skip if already handled by KNN

        strategy_to_apply = chosen_strategies[col]
        is_numeric = col in numerical_features_to_impute
        
        if strategy_to_apply == 'fill_value':
            fill_val = None
            while True:
                fill_val_input = input(f"Enter value to fill missing in '{col}': ").strip()
                if is_numeric:
                    try:
                        fill_val = float(fill_val_input)
                        break
                    except ValueError:
                        print("Invalid input for numeric column. Please enter a number.")
                else: # Categorical/Object
                    fill_val = fill_val_input
                    break
            df_processed[col].fillna(fill_val, inplace=True)
            if verbose: print(f"  Filled missing values in '{col}' with custom value '{fill_val}'.")

        elif primary_key_cols and all(c in df_processed.columns for c in primary_key_cols):
            # Determine actual grouping columns for imputation based on type and primary key structure
            actual_grouping_cols = primary_key_cols
            if is_numeric and len(primary_key_cols) > 1:
                # For numerical columns, group by the first PK column for broader context
                actual_grouping_cols = [primary_key_cols[0]] 
                if verbose:
                    print(f"  (Using '{primary_key_cols[0]}' for grouping column '{col}')")

            grouped = df_processed.groupby(actual_grouping_cols)

            if is_numeric:
                global_fill_value = None
                if strategy_to_apply == 'mean':
                    global_fill_value = df_processed[col].mean()
                elif strategy_to_apply == 'median':
                    global_fill_value = df_processed[col].median()
                # KNN handled in Phase 2
                else: # Should not happen if validation is correct
                    if verbose: print(f"  Warning: Invalid numeric strategy '{strategy_to_apply}' for '{col}', defaulting to global mean.")
                    global_fill_value = df_processed[col].mean()
                
                df_processed[col] = df_processed[col].fillna(
                    grouped[col].transform(strategy_to_apply).fillna(global_fill_value)
                )
                if verbose: print(f"  Filled missing values in '{col}' with group {strategy_to_apply} (fallback to global {strategy_to_apply}).")
                
                # --- Rounding and type conversion for numerical columns ---
                if 'int' in str(original_dtypes[col]): 
                    if verbose: print(f"  Rounding and converting '{col}' to integer type.")
                    df_processed[col] = df_processed[col].round().astype(pd.Int64Dtype())

            else: # Categorical/Object with primary key
                if strategy_to_apply == 'mode':
                    global_mode_series = df_processed[col].mode()
                    global_fill_value = global_mode_series.iloc[0] if not global_mode_series.empty else np.nan

                    for name, group in grouped:
                        group_mode_series = group[col].mode()
                        fill_val_for_group = None
                        if not group_mode_series.empty:
                            fill_val_for_group = group_mode_series.iloc[0]
                        else:
                            fill_val_for_group = global_fill_value

                        if pd.notna(fill_val_for_group):
                            df_processed.loc[group.index, col] = df_processed.loc[group.index, col].fillna(fill_val_for_group)
                    if verbose: print(f"  Filled missing values in '{col}' with group mode (fallback to global mode).")
                else: # Should not happen if validation is correct
                    if verbose: print(f"  Warning: Invalid categorical strategy '{strategy_to_apply}' for '{col}'. Defaulting to global mode.")
                    mode_val_series = df_processed[col].mode()
                    if not mode_val_series.empty:
                        df_processed[col].fillna(mode_val_series.iloc[0], inplace=True)

        else: # No primary key provided or invalid primary key columns for grouping
            if is_numeric:
                if strategy_to_apply == 'mean':
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    if verbose: print(f"  Filled missing values in '{col}' with global mean.")
                elif strategy_to_apply == 'median':
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    if verbose: print(f"  Filled missing values in '{col}' with global median.")
                # KNN handled in Phase 2
                else: # Should not happen
                    if verbose: print(f"  Warning: Invalid numeric strategy '{strategy_to_apply}' for '{col}', defaulting to global mean.")
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                
                # --- Rounding and type conversion for numerical columns ---
                if 'int' in str(original_dtypes[col]): 
                    if verbose: print(f"  Rounding and converting '{col}' to integer type.")
                    df_processed[col] = df_processed[col].round().astype(pd.Int64Dtype())

            else: # Categorical/Object without primary key
                if strategy_to_apply == 'mode':
                    mode_val_series = df_processed[col].mode()
                    if not mode_val_series.empty:
                        df_processed[col].fillna(mode_val_series.iloc[0], inplace=True)
                        if verbose: print(f"  Filled missing values in '{col}' with global mode: {mode_val_series.iloc[0]}.")
                    else:
                        if verbose: print(f"  Could not determine global mode for '{col}'. Skipping fill.")
                else: # Should not happen
                    if verbose: print(f"  Warning: Invalid categorical strategy '{strategy_to_apply}' for '{col}'. Defaulting to global mode.")
                    mode_val_series = df_processed[col].mode()
                    if not mode_val_series.empty:
                        df_processed[col].fillna(mode_val_series.iloc[0], inplace=True)


    if verbose:
        print("\n--- Missing values after filling (summary for imputed columns) ---")
        if missing_cols_overall:
            print(df_processed[missing_cols_overall].isnull().sum())
        else:
            print("No columns were identified for imputation, or all missing values were already handled.")

    return df_processed


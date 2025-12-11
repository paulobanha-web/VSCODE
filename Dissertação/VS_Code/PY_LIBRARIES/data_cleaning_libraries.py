################################################################################################################################################
'''Python Libraries'''
################################################################################################################################################
import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime 
from scipy.stats import zscore
################################################################################################################################################
'''Functions'''
################################################################################################################################################
# Function to read data based on file extension
#import pandas as pd
#import os
################################################################################################################################################
def load_data(filepath):

    _, file_ext = os.path.splitext(filepath)

    if file_ext == '.csv':
        return pd.read_csv(filepath)
    elif file_ext == '.json':
        return pd.read_json(filepath)
    elif file_ext == '.xlsx':
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")

################################################################################################################################################
# Function to campare two csv files
#import pandas as pd
################################################################################################################################################
def compare_csv_files(file1, file2, primary_key_cols):
    """
    Compares two CSV files based on primary key columns and prints the
    differences in values for each row.

    Args:
        file1: Path to the first CSV file.
        file2: Path to the second CSV file.
        primary_key_cols: A list of column names representing the primary key.
    """
    try:
        # Load the CSV files into pandas DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Set the primary key columns as index
        df1 = df1.set_index(primary_key_cols)
        df2 = df2.set_index(primary_key_cols)

        # Get common columns
        common_cols = list(set(df1.columns) & set(df2.columns))

        # Select only common columns for comparison
        df1 = df1[common_cols]
        df2 = df2[common_cols]

        # Check if either DataFrame is empty after selecting common columns
        if df1.empty or df2.empty:
            print("No common columns found between the CSV files.")
            return

        # Check if DataFrames have the same shape after selecting common columns
        if df1.shape != df2.shape:
            print("DataFrames have different shapes after selecting common columns:")
            print(f"DataFrame 1 shape: {df1.shape}")
            print(f"DataFrame 2 shape: {df2.shape}")
            return

        # Find rows with differences
        diff_mask = (df1 != df2).any(axis=1)
        diff_rows = df1[diff_mask]

        # Print the differences
        if diff_rows.empty:
            print("No differences found between the CSV files.")
        else:
            print("Differences found:")
            for index, row in diff_rows.iterrows():
                print(f"Primary Key: {index}")
                for col in common_cols:
                    if row[col] != df2.loc[index, col]:
                        print(f"  Column: {col}")
                        print(f"    df1 Value: {row[col]}")
                        print(f"    df2 Value: {df2.loc[index, col]}")
                        print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")

################################################################################################################################################
# Function to calculate correlation matrix for numerical features and print out heat map
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
################################################################################################################################################
def analyze_numerical(df: pd.DataFrame, correlation: float = 0.9, numerical_cols_to_exclude: list = None,
                      numerical_features: list = None):
  """Analyzes numerical columns of a dataframe.

  Calculates and prints the correlation matrix and heatmap for numerical columns.

  Args:
    df (pd.DataFrame): The input pandas DataFrame.
    correlation (float, optional): The correlation threshold to identify highly correlated features. Defaults to 0.9.
    numerical_cols_to_exclude (list, optional): A list of numerical column names to exclude from analysis.
                                                Defaults to None (no columns excluded).
    numerical_features (list, optional): A pre-defined list of numerical column names to analyze.
                                         If None, numerical columns will be identified based on dtype.
                                         Defaults to None.

  Returns:
    list: A list of numerical column names that were analyzed.
  """

  if numerical_cols_to_exclude is None:
    numerical_cols_to_exclude = []

  # 1. Determine Numerical Columns for Analysis:
  numerical_cols = []
  if numerical_features is not None:
    # Use the provided list of numerical features
    numerical_cols = [col for col in numerical_features if col in df.columns and col not in numerical_cols_to_exclude]
    if not numerical_cols:
      print("Warning: The provided 'numerical_features' list is empty or contains no valid columns in the DataFrame after excluding specified columns.")
      return []
  else:
    # Calculate numerical columns based on dtype and exclusion list
    numerical_cols = [col for col in df.columns
                      # Using is_numeric_dtype for broader compatibility
                      if pd.api.types.is_numeric_dtype(df[col])
                      and col not in numerical_cols_to_exclude]
  
  numerical_columns_count = len(numerical_cols)
  if numerical_columns_count == 0:
    print("No numerical columns found for analysis after exclusions or in the provided list.")
    return []

  print(f"The number of numerical columns being analyzed is: {numerical_columns_count}")

  # Calculate correlation matrix:
  # Ensure only numerical columns are used for correlation, dropping any non-numeric if they somehow made it through
  correlation_matrix = df[numerical_cols].select_dtypes(include=[np.number]).corr()
  
  if correlation_matrix.empty:
      print("No suitable numerical columns to calculate correlation matrix.")
      return numerical_cols

  plt.figure(figsize=(19, 15))  # Adjust figure size as desired
  sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 18})
  tick_label_size = 14 # Choose your desired size
  plt.xticks(fontsize=tick_label_size)
  plt.yticks(fontsize=tick_label_size)
  plt.title("Correlation Heatmap")
  plt.tight_layout()
  plt.show()

  # Use k=-1 to exclude self-correlation from the `stack()` result explicitly
  high_corr_features = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)) 
  high_corr_features = high_corr_features.stack()
  high_corr_features = high_corr_features[
      ((high_corr_features > correlation) | (abs(high_corr_features) == 1))] # Include correlations > threshold OR == 1.0

  # Convert the Series to a string and remove the dtype information
  high_corr_str = high_corr_features.to_string()
  high_corr_str = high_corr_str.replace('dtype: float64', '')

  if high_corr_features.empty:
    print(f"No features found with correlation > {correlation} (excluding self-correlation).")
  else:
    print(f"The columns with higher correlation (>{correlation}) are:\n{high_corr_str}")

  return numerical_cols

################################################################################################################################################
# Function to calculate correlation matrix for categorical features and print out heat map
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime
#from sklearn.preprocessing import OrdinalEncoder
################################################################################################################################################
def analyze_categorical(df: pd.DataFrame, correlation_threshold: float = 0.9, 
                        categorical_cols_to_exclude: list = None,
                        categorical_features: list = None):
    """Analyzes categorical columns of a DataFrame.

    Encodes specified or auto-detected categorical columns using OrdinalEncoder,
    handles missing values by imputing with mode after encoding (NaNs are preserved by OE
    and then filled), calculates and prints the correlation matrix and heatmap,
    displays the mapping between original and encoded values, and highlights columns
    with correlations above a specified threshold. Handles mixed data types in
    columns by converting datetime objects to strings before encoding.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        correlation_threshold (float, optional): The threshold for correlation above which columns
                                                 are highlighted. Defaults to 0.9.
        categorical_cols_to_exclude (list, optional): A list of categorical column names to exclude
                                                      from analysis. Defaults to None (no columns excluded).
        categorical_features (list, optional): A pre-defined list of column names to analyze as categorical.
                                              If None, columns of 'object' dtype will be automatically detected
                                              and treated as categorical.
                                              Defaults to None.

    Returns:
        None (prints results, heatmaps, and encoded value mappings)

    Notes:
    - If `categorical_features` is provided, numerical columns within this list (e.g., binary 0/1 integers,
      or low-cardinality numbers like satisfaction scores) will be treated as categorical and encoded.
      OrdinalEncoder treats each unique value (number or string) as a distinct category.
    - Missing values introduced by OrdinalEncoder (due to `unknown_value=np.nan`) are filled with the mode.
    """

    # 1. Create a copy of the DataFrame to avoid modifying the original:
    df_encoded = df.copy()

    if categorical_cols_to_exclude is None:
        categorical_cols_to_exclude = []

    # 2. Determine Categorical Columns for Analysis:
    cols_to_analyze_as_categorical = []

    if categorical_features is not None:
        # User provided a list; filter it by existence in DataFrame and exclusions
        cols_to_analyze_as_categorical = [col for col in categorical_features 
                                          if col in df_encoded.columns and col not in categorical_cols_to_exclude]
        if not cols_to_analyze_as_categorical:
            print("Warning: The provided 'categorical_features' list is empty or contains no valid columns in the DataFrame after exclusions.")
            return
        print(f"Using provided categorical columns: {', '.join(cols_to_analyze_as_categorical)}")
    else:
        # Auto-detect 'object' dtype columns not in exclusions
        cols_to_analyze_as_categorical = [col for col in df_encoded.columns
                                          if pd.api.types.is_object_dtype(df_encoded[col]) # Strictly looking for object dtype if not provided
                                          and col not in categorical_cols_to_exclude]
        if cols_to_analyze_as_categorical:
            print(f"Automatically detected object (categorical) columns: {', '.join(cols_to_analyze_as_categorical)}")
        else:
            print("No object (categorical) columns found for analysis based on automatic detection.")
            return # Exit if no columns to analyze

    categorical_columns_count = len(cols_to_analyze_as_categorical)
    print(f"The number of categorical columns being analyzed is: {categorical_columns_count}")

    if categorical_columns_count == 0:
        print("No categorical columns found for analysis after exclusions or in the provided list.")
        return

    # 3. Handle potential mixed data types by converting datetimes to strings
    # This loop runs on the identified `cols_to_analyze_as_categorical`
    for col in cols_to_analyze_as_categorical:
        # Check if the column contains any datetime objects (pd.Timestamp) or python datetime objects
        # Using .apply(type) and checking against known datetime types
        # More robust check: check first non-null element, or a sample
        if df_encoded[col].dropna().apply(lambda x: isinstance(x, (pd.Timestamp, datetime))).any():
            if df_encoded[col].dtype != 'object': # Ensure it's not already converted
                df_encoded[col] = df_encoded[col].astype(str)
                print(f"Converted datetime-like values in column '{col}' to string type for encoding.")


    # 4. Encoding Categorical Columns:
    # OrdinalEncoder treats NaNs as a separate category if not explicitly handled,
    # but `handle_unknown='use_encoded_value', unknown_value=np.nan` ensures new unseen values become NaN.
    # We will then explicitly fill these NaNs with the mode after encoding.
    ordinal_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=np.nan # New unknown values will become NaN
    )
    encoded_values_mapping = {}

    # Apply fit_transform only to the selected categorical columns
    # Create a temporary DataFrame for encoding to ensure dtypes are consistent for OE
    # Convert selected columns to object/string dtype if they aren't already, for consistent OE behavior
    temp_df_for_encoding = df_encoded[cols_to_analyze_as_categorical].astype(str)
    
    # Fit and transform the identified categorical columns
    df_encoded[cols_to_analyze_as_categorical] = ordinal_encoder.fit_transform(temp_df_for_encoding)

    # Store the mappings
    for i, col in enumerate(cols_to_analyze_as_categorical):
        # OrdinalEncoder.categories_ holds arrays of unique values for each fitted column
        encoded_values_mapping[col] = dict(
            zip(ordinal_encoder.categories_[i],
                range(len(ordinal_encoder.categories_[i])))
        )
    
    # 5. Handle Missing Values AFTER Encoding (any NaNs from original data or `unknown_value=np.nan`)
    print("\nHandling missing values in encoded categorical columns (filling with mode)...")
    for col in cols_to_analyze_as_categorical:
        if df_encoded[col].isnull().any():
            # Calculate mode based on the *encoded* values
            mode_val = df_encoded[col].mode()
            if not mode_val.empty:
                df_encoded[col].fillna(mode_val.iloc[0], inplace=True)
                print(f"  Filled missing values in '{col}' with encoded mode: {mode_val.iloc[0]}.")
            else:
                print(f"  Could not determine mode for encoded column '{col}'. Missing values remain.")
    
    # 6. Correlation Analysis:
    # Ensure correlation is calculated only on the now-numeric (encoded) categorical columns
    correlation_matrix = df_encoded[cols_to_analyze_as_categorical].corr()

    if correlation_matrix.empty:
        print("No suitable categorical columns to calculate correlation matrix after encoding.")
        return

    plt.figure(figsize=(19, 15))  # Adjust figure size as desired
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8})
    plt.title("Correlation Heatmap (Categorical Columns - Ordinal Encoded)")
    plt.tight_layout()
    plt.show()

    # 7. Find and print highly correlated features
    # Use k=-1 to exclude self-correlation from the `stack()` result
    high_corr_features = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool)) 
    high_corr_features = high_corr_features.stack()
    high_corr_features = high_corr_features[
        ((high_corr_features > correlation_threshold) | (abs(high_corr_features) == 1))] 

    # Convert the Series to a string and remove the dtype information
    high_corr_str = high_corr_features.to_string()
    high_corr_str = high_corr_str.replace('dtype: float64', '')

    if high_corr_features.empty:
        print(f"\nNo features found with correlation > {correlation_threshold} (excluding self-correlation).")
    else:
        print(f"\nThe columns with higher correlation (>{correlation_threshold}) are:\n{high_corr_str}")

    # 8. Display Encoded Value Mappings:
    print("\n--- Encoded Value Mappings ---")
    for col, mapping in encoded_values_mapping.items():
        print(f"\nColumn: {col}")
        # Sort the mapping by encoded value for clearer display
        sorted_mapping = sorted(mapping.items(), key=lambda item: item[1])
        for original_value, encoded_value in sorted_mapping:
            print(f"  '{original_value}' -> {encoded_value}")

################################################################################################################################################
# Function to calculate PCA for numerical features, advise for columns to keep and drop and prints out results
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
################################################################################################################################################
def analyze_numerical_pca(df: pd.DataFrame, numerical_cols_to_exclude: list = None,
                          numerical_features: list = None, # New parameter
                          variance_threshold: float = 0.95, 
                          imputation_strategy: str = 'mean'):
    """Analyzes numerical columns of a dataframe using PCA.

    Handles missing values using imputation before applying PCA.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        numerical_cols_to_exclude (list, optional): A list of numerical column names to exclude from analysis.
                                                    Defaults to None (no columns excluded).
        numerical_features (list, optional): A pre-defined list of numerical column names to analyze.
                                            If None, numerical columns will be identified based on dtype ('int64' or 'float64').
                                            Defaults to None.
        variance_threshold (float, optional): The desired cumulative explained variance threshold (default 0.95).
        imputation_strategy (str, optional): The strategy to use for imputation ('mean', 'median', 'most_frequent', 'constant').
                                             Defaults to 'mean'.

    Returns:
        A tuple containing:
          - n_components (int): The suggested number of components to keep.
          - explained_variance_ratio (np.array): The explained variance ratio for each component.
          - features_to_keep (list): A list of original feature names that contribute most to the principal components.
          - features_to_drop (list): A list of original feature names that were identified for dropping based on PCA.
        Returns (0, np.array([]), [], []) if no numerical columns are found for analysis.
    """

    if numerical_cols_to_exclude is None:
        numerical_cols_to_exclude = []

    # 1. Determine Numerical Columns for Analysis:
    cols_to_analyze_as_numerical = []

    if numerical_features is not None:
        # Use the provided list of numerical features
        cols_to_analyze_as_numerical = [col for col in numerical_features
                                        if col in df.columns and col not in numerical_cols_to_exclude]
        if not cols_to_analyze_as_numerical:
            print("Warning: The provided 'numerical_features' list is empty or contains no valid columns in the DataFrame after exclusions.")
            return 0, np.array([]), [], []
        print(f"Using provided numerical columns: {', '.join(cols_to_analyze_as_numerical)}")
    else:
        # Calculate numerical columns based on dtype and exclusion list
        cols_to_analyze_as_numerical = [col for col in df.columns
                                        # Using is_numeric_dtype for broader compatibility
                                        if pd.api.types.is_numeric_dtype(df[col]) 
                                        and col not in numerical_cols_to_exclude]
        if cols_to_analyze_as_numerical:
            print(f"Automatically detected numerical columns: {', '.join(cols_to_analyze_as_numerical)}")
        else:
            print("No numerical columns found for analysis based on automatic detection.")
            return 0, np.array([]), [], []

    num_components_analyzed = len(cols_to_analyze_as_numerical)
    if num_components_analyzed == 0:
        print("No numerical columns found for PCA analysis after exclusions or in the provided list.")
        return 0, np.array([]), [], []

    print(f"Number of numerical columns being analyzed: {num_components_analyzed}")

    # 2. Standardize the data (with imputation):
    x = df[cols_to_analyze_as_numerical].values

    # Imputation using SimpleImputer
    imputer = SimpleImputer(strategy=imputation_strategy)
    x_imputed = imputer.fit_transform(x)

    x_scaled = StandardScaler().fit_transform(x_imputed)

    # Check if there's enough data for PCA after imputation and scaling
    if x_scaled.shape[0] < 2 or x_scaled.shape[1] < 1:
        print("Not enough data (rows or columns) after imputation for PCA. Requires at least 2 rows and 1 column.")
        return 0, np.array([]), [], []

    # 3. Perform PCA:
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(x_scaled)

    # If PCA determined 0 components (e.g., all variance is negligible or data is constant)
    if pca.n_components_ == 0:
        print(f"PCA could not find components explaining {variance_threshold*100:.1f}% variance. "
              "This might happen if all features have zero variance or are highly correlated reducing dimensions to 0.")
        return 0, np.array([]), [], []

    # 4. Get the features to keep and drop:
    # Create a DataFrame with component loadings
    loadings = pd.DataFrame(pca.components_.T, 
                            columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
                            index=cols_to_analyze_as_numerical)

    # Get features with highest absolute loading for each PC
    features_to_keep_set = set()
    for pc_col in loadings.columns:
        feature_with_max_loading = loadings[pc_col].abs().idxmax()
        features_to_keep_set.add(feature_with_max_loading)
    
    features_to_keep = list(features_to_keep_set)

    # Get features to drop (all analyzed numerical columns except those to keep)
    features_to_drop = list(set(cols_to_analyze_as_numerical) - features_to_keep_set)


    # 5. Calculate cumulative explained variance:
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


    # 6. Plot cumulative explained variance vs. number of components:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100:.1f}% Variance Threshold')
    
    if pca.n_components_ > 0:
        plt.axvline(x=pca.n_components_, color='b', linestyle=':', label=f'{pca.n_components_} Components Selected')
        #plt.text(pca.n_components_ + 0.5, 0.1, f'{pca.n_components_} Components', color='blue', ha='left')
        plt.text(pca.n_components_ + 0.1, 0.1, 
             f'{pca.n_components_} Components', 
             color='blue', ha='center', fontsize=12)

    #plt.text(0.5, variance_threshold - 0.05, f'{variance_threshold*100:.1f}% variance', color='red', fontsize=12)
    max_components = len(cumulative_variance)
    plt.text(max_components * 0.9, variance_threshold - 0.02, f'{variance_threshold*100:.1f}% variance', 
         color='red', fontsize=12, ha='right')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, len(cumulative_variance) + 1))
    #plt.tight_layout()
    plt.show()

    # 7. Determine number of components to keep:
    n_components = pca.n_components_
    print(f"Suggested number of components to keep: {n_components}")

    # 8. Return results:
    return n_components, pca.explained_variance_ratio_, features_to_keep, features_to_drop

################################################################################################################################################
# Function to calculate PCA for categorical features, advise for columns to keep and drop and prints out results
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from datetime import datetime
#from sklearn.preprocessing import OrdinalEncoder
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
################################################################################################################################################
def analyze_categorical_pca(df: pd.DataFrame, 
                            categorical_cols_to_exclude: list = None,
                            categorical_features: list = None, # New parameter
                            variance_threshold: float = 0.95, 
                            imputation_strategy: str = 'most_frequent'):
    """Analyzes categorical columns of a DataFrame using PCA.

    Encodes specified or auto-detected categorical columns using OrdinalEncoder,
    handles missing values using imputation, calculates and prints the cumulative
    explained variance vs. number of components, and suggests components that can
    potentially be reduced based on a variance threshold. Also displays the mapping
    between original and encoded values for categorical columns. Handles mixed
    data types in columns by converting datetime objects to strings.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        categorical_cols_to_exclude (list, optional): A list of categorical column names to exclude from analysis.
                                                      Defaults to None (no columns excluded).
        categorical_features (list, optional): A pre-defined list of column names to analyze as categorical.
                                              If None, columns of 'object' dtype will be automatically detected
                                              and treated as categorical.
                                              Defaults to None.
        variance_threshold (float, optional): The desired cumulative explained variance threshold (default 0.95).
        imputation_strategy (str, optional): The strategy to use for imputation ('most_frequent', 'constant', etc.).
                                             Defaults to 'most_frequent'.

    Returns:
        A tuple containing:
            - n_components (int): The suggested number of components to keep.
            - explained_variance_ratio (np.array): The explained variance ratio for each component.
            - features_to_keep (list): A list of original feature names that contribute most to the principal components.
            - features_to_drop (list): A list of original feature names that were identified for dropping based on PCA.
            - encoded_values_mapping (dict): A dictionary containing the mapping between original
                                             and encoded values for categorical columns.
        Returns (0, np.array([]), [], [], {}) if no categorical columns are found for analysis.

    Notes:
    - If `categorical_features` is provided, numerical columns within this list (e.g., binary 0/1 integers,
      or low-cardinality numbers like satisfaction scores) will be treated as categorical and encoded.
      OrdinalEncoder treats each unique value (number or string) as a distinct category.
    - Missing values introduced by OrdinalEncoder (due to `unknown_value=np.nan`) or originally present
      are filled using the specified `imputation_strategy`.
    """

    if categorical_cols_to_exclude is None:
        categorical_cols_to_exclude = []

    # 1. Determine Categorical Columns for Analysis:
    cols_to_analyze_as_categorical = []

    if categorical_features is not None:
        # User provided a list; filter it by existence in DataFrame and exclusions
        cols_to_analyze_as_categorical = [col for col in categorical_features 
                                          if col in df.columns and col not in categorical_cols_to_exclude]
        if not cols_to_analyze_as_categorical:
            print("Warning: The provided 'categorical_features' list is empty or contains no valid columns in the DataFrame after exclusions.")
            return 0, np.array([]), [], [], {}
        print(f"Using provided categorical columns: {', '.join(cols_to_analyze_as_categorical)}")
    else:
        # Auto-detect 'object' dtype columns not in exclusions
        cols_to_analyze_as_categorical = [col for col in df.columns
                                          if pd.api.types.is_object_dtype(df[col]) # Strictly looking for object dtype if not provided
                                          and col not in categorical_cols_to_exclude]
        if cols_to_analyze_as_categorical:
            print(f"Automatically detected object (categorical) columns: {', '.join(cols_to_analyze_as_categorical)}")
        else:
            print("No object (categorical) columns found for analysis based on automatic detection.")
            return 0, np.array([]), [], [], {} # Exit if no columns to analyze

    num_components_analyzed = len(cols_to_analyze_as_categorical)
    if num_components_analyzed == 0:
        print("No categorical columns found for PCA analysis after exclusions or in the provided list.")
        return 0, np.array([]), [], [], {}

    print(f"Number of categorical columns being analyzed: {num_components_analyzed}")

    # Create a copy of the DataFrame for encoding to avoid modifying the original
    df_encoded = df[cols_to_analyze_as_categorical].copy()

    # 2. Handle potential mixed data types by converting datetimes to strings
    for col in cols_to_analyze_as_categorical:
        if df_encoded[col].dropna().apply(lambda x: isinstance(x, (pd.Timestamp, datetime))).any():
            if df_encoded[col].dtype != 'object': # Avoid unnecessary conversion if already object
                df_encoded[col] = df_encoded[col].astype(str)
                print(f"Converted datetime-like values in column '{col}' to string type for encoding.")


    # 3. Encoding Categorical Columns:
    # OrdinalEncoder treats NaNs as a separate category if not explicitly handled,
    # but `handle_unknown='use_encoded_value', unknown_value=np.nan` ensures new unseen values become NaN.
    # We will then explicitly fill these NaNs with the imputation strategy.
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    encoded_values_mapping = {}

    # Convert selected columns to string dtype before encoding to ensure consistent behavior for OE
    # This is important if `categorical_features` includes numerical columns (e.g., 0/1, 1-5 ratings)
    temp_df_for_encoding = df_encoded[cols_to_analyze_as_categorical].astype(str)
    
    # Fit and transform the identified categorical columns
    df_encoded[cols_to_analyze_as_categorical] = ordinal_encoder.fit_transform(temp_df_for_encoding)

    # Store the mappings
    for i, col in enumerate(cols_to_analyze_as_categorical):
        encoded_values_mapping[col] = dict(zip(ordinal_encoder.categories_[i], range(len(ordinal_encoder.categories_[i]))))

    # 4. Standardize the data (with imputation):
    x = df_encoded[cols_to_analyze_as_categorical].values

    # Imputation using SimpleImputer (applies to NaNs from original data or `unknown_value`)
    imputer = SimpleImputer(strategy=imputation_strategy)
    x_imputed = imputer.fit_transform(x)

    x_scaled = StandardScaler().fit_transform(x_imputed)

    # Check if there's enough data for PCA after imputation and scaling
    if x_scaled.shape[0] < 2 or x_scaled.shape[1] < 1:
        print("Not enough data (rows or columns) after imputation for PCA. Requires at least 2 rows and 1 column.")
        return 0, np.array([]), [], [], {}

    # 5. Perform PCA:
    pca = PCA(n_components=variance_threshold)
    pca.fit(x_scaled)

    # If PCA determined 0 components (e.g., all variance is negligible or data is constant)
    if pca.n_components_ == 0:
        print(f"PCA could not find components explaining {variance_threshold*100:.1f}% variance. "
              "This might happen if all features have zero variance or are highly correlated reducing dimensions to 0.")
        return 0, np.array([]), [], [], {}

    # 6. Get the features to keep and drop:
    # Create a DataFrame with component loadings
    loadings = pd.DataFrame(pca.components_.T, 
                            columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
                            index=cols_to_analyze_as_categorical)

    # Get features with highest absolute loading for each PC
    features_to_keep_set = set() # Use a set to automatically handle duplicates
    for pc_col in loadings.columns:
        feature_with_max_loading = loadings[pc_col].abs().idxmax()
        features_to_keep_set.add(feature_with_max_loading)
    
    features_to_keep = list(features_to_keep_set)

    # Get features to drop (all analyzed categorical columns except those to keep)
    features_to_drop = list(set(cols_to_analyze_as_categorical) - features_to_keep_set)

    # 7. Calculate cumulative explained variance:
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 8. Plot cumulative explained variance vs. number of components:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance (Categorical Columns)')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100:.1f}% Variance Threshold')
    
    if pca.n_components_ > 0:
        plt.axvline(x=pca.n_components_, color='b', linestyle=':', label=f'{pca.n_components_} Components Selected')
        #plt.text(pca.n_components_ + 0.5, 0.1, f'{pca.n_components_} Components', color='blue', ha='left')

    plt.text(0.5, variance_threshold - 0.05, f'{variance_threshold*100:.1f}% variance', color='red', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, len(cumulative_variance) + 1))
    #plt.tight_layout()
    plt.show()

    # 9. Determine number of components to keep:
    n_components = pca.n_components_
    print(f"Suggested number of components to keep: {n_components}")

    # 10. Display Encoded Value Mappings:
    print("\n--- Encoded Value Mappings ---")
    for col, mapping in encoded_values_mapping.items():
        print(f"\nColumn: {col}")
        sorted_mapping = sorted(mapping.items(), key=lambda item: item[1])
        for original_value, encoded_value in sorted_mapping:
            print(f"  '{original_value}' -> {encoded_value}")

    # 11. Return results:
    return n_components, pca.explained_variance_ratio_, features_to_keep, features_to_drop, encoded_values_mapping


################################################################################################################################################  
# Function to evaluate classe on given columns
#import pandas as pd
################################################################################################################################################
def evaluate_classes_interactive(df, known_class_column=None):
    """
    Evaluates possible classes in a Pandas DataFrame interactively, allowing
    the user to choose the columns to consider, focusing on categorical features
    or a specifically known class column (even if numeric).

    Args:
        df (pandas.DataFrame): The DataFrame to evaluate.
        known_class_column (str, optional): A column name that is known to be
                                            a class column, even if it's numeric.
                                            This column will be validated and
                                            prioritized for user selection.
                                            Defaults to None.

    Returns:
        tuple: A tuple containing:
               - pandas.DataFrame: A DataFrame summarizing the unique classes and their counts
                 for the selected columns.
               - list: A list of the column names selected by the user.
    """

    all_eligible_cols = []

    # If a known_class_column is provided, add it to eligible columns
    if known_class_column:
        if known_class_column not in df.columns:
            print(f"Warning: Known class column '{known_class_column}' not found in DataFrame. Ignoring this parameter.")
        else:
            all_eligible_cols.append(known_class_column)

    # Get a list of categorical columns (object, category, bool)
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_eligible_cols.extend([col for col in categorical_cols if col not in all_eligible_cols])

    # Also include numeric columns for interactive selection, if not already added by known_class_column
    # The user might want to treat int/float as categories (e.g., status codes)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_eligible_cols.extend([col for col in numeric_cols if col not in all_eligible_cols])


    if not all_eligible_cols:
        print("No eligible columns (categorical or numeric) found to evaluate.")
        return pd.DataFrame(), []

    selected_columns = []

    # --- Step 1: Prioritize known_class_column if provided and valid ---
    if known_class_column and known_class_column in df.columns:
        print(f"\nPotential class column suggested: '{known_class_column}' (from 'known_class_column' parameter).")
        print(f"Unique values in '{known_class_column}': {df[known_class_column].nunique()}")
        print(f"Top 5 counts in '{known_class_column}':\n{df[known_class_column].value_counts().head()}")

        user_choice = input(f"Do you want to evaluate '{known_class_column}' as a class column? (yes/no): ").lower()
        if user_choice == 'yes':
            selected_columns.append(known_class_column)
            # Remove it from the list of all_eligible_cols to prevent double-prompting
            if known_class_column in all_eligible_cols:
                all_eligible_cols.remove(known_class_column)
        elif user_choice == 'no':
            print("Skipping known class column. Proceeding to interactive selection.")
        # If input is neither 'yes' nor 'no', it will proceed to interactive selection below


    # --- Step 2: Interactive selection from the remaining eligible columns ---
    print("\nSelect additional columns to evaluate for classes (enter comma-separated column numbers).")
    print("This includes categorical and numeric columns you might want to treat as categories:")
    for i, column in enumerate(all_eligible_cols):
        print(f"{i + 1}. {column} (Type: {df[column].dtype}, Unique: {df[column].nunique()})")

    while True:
        try:
            selected_columns_indices_str = input("Enter your choices (e.g., 1,3,5), or press Enter to skip: ")
            if not selected_columns_indices_str.strip(): # User pressed Enter
                if not selected_columns: # If no columns were selected at all (not even known_class_column)
                    print("No columns selected for class evaluation.")
                    return pd.DataFrame(), []
                else: # If known_class_column was selected, just proceed with it
                    break
            
            # Process input for additional selections
            current_selection_indices = [int(x.strip()) for x in selected_columns_indices_str.split(",") if x.strip()]
            
            # Temporarily store new selections to add to 'selected_columns'
            new_selections = []
            for i in current_selection_indices:
                if 1 <= i <= len(all_eligible_cols):
                    col_name = all_eligible_cols[i - 1]
                    if col_name not in selected_columns: # Avoid adding duplicates if known_class_column was already chosen
                        new_selections.append(col_name)
                else:
                    print(f"Warning: Column number {i} is out of range. Skipping.")
            
            if new_selections:
                selected_columns.extend(new_selections)
                break
            elif not selected_columns: # If no new valid selections and no previous selections
                print("Invalid selection. Please enter valid numbers from the list, or press Enter to skip.")
            else: # If no new valid selections, but some were already selected (e.g., via known_class_column)
                print("No new valid columns selected. Proceeding with previously selected columns.")
                break # Exit loop if no new valid selections
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or press Enter to skip.")
        except IndexError: # This might occur if selected_columns_indices is empty but try to access it
            print("Invalid selection. Please enter numbers within the range of the list.")

    if not selected_columns:
        print("No columns were ultimately selected for class evaluation.")
        return pd.DataFrame(), []

    # Initialize class_summary list
    class_summary = []

    # Iterate through the selected columns to evaluate classes
    for column in selected_columns:
        print(f"\n--- Evaluating column: '{column}' ---")
        column_data = df[column].dropna()
        if column_data.empty:
            print(f"Column '{column}' is empty after dropping NaNs. Skipping.")
            continue

        unique_classes = column_data.unique()
        class_counts = column_data.value_counts()

        print(f"Total unique classes: {len(unique_classes)}")
        print(f"Top 10 classes and their counts in '{column}':")
        print(class_counts.head(10)) # Display top 10 for quick overview

        for class_value in unique_classes:
            class_summary.append([column, class_value, class_counts[class_value]])

    # Create the summary DataFrame
    summary_df = pd.DataFrame(class_summary, columns=['Column', 'Class', 'Count'])
    summary_df = summary_df.sort_values(by=['Column', 'Count'], ascending=[True, False])

    # Return the summary DataFrame and the list of selected columns
    return summary_df, selected_columns

################################################################################################################################################  
# Function to check for duplicated columns with user input 
#import pandas as pd
################################################################################################################################################
import pandas as pd

def handle_duplicate_cols(df):
    """
    Checks for duplicate columns, suggests deletions, and removes them based on user input.
    The function groups duplicate columns by their content and displays them.
    Users can select specific columns to delete by their corresponding number (comma-separated).
    If a group of duplicates is found, at least one column from that group must be kept,
    unless the user explicitly types 'none' for that group to keep all.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame after processing duplicate columns.
    """
    current_df = df.copy() # Work on a copy to avoid modifying original until confirmed

    # Transpose and find duplicate rows (which are duplicate columns in original df)
    df_transposed = current_df.T

    # Get boolean series for duplicated rows in transposed df, keeping the first occurrence
    # to identify the "original" in a duplicate set
    # Using keep='first' ensures we only get the *actual* duplicates, not the originals
    duplicated_mask = df_transposed.duplicated(keep='first')

    # Get all column names that are duplicates of some other column based on content
    all_duplicate_column_names_identified = df_transposed[duplicated_mask].index.tolist()

    if not all_duplicate_column_names_identified:
        print("No duplicate columns found.")
        return current_df

    print("Duplicate columns found based on content:")

    duplicate_groups = {}
    for col_name in current_df.columns:
        # Convert Series to a hashable type (tuple of values).
        # Using .to_string() and splitting lines is a robust way to handle mixed types
        # and ensure consistent hashing across identical columns.
        col_series = current_df[col_name]
        col_hash = tuple(col_series.to_string(index=False, header=False, float_format='%.10f').splitlines())

        if col_hash not in duplicate_groups:
            duplicate_groups[col_hash] = []
        duplicate_groups[col_hash].append(col_name)

    # Filter out groups that are not duplicates (i.e., only contain one column)
    # And only consider groups where at least one column in the group was identified as a duplicate.
    # This refinement ensures we only interact with actual duplicate groups
    filtered_duplicate_groups = {
        k: v for k, v in duplicate_groups.items()
        if len(v) > 1 and any(col in all_duplicate_column_names_identified for col in v)
    }

    if not filtered_duplicate_groups:
        print("No actual groups of identical duplicate columns found after filtering.")
        return current_df

    columns_to_drop_overall = []

    for i, (content_hash, cols_in_group) in enumerate(filtered_duplicate_groups.items()):
        print(f"\n--- Duplicate Group {i + 1} ---")
        print("Columns with identical content:")
        # Enumerate columns for easy selection by number
        numbered_cols_in_group = {idx + 1: col for idx, col in enumerate(cols_in_group)}
        for num, col_name in numbered_cols_in_group.items():
            print(f"  {num}. {col_name}")

        # Display content preview
        print(f"\nContent preview of these columns (first few rows):")
        if len(current_df) <= 10:
            print(current_df[cols_in_group].to_string())
        else:
            print(current_df[cols_in_group].head(10).to_string())
            print(f"... Showing first 10 of {len(current_df)} rows for these columns.")

        prompt = (
            f"\nFor group {i+1} ({', '.join(cols_in_group)}):\n"
            "Select columns to delete by number (e.g., 1,3). You must keep at least one column.\n"
            "Or type 'none' to keep all columns in this group: "
        )
        input_str = input(prompt).strip()

        if input_str.lower() == 'none':
            print(f"Keeping all columns in group {i+1}.")
            continue

        selected_nums_to_delete_str = [x.strip() for x in input_str.split(',') if x.strip()]
        
        valid_selected_cols_for_group = []
        invalid_nums = []

        for num_str in selected_nums_to_delete_str:
            try:
                num = int(num_str)
                if num in numbered_cols_in_group:
                    valid_selected_cols_for_group.append(numbered_cols_in_group[num])
                else:
                    invalid_nums.append(num_str)
            except ValueError:
                invalid_nums.append(num_str)

        if invalid_nums:
            print(f"Warning: Invalid input numbers or non-numeric entries: {', '.join(invalid_nums)}. Ignoring these.")

        # Check if the user is attempting to delete ALL columns in this group
        if len(valid_selected_cols_for_group) == len(cols_in_group):
            print(f"Error: You must keep at least one column from group {i+1}. No columns deleted from this group.")
        elif valid_selected_cols_for_group:
            columns_to_drop_overall.extend(valid_selected_cols_for_group)
            print(f"Marked for deletion from group {i+1}: {valid_selected_cols_for_group}")
        else:
            print("No valid column numbers selected for deletion from this group.")

    if columns_to_drop_overall:
        print(f"\nFinal columns to be deleted: {columns_to_drop_overall}")
        current_df = current_df.drop(columns=columns_to_drop_overall)
        print("Duplicate columns removed.")
    else:
        print("\nNo duplicate columns were selected for deletion.")

    return current_df

################################################################################################################################################
# Function to check for duplicated rows with user input 

################################################################################################################################################
def handle_duplicate_rows(df):
    """
    Checks for duplicate rows, groups them, and allows the user to delete
    all suggested duplicates or select specific ones by index.

    Args:
        df: The pandas DataFrame to process.

    Returns:
        The DataFrame with the selected duplicate rows removed.
    """
    # Identify all duplicate rows. keep=False marks all duplicates as True.
    duplicate_rows = df[df.duplicated(keep=False)]

    if duplicate_rows.empty:
        print("No duplicate rows found.")
        return df  # Return the original DataFrame if no duplicates

    # Group the duplicate row indices.
    duplicate_groups = []
    seen_indices = set()
    for index in duplicate_rows.index:
        if index not in seen_indices:
            group_indices = list(df[df.apply(lambda row: row.equals(df.loc[index]), axis=1)].index)
            seen_indices.update(group_indices)
            duplicate_groups.append(group_indices)

    print("Duplicate Groups:")
    for i, group in enumerate(duplicate_groups):
        print(f"Group {i+1}: Indices {group}")
        #display(df.loc[group])
        print(df.loc[group].to_string())

    # Suggest rows to delete, keeping the first row of each group.
    suggested_delete_indices = []
    for group in duplicate_groups:
        suggested_delete_indices.extend(group[1:])  # Keep first, delete the rest

    if suggested_delete_indices:
        print(f"\nSuggested rows to delete: {suggested_delete_indices}")

        # Get user input on which rows to delete or to keep all
        prompt = "Select rows to delete (comma-separated row indices), type 'all' to delete all suggested, or type 'none' to keep all suggested rows:"
        input_str = input(prompt).strip().lower()

        if input_str == 'all':
            df = df.drop(index=suggested_delete_indices)
            print(f"Removed all suggested rows: {suggested_delete_indices}")
        elif input_str == 'none':
            print("Keeping all suggested rows.")
        # No action needed, the DataFrame 'df' remains unchanged
        else:
            selected_rows_to_delete_str = [x.strip() for x in input_str.split(',') if x.strip()]
            selected_rows_to_delete = []
            for item in selected_rows_to_delete_str:
                if item.isdigit():
                    selected_rows_to_delete.append(int(item))
                else:
                    print(f"Warning: '{item}' is not a valid row index and will be ignored.")

            valid_selected_rows = [index for index in selected_rows_to_delete if index in suggested_delete_indices]

            if valid_selected_rows:
                df = df.drop(index=valid_selected_rows)
                print(f"Removed rows with indices: {valid_selected_rows}")
            else:
                print("No valid row indices selected for deletion.")
    else:
        print("No rows suggested for deletion.")

    return df

################################################################################################################################################
# Function to handle with a given threshold missing values on columns and rows 
#import pandas as pd
################################################################################################################################################
def handle_missing_values_threshold_col_row(df: pd.DataFrame, threshold_col: float = None, threshold_row: float = None, verbose: bool = True) -> pd.DataFrame:
    """Handles missing values in a DataFrame, allowing user to choose columns or rows to drop using numbers.

    Args:
        df: The input DataFrame.
        threshold_col: The threshold for missing values in columns (percentage).
                       If None, column handling is skipped.
        threshold_row: The threshold for missing values in rows (percentage).
                       If None, row handling is skipped.
        verbose: Whether to print output (default: True).

    Returns:
        The DataFrame with missing values handled, or None if input DataFrame is None or becomes empty.
    """
    if df is None:
        if verbose:
            print("Input DataFrame is None. Returning None.")
        return None

    df_copy = df.copy()

    # Flags to track if any items *would have* prompted a user interaction
    # These are determined by re-evaluating the threshold condition
    cols_would_prompt = False
    rows_would_prompt = False

    # Handle columns if threshold_col is provided
    if threshold_col is not None:
        if verbose:
            print(f"\n--- Handling columns with missing values (threshold: {threshold_col:.2f}%) ---")
        
        # --- Duplicate logic to check if columns would exceed threshold ---
        # Axis 0 in sum() is for summing down columns, to get column-wise missing values
        # df.shape[0] is number of rows
        col_missing_percentage = (df_copy.isnull().sum(axis=0) / df_copy.shape[0]) * 100
        cols_exceeding_threshold = col_missing_percentage[col_missing_percentage > threshold_col]
        cols_would_prompt = cols_exceeding_threshold.any()
        # --- End of duplicated logic ---

        df_copy = _handle_missing_axis_numbered(df_copy, threshold_col, axis=0, verbose=verbose) # axis=1 for columns (as per your _handle_missing_axis_numbered)

        if df_copy is None:
            if verbose:
                print("DataFrame became empty after column handling. Returning None.")
            return None
        elif df_copy.empty and verbose:
            print("Warning: DataFrame is now empty after column handling.")

    # Handle rows if threshold_row is provided
    if threshold_row is not None:
        if verbose:
            print(f"\n--- Handling rows with missing values (threshold: {threshold_row:.2f}%) ---")
        
        # --- Duplicate logic to check if rows would exceed threshold ---
        # Axis 1 in sum() is for summing across rows, to get row-wise missing values
        # df.shape[1] is number of columns
        row_missing_percentage = (df_copy.isnull().sum(axis=1) / df_copy.shape[1]) * 100
        rows_exceeding_threshold = row_missing_percentage[row_missing_percentage > threshold_row]
        rows_would_prompt = rows_exceeding_threshold.any()
        # --- End of duplicated logic ---

        df_copy = _handle_missing_axis_numbered(df_copy, threshold_row, axis=1, verbose=verbose) # axis=0 for rows (as per your _handle_missing_axis_numbered)

        if df_copy is None:
            if verbose:
                print("DataFrame became empty after row handling. Returning None.")
            return None
        elif df_copy.empty and verbose:
            print("Warning: DataFrame is now empty after row handling.")

    # --- Consolidated message for no action needed ---
    if verbose:
        # If column handling was requested AND no columns were found that would trigger a prompt
        if threshold_col is not None and not cols_would_prompt:
            print(f"\nNo columns found with more than {threshold_col:.2f}% missing values to consider for removal.")
        
        # If row handling was requested AND no rows were found that would trigger a prompt
        if threshold_row is not None and not rows_would_prompt:
            print(f"\nNo rows found with more than {threshold_row:.2f}% missing values to consider for removal.")
            
    return df_copy

def _handle_missing_axis_numbered(df, threshold, axis, verbose=True):
    """Handles missing values for a specific axis (columns or rows) using numbered selection.

    Args:
        df: The input DataFrame.
        threshold: The threshold for missing values (percentage).
        axis: 0 for columns, 1 for rows.
        verbose: Whether to print output (default: True).

    Returns:
        The modified DataFrame, or None if the DataFrame becomes None after dropping.
    """
    missing_percentage = (df.isnull().sum(axis=axis) / df.shape[axis]) * 100
    exceeding_threshold = missing_percentage[missing_percentage > threshold]

    if exceeding_threshold.any():
        axis_label = "columns" if axis == 0 else "rows"
        item_label = "column" if axis == 0 else "row"
        index_names = exceeding_threshold.index.tolist()
        numbered_items = {i + 1: name for i, name in enumerate(index_names)}

        if verbose:
            print(f"{axis_label.capitalize()} exceeding {threshold}% missing values:")
            for number, name in numbered_items.items():
                print(f"{number}: {name} ({exceeding_threshold[name]:.2f}%)")

        while True:
            response = input(f"Enter the numbers of the {item_label}(s) you want to drop "
                             f"(comma-separated, 'all' to drop all, or 'none' to keep all): ").strip().lower()

            if response == 'all':
                items_to_drop = list(numbered_items.values())
                if axis == 0:
                    df = df.drop(columns=items_to_drop)
                else:
                    df = df.drop(index=items_to_drop)
                if verbose:
                    print(f"All specified {axis_label} dropped: {items_to_drop}.")
                break
            elif response == 'none':
                if verbose:
                    print(f"No {axis_label} dropped.")
                break
            else:
                selected_numbers_str = [x.strip() for x in response.split(',') if x.strip()]
                selected_numbers = [int(num) for num in selected_numbers_str if num.isdigit()]
                valid_numbers = [num for num in selected_numbers if num in numbered_items]
                items_to_drop = [numbered_items[num] for num in valid_numbers]

                if items_to_drop:
                    if axis == 0:
                        df = df.drop(columns=items_to_drop)
                    else:
                        df = df.drop(index=items_to_drop)
                    if verbose:
                        print(f"{axis_label.capitalize()} dropped: {items_to_drop}.")
                    break
                elif selected_numbers:
                    invalid_numbers = [num for num in selected_numbers if num not in numbered_items]
                    if verbose:
                        print(f"Invalid numbers entered: {invalid_numbers}. Please enter valid numbers or 'all'/'none'.")
                elif selected_numbers_str:
                    if verbose:
                        print("Invalid input. Please enter numbers, 'all', or 'none'.")
                else:
                    if verbose:
                        print("No valid input received.")

    return df if not df.empty else None  # Return DataFrame or None if empty

################################################################################################################################################
#Function to describe dataset and help to define numerical continuos features and categorical discrete features
#import pandas as pd
#import numpy as np
#from scipy.stats import zscore
################################################################################################################################################
def describe_dataframe_custom_out(df: pd.DataFrame, exclude_columns: list = None,
                                  zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5,
                                  categorical_threshold_unique: int = 5,
                                  max_unique_ratio_for_categorical: float = 0.1,
                                  force_categorical_columns: (list | dict) = None # <--- UPDATED TYPE HINT
                                 ) -> dict:
    """
    Generates a descriptive summary for a Pandas DataFrame, detailing each column,
    with the option to exclude specific columns and dividing the output into
    separate tables for numerical and categorical features, with user reclassification.
    Includes an option to bypass interactive classification for automatic default acceptance.

    For numerical columns, it provides:
    - Data Type
    - Minimum Value
    - Maximum Value
    - Mean
    - Median
    - Mode(s)
    - Z-Score Outliers (count of values beyond zscore_threshold)
    - IQR Outliers (count of values outside Q1 - (iqr_multiplier * IQR) and Q3 + (iqr_multiplier * IQR))

    For categorical columns, it provides:
    - Data Type
    - Number of Unique Values
    - Mode(s)
    - Missing Values
    - Missing %

    Args:
        df (pd.DataFrame): The input DataFrame to describe.
        exclude_columns (list, optional): A list of column names to exclude from the description.
                                         Defaults to None (no columns excluded).
        zscore_threshold (float, optional): The absolute Z-score threshold to consider a value an outlier.
                                            Defaults to 3.0. Only applies to numerical columns.
        iqr_multiplier (float, optional): The multiplier for the IQR to define the outlier bounds.
                                          Defaults to 1.5 (Tukey's fences). Only applies to numerical columns.
        categorical_threshold_unique (int, optional): For numerical columns, if the number of unique
                                                       values is less than or equal to this threshold,
                                                       it will be initially considered for categorical classification.
        max_unique_ratio_for_categorical (float, optional): For numerical columns considered for categorical
                                                             reclassification (based on `categorical_threshold_unique`),
                                                             if their unique value count is also less than
                                                             this ratio of total rows, they will be initially
                                                             classified as categorical. Binary (0/1) numeric columns are also reclassified.
        force_categorical_columns (list or dict, optional):
            - If a list: A list of column names to be treated as categorical without binning.
            - If a dict: Keys are column names to be forced as categorical, and values are either:
                - None: Treat as categorical without binning (e.g., specific integer codes).
                - A list of bin edges (e.g., [0, 18, 65, np.inf]): Numerical values will be
                  binned into categories based on these edges.
            Defaults to None.

    Returns:
        dict: A dictionary containing two Pandas DataFrames and two lists:
              - 'numerical_summary': Description of numerical columns.
              - 'categorical_summary': Description of categorical columns.
              - 'numerical_features': A list of column names classified as numerical.
              - 'categorical_features': A list of column names classified as categorical.
              Returns empty DataFrames/lists if no columns of a certain type are found.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if df.empty:
        print("The DataFrame is empty. No description can be generated.")
        return {'numerical_summary': pd.DataFrame(), 'categorical_summary': pd.DataFrame(),
                'numerical_features': [], 'categorical_features': []}

    # --- Refactoring starts here ---
    # Normalize force_categorical_columns to always be a dictionary
    if force_categorical_columns is None:
        _force_categorical_dict = {}
    elif isinstance(force_categorical_columns, list):
        _force_categorical_dict = {col: None for col in force_categorical_columns}
    elif isinstance(force_categorical_columns, dict):
        _force_categorical_dict = force_categorical_columns
    else:
        raise TypeError("`force_categorical_columns` must be a list or a dictionary.")
    # --- Refactoring ends here ---

    temp_column_descriptions = []

    if exclude_columns is None:
        exclude_columns = []

    # First pass: Collect initial descriptions and heuristic classifications
    for column in df.columns:
        if column in exclude_columns:
            continue

        col_data = df[column]
        col_type = str(col_data.dtype)
        num_missing = col_data.isnull().sum()
        total_rows = len(col_data)
        missing_percentage = (num_missing / total_rows * 100).round(2) if total_rows > 0 else 0

        column_description = {
            'Column Name': column,
            'Data Type': col_type,
            'Missing Values': num_missing,
            'Missing %': f"{missing_percentage}%",
            'Min Value': np.nan, 'Max Value': np.nan, 'Mean': np.nan, 'Median': np.nan,
            'Mode(s)': 'N/A', 'Z-Score Outliers': 'N/A', 'IQR Outliers': 'N/A', 'Unique Values': 'N/A',
            '_is_numeric_original': False,
            '_is_heuristic_categorical': False,
            '_reclassification_reason': None,
            '_forced_categorical': False,
            '_binning_applied': False
        }

        # Check if column is in force_categorical_columns (using the normalized dict)
        if column in _force_categorical_dict: # <--- USE NORMALIZED DICT HERE
            column_description['_forced_categorical'] = True
            bin_edges = _force_categorical_dict[column] # <--- USE NORMALIZED DICT HERE
            
            if bin_edges is not None:
                # Apply binning if bin edges are provided
                try:
                    # Make a copy to avoid modifying the original df
                    # Create the categorical bins based on the original data, then update col_data
                    original_col_data = df[column] # Get original data for binning
                    binned_data = pd.cut(original_col_data, bins=bin_edges, include_lowest=True, right=False)
                    col_data = binned_data # Update col_data for subsequent stats (like unique values, mode)
                    column_description['_binning_applied'] = True
                    column_description['_reclassification_reason'] = "forced categorical with binning"
                    # Update col_type to reflect the new categorical nature
                    column_description['Data Type'] = 'category (binned)'
                except Exception as e:
                    print(f"Warning: Could not apply binning to column '{column}'. Error: {e}")
                    # Revert to original data if binning fails, and proceed with default heuristic
                    column_description['_forced_categorical'] = False # Revert
                    column_description['_binning_applied'] = False # Revert
                    column_description['_reclassification_reason'] = None # Revert
                    # IMPORTANT: Reset col_data to its original form if binning failed
                    col_data = df[column]
            else:
                column_description['_reclassification_reason'] = "forced categorical by user"
                # If not binned, just mark as forced categorical, original dtype might remain for now
                # but it will be treated as categorical in final output.

        is_numeric_dtype_original = pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data)

        # If it was forced categorical and successfully binned, it's not "numeric_original" for the rest of the checks
        if column_description['_binning_applied']:
             is_numeric_dtype_original = False # This ensures binned columns are processed as categorical from here

        column_description['_is_numeric_original'] = is_numeric_dtype_original


        if is_numeric_dtype_original: # Original numeric dtype, not forced binned
            numeric_col_data = pd.to_numeric(col_data, errors='coerce').dropna()

            if not numeric_col_data.empty:
                column_description['Min Value'] = numeric_col_data.min()
                column_description['Max Value'] = numeric_col_data.max()
                column_description['Mean'] = numeric_col_data.mean()
                column_description['Median'] = numeric_col_data.median()

                modes = numeric_col_data.mode()
                if len(modes) == 1:
                    column_description['Mode(s)'] = modes[0]
                else:
                    column_description['Mode(s)'] = ', '.join(map(str, modes.tolist()))

                if numeric_col_data.nunique() > 1:
                    try:
                        z_scores = np.abs(zscore(numeric_col_data))
                        outliers_count_zscore = (z_scores > zscore_threshold).sum()
                        column_description['Z-Score Outliers'] = outliers_count_zscore
                    except Exception as e:
                        column_description['Z-Score Outliers'] = f"Error: {e}"
                else:
                    column_description['Z-Score Outliers'] = 0

                if numeric_col_data.nunique() > 1 and len(numeric_col_data) >= 2:
                    Q1 = numeric_col_data.quantile(0.25)
                    Q3 = numeric_col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - (iqr_multiplier * IQR)
                    upper_bound = Q3 + (iqr_multiplier * IQR)
                    outliers_count_iqr = ((numeric_col_data < lower_bound) | (numeric_col_data > upper_bound)).sum()
                    column_description['IQR Outliers'] = outliers_count_iqr
                else:
                    column_description['IQR Outliers'] = 0

            if total_rows > 0:
                unique_count = col_data.nunique()
                unique_ratio = unique_count / total_rows

                if (unique_count <= categorical_threshold_unique and unique_ratio < max_unique_ratio_for_categorical) or \
                   (unique_count == 2 and (col_data.min() == 0 or pd.isna(col_data.min())) and (col_data.max() == 1 or pd.isna(col_data.max()))):
                    column_description['_is_heuristic_categorical'] = True
                    if unique_count <= categorical_threshold_unique and unique_ratio < max_unique_ratio_for_categorical:
                        column_description['_reclassification_reason'] = f"low unique count ({unique_count}) and ratio ({unique_ratio:.2f})"
                    else:
                        column_description['_reclassification_reason'] = "binary (0/1) values"

            column_description['Unique Values'] = col_data.nunique()

        else: # Column is not numeric dtype originally (e.g., object, category, boolean, or forced binned)
            column_description['_is_heuristic_categorical'] = True # Always heuristic categorical if not numeric original
            column_description['Unique Values'] = col_data.nunique()
            modes = col_data.mode()
            if not modes.empty:
                if len(modes) == 1:
                    column_description['Mode(s)'] = modes[0]
                else:
                    column_description['Mode(s)'] = ', '.join(map(str, modes.tolist()))
            else:
                column_description['Mode(s)'] = 'N/A (No mode found)'

        temp_column_descriptions.append(column_description)

    final_numerical_description_list = []
    final_categorical_description_list = []

    # User input to bypass interactive reclassification
    should_prompt_reclassification = False
    reclassification_candidates_exist = any(desc['_is_numeric_original'] and desc['_is_heuristic_categorical'] and not desc['_forced_categorical'] for desc in temp_column_descriptions)

    if reclassification_candidates_exist:
        while True:
            bypass_input = input("\nDo you want to accept the default (heuristic) classification for all columns, "
                                 "or interactively define types for each suggested reclassification? (y/n): ").strip().lower()
            if bypass_input == 'y':
                should_prompt_reclassification = False
                print("Accepting default classifications. No interactive reclassification will occur.")
                break
            elif bypass_input == 'n':
                should_prompt_reclassification = True
                print("Proceeding with interactive reclassification for suggested columns.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        print("\nNo columns were initially classified as numerical but heuristically as categorical (and not forced). Skipping reclassification prompt.")


    # Second pass: Apply reclassification based on user input or heuristic, prioritizing 'force_categorical_columns'
    for desc in temp_column_descriptions:
        column = desc['Column Name']
        is_numeric_original = desc['_is_numeric_original']
        is_heuristic_categorical = desc['_is_heuristic_categorical']
        reclassification_reason = desc['_reclassification_reason']
        is_forced_categorical = desc['_forced_categorical']

        is_final_categorical = is_heuristic_categorical # Default to heuristic classification

        if is_forced_categorical:
            is_final_categorical = True # Force it to be categorical if the user specified
        elif is_numeric_original and is_heuristic_categorical and should_prompt_reclassification:
            # Only prompt if bypass was 'no' and column is a reclassification candidate AND NOT forced
            while True:
                prompt_msg = (f"\nColumn '{column}' is numerically typed ('{desc['Data Type']}'), "
                              f"but was reclassified as categorical due to {reclassification_reason}. "
                              f"Do you want to treat '{column}' as categorical (y) or numerical (n)? (y/n): ")
                response = input(prompt_msg).strip().lower()
                if response == 'y':
                    is_final_categorical = True
                    break
                elif response == 'n':
                    is_final_categorical = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

        final_desc = {k: v for k, v in desc.items() if not k.startswith('_')} # Clean up internal flags

        if is_final_categorical:
            # Clear numerical-specific fields for categorical columns
            final_desc['Min Value'] = np.nan
            final_desc['Max Value'] = np.nan
            final_desc['Mean'] = np.nan
            final_desc['Median'] = np.nan
            final_desc['Z-Score Outliers'] = np.nan
            final_desc['IQR Outliers'] = np.nan

            # Ensure Unique Values and Mode(s) are correctly populated for categorical
            # This part is crucial if the column was originally numeric and now made categorical
            # Use the actual column data from the original df for accurate counts/modes
            # unless it was binned, in which case col_data would have been updated
            if column_description['_binning_applied']:
                 col_for_stats = col_data # Use the binned data
            else:
                 col_for_stats = df[column] # Use original if not binned

            if final_desc['Unique Values'] == 'N/A' or pd.isna(final_desc['Unique Values']):
                final_desc['Unique Values'] = col_for_stats.nunique()

            if final_desc['Mode(s)'] == 'N/A' or final_desc['Mode(s)'] == 'N/A (No mode found)':
                modes = col_for_stats.mode()
                if not modes.empty:
                    if len(modes) == 1:
                        final_desc['Mode(s)'] = modes[0]
                    else:
                        final_desc['Mode(s)'] = ', '.join(map(str, modes.tolist()))
                else:
                    final_desc['Mode(s)'] = 'N/A (No mode found)'

            final_categorical_description_list.append(final_desc)
        else: # is_final_numerical
            # Clear categorical-specific fields for numerical columns
            final_desc['Unique Values'] = np.nan

            final_numerical_description_list.append(final_desc)

    numerical_output_columns = [
        'Column Name', 'Data Type', 'Missing Values', 'Missing %',
        'Min Value', 'Max Value', 'Mean', 'Median', 'Mode(s)',
        'Z-Score Outliers', 'IQR Outliers'
    ]
    categorical_output_columns = [
        'Column Name', 'Data Type', 'Missing Values', 'Missing %',
        'Unique Values', 'Mode(s)'
    ]

    numerical_description_df = pd.DataFrame(final_numerical_description_list)
    categorical_description_df = pd.DataFrame(final_categorical_description_list)

    numerical_description_df = numerical_description_df.reindex(columns=numerical_output_columns, fill_value=np.nan)
    categorical_description_df = categorical_description_df.reindex(columns=categorical_output_columns, fill_value=np.nan)

    # Extract column names into lists for convenience
    numerical_features = numerical_description_df['Column Name'].tolist()
    categorical_features = categorical_description_df['Column Name'].tolist()

    return {
        'numerical_summary': numerical_description_df,
        'categorical_summary': categorical_description_df,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }

################################################################################################################################################
#Function to compare dataframes
#import pandas as pd
################################################################################################################################################
def compare_dataframes_optimized(df1, df2, primary_key_cols=None):
    """
    Compares two pandas DataFrames and returns a DataFrame highlighting the differences
    using an optimized iterative approach. Includes primary key columns in the result
    for easy row identification.

    Args:
        df1 (pd.DataFrame): The first DataFrame for comparison.
        df2 (pd.DataFrame): The second DataFrame for comparison.
        primary_key_cols (list, optional): A list of column names that uniquely identify
                                           each row. These columns will be included
                                           in the result DataFrame to help identify
                                           the rows where differences were found.
                                           Defaults to None (no primary keys specified).

    Returns:
        pd.DataFrame: A DataFrame containing only the rows and columns where
                      values differ between df1 and df2. The differing values
                      from both DataFrames are shown side-by-side.
                      Primary key columns are prepended to the result for context.
                      Returns None if the DataFrames have different shapes
                      or column names, or if primary key columns are missing.
    """
    if df1.shape != df2.shape:
        print("DataFrames have different shapes. Cannot compare.")
        return None
    if not df1.columns.equals(df2.columns):
        print("DataFrames have different column names. Cannot compare.")
        return None

    if primary_key_cols is None:
        primary_key_cols = []
    
    # Validate primary key columns
    if not all(col in df1.columns for col in primary_key_cols):
        missing_pks = [col for col in primary_key_cols if col not in df1.columns]
        print(f"Error: One or more primary key columns not found in DataFrames: {missing_pks}")
        return None

    # Create temporary DataFrames with a default integer index for comparison.
    # We will use these temporary indices to track differences and later retrieve
    # the corresponding primary key values.
    df1_temp = df1.reset_index(drop=True)
    df2_temp = df2.reset_index(drop=True)

    # Dictionary to store the differing values, keyed by row index (from temp DFs) and column name
    differences_dict = {}
    
    # Track the actual row indices (from the temp DFs) where differences occurred
    differing_row_indices = set()

    # Iterate through each column to find differences
    for col in df1_temp.columns:
        # Compare element-wise, handling NaN/None differences appropriately.
        # (A != B) handles numerical NaNs (NaN != NaN is True).
        # (A.isna() != B.isna()) handles cases where one is NaN/None and the other is not.
        col_diff_mask = (df1_temp[col] != df2_temp[col]) | (df1_temp[col].isna() != df2_temp[col].isna())
        
        # Get the indices where the column differs
        diff_indices = df1_temp[col_diff_mask].index

        # If there are differences in this column
        if not diff_indices.empty:
            for idx in diff_indices:
                differing_row_indices.add(idx) # Keep track of unique rows with differences
                # Store the differing values
                if idx not in differences_dict:
                    differences_dict[idx] = {}
                differences_dict[idx][f'{col}_df1'] = df1_temp.loc[idx, col]
                differences_dict[idx][f'{col}_df2'] = df2_temp.loc[idx, col]

    if not differences_dict:
        print("DataFrames are identical.")
        return pd.DataFrame() # Return an empty DataFrame

    # Convert the differences dictionary to a DataFrame
    # The index of diff_df will be the row indices from df1_temp/df2_temp where differences were found
    diff_df = pd.DataFrame.from_dict(differences_dict, orient='index')

    # Add primary key columns to the result DataFrame if specified
    if primary_key_cols:
        # Retrieve the primary key values for the rows that have differences
        # Use the `differing_row_indices` to select these rows from `df1_temp`
        pk_values_for_diff_rows = df1_temp.loc[list(differing_row_indices), primary_key_cols]
        
        # Ensure the index of pk_values_for_diff_rows matches diff_df's index
        pk_values_for_diff_rows = pk_values_for_diff_rows.set_axis(list(differing_row_indices))

        # Join the PK values with the differences DataFrame.
        # Since both DataFrames have the same index (the differing row indices), `join` works well.
        diff_df = pk_values_for_diff_rows.join(diff_df, how='left')

        # Ensure primary key columns are at the very beginning of the result DataFrame
        all_diff_cols = list(diff_df.columns)
        reordered_columns_with_pk = primary_key_cols + [col for col in all_diff_cols if col not in primary_key_cols]
        diff_df = diff_df[reordered_columns_with_pk]

    # Reorder the remaining columns to ensure df1 and df2 values for changed columns are side-by-side.
    # This also ensures the original column order is maintained for the differing value pairs.
    final_display_columns = []
    
    # First, add the primary key columns if they were included
    if primary_key_cols:
        final_display_columns.extend(primary_key_cols)

    # Then, iterate through the original DataFrame's columns to add the _df1/_df2 pairs
    for col in df1.columns:
        # Add the _df1/_df2 pair only if they exist in the diff_df and are not PK columns (already added)
        if f'{col}_df1' in diff_df.columns and col not in primary_key_cols:
            final_display_columns.append(f'{col}_df1')
            final_display_columns.append(f'{col}_df2')
    
    # Filter final_display_columns to only include columns that actually exist in diff_df.
    # This handles cases where some expected _df1/_df2 pairs might not be generated if there were no differences for them.
    final_display_columns_filtered = [col for col in final_display_columns if col in diff_df.columns]

    return diff_df[final_display_columns_filtered]

################################################################################################################################################
# Function to reduce dimensiinality
#import pandas as pd
################################################################################################################################################

def reduce_dimensionality(
    df: pd.DataFrame,
    numerical_features_to_keep: list = None,
    categorical_features_to_keep: list = None,
    primary_key_cols: list = None,
    other_columns_to_keep: list = None
) -> pd.DataFrame:
    """
    Creates a new DataFrame containing only the specified primary key columns,
    numerical features, categorical features, and any other explicitly listed columns.
    All columns not in these lists will be excluded.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_features_to_keep (list, optional): A list of numerical column names to retain. Defaults to None.
        categorical_features_to_keep (list, optional): A list of categorical column names to retain. Defaults to None.
        primary_key_cols (list, optional): A list of primary key column names to retain. Defaults to None.
        other_columns_to_keep (list, optional): A list of any other additional column names to retain. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with only the specified columns.
                      Returns an empty DataFrame if no valid columns are specified or found.
    """
    # Initialize an empty list to collect all desired columns
    all_columns_to_keep_raw = []

    # Add primary key columns if provided
    if primary_key_cols:
        all_columns_to_keep_raw.extend(primary_key_cols)
    
    # Add numerical features if provided
    if numerical_features_to_keep:
        all_columns_to_keep_raw.extend(numerical_features_to_keep)
        
    # Add categorical features if provided
    if categorical_features_to_keep:
        all_columns_to_keep_raw.extend(categorical_features_to_keep)

    # Add other specified columns if provided
    if other_columns_to_keep:
        all_columns_to_keep_raw.extend(other_columns_to_keep)

    # Filter out any columns from the combined list that do not exist in the original DataFrame
    # This also acts as an implicit way to handle None or empty lists passed as args.
    existing_columns_to_keep = [col for col in all_columns_to_keep_raw if col in df.columns]
    
    # Remove duplicates while preserving order (e.g., if a column is listed in multiple 'to_keep' lists)
    # Using dict.fromkeys to maintain order and uniqueness
    final_columns_to_keep = list(dict.fromkeys(existing_columns_to_keep))

    if not final_columns_to_keep:
        print("Warning: No valid columns to keep were found in the DataFrame based on your input lists. Returning an empty DataFrame.")
        return pd.DataFrame()

    print(f"Reducing dimensionality. Keeping {len(final_columns_to_keep)} columns from the original DataFrame ({df.shape[1]} columns):")
    print(f"  Columns to be included: {', '.join(final_columns_to_keep)}")

    return df[final_columns_to_keep].copy()

################################################################################################################################################
#Function to identify primary key
#import pandas as pd
#import itertools
################################################################################################################################################

def get_primary_key_columns(df, exclude_columns=[], max_key_size=3):
    """Attempts to identify and print primary key column names of a DataFrame.

    This function makes some assumptions to identify potential primary key
    candidates:
    - Columns with unique values for each row are considered.
    - If multiple single-column candidates are found, it prompts the user to choose.
    - If no clear candidate is found, it prints a message.
    - You can exclude columns from consideration using `exclude_columns`.
    - Instead of `max_key_size`, it suggests composite key sizes and prompts
      the user for confirmation or to create an index.
    - Includes an 'exit' option at prompts to stop the function.

    Args:
        df: The input Pandas DataFrame.
        exclude_columns: A list of column names to exclude from primary key detection.
        max_key_size: The maximum number of columns to consider for a composite key
                      (used as a suggestion for search depth).

    Returns:
        tuple: A tuple containing:
               - The modified Pandas DataFrame (if an index column was added, otherwise the original df).
               - A list of the primary key column names (or the name of the new index column) if found and accepted,
                 otherwise None. If the user chooses to 'exit', returns (original_df, None).
    """

    eligible_cols = [col for col in df.columns if col not in exclude_columns]
    current_df = df.copy() # Work on a copy of the DataFrame

    # Helper function to handle user input for choices
    def get_user_choice(prompt_message):
        while True:
            user_input = input(prompt_message).lower()
            if user_input == 'exit':
                print("Operation cancelled by user.")
                return 'exit'
            return user_input

    # 1. Check for single-column primary key
    unique_value_cols = [col for col in eligible_cols
                         if current_df[col].nunique() == current_df.shape[0]]

    if unique_value_cols:
        if len(unique_value_cols) == 1:
            chosen_pk = unique_value_cols[0]
            print(f"Potential primary key column: '{chosen_pk}'")
            user_choice = get_user_choice("Do you accept this as the primary key? (yes/no/create_index/exit): ")

            if user_choice == 'yes':
                return current_df, [chosen_pk]
            elif user_choice == 'create_index':
                current_df = current_df.reset_index(drop=False)
                new_index_col_name = 'index_column'
                if new_index_col_name in current_df.columns:
                    i = 1
                    while f'index_column_{i}' in current_df.columns:
                        i += 1
                    new_index_col_name = f'index_column_{i}'
                current_df = current_df.rename(columns={'index': new_index_col_name})
                print(f"Created a new index column: '{new_index_col_name}'")
                return current_df, [new_index_col_name]
            elif user_choice == 'exit':
                return df, None # Return original df and None
            else:
                print("Searching for composite keys or new index creation.")
        else: # Multiple single-column PK candidates
            print("Multiple potential single primary key columns found:")
            for i, col in enumerate(unique_value_cols):
                print(f"{i + 1}. '{col}'")
            print("Please choose one by number, or type 'no' to search for composite keys, or 'create_index' to add a new index column, or 'exit' to stop.")

            while True:
                user_input = get_user_choice("Your choice: ")
                if user_input == 'exit':
                    return df, None # Return original df and None

                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(unique_value_cols):
                        chosen_pk = unique_value_cols[idx]
                        print(f"You selected '{chosen_pk}' as the primary key.")
                        return current_df, [chosen_pk]
                    else:
                        print("Invalid number. Please try again.")
                elif user_input == 'no':
                    print("Proceeding to search for composite keys.")
                    break # Exit loop to continue to composite key search
                elif user_input == 'create_index':
                    current_df = current_df.reset_index(drop=False)
                    new_index_col_name = 'index_column'
                    if new_index_col_name in current_df.columns:
                        i = 1
                        while f'index_column_{i}' in current_df.columns:
                            i += 1
                        new_index_col_name = f'index_column_{i}'
                    current_df = current_df.rename(columns={'index': new_index_col_name})
                    print(f"Created a new index column: '{new_index_col_name}'")
                    return current_df, [new_index_col_name]
                else:
                    print("Invalid input. Please enter a number, 'no', 'create_index', or 'exit'.")


    # 2. Check for composite primary key
    for key_size in range(2, max_key_size + 1):
        for key_cols in itertools.combinations(eligible_cols, key_size):
            if current_df.groupby(list(key_cols)).size().max() == 1:
                print(f"Potential composite primary key column(s) (size {key_size}):", list(key_cols))
                user_choice = get_user_choice("Do you accept this as the primary key? (yes/no/create_index/exit): ")
                if user_choice == 'yes':
                    return current_df, list(key_cols)
                elif user_choice == 'create_index':
                    current_df = current_df.reset_index(drop=False)
                    new_index_col_name = 'index_column'
                    if new_index_col_name in current_df.columns:
                        i = 1
                        while f'index_column_{i}' in current_df.columns:
                            i += 1
                        new_index_col_name = f'index_col_name_{i}' # Corrected variable name
                    current_df = current_df.rename(columns={'index': new_index_col_name})
                    print(f"Created a new index column: '{new_index_col_name}'")
                    return current_df, [new_index_col_name]
                elif user_choice == 'exit':
                    return df, None # Return original df and None
                else:
                    print("Continuing search or new index creation.")
        if key_size < max_key_size:
            cont_choice = get_user_choice(f"No composite key of size {key_size} found. Do you want to check for larger composite keys (up to {max_key_size})? (yes/no/exit): ")
            if cont_choice == 'no':
                break
            elif cont_choice == 'exit':
                return df, None # Return original df and None

    print("Could not determine a primary key based on unique values.")
    user_final_choice = get_user_choice("Do you want to create a new index column? (yes/no/exit): ")
    if user_final_choice == 'yes':
        current_df = current_df.reset_index(drop=False)
        new_index_col_name = 'index_column'
        if new_index_col_name in current_df.columns:
            i = 1
            while f'index_column_{i}' in current_df.columns:
                i += 1
            new_index_col_name = f'index_column_{i}'
        current_df = current_df.rename(columns={'index': new_index_col_name})
        print(f"Created a new index column: '{new_index_col_name}'")
        return current_df, [new_index_col_name]
    elif user_final_choice == 'exit':
        return df, None # Return original df and None
    else:
        return current_df, None # Return the original df and None for PK if not created    
################################################################################################################################################
# Function to check if there are duplicates on a given primary key
#import pandas as pd
################################################################################################################################################
def check_primary_key_duplicates(df, primary_key_cols):
    """Checks for duplicate rows based on primary key columns in a DataFrame.

    Args:
        df: The input pandas DataFrame.
        primary_key_cols: A list of column names representing the primary key.

    Returns:
        bool: True if duplicates are found, False otherwise.
    """

    # Check for duplicates in the primary key columns
    duplicates = df.duplicated(subset=primary_key_cols, keep=False)

    if duplicates.any():
        print("Duplicate rows found based on primary key:")
        print(df[duplicates])  # Print the duplicate rows
        return True
    else:
        print("No duplicate rows found based on primary key.")
        return False    
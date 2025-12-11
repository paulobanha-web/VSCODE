################################################################################################################################################
'''Python Libraries'''
################################################################################################################################################
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import display # For interactive use in Jupyter environments
import time # For small delays in interactive displays
################################################################################################################################################

def get_numerical_outliers(data, column_to_analyze, method, threshold, primary_key_columns, all_numerical_features, group_by_pks_for_detection=True, verbose_notes=True):
    """
    Detects numerical outliers using specified method. Can perform global or grouped detection.
    Returns outlier indices and scores (for LOF/KNN/IsolationForest) mapped back to the original DataFrame's index.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        column_to_analyze (str): The specific numerical column for which outliers are being identified and visualized.
        method (str): Outlier detection method ('zscore', 'iqr', 'lof', 'knn', 'isolationforest').
        threshold (float): The threshold value for outlier detection.
        primary_key_columns (list): List of primary key column names to group data by for detection (if group_by_pks_for_detection is True).
        all_numerical_features (list): List of ALL numerical feature columns to be used for multivariate methods.
        group_by_pks_for_detection (bool): If True, detection is grouped by primary keys; otherwise, it's global.
        verbose_notes (bool): If True, prints notes about skipped groups due to insufficient data.
    """
    all_outlier_indices = pd.Index([])
    full_lof_scores = pd.Series(np.nan, index=data.index, dtype=float)
    full_knn_distances = pd.Series(np.nan, index=data.index, dtype=float)
    full_if_anomaly_scores = pd.Series(np.nan, index=data.index, dtype=float)

    pk_cols_list = primary_key_columns if isinstance(primary_key_columns, list) else [primary_key_columns]
    valid_pk_cols = [col for col in pk_cols_list if col in data.columns]
    
    # Determine the iterator for groups
    if not group_by_pks_for_detection or not valid_pk_cols:
        # Global detection: treat the entire data as one group
        if verbose_notes and group_by_pks_for_detection and not valid_pk_cols:
            print("Warning: No valid primary key columns provided for grouped detection. Performing global outlier detection.")
        elif verbose_notes and not group_by_pks_for_detection:
            print("Note: Performing GLOBAL multivariate outlier detection. Primary keys are NOT used for grouping during detection.")
        groups_iterator = [(None, data.copy())] # Pass a copy to avoid SettingWithCopyWarning
    else:
        # Grouped detection: iterate through primary key groups
        groups_iterator = data.groupby(valid_pk_cols)

    for group_keys, group_df in groups_iterator:
        current_outlier_indices = pd.Index([])
        scores_in_temp = None

        if method in ['zscore', 'iqr']:
            # Univariate methods always use only the 'column_to_analyze'
            temp_data_for_method = group_df[column_to_analyze].dropna()
            original_indices_in_group = temp_data_for_method.index
            X = temp_data_for_method.values.reshape(-1, 1)

            if temp_data_for_method.empty:
                continue

            if method == 'zscore':
                mean_val = temp_data_for_method.mean()
                std_val = temp_data_for_method.std()
                if std_val == 0:
                    current_outlier_indices = pd.Index([])
                else:
                    z = np.abs((temp_data_for_method - mean_val) / std_val)
                    current_outlier_indices = original_indices_in_group[z > threshold]
            elif method == 'iqr':
                Q1 = temp_data_for_method.quantile(0.25)
                Q3 = temp_data_for_method.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    current_outlier_indices = pd.Index([])
                else:
                    current_outlier_indices = original_indices_in_group[(temp_data_for_method < (Q1 - 1.5 * IQR)) | (temp_data_for_method > (Q3 + 1.5 * IQR))]

        else: # Multivariate methods: LOF, KNN, IsolationForest
            # These methods use 'all_numerical_features' provided
            features_for_detection = [f for f in all_numerical_features if f in group_df.columns]
            
            if not features_for_detection:
                if verbose_notes:
                    print(f"Warning: No valid numerical features available for detection in group '{group_keys}'. Skipping.")
                continue

            temp_data_for_method = group_df[features_for_detection].dropna()
            original_indices_in_group = temp_data_for_method.index
            X = temp_data_for_method.values

            # Check for sufficient samples for sklearn models
            if len(original_indices_in_group) < 2:
                if verbose_notes and group_by_pks_for_detection: # Only print notes if we're in grouped mode
                    print(f"Note: Group '{group_keys}' has only {len(original_indices_in_group)} non-NaN data points across numerical features. Skipping {method.upper()} calculation for this group due to insufficient data.")
                continue
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X)

            if method == 'lof':
                n_neighbors_adjusted = min(5, len(scaled_data) - 1)
                n_neighbors_adjusted = max(1, n_neighbors_adjusted) 
                lof = LocalOutlierFactor(n_neighbors=n_neighbors_adjusted, contamination='auto')
                lof.fit(scaled_data)
                scores_in_temp = -lof.negative_outlier_factor_
                current_outlier_indices = original_indices_in_group[scores_in_temp > threshold]
                full_lof_scores.loc[original_indices_in_group] = scores_in_temp
            
            elif method == 'knn':
                n_neighbors_adjusted = min(5, len(scaled_data))
                n_neighbors_adjusted = max(1, n_neighbors_adjusted) 
                knn = NearestNeighbors(n_neighbors=n_neighbors_adjusted)
                knn.fit(scaled_data)
                distances, _ = knn.kneighbors(scaled_data)
                
                if distances.shape[1] > 0: 
                    scores_in_temp = distances[:, n_neighbors_adjusted - 1]
                else:
                    scores_in_temp = np.array([]) 
                    
                current_outlier_indices = original_indices_in_group[scores_in_temp > threshold]
                full_knn_distances.loc[original_indices_in_group] = scores_in_temp

            elif method == 'isolationforest':
                if_model = IsolationForest(random_state=42, contamination='auto') 
                if_model.fit(scaled_data)
                
                scores_in_temp = -if_model.score_samples(scaled_data)
                
                current_outlier_indices = original_indices_in_group[scores_in_temp > threshold]
                full_if_anomaly_scores.loc[original_indices_in_group] = scores_in_temp

        all_outlier_indices = all_outlier_indices.union(current_outlier_indices)

    return all_outlier_indices, full_lof_scores, full_knn_distances, full_if_anomaly_scores

def get_categorical_outliers(data, column, threshold, primary_key_columns):
    """
    Detects categorical outliers based on frequency, optionally grouping by primary keys.
    """
    all_outlier_indices = pd.Index([])
    
    pk_cols_list = primary_key_columns if isinstance(primary_key_columns, list) else [primary_key_columns]
    valid_pk_cols = [col for col in pk_cols_list if col in data.columns]

    if not valid_pk_cols:
        print("Warning: No valid primary key columns provided or found. Performing global categorical outlier detection.")
        groups_iterator = [(None, data)]
    else:
        groups_iterator = data.groupby(valid_pk_cols)

    for group_keys, group_df in groups_iterator:
        clean_column_data = group_df[column].dropna()
        if clean_column_data.empty:
            continue

        freq_table = clean_column_data.value_counts(normalize=True)
        outlier_categories_in_group = freq_table[freq_table < threshold].index
        
        current_outlier_indices = group_df[group_df[column].isin(outlier_categories_in_group) & group_df[column].notna()].index
        all_outlier_indices = all_outlier_indices.union(current_outlier_indices)
        
    return all_outlier_indices

def visualize_numerical_outliers(data, column, outliers_indices, primary_key_col, plot_type, threshold, lof_scores=None, knn_distances=None, if_anomaly_scores=None):
    """
    Visualizes numerical outliers.
    primary_key_col is expected to be a single primary key column name for visualization (e.g., 'part_id').
    """
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'scatter':
        is_outlier = data.index.isin(outliers_indices)
        outliers_data = data[is_outlier]
        non_outliers_data = data[~is_outlier]

        use_scores_for_coloring = False
        color_data = None
        color_label = ''
        title_suffix = ''

        if if_anomaly_scores is not None and not if_anomaly_scores.isnull().all():
            use_scores_for_coloring = True
            color_data = if_anomaly_scores
            color_label = 'Isolation Forest Anomaly Score'
            title_suffix = '(Colored by IF Score)'
        elif lof_scores is not None and not lof_scores.isnull().all():
            use_scores_for_coloring = True
            color_data = lof_scores
            color_label = 'LOF Outlier Factor'
            title_suffix = '(Colored by LOF Score)'
        elif knn_distances is not None and not knn_distances.isnull().all():
            use_scores_for_coloring = True
            color_data = knn_distances
            color_label = 'KNN Distance'
            title_suffix = '(Colored by KNN Distance)'

        if use_scores_for_coloring and color_data is not None:
            scatter = plt.scatter(non_outliers_data.index, non_outliers_data[column],
                                  c=color_data.loc[non_outliers_data.index], cmap='viridis', alpha=0.7, zorder=1)
            plt.colorbar(scatter, label=color_label)
            plt.title(f'Scatter Chart of {column} {title_suffix}')
        elif primary_key_col and primary_key_col in data.columns:
            sns.scatterplot(x=non_outliers_data.index, y=non_outliers_data[column],
                            data=non_outliers_data, hue=data.loc[non_outliers_data.index, primary_key_col].astype(str), zorder=1)
            plt.legend(title=primary_key_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Scatter Chart of {column} (Colored by {primary_key_col})')
        else:
            sns.scatterplot(x=non_outliers_data.index, y=non_outliers_data[column], data=non_outliers_data, color='skyblue', zorder=1)
            plt.title(f'Scatter Chart of {column}')
            plt.legend([],[], frameon=False)

        if not outliers_indices.empty:
            sns.scatterplot(x=outliers_data.index, y=outliers_data[column], color='red', s=70, label='Outliers', zorder=5)

            handles, labels = plt.gca().get_legend_handles_labels()
            if 'Outliers' not in labels:
                outlier_handle = plt.Line2D([0], [0], marker='o', color='red', linestyle='', markersize=8, label='Outliers')
                handles.append(outlier_handle)
                labels.append('Outliers')
                plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xlabel('Data Index')
        plt.ylabel(column)
        plt.xticks(rotation=0)

    elif plot_type == 'boxplot':
        if primary_key_col and primary_key_col in data.columns:
            sns.boxplot(y=column, x=data[primary_key_col].astype(str), data=data)
            plt.title(f'Box Plot of {column} (Grouped by {primary_key_col})')
            plt.xlabel(primary_key_col)
            plt.xticks(rotation=45, ha='right')
        else:
            sns.boxplot(y=column, data=data)
            plt.title(f'Box Plot of {column} (Overall)')
            plt.xlabel('')
            plt.xticks(rotation=0)
        plt.ylabel(column)

    elif plot_type == 'histogram' and (lof_scores is not None or knn_distances is not None or if_anomaly_scores is not None):
        scores = None
        method_name = ''
        if if_anomaly_scores is not None and not if_anomaly_scores.isnull().all():
            scores = if_anomaly_scores.dropna()
            method_name = 'Isolation Forest Anomaly Score'
        elif lof_scores is not None and not lof_scores.isnull().all():
            scores = lof_scores.dropna()
            method_name = 'LOF Score'
        elif knn_distances is not None and not knn_distances.isnull().all():
            scores = knn_distances.dropna()
            method_name = 'KNN Distance'

        if scores is not None and not scores.empty:
            plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
            plt.xlabel(method_name)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {method_name} for {column}')
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Outlier Threshold ({threshold})')
            plt.legend()
        else:
            print(f"No non-NaN {method_name} values to plot for histogram.")
    
    plt.tight_layout()
    plt.show()


def visualize_categorical_outliers(data, column, outliers_indices, primary_key_col):
    """
    Visualizes categorical outliers using a horizontal bar chart.
    primary_key_col is expected to be a single primary key column name.
    """
    plt.figure(figsize=(10, 8))
    value_counts = data[column].value_counts().sort_values(ascending=False)
    
    outlier_categories_in_data = data.loc[outliers_indices, column].dropna().unique()
    is_outlier_category_mask = value_counts.index.isin(outlier_categories_in_data)
    
    colors = ['skyblue' if not is_outlier else 'red' for is_outlier in is_outlier_category_mask]
    
    sns.barplot(x=value_counts.values, y=value_counts.index, hue=value_counts.index, palette=colors, legend=False)
    plt.xlabel('Frequency')
    plt.ylabel(column)
    plt.title(f'Frequency of Categories in {column} (Outliers in Red)')
    plt.tight_layout()
    plt.show()

def analyze_outliers(data, primary_key_columns, numerical_features, categorical_features, verbose_notes=True):
    """
    Analyzes and visualizes outliers in a given DataFrame column.
    Allows user to choose between univariate or multivariate analysis and then select method.
    Outlier detection can be performed within primary key groups or globally for multivariate.

    Args:
        data (pd.DataFrame): The input DataFrame.
        primary_key_columns (list): List of primary key column names.
        numerical_features (list): List of all numerical feature column names to consider for analysis.
        categorical_features (list): List of all categorical feature column names to consider for analysis.
        verbose_notes (bool): If True, prints notes about skipped groups due to insufficient data for multivariate methods.
    """
    if not isinstance(primary_key_columns, list):
        primary_key_columns = [primary_key_columns]

    all_analysable_features = []
    if numerical_features:
        all_analysable_features.extend(numerical_features)
    if categorical_features:
        all_analysable_features.extend(categorical_features)

    final_analysable_cols = [col for col in all_analysable_features 
                              if col not in primary_key_columns and col in data.columns]

    if not final_analysable_cols:
        print("No columns available for outlier analysis after excluding primary keys or specified features not found.")
        return

    while True:
        print("\nAvailable columns for analysis:")
        for i, col in enumerate(final_analysable_cols):
            print(f"{i + 1}. {col}")
        try:
            col_index = int(input("Enter the number of the column to analyze: ")) - 1
            if 0 <= col_index < len(final_analysable_cols):
                column_to_analyze = final_analysable_cols[col_index]
                break
            else:
                print("Invalid column number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    plot_pk_col = primary_key_columns[0] if primary_key_columns and primary_key_columns[0] in data.columns else None

    if column_to_analyze in numerical_features:
        group_by_pks_for_detection = True # Default for univariate and initial for multivariate
        
        while True:
            analysis_type_choice = input("Do you want to perform (1) Univariate or (2) Multivariate outlier analysis? Enter choice (1/2): ").strip()
            if analysis_type_choice == '1':
                print("You chose Univariate analysis (Z-score, IQR).")
                valid_methods = ['zscore', 'iqr']
                method_prompt = f"Enter outlier detection method for {column_to_analyze} (zscore, iqr): "
                break
            elif analysis_type_choice == '2':
                print("You chose Multivariate analysis (using LOF, KNN, or Isolation Forest algorithms).")
                
                # Ask about grouped vs. global for multivariate only
                while True:
                    scope_choice = input("Do you want (1) Grouped multivariate detection or (2) Global multivariate detection? Enter choice (1/2): ").strip()
                    if scope_choice == '1':
                        group_by_pks_for_detection = True
                        print("You chose Grouped multivariate detection.")
                        break
                    elif scope_choice == '2':
                        group_by_pks_for_detection = False
                        print("You chose Global multivariate detection.")
                        break
                    else:
                        print("Invalid choice. Please enter '1' or '2'.")

                valid_methods = ['lof', 'knn', 'isolationforest']
                method_prompt = f"Enter outlier detection method for {column_to_analyze} (lof, knn, isolationforest): "
                break
            else:
                print("Invalid choice. Please enter '1' or '2'.")

        while True:
            method = input(method_prompt).lower()
            if method in valid_methods:
                if method in ['lof', 'knn', 'isolationforest'] and len(numerical_features) == 1 and group_by_pks_for_detection == False:
                    print(f"Warning: '{method}' is typically for multivariate analysis. You chose Global detection but only provided one numerical feature '{numerical_features[0]}'. It will run as if on a single global feature.")
                elif method in ['lof', 'knn', 'isolationforest'] and len(numerical_features) == 1 and group_by_pks_for_detection == True:
                     print(f"Warning: '{method}' is typically for multivariate analysis. You chose Grouped detection but only provided one numerical feature '{numerical_features[0]}'. It will run as if on a single feature within groups.")
                break
            else:
                print(f"Invalid method for selected analysis type. Please choose from {', '.join(valid_methods)}.")
        
        default_threshold = 0
        if method == 'zscore':
            default_threshold = 3.0
        elif method == 'iqr':
            default_threshold = 1.5
        elif method in ['lof', 'knn']:
            default_threshold = 1.5
        elif method == 'isolationforest':
            default_threshold = 1.5
            print(f"Note: For Isolation Forest, a higher score (after negation) indicates a stronger anomaly. Default threshold {default_threshold} means points with a negated score above {default_threshold} are outliers.")

        try:
            threshold_input = input(f"Enter threshold value for {column_to_analyze} (default: {default_threshold}): ").strip()
            threshold = float(threshold_input) if threshold_input else default_threshold
        except ValueError:
            print(f"Invalid threshold value. Using default: {default_threshold}.")
            threshold = default_threshold

        while True:
            plot_type_options = ['scatter', 'boxplot']
            if method in ['lof', 'knn', 'isolationforest']:
                plot_type_options.append('histogram')
            plot_type_prompt = f"Enter plot type ({', '.join(plot_type_options)}): "
            
            plot_type = input(plot_type_prompt).lower()
            if plot_type in plot_type_options:
                break
            else:
                print("Invalid plot type.")

        # Pass all numerical features and the group_by_pks_for_detection flag
        outliers_indices, lof_scores, knn_distances, if_anomaly_scores = get_numerical_outliers(
            data, column_to_analyze, method, threshold, primary_key_columns, 
            numerical_features, # Pass all numerical features for multivariate methods
            group_by_pks_for_detection, verbose_notes=verbose_notes
        )
        
        visualize_numerical_outliers(
            data, column_to_analyze, outliers_indices, plot_pk_col, plot_type, threshold, 
            lof_scores, knn_distances, if_anomaly_scores
        )

    elif column_to_analyze in categorical_features:
        threshold = 0.05
        print(f"Using frequency-based outlier detection with threshold: {threshold} for categorical column '{column_to_analyze}'.")
        
        outliers_indices = get_categorical_outliers(data, column_to_analyze, threshold, primary_key_columns)
        
        visualize_categorical_outliers(data, column_to_analyze, outliers_indices, plot_pk_col)
    else:
        print(f"Column '{column_to_analyze}' is not identified as either a numerical or categorical feature.")
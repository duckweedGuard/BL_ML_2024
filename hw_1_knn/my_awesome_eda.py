import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def run_eda(df: pd.DataFrame):
    print('\nᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ ᶘಠᴥಠᶅ')
    shape_df = df.shape
    print(f'\nNumber of observations (rows) is {shape_df[0]}, number of parameters (features, columns) is {shape_df[1]}.')
    cols_names = df.columns
    numeric_cols = []
    str_cols = []
    categorial_cols = []
    for col in cols_names:
        if len(np.unique(df[col].dropna().values)) <= shape_df[0]*0.03:
            categorial_cols.append(col)
        elif is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            str_cols.append(col)
    print(f'\nAccording to the decision rule (if a feature has less than 3% {shape_df[0]*0.03} number of observations (rows) of unique values, it is considered categorical number of parameters (features, columns) are divided into factor (categorical), numerical and string variables as follows:')
    print(f'  - Factor (categorical) variables: {", ".join(categorial_cols)}.')
    print(f'  - Numerical variables: {", ".join(numeric_cols)}.')
    print(f'  - String variables: {", ".join(str_cols)}.')
    
    print('\nDescribtion of factor (categorical) variables:')
    for cat_col in categorial_cols:
        value, counts  = np.unique(df[cat_col].dropna().values, return_counts=True)
        total_counts = df[cat_col].dropna().shape[0]
        #print(df[cat_col].value_counts().sort_index())
        print(f' * Output counts and frequencies of values in variable {cat_col}:')
        for i in range(len(value)):
            print(f'   {round(value[i], 3)}: count = {counts[i]}, frequency = {round(counts[i]/total_counts*100, 2)}%')
    
    print('\nDescribtion of numerical variables:')
    for num_col in numeric_cols:
        min_ = df[num_col].dropna().min()
        max_ = df[num_col].dropna().max()
        std_ = df[num_col].dropna().std()
        quartiles = df[num_col].quantile([0.25, 0.5, 0.75])
        median = df[num_col].dropna().median()
        
        print(f' * Output counts and frequencies of values in variable {num_col}:')
        print(f'   min {round(min_, 2)};  max {round(max_, 2)};  std {round(std_, 2)};  quartile 0.25: {round(quartiles[0.25], 2)}, quartile 0.75: {round(quartiles[0.75], 2)};  median {round(quartiles[0.5], 2)}')
    
    print('\nOutliers:')
    for num_col in numeric_cols:
        q1 = df[num_col].quantile(0.25)
        q3 = df[num_col].quantile(0.75)
        iqr = 1.5*(q3 - q1)
        
        lower = q1 - iqr
        upper = q3 + iqr
        print(f'Variable {num_col}: outliers are below {round(lower,3)} and above {round(upper,3)}.')
        outliers = df[(df[num_col] < lower) | (df[num_col] > upper)].shape[0]
        print(f' * Number of outliers in variable {num_col}: {outliers}')
    
    
    print(f'\nMissing values total: {df.isna().sum().sum()}')
    
    print(f'\nNumber of rows that contain missing values: {df.shape[0] - df.dropna().shape[0]}')
    
    print(f'\nThese columns contain missing values: ')
    
    list_column_na = df.columns[df.isna().any()].tolist()
    list_number_column_na = []
    for col in list_column_na:
        list_number_column_na.append(df[col].isna().sum())

    for i in range(len(list_column_na)):
        print(f' * {list_column_na[i]}: {list_number_column_na[i]}')
    
    print(f'\nNumber of duplicated rows: {df.duplicated().sum()}')
    
        
    list_proportions_column_na = []
    for i in range(len(list_column_na)):
        list_proportions_column_na.append(list_number_column_na[i]/shape_df[0]*100)

    print('\n\nMissing values counts & proportions')

    missing_count = df.apply(lambda x: x.isnull().sum())
    missing_rate = missing_count / len(df)
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(df.columns):
        plt.bar(col, missing_rate.loc[col])
        plt.title('Missing values proportions')
        plt.ylabel('Missing value rate')
        plt.xlabel('Vaule')
    plt.tight_layout()
    plt.show()
    None
    
    print('\n\nHeatmap correlations for all variables in the dataset:')
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, label=col)
    plt.show()
    
    print(f'\n\nHistogram with a boxplot above it for numeric variables {", ".join(numeric_cols)}:')
    warnings.simplefilter(action="ignore", category=FutureWarning)
    for col in numeric_cols:
        sns.set(style="ticks")
        x = df[col]
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(8, 6), dpi=80,
                                    gridspec_kw={"height_ratios": (.15, .85)})

        sns.boxplot(x, ax=ax_box, color='c', medianprops={"color": "lime", "linewidth": 2})
        sns.distplot(x, ax=ax_hist, color='cyan')
        ax_box.set(yticks=[])
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True)

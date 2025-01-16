import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Load Multiple CSVs and Combine
def load_and_combine_csvs(folder_path):
    #csv_files = glob.glob(f"{folder_path}/*.csv")
    #dataframes = [pd.read_csv(file, low_memory=False) for file in csv_files]
    #combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = pd.read_csv('datasets/LDAP.csv', low_memory=False)
    return combined_df


# Step 2: Descriptive Analytics
def descriptive_analytics(df):
    """
    print("Dataset Overview:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    """

    # Mean, Median, and Mode
    print("\nCentral Tendency:")
    for column in df.select_dtypes(include=np.number).columns:
        print(f"{column}:")
        print(f"  Mean: {df[column].mean()}")
        print(f"  Median: {df[column].median()}")
        print(f"  Mode: {df[column].mode().iloc[0]}")

    # Standard Deviation and Variance
    print("\nDispersion:")
    for column in df.select_dtypes(include=np.number).columns:
        print(f"{column}:")
        print(f"  Standard Deviation: {df[column].std(skipna=True)}")
        print(f"  Variance: {df[column].var(skipna=True)}")
        print(f"  Range: {df[column].max() - df[column].min(skipna=True)}")

    # Correlation
    print("\nCorrelation Matrix:")
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    print("Non-numeric columns:", non_numeric_columns)
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr(skipna=True)
    print(correlation_matrix)

    # Visualization of Correlation Matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Simplified Correlation Matrix Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Step 3: Visualization for Insights
def visualize_data(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    num_columns = len(numeric_columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(num_columns, 2, figsize=(12, 5 * num_columns))
    fig.tight_layout(pad=5.0)  # Adjust spacing between plots

    # Loop through each numerical column and create plots
    for i, column in enumerate(numeric_columns):
        # Histogram (Left Subplot)
        sns.histplot(df[column], kde=True, bins=500, ax=axes[i, 0], color='blue', skipna=True)
        axes[i, 0].set_title(f'Distribution of {column}')
        axes[i, 0].set_xlabel(column)
        axes[i, 0].set_ylabel('Frequency')

        # Boxplot (Right Subplot)
        sns.boxplot(x=df[column], ax=axes[i, 1], color='orange', skipna=True)
        axes[i, 1].set_title(f'Boxplot of {column}')
        axes[i, 1].set_xlabel(column)

    # Add the correlation heatmap as a separate plot
    correlation_matrix = df[numeric_columns].corr(skipna=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, skipna=True)
    ax.set_title("Correlation Matrix Heatmap")

    # Show the plots
    plt.show()

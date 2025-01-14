import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Load Multiple CSVs and Combine
def load_and_combine_csvs(folder_path):
    csv_files = glob.glob(f"{folder_path}/*.csv")
    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


# Step 2: Descriptive Analytics
def descriptive_analytics(df):
    print("Dataset Overview:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

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
        print(f"  Standard Deviation: {df[column].std()}")
        print(f"  Variance: {df[column].var()}")
        print(f"  Range: {df[column].max() - df[column].min()}")

    # Correlation
    print("\nCorrelation Matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    # Visualization of Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()


# Step 3: Visualization for Insights
def visualize_data(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column], color='orange')
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()


# Main Execution
folder_path = "path_to_your_csv_files"  # Replace with the path to your folder containing CSVs
df = load_and_combine_csvs(folder_path)

descriptive_analytics(df)
visualize_data(df)

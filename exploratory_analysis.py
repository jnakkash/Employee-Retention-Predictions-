#!/usr/bin/env python3
"""
Exploratory Data Analysis for Salifort Motors Employee Retention Project
This script performs initial data exploration to understand the dataset structure,
distributions, correlations, and identify potential patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Create output directory for visualizations if it doesn't exist
Path("../results/analyzed_data").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the dataset from the raw data directory."""
    try:
        data_path = os.path.join('..', 'data', 'raw', 'HR_capstone_dataset.csv')
        df = pd.read_csv(data_path)
        print(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_eda(df):
    """Perform basic exploratory data analysis."""
    print("\n=== DATASET INFORMATION ===")
    print(df.info())
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe(include='all').T)
    
    print("\n=== CHECKING FOR MISSING VALUES ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if sum(missing_values) > 0 else "No missing values found.")
    
    print("\n=== CHECKING FOR DUPLICATES ===")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    return df

def explore_target_variable(df):
    """Explore the target variable distribution."""
    print("\n=== TARGET VARIABLE DISTRIBUTION ===")
    target_counts = df['left'].value_counts()
    print(target_counts)
    print(f"Percentage of employees who left: {100 * target_counts[1] / len(df):.2f}%")
    
    # Create a copy of df with human-readable target values
    df_visual = df.copy()
    df_visual['Employment Status'] = df_visual['left'].map({0: 'Stayed', 1: 'Left'})
    
    # Visualize target distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Employment Status', data=df_visual)
    plt.title('Employee Retention Status', fontsize=16)
    plt.xlabel('')
    plt.ylabel('Number of Employees', fontsize=12)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   fontsize=12)
    
    plt.savefig('../results/analyzed_data/employee_retention_status.png', dpi=300, bbox_inches='tight')
    
    return target_counts

def explore_numeric_features(df):
    """Explore numeric features distributions and relationships with target."""
    print("\n=== NUMERIC FEATURES ANALYSIS ===")
    
    # Create a copy of df with human-readable target values
    df_visual = df.copy()
    df_visual['Employment Status'] = df_visual['left'].map({0: 'Stayed', 1: 'Left'})
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'left']  # Skip target variable
    
    # Distribution of numeric features
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df_visual, x=col, hue='Employment Status', multiple='stack', bins=20)
        plt.title(f'Distribution of {col.replace("_", " ").title()}', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/analyzed_data/numeric_distributions.png', dpi=300, bbox_inches='tight')
    
    # Box plots for numeric features by target
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x='Employment Status', y=col, data=df_visual)
        plt.title(f'{col.replace("_", " ").title()} by Employment Status', fontsize=12)
        plt.ylabel(col.replace("_", " ").title(), fontsize=10)
        plt.xlabel('')
    plt.tight_layout()
    plt.savefig('../results/analyzed_data/numeric_boxplots.png', dpi=300, bbox_inches='tight')
    
    return numeric_cols

def explore_categorical_features(df):
    """Explore categorical features distributions and relationships with target."""
    print("\n=== CATEGORICAL FEATURES ANALYSIS ===")
    
    # Create a copy of df with human-readable target values
    df_visual = df.copy()
    df_visual['Employment Status'] = df_visual['left'].map({0: 'Stayed', 1: 'Left'})
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Count plots for categorical features
    plt.figure(figsize=(15, 15))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(len(categorical_cols), 1, i)
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df_visual, order=order, hue='Employment Status')
        plt.title(f'Distribution of {col.replace("_", " ").title()} by Employment Status', fontsize=14)
        plt.ylabel('Number of Employees', fontsize=12)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/analyzed_data/categorical_distributions.png', dpi=300, bbox_inches='tight')
    
    # Calculate attrition rate by category
    for col in categorical_cols:
        print(f"\nAttrition rate by {col}:")
        attrition_by_group = df.groupby(col)['left'].mean().sort_values(ascending=False) * 100
        print(attrition_by_group)
        
        # Visualize attrition rate by category
        plt.figure(figsize=(12, 6))
        attrition_by_group.plot(kind='bar', color='coral')
        plt.title(f'Attrition Rate by {col.replace("_", " ").title()}', fontsize=14)
        plt.ylabel('Attrition Rate (%)', fontsize=12)
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        # Add percentage labels on top of bars
        for i, v in enumerate(attrition_by_group):
            plt.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'../results/analyzed_data/attrition_rate_by_{col}.png', dpi=300, bbox_inches='tight')
    
    return categorical_cols

def explore_correlations(df):
    """Explore correlations between features."""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Calculate correlation matrix for numeric features
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    print("Correlation with target variable (left):")
    print(correlation_matrix['left'].sort_values(ascending=False))
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, mask=mask)
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/analyzed_data/correlation_matrix.png', dpi=300, bbox_inches='tight')
    
    return correlation_matrix

def explore_feature_interactions(df):
    """Explore interactions between important features."""
    print("\n=== FEATURE INTERACTIONS ===")
    
    # Create a copy of df with human-readable target values
    df_visual = df.copy()
    df_visual['Employment Status'] = df_visual['left'].map({0: 'Stayed', 1: 'Left'})
    
    # Satisfaction level vs last evaluation by target
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x='satisfaction_level', 
        y='last_evaluation', 
        hue='Employment Status', 
        size='average_montly_hours',
        sizes=(20, 200),
        data=df_visual,
        palette=["#1f77b4", "#d62728"]  # Blue for stayed, red for left
    )
    plt.title('Satisfaction Level vs Last Evaluation Score', fontsize=16)
    plt.xlabel('Satisfaction Level', fontsize=12)
    plt.ylabel('Last Evaluation Score', fontsize=12)
    legend = scatter.legend(fontsize=10, title_fontsize=12)
    plt.savefig('../results/analyzed_data/satisfaction_vs_evaluation.png', dpi=300, bbox_inches='tight')
    
    # Satisfaction level vs time spent in company by target
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='time_spend_company', y='satisfaction_level', hue='Employment Status', data=df_visual,
               palette=["#1f77b4", "#d62728"])
    plt.title('Satisfaction Level by Years at Company', fontsize=16)
    plt.xlabel('Years at Company', fontsize=12)
    plt.ylabel('Satisfaction Level', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('../results/analyzed_data/satisfaction_vs_time.png', dpi=300, bbox_inches='tight')
    
    # Monthly hours vs number of projects by target
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='number_project', y='average_montly_hours', hue='Employment Status', data=df_visual,
               palette=["#1f77b4", "#d62728"])
    plt.title('Monthly Hours by Number of Projects', fontsize=16)
    plt.xlabel('Number of Projects', fontsize=12)
    plt.ylabel('Average Monthly Hours', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('../results/analyzed_data/hours_vs_projects.png', dpi=300, bbox_inches='tight')

def save_processed_data(df):
    """Save the processed dataset."""
    Path("../data/processed").mkdir(parents=True, exist_ok=True)
    processed_path = os.path.join('..', 'data', 'processed', 'processed_hr_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to {processed_path}")

def main():
    """Main function to execute EDA."""
    print("Starting Exploratory Data Analysis...")
    
    # Load dataset
    df = load_data()
    if df is None:
        return
    
    # Perform EDA
    df = basic_eda(df)
    target_counts = explore_target_variable(df)
    numeric_cols = explore_numeric_features(df)
    categorical_cols = explore_categorical_features(df)
    correlation_matrix = explore_correlations(df)
    explore_feature_interactions(df)
    
    # Save processed data
    save_processed_data(df)
    
    print("\nExploratory Data Analysis completed.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Model Training for Salifort Motors Employee Retention Project
This script trains multiple machine learning models for predicting employee 
turnover and selects the best performing model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import class_weight

# Create output directories if they don't exist
Path("../results/model_comparison").mkdir(parents=True, exist_ok=True)
Path("../models").mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data():
    """Load the processed data for modeling."""
    try:
        # Use processed data
        data_path = os.path.join('..', 'data', 'processed', 'processed_hr_data.csv')
        df = pd.read_csv(data_path)
        print(f"Successfully loaded processed dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_data_for_modeling(df):
    """Prepare data for modeling by splitting features and target."""
    print("\n=== PREPARING DATA FOR MODELING ===")
    
    # Set target variable
    target_col = 'left'
    
    # Check if target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return None, None, None, None, None, None
    
    # Create a copy for the unscaled version
    df_unscaled = df.copy()
    
    # Handle categorical variables
    print("Converting categorical variables to numeric...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Use pandas get_dummies to one-hot encode categorical variables
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Simple scaling for numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    for col in numeric_cols:
        if col != target_col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Splitting features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Check class distribution
    class_distribution = y.value_counts(normalize=True) * 100
    print(f"Class distribution:\n{class_distribution}")
    
    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weights_dict}")
    
    # Split into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Get feature names for feature importance later
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names, class_weights_dict

def train_logistic_regression(X_train, y_train, class_weights):
    """Train a Logistic Regression model."""
    print("\n=== TRAINING LOGISTIC REGRESSION MODEL ===")
    
    # Grid search parameters
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', None],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'class_weight': [class_weights, 'balanced', None]
    }
    
    # Create base model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Get best model
    best_lr = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    return best_lr, grid_search.best_params_, grid_search.best_score_, train_time

def train_random_forest(X_train, y_train, class_weights):
    """Train a Random Forest model."""
    print("\n=== TRAINING RANDOM FOREST MODEL ===")
    
    # Grid search parameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [class_weights, 'balanced', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Get best model
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    return best_rf, grid_search.best_params_, grid_search.best_score_, train_time

def train_decision_tree(X_train, y_train, class_weights):
    """Train a Decision Tree model."""
    print("\n=== TRAINING DECISION TREE MODEL ===")
    
    # Grid search parameters
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [class_weights, 'balanced', None]
    }
    
    # Create base model
    dt = DecisionTreeClassifier(random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Get best model
    best_dt = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    return best_dt, grid_search.best_params_, grid_search.best_score_, train_time

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the trained model on the test set."""
    print(f"\n=== EVALUATING {model_name.upper()} MODEL ===")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks([0.5, 1.5], ['Stayed', 'Left'])
    plt.yticks([0.5, 1.5], ['Stayed', 'Left'], rotation=0)
    plt.tight_layout()
    plt.savefig(f'../results/model_comparison/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../results/model_comparison/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}

def plot_feature_importance(model, feature_names, model_name):
    """Plot the feature importance for the model."""
    print(f"\n=== PLOTTING FEATURE IMPORTANCE FOR {model_name.upper()} ===")
    
    # Get feature importance
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not have feature importances.")
            return None
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 10 features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
        plt.title(f'Top 10 Feature Importance - {model_name}', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'../results/model_comparison/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        
        # Save top features to JSON
        top_features = [
            {'feature': feature, 'importance': float(importance)}
            for feature, importance in zip(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
        ]
        
        feature_data = {
            'model_name': model_name,
            'top_features': top_features
        }
        
        with open(f'../results/model_comparison/feature_importance.json', 'w') as f:
            json.dump(feature_data, f, indent=4)
        
        return importance_df
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None

def save_model(model, model_name):
    """Save the trained model to disk."""
    print(f"\n=== SAVING {model_name.upper()} MODEL ===")
    
    model_path = os.path.join('..', 'models', f'{model_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")

def compare_models(models_metrics):
    """Compare models based on their performance metrics."""
    print("\n=== COMPARING MODEL PERFORMANCE ===")
    
    # Convert metrics to DataFrame for easier comparison
    metrics_df = pd.DataFrame.from_dict(models_metrics, orient='index')
    
    # Print metrics
    print(metrics_df)
    
    # Plot comparison of metrics
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/model_comparison/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save metrics to JSON
    with open('../results/model_comparison/model_metrics.json', 'w') as f:
        json.dump(models_metrics, f, indent=4)
    
    # Find the best model based on F1 score
    best_model = metrics_df['f1'].idxmax()
    print(f"\nBest model based on F1 score: {best_model}")
    
    return best_model

def main():
    """Main function to train and evaluate models."""
    print("Starting model training...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, class_weights = prepare_data_for_modeling(df)
    if X_train is None:
        return
    
    # Train models
    models = {}
    model_params = {}
    cv_scores = {}
    train_times = {}
    
    # Train Logistic Regression
    lr_model, lr_params, lr_cv, lr_time = train_logistic_regression(X_train, y_train, class_weights)
    models['Logistic_Regression'] = lr_model
    model_params['Logistic_Regression'] = lr_params
    cv_scores['Logistic_Regression'] = lr_cv
    train_times['Logistic_Regression'] = lr_time
    
    # Train Random Forest
    rf_model, rf_params, rf_cv, rf_time = train_random_forest(X_train, y_train, class_weights)
    models['Random_Forest'] = rf_model
    model_params['Random_Forest'] = rf_params
    cv_scores['Random_Forest'] = rf_cv
    train_times['Random_Forest'] = rf_time
    
    # Train Decision Tree
    dt_model, dt_params, dt_cv, dt_time = train_decision_tree(X_train, y_train, class_weights)
    models['Decision_Tree'] = dt_model
    model_params['Decision_Tree'] = dt_params
    cv_scores['Decision_Tree'] = dt_cv
    train_times['Decision_Tree'] = dt_time
    
    # Evaluate models
    models_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        models_metrics[model_name] = metrics
        
        # Plot feature importance
        plot_feature_importance(model, feature_names, model_name)
        
        # Save model
        save_model(model, model_name)
    
    # Compare models and select the best one
    best_model_name = compare_models(models_metrics)
    
    # Save the best model as "Best_Model"
    best_model = models[best_model_name]
    save_model(best_model, 'Best_Model')
    
    # Combine ROC curves for comparison
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/model_comparison/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\nModel training and evaluation completed.")
    print(f"Best model: {best_model_name}")
    
if __name__ == "__main__":
    main() 
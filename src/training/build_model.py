"""
build_model.py: this script trains a random forest model for each target label (i.e., CQ, ARG, SMU, ITO)

developer: pankaj chejara
date: 6th May 2025

"""

# import required libraries
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, RocCurveDisplay)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

import argparse
import os
from pathlib import Path

from utils import config
import json
import joblib

from datetime import datetime

def build_model(X,y,dimension, result_repo, model_repo):
    """
    Build machine learning model for prediction.

    This function trains a random forest classifier using given X, y.

    Args:
    -----
        X (DataFrame)- Pandas Dataframe containing features

        y (List) - Target 

        dimensions (str) - Name of target label

        results_repo (str) - Directory to store results

        model_repo (str) - Directory to store developed models

    Returns:
    ----

        
    """

    train_auc_plot = result_repo /  f"train_performance_{dimension}.png"
    test_auc_plot = result_repo /  f"test_performance_{dimension}.png"
    performance_file =result_repo / f"test_performance_metrics_{dimension}.csv"
    model_file = model_repo / f"trained_models/random_forest_{dimension}.joblib"
    feature_file = result_repo / f"feature_importance_{dimension}.csv"

    # 1. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # 2. Set up 5-fold CV for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20]
    }
    
    rf = RandomForestClassifier(random_state=42)
    cv = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
    )
    
    # Perform grid search
    cv.fit(X_train, y_train)
    
    # 3. Train best model on entire training set
    best_rf = RandomForestClassifier(
        n_estimators=cv.best_params_['n_estimators'],
        max_depth=cv.best_params_['max_depth'],
        random_state=42
    )
    best_rf.fit(X_train, y_train)
    
    # 4. Get cross-validation AUC scores from the best model
    cv_auc_scores = cross_val_score(
        best_rf, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # 5. Calculate mean and std of AUC scores
    mean_auc = np.mean(cv_auc_scores)
    std_auc = np.std(cv_auc_scores)
    
    # 6. Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot individual CV folds
    plt.plot(range(1, 6), cv_auc_scores, 'o-', color='gray', 
             alpha=0.3, label='Individual folds')
    
    # Plot mean line
    plt.axhline(y=mean_auc, color='b', linestyle='--', 
                label=f'Mean AUC ({mean_auc:.3f})')
    
    # 7. Plot standard deviation band
    plt.fill_between(
        x=[0, 6],
        y1=mean_auc - std_auc,
        y2=mean_auc + std_auc,
        color='blue',
        alpha=0.1,
        label=f'±1 std dev ({std_auc:.3f})'
    )
    
    # Formatting
    plt.title(f'5-Fold Cross-Validation AUC Scores {dimension}\n(Mean ± Standard Deviation)')
    plt.xlabel('Fold Number')
    plt.ylabel('AUC Score')
    plt.xlim(0.5, 5.5)
    plt.ylim(0, 1.05)
    plt.xticks(range(1, 6))
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    
    # Add text annotation for mean and std
    plt.text(
        5.1, mean_auc - std_auc - 0.05, 
        f'Mean: {mean_auc:.3f}\nStd: {std_auc:.3f}',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    #plt.show()
    plt.savefig(train_auc_plot,format='png')
    
    # 8. Evaluate on test set and plot AUC
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Plot ROC Curve
    RocCurveDisplay.from_estimator(best_rf, X_test, y_test)
    plt.title(f'ROC AUC = {roc_auc_score(y_test, y_proba):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    #plt.show()
    plt.savefig(test_auc_plot,format='png')

    
    # 9. Create performance metrics table
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }

    
    
    metrics_df = pd.DataFrame.from_dict(
        metrics, 
        orient='index', 
        columns=['Score']
    ).round(3)
    
    print("\nTest Set Performance:")
    print(metrics)
    metrics_df.to_csv(performance_file,index=False)
    
    # Optional: Feature Importance Plot
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    importance_df = get_feature_importance_df(best_rf,X.columns)
    importance_df.to_csv(feature_file,index=False)
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[sorted_idx], align='center')
    plt.xticks(range(X_train.shape[1]), X.columns[sorted_idx], rotation=90)
    plt.tight_layout()
    #plt.show()

    best_rf.metrics_ = metrics
    best_rf.training_date_ = datetime.today().strftime('%d-%m-%Y')
    # Save the trained model
    joblib.dump(best_rf, model_file)


def get_feature_importance_df(model, feature_names=None):
    """
    Extract feature importances from a trained Random Forest model and return as a sorted DataFrame.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        A trained Random Forest classifier model
    feature_names : list, optional
        List of feature names. If None, will use generic names (feature_0, feature_1, etc.)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with two columns: 'feature' and 'importance', sorted by importance (descending)
    """
    # Validate input
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Model must be a RandomForestClassifier")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    elif len(feature_names) != len(importances):
        raise ValueError("Length of feature_names must match number of features in the model")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Reset index (optional)
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="ML Model Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        "--dimension", 
        type=str,
        required=True,
        help="Name of target variable for building ML model"
    )
    
    
    
    # Parse arguments
    args = parser.parse_args()
    
    dimension = args.dimension

    data_file_path = config.settings.paths.PROCESSED_DATA
    json_file_path = config.settings.paths.FEATURE_NAMES
    model_repo = config.settings.paths.MODELS_DIR
    result_repo = config.settings.paths.RESULTS_DIR

    with open(json_file_path, 'r') as f:
        features = json.load(f)

    # Loading data
    df = pd.read_csv(data_file_path)
    X = df[features]
    y = df[dimension].to_list()

    # Building ML model
    build_model(X,y,dimension,result_repo,model_repo)

if __name__ == "__main__":
    main()

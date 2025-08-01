# -*- coding: utf-8 -*-
"""diabetes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hFr9TXuHPf99ta7qmvKTlO0zOvQ5rjnJ
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,roc_curve,recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.combine import SMOTETomek
import shap

def prediction_diabetes(test):
    Y_RF = RF.predict(test)
    Y_XG = XG.predict(test)
    Y_LR = LR.predict(test)
    return Y_RF,Y_XG,Y_LR

def DataOutput_diabetes():
    return X_train, X_test, Y_train, Y_test

# model performance table
def get_results_df(Y_test, y_pred_RF, y_score_RF, y_pred_XG, y_score_XG, y_pred_LR, y_score_LR):
    models = {
        'Random Forest': (y_pred_RF, y_score_RF),
        'XGBoost': (y_pred_XG, y_score_XG),
        'Logistic Regression': (y_pred_LR, y_score_LR)
    }

    results = []
    for model_name, (y_pred, y_prob) in models.items():
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        roc_auc = roc_auc_score(Y_test, y_prob)

        results.append({
            'Model': model_name,
            'ROC AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    results_df = pd.DataFrame(results).round(3)
    return results_df

def plot_roc_curve():
    #auc_rf, auc_xgb, auc_lg
    roc_diabetes, ax = plt.subplots(figsize=(8, 8))
    plt.title('Diabetes: Comparision of ROC Curves', fontsize=18)
    plt.plot(false_positive_rate7, true_positive_rate7, color = 'red', label = f'Random Forest (AUC={auc_rf:.3f})', linewidth=2)
    plt.plot(false_positive_rate6, true_positive_rate6, color = 'limegreen', label = f'XGBoost (AUC={auc_xgb:.3f})', linewidth=2)
    plt.plot(false_positive_rate2, true_positive_rate2, color = 'blue', label = f'Logistic Regression (AUC={auc_lg:.3f})', linewidth=2)
    plt.plot([0, 1], ls="--")
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.legend(fontsize=12)
    # Diagonal line
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    return roc_diabetes

def plot_shap_diabetes(figx):
    ## SHAP ## XGBoost
    explainer = shap.TreeExplainer(XG)
    shap_values = explainer.shap_values(X_test)
    # Create figure with desired size
    fig, ax = plt.subplots(figsize=(figx, figx))
    shap.summary_plot(shap_values, X_test, max_display=5, show=False) #beeswarm #plot global feature importance
    # Get current figure object
    shap_diabetes = plt.gcf()
    plt.title("Diabetes: Top 5 Feature Contributions", fontsize=16)
    plt.tight_layout()
    return shap_diabetes
    # RF is too slow, KNN and LR do not support shap

#url = "https://drive.google.com/file/d/1PoO03AqSY645yZIXZuB1v7-WsEXZKWk9/view?usp=sharing"
#path = "https://drive.google.com/uc?export=download&id="+url.split("/")[-2]
path = "./Data/diabetes_prediction_dataset.csv"
data = pd.read_csv(path)
# Remove Duplicates
data = data.drop_duplicates()

# Identify numerical and categorical features
#num_features = data.select_dtypes(include="number").columns
#cat_features = data.select_dtypes(exclude="number").columns
categorical = ['gender', 'smoking_history']
numerical = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# One-hot encode categorical variables
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = pd.DataFrame(ohe.fit_transform(data[categorical]),
                           columns=ohe.get_feature_names_out(categorical),
                           index=data.index)

# Combine numerical and encoded categorical columns
X = pd.concat([data[numerical], encoded_cat], axis=1)
Y = data['diabetes']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 8.8% diabetes, so using SMOTE
# Intializing our data sampling algorithm
smt = SMOTETomek(random_state=42)
X_train, Y_train = smt.fit_resample(X_train, Y_train)

## models
RF = RandomForestClassifier(n_estimators=100, max_depth = None, random_state=47 , n_jobs = 3)
RF.fit(X_train,Y_train)
y_pred_RF=RF.predict(X_test)
# calculate probability
y_score_RF = RF.predict_proba(X_test)[:,1]
false_positive_rate7, true_positive_rate7, threshold7 = roc_curve(Y_test, y_score_RF)


XG = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100)
XG.fit(X_train, Y_train)
y_pred_XG = XG.predict(X_test)
# calculate probability
y_score_XG = XG.predict_proba(X_test)[:,1]
false_positive_rate6, true_positive_rate6, threshold6 = roc_curve(Y_test, y_score_XG)

LR = LogisticRegression(C=10, solver='liblinear', random_state=42)
LR.fit(X_train,Y_train)
y_pred_LR=LR.predict(X_test)
# calculate probability
y_score_LR = LR.predict_proba(X_test)[:,1]
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(Y_test, y_score_LR)

# call the function
results_df_diabetes = get_results_df(Y_test, y_pred_RF, y_score_RF, y_pred_XG, y_score_XG, y_pred_LR, y_score_LR)

# Calculate AUC scores
auc_rf  = roc_auc_score(Y_test, y_score_RF)
auc_xgb = roc_auc_score(Y_test, y_score_XG)
auc_lg = roc_auc_score(Y_test, y_score_LR)

# Create figure and axis
roc_diabetes = plot_roc_curve()
shap_diabetes = plot_shap_diabetes(5)


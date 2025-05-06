#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Disease Prediction and XAI via SDOH Project Framework
# Author: Stephan Mabry

# ================================
# SECTION 1: Library Imports and Setup
# ================================
#Basics
import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

#Models
import xgboost as xgb
from xgboost import XGBClassifier

#XAI
# import shap  #imported in function
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer

#SKLearn Components
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier

## NN Tools
import tensorflow as tf  #used for XAI
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set directories
DATA_DIR = r"D:\School\Research\DxByXAI\Data"
PLOTS_DIR = "../Plots"
COLUMN_SPECS_PATH = "Column_Specs_and_Names_trimmed.csv"
DATA_FILE = "h233.dat"
TARGET_FIELD = "drOfficeVisitsOthCombinedAmt21"

SDOH_FIELDS = [
    "sdohAffordableHousing", "sdohAttendChurchOrServices", "sdohMeetingsPerYear",
    "sdohDateCompletedMonth", "sdohDateCompletedYear", "sdohHelpFromCommunity",
    "sdohFeelLackCompanionship", "sdoh30DaysAvgDaysPerWkModExrcs", "sdoh12MosContactCollection",
    "sdohDiscriminationInHealthcare", "sdohDiscriminationInHousing", "sdohDiscriminationApplyingJobs",
    "sdohDiscriminationByPolice", "sdohDiscriminationApplyingServices", "sdohDiscriminationStores",
    "sdohDiscriminationAtWork", "sdohUsedElectronicNicotineProd", "sdohHelpFromFamily",
    "sdohHowOftenForced", "sdohHelpFromFriends", "sdohSeeOthersPerWeek",
    "sdohPlacesForHealthyFood", "sdohLivedWithAlcoholic18", "sdohHowOftenAbused18",
    "sdohLivedWithMentalIllness18", "sdohLivedWithSplitHome18", "sdohLivedWithDrugs18",
    "sdohLivedWithSentenced18", "sdohSatisfiedWithHome", "sdohHowOftenChildHurt",
    "sdohHowOftenChildInsult", "sdohHowOftenInsulted", "sdohFeelIsolated",
    "sdoh12MosPayRentmortgLate", "sdoh12MosPayUtilityLate", "sdohFeelLeftOut",
    "sdohSatisfiedWithLife", "sdohPlacesForMedicalCare", "sdoh30DaysMinPerDayModExrcs",
    "sdoh12MosMissCardOrLoanPymt", "sdoh12MosFoodRanOut", "sdoh12MosNoTransDaily",
    "sdohPlacesForParksplay", "sdohHowHardPayBasics", "sdohHowOftenHurtByOthers",
    "sdohHomeProblemCook", "sdohHomeProblemHeat", "sdohHomeProblemLead",
    "sdohHomeProblemWaterLeaks", "sdohHomeProblemMold", "sdohNoHomeProblems",
    "sdohHomeProblemPests", "sdohHomeProblemSmokeDetector", "sdohRelationshipRespondentToAdult",
    "sdohAccessPublicTransportation", "sdohHowOftenScreamCurse", "sdohSafeFromCrimeviolence",
    "sdoh12MosThreatUtilityOff", "sdohHowOftenStress", "sdohHowOftenAskedToTouch",
    "sdohHowOftenChildTouched", "sdohHowOftenThreatenedHarm", "sdohTelephoneOthersPerWeek",
    "sdohCoverUnexpectedExpense", "sdoh12MosWorriedAboutFood"
]

DIAGNOSIS_FIELDS_TEST = [
    "adhdaddDiagnosis517"
]

DIAGNOSIS_FIELDS = [
    "adhdaddDiagnosis517", "anginaDiagnosis17", "arthritisDiagnosis17", "typeOfArthritisDiagnosed17",
    "asthmaDiagnosis", "cancerDiagnosedBladder17", "cancerDiagnosedBreast17", "cancerDiagnosedCervical17",
    "cancerDiagnosedColon17", "cancerDiagnosedLung17", "cancerDiagnosedLymphomaNonhodgkins17",
    "cancerDiagnosedSkinMelanoma17", "cancerDiagnosis17", "cancerDiagnosedOther17",
    "cancerDiagnosedProstate17", "cancerDiagnosedskinunknownType17", "cancerDiagnosedSkinnonmelano17",
    "cancerDiagnosedUterine17", "highCholesterolDiagnosis17", "diabetesDiagnosis",
    "dcsDiabetesDiagnosisByHealthProf", "emphysemaDiagnosis17", "strokeDiagnosis17"
]

os.chdir(DATA_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ================================
# SECTION 1: Fixing Bugs in frameworks... 
# ================================
# --- Wrapper fix for bug in scikit-learn when building ensemble models ---
class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.classes_ = np.array([0, 1])  # Default for binary classification

    def fit(self, X, y):
        self.model = build_nn_model(self.input_dim)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = (self.model.predict(X) > 0.5).astype(int).flatten()
        return preds

    def predict_proba(self, X):
        proba = self.model.predict(X)
        return np.hstack([1 - proba, proba])



# ================================
# SECTION 2: Utility Functions
# ================================
def save_plot(fig, filename):
    plt.show()  # Display the plot before saving
    fig.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close(fig)  # Clean up after saving


def load_layout_from_csv(layout_path):
    layout_df = pd.read_csv(layout_path)
    colspecs = [ast.literal_eval(spec) for spec in layout_df['colspecs']]
    colnames = layout_df['colnames'].tolist()
    return colspecs, colnames

def load_fixed_width_data_all(data_path, layout_path):
    colspecs, colnames = load_layout_from_csv(layout_path)
    df = pd.read_fwf(data_path, colspecs=colspecs, names=colnames)
    return df
# ================================
# SECTION 4: Preprocessing Function
# ================================

def prepare_sdoh_diagnosis_xy(df):
    # Only keep the columns we care about
    all_fields = SDOH_FIELDS + DIAGNOSIS_FIELDS
    df_subset = df[all_fields].copy()

    # Drop rows that already have NaNs (after previous cleaning)
    df_clean = df_subset.dropna(subset=all_fields)

    # Split into inputs and targets
    X = df_clean[SDOH_FIELDS].copy()
    Y = df_clean[DIAGNOSIS_FIELDS].copy()

    return X, Y

def clean_diagnosis_labels(y_df):
    """
    Converts all diagnosis values:
    - 1 → 1 (Yes)
    - Anything else → 0 (No)
    """
    return y_df.where(y_df == 1, 0)

  
# ================================
# SECTION 5: Plotting Functions
# ================================
def plot_histograms(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['TOTEXP'], bins=50, ax=ax[0])
    ax[0].set_title("Raw Cost Distribution")
    sns.histplot(df['LOG_COST'], bins=50, ax=ax[1])
    ax[1].set_title("Log-Transformed Cost Distribution")
    save_plot(fig, "cost_histograms.png")

def plot_categorical_distribution(df):
    for col in ['Race', 'Sex', 'Income']:  # Replace with actual column names
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(f"Distribution of {col}")
        save_plot(fig, f"{col.lower()}_distribution.png")

def plot_missing_data(df):
    fig = plt.figure()
    msno.matrix(df)
    save_plot(fig, "missing_data_matrix.png")

    fig = plt.figure()
    msno.bar(df)
    save_plot(fig, "missing_data_bar.png")

def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    save_plot(fig, "correlation_matrix.png")

# ================================
# SECTION 6: Model Training
# ================================
# --- Models ---
def build_nn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_xgb_model():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

def build_dt_model():
    return DecisionTreeClassifier(max_depth=5, random_state=42)

# --- Training ---
def train_nn_model(X_train, y_train, input_dim):
    model = build_nn_model(input_dim)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    return model

def train_sklearn_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# --- Predictions --
def predict_nn_model(model, X_val):
    return (model.predict(X_val) > 0.5).astype(int).flatten()

def predict_sklearn_model(model, X_val):
    return model.predict(X_val)

# --- Ensemble ---
def build_ensemble_model(input_dim):
    estimators = [
        ('nn', KerasModelWrapper(input_dim=input_dim)),
        ('xgb', build_xgb_model()),
        ('dt', build_dt_model())
    ]
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )
    return ensemble


# -- Evaluation --
def evaluate_predictions(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

def run_models_for_diagnosis(X, y, diag_name):
    results = []
    xai_results = {}
  
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
     # Tests for SHAP - so we have apples to apples comparison of items. Explain the same 100 sample records.
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], size=100, replace=False)]
    val_sample = X_val_scaled[:100]
    
    input_dim = X_train_scaled.shape[1]

    # === 1. Neural Network
    nn = train_nn_model(X_train_scaled, y_train, input_dim)
    nn_preds = predict_nn_model(nn, X_val_scaled)
    nn_metrics = evaluate_predictions(y_val, nn_preds)
    nn_metrics.update({'diagnosis': diag_name, 'model': 'NN'})
    results.append(nn_metrics)

    xai_results['NN'] = run_xai_for_nn(nn, X_train_scaled, X_val_scaled, y_val, diag_name, background, val_sample, feature_names = X_train.columns.tolist())

    # === 2. XGBoost
    xgb = train_sklearn_model(build_xgb_model(), X_train_scaled, y_train)
    xgb_preds = predict_sklearn_model(xgb, X_val_scaled)
    xgb_metrics = evaluate_predictions(y_val, xgb_preds)
    xgb_metrics.update({'diagnosis': diag_name, 'model': 'XGB'})
    results.append(xgb_metrics)

    xai_results['XGB'] = run_xai_for_model(xgb, X_train_scaled, X_val_scaled, y_val, diag_name, background, val_sample, feature_names = X_train.columns.tolist())

    # === 3. Decision Tree
    dt = train_sklearn_model(build_dt_model(), X_train_scaled, y_train)
    dt_preds = predict_sklearn_model(dt, X_val_scaled)
    dt_metrics = evaluate_predictions(y_val, dt_preds)
    dt_metrics.update({'diagnosis': diag_name, 'model': 'DT'})
    results.append(dt_metrics)

    xai_results['DT'] = run_xai_for_model(dt, X_train_scaled, X_val_scaled, y_val, diag_name, background, val_sample, feature_names = X_train.columns.tolist())

    # === 4. Ensemble
    ensemble = build_ensemble_model(input_dim)
    ensemble = train_sklearn_model(ensemble, X_train_scaled, y_train)
    ensemble_preds = predict_sklearn_model(ensemble, X_val_scaled)
    ensemble_metrics = evaluate_predictions(y_val, ensemble_preds)
    ensemble_metrics.update({'diagnosis': diag_name, 'model': 'Ensemble'})
    results.append(ensemble_metrics)

    xai_results['Ensemble'] = run_xai_for_model(ensemble, X_train_scaled, X_val_scaled, y_val, diag_name, background, val_sample, model_type='generic', feature_names = X_train.columns.tolist())

    print(f"✅ Finished training + XAI on {diag_name}")
    return results, xai_results

# --- Master Driver ---
def run_model_per_diagnosis(X, Y):
    all_results = []
    all_xai_results = {}

    for diag in DIAGNOSIS_FIELDS:
        y = Y[diag]
        if y.nunique() < 2:
            print(f"Skipping {diag} due to only one class present.")
            continue

        results, xai_results = run_models_for_diagnosis(X, y, diag)
        all_results.extend(results)
        all_xai_results[diag] = xai_results

    return pd.DataFrame(all_results), all_xai_results

# ================================
# SECTION X: REPORTING TOOLS
# ================================
def print_xai_summary(results_df, xai_results_dict):
    print("\n================= MODEL PERFORMANCE SUMMARY =================")
    print(results_df.to_string(index=False))

    # Save model performance
    results_df.to_csv(os.path.join(DATA_DIR, "model_results.csv"), index=False)
    print("✅ Saved model performance results to model_results.csv")

    print("\n================= XAI RESULTS SUMMARY =================")
    flat_rows = []
    for diag_name, model_xai_dict in xai_results_dict.items():
        print(f"\n📋 Diagnosis: {diag_name}")
        for model_name, xai_methods in model_xai_dict.items():
            print(f"  🧠 Model: {model_name}")
            for method, output in xai_methods.items():
                output_str = str(output)
                print(f"    - {method}: {output_str[:500]}...")

                for feature, score in output:
                    flat_rows.append({
                        'Diagnosis': diag_name,
                        'Model': model_name,
                        'XAI_Method': method,
                        'Feature': feature,
                        'Score': score
                    })

    xai_df = pd.DataFrame(flat_rows)
    xai_df.to_csv(os.path.join(DATA_DIR, "xai_summary.csv"), index=False)
    print("\n✅ Saved XAI summary results to xai_summary.csv")

# --- Performance + XAI Reporting Plots ---
def generate_performance_and_xai_plots(results_df, xai_results_dict, plots_dir):
    import seaborn as sns
    # --- Model Performance Bar Plot ---
    perf_melted = results_df.melt(id_vars=['diagnosis', 'model'],
                                  value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
                                  var_name='metric', value_name='value')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=perf_melted, x='model', y='value', hue='metric')
    plt.title("Model Performance Metrics by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_performance_metrics.png"))
    plt.close()

    # --- Top Feature Attribution Plot by XAI Method ---
    for diag_name, model_xai_dict in xai_results_dict.items():
        for model_name, method_dict in model_xai_dict.items():
            for method, feature_scores in method_dict.items():
                if not isinstance(feature_scores, list):
                    continue
                feature_names = [x[0] for x in feature_scores[:10]]
                scores = [x[1] for x in feature_scores[:10]]
                plt.figure(figsize=(10, 5))
                sns.barplot(x=scores, y=feature_names)
                plt.title(f"{method} Top Features for {model_name} on {diag_name}")
                plt.tight_layout()
                fname = f"{diag_name}_{model_name}_{method}_top_features.png".replace(" ", "_")
                plt.savefig(os.path.join(plots_dir, fname))
                plt.close()

    print("✅ Plots saved to:", plots_dir)
    
# ================================
# SECTION 7: XAI METHODS FRAMEWORK
# ================================
# --- SHAP ---
def explain_with_shap(model, background, X_val, model_type, feature_names=None):
    import shap  ## Ran into some conflicts with TF training. Moved to here to avoid that conflict.
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'deep':
        explainer = shap.DeepExplainer(model, background)
    else:
        explainer = shap.KernelExplainer(model.predict, background)

    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_shap[:, 0] if mean_shap.ndim == 2 else mean_shap)[-10:][::-1]

    results = []
    for idx in top_features:
        score = mean_shap[idx][0] if mean_shap.ndim == 2 else mean_shap[idx]
        name = feature_names[idx] if feature_names is not None else f"feature_{idx}"
        results.append((name, float(score)))

    return results

# --- LIME ---
def explain_with_lime(model, X_train, X_val, instance_idx=0, feature_names=None):

    def keras_predict_proba(x):
        preds = model.predict(x)
        return np.hstack([1 - preds, preds]) if preds.ndim == 2 else np.vstack([1 - preds, preds]).T

    explainer = LimeTabularExplainer(X_train, mode='classification', feature_names=feature_names)
    exp = explainer.explain_instance(X_val[instance_idx], keras_predict_proba)

    results = []
    for term, val in exp.as_list():
        if feature_names:
            try:
                # If term is like '33 <= -0.54', extract index
                idx = int(term.split('<=')[0].strip())
                name = feature_names[idx] if idx < len(feature_names) else term
            except ValueError:
                # If term already has feature name
                name = term
            results.append((name, val))
        else:
            results.append((term, val))

    return results


# --- Integrated Gradients (NN Only) ---
def explain_with_integrated_gradients(model, X_val, instance_idx=0, feature_names=None):
    x = tf.convert_to_tensor(X_val[instance_idx:instance_idx+1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
    grads = tape.gradient(preds, x).numpy()
    integrated_grads = (x.numpy() * grads).squeeze()

    results = []
    for i, score in enumerate(integrated_grads):
        name = feature_names[i] if feature_names else f"feature_{i}"
        results.append((name, float(score)))

    plt.bar([r[0] for r in results], [r[1] for r in results])
    plt.title("Integrated Gradients Attribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return results

# --- Saliency Map (NN Only) ---
def explain_with_saliency_map(model, X_val, instance_idx=0, feature_names=None):
    x = tf.convert_to_tensor(X_val[instance_idx:instance_idx+1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
    grads = tape.gradient(preds, x).numpy()
    saliency = grads.squeeze()

    results = []
    for i, score in enumerate(saliency):
        name = feature_names[i] if feature_names else f"feature_{i}"
        results.append((name, float(score)))

    plt.bar([r[0] for r in results], [r[1] for r in results])
    plt.title("Saliency Map Attribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return results


# --- Permutation Importance ---
def compute_permutation_importance(model, X_val, y_val, feature_names=None):
    from sklearn.metrics import accuracy_score, make_scorer
    from sklearn.inspection import permutation_importance

    def wrapped_predict(X):
        return (model.predict(X) > 0.5).astype(int).flatten()

    class Wrapper:
        def __init__(self, model):
            self.model = model
        def fit(self, X, y):
            return self
        def predict(self, X):
            return wrapped_predict(X)
        def score(self, X, y):
            return accuracy_score(y, self.predict(X))

    wrapped_model = Wrapper(model)
    scorer = make_scorer(accuracy_score)
    r = permutation_importance(wrapped_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1, scoring=scorer)

    sorted_idx = r.importances_mean.argsort()
    names = [feature_names[i] if feature_names else f"feature_{i}" for i in sorted_idx]
    scores = r.importances_mean[sorted_idx]

    plt.barh(names, scores)
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.show()

    return list(zip(names, scores))

# --- Feature Ablation ---
def compute_feature_ablation(model, X_val, y_val, feature_names=None):
    from sklearn.metrics import accuracy_score

    def get_accuracy(m, X, y):
        try:
            return m.score(X, y)
        except AttributeError:
            y_pred = (m.predict(X) > 0.5).astype(int).flatten()
            return accuracy_score(y, y_pred)

    base_score = get_accuracy(model, X_val, y_val)
    drop_losses = []

    for i in range(X_val.shape[1]):
        X_val_copy = X_val.copy()
        X_val_copy[:, i] = 0  # zero out the i-th feature
        score = get_accuracy(model, X_val_copy, y_val)
        name = feature_names[i] if feature_names else f"feature_{i}"
        drop_losses.append((name, base_score - score))

    drop_losses.sort(key=lambda x: -x[1])
    plt.bar([k for k, _ in drop_losses], [v for _, v in drop_losses])
    plt.title("Feature Ablation Impact")
    plt.xlabel("Feature")
    plt.ylabel("Drop in Accuracy")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return drop_losses


# ================================
# XAI Runner Functions
# ================================

def run_xai_for_nn(model, X_train, X_val, y_val, diag_name, background, val_sample, model_type='deep', feature_names = 'none'):
    print(f"\n🔍 Running XAI methods for NN on {diag_name}")
    results = {
        'SHAP': explain_with_shap(model, background, val_sample, model_type=model_type, feature_names=feature_names),
        'LIME': explain_with_lime(model, X_train, X_val, feature_names=feature_names),
        'Integrated_Gradients': explain_with_integrated_gradients(model, X_val, feature_names=feature_names),
        'Saliency': explain_with_saliency_map(model, X_val, feature_names=feature_names),
        'Permutation_Importance': compute_permutation_importance(model, X_val, y_val, feature_names=feature_names),
        'Feature_Ablation': compute_feature_ablation(model, X_val, y_val, feature_names=feature_names)
    }
    return results

def run_xai_for_model(model, X_train, X_val, y_val, diag_name, background, val_sample, model_type='tree', feature_names = 'none'):
    print(f"\n🔍 Running XAI methods for model on {diag_name}")
    results = {
        'SHAP': explain_with_shap(model, background, val_sample, model_type=model_type, feature_names=feature_names),
        'LIME': explain_with_lime(model, X_train, X_val, feature_names=feature_names),
        'Permutation_Importance': compute_permutation_importance(model, X_val, y_val, feature_names=feature_names),
        'Feature_Ablation': compute_feature_ablation(model, X_val, y_val, feature_names=feature_names)
    }
    return results

    
# ================================
# SECTION 8: Main Driver
# ================================


def main():
    print("Loading Data")
    df = load_fixed_width_data_all(DATA_FILE, COLUMN_SPECS_PATH)
    print("Data Loaded")

    print("🔹 Filtering and Splitting into X (SDOH) and Y (Diagnoses)")
    X, Y = prepare_sdoh_diagnosis_xy(df)
    Y = clean_diagnosis_labels(Y)

    print(f"✅ Features (X) shape: {X.shape}, Targets (Y) shape: {Y.shape}")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print("🧠 Starting training for each diagnosis condition...")
    results_df, xai_results_dict = run_model_per_diagnosis(X, Y)

    print_xai_summary(results_df, xai_results_dict)
    generate_performance_and_xai_plots(results_df, xai_results_dict, PLOTS_DIR)


if __name__ == "__main__":
    main()


# In[ ]:





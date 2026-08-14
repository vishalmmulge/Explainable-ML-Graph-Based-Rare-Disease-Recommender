"""SHAP explainability for Rare Disease Recommendation System."""

import numpy as np
import pandas as pd
import pickle
import joblib
import shap
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model(model_name='best_model'):
    """Load trained ML model."""
    model_path = MODEL_DIR / f"{model_name}.pkl"
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    return data['model'], data['scaler'], data['label_encoder']


def load_hierarchical_model():
    """Load hierarchical model components."""
    model_path = MODEL_DIR / "hierarchical_model.pkl"
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    return data


def load_processed_data():
    """Load processed data."""
    X = np.load(PROCESSED_DATA_DIR / "X.npy")
    y = np.load(PROCESSED_DATA_DIR / "y.npy")
    
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(SYMPTOM_VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(PROCESSED_DATA_DIR / "symptom_features.pkl", 'rb') as f:
        symptom_features = pickle.load(f)
    
    return X, y, label_encoder, vocab, symptom_features


def create_shap_explainer(model, X_background, scaler=None):
    """Create SHAP explainer."""
    print("Creating SHAP explainer...")
    
    if scaler is not None:
        X_background_scaled = scaler.transform(X_background)
    else:
        X_background_scaled = X_background
    
    # Use TreeExplainer for XGBoost with limited classes to avoid memory issues
    if hasattr(model, 'feature_importances_'):
        try:
            explainer = shap.TreeExplainer(model)
        except MemoryError:
            print("  TreeExplainer memory error, falling back to KernelExplainer...")
            explainer = shap.KernelExplainer(model.predict_proba, X_background_scaled[:50])
    elif hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_background_scaled, feature_perturbation="interventional")
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_background_scaled[:100])
    
    return explainer


def explain_prediction(explainer, X_sample, scaler=None, top_k=10):
    """Generate SHAP explanation for a single prediction."""
    if scaler is not None:
        X_scaled = scaler.transform(X_sample.reshape(1, -1))
    else:
        X_scaled = X_sample.reshape(1, -1)
    
    shap_values = explainer.shap_values(X_scaled)
    
    if isinstance(shap_values, list):
        return shap_values
    else:
        return shap_values


def get_top_contributing_features(shap_values, feature_names, class_idx, top_k=10):
    """Get top contributing features for a specific class."""
    if isinstance(shap_values, list):
        vals = shap_values[class_idx][0]
    elif shap_values.ndim == 3:
        vals = shap_values[0, :, class_idx]
    elif shap_values.ndim == 2:
        if shap_values.shape[0] == len(feature_names):
            vals = shap_values[:, class_idx]
        else:
            vals = shap_values[0, :]
    else:
        vals = shap_values[0]
    
    top_pos_idx = np.argsort(vals)[::-1][:top_k]
    top_neg_idx = np.argsort(vals)[:top_k]
    
    pos_features = [(feature_names[i], float(vals[i])) for i in top_pos_idx if vals[i] > 0]
    neg_features = [(feature_names[i], float(vals[i])) for i in top_neg_idx if vals[i] < 0]
    
    return pos_features, neg_features


def generate_global_shap_summary(explainer, X_sample, feature_names, output_path):
    """Generate global SHAP summary plot."""
    print("Generating SHAP summary plot...")
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=0)
        sv = np.mean(np.abs(sv), axis=0)
    elif shap_values.ndim == 3:
        sv = np.mean(np.abs(shap_values), axis=2)
    else:
        sv = shap_values
    
    shap.summary_plot(sv, X_sample, feature_names=feature_names, 
                     show=False, max_display=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def generate_class_shap_plot(explainer, X_sample, feature_names, class_idx, label_encoder, output_path):
    """Generate SHAP plot for a specific class."""
    print(f"Generating SHAP plot for class {class_idx}...")
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, class_idx]
    else:
        sv = shap_values
    
    shap.summary_plot(sv, X_sample, feature_names=feature_names, 
                     show=False, max_display=20)
    
    plt.title(f"SHAP values for {label_encoder.classes_[class_idx]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    print("="*60)
    print("SHAP EXPLAINABILITY (Hierarchical Model)")
    print("="*60)
    
    # Load hierarchical model components for group/type explainability
    hier_data = load_hierarchical_model()
    group_model = hier_data['group_model']
    group_scaler = hier_data['scaler_group']
    group_le = hier_data['group_le']
    type_model = hier_data['type_model']
    type_scaler = hier_data['scaler_type']
    type_le = hier_data['type_le']
    
    X, y, disease_le, vocab, symptom_features = load_processed_data()
    
    # Use group model for SHAP (3 classes, fast)
    print("\n--- Group Classifier SHAP ---")
    n_background = min(50, len(X))
    bg_indices = np.random.choice(len(X), n_background, replace=False)
    X_background = X[bg_indices]
    
    group_explainer = create_shap_explainer(group_model, X_background, group_scaler)
    
    test_idx = 0
    X_test = X[test_idx:test_idx+1]
    
    print(f"Test sample: {disease_le.classes_[test_idx]}")
    
    # Group prediction
    X_test_group = group_scaler.transform(X_test)
    group_proba = group_model.predict_proba(X_test_group)
    group_pred = np.argmax(group_proba, axis=1)[0]
    print(f"Predicted Group: {group_le.inverse_transform([group_pred])[0]}")
    
    group_shap = explain_prediction(group_explainer, X_test[0], group_scaler)
    pos_feats, neg_feats = get_top_contributing_features(
        group_shap, symptom_features, group_pred, top_k=10
    )
    
    print("Top positive contributors (Group):")
    for feat, val in pos_feats[:5]:
        print(f"  {feat}: {val:.4f}")
    
    # Type prediction
    print("\n--- Type Classifier SHAP ---")
    type_explainer = create_shap_explainer(type_model, X_background, type_scaler)
    X_test_type = type_scaler.transform(X_test)
    type_proba = type_model.predict_proba(X_test_type)
    type_pred = np.argmax(type_proba, axis=1)[0]
    print(f"Predicted Type: {type_le.inverse_transform([type_pred])[0]}")
    
    type_shap = explain_prediction(type_explainer, X_test[0], type_scaler)
    pos_feats, neg_feats = get_top_contributing_features(
        type_shap, symptom_features, type_pred, top_k=10
    )
    
    print("Top positive contributors (Type):")
    for feat, val in pos_feats[:5]:
        print(f"  {feat}: {val:.4f}")
    
    # Generate summary plots
    n_sample = min(100, len(X))
    sample_idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[sample_idx]
    
    X_sample_group = group_scaler.transform(X_sample)
    generate_global_shap_summary(group_explainer, X_sample_group, symptom_features, 
                                FIGURES_DIR / "shap_summary_group.png")
    
    X_sample_type = type_scaler.transform(X_sample)
    generate_global_shap_summary(type_explainer, X_sample_type, symptom_features, 
                                FIGURES_DIR / "shap_summary_type.png")
    
    print("="*60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
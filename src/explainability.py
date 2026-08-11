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
    
    # Scale background data if needed
    if scaler is not None:
        X_background_scaled = scaler.transform(X_background)
    else:
        X_background_scaled = X_background
    
    # Use LinearExplainer for logistic regression (fast)
    if hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_background_scaled, feature_perturbation="interventional")
    else:
        # Fallback to KernelExplainer (slower but works for any model)
        explainer = shap.KernelExplainer(model.predict_proba, X_background_scaled[:100])
    
    return explainer


def explain_prediction(explainer, X_sample, scaler=None, top_k=10):
    """Generate SHAP explanation for a single prediction."""
    if scaler is not None:
        X_scaled = scaler.transform(X_sample.reshape(1, -1))
    else:
        X_scaled = X_sample.reshape(1, -1)
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_scaled)
    
    # For multi-class, shap_values is list of arrays per class
    if isinstance(shap_values, list):
        # Return SHAP values for all classes
        return shap_values
    else:
        return shap_values


def get_top_contributing_features(shap_values, feature_names, class_idx, top_k=10):
    """Get top contributing features for a specific class."""
    # For LinearExplainer with multi-class, shap_values shape is (n_samples, n_features, n_classes)
    # or (n_classes, n_features)
    if isinstance(shap_values, list):
        vals = shap_values[class_idx][0]  # First sample, specific class
    elif shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        vals = shap_values[0, :, class_idx]
    elif shap_values.ndim == 2:
        # Shape: (n_classes, n_features) or (n_samples, n_features)
        if shap_values.shape[0] == len(feature_names):
            vals = shap_values[:, class_idx]
        else:
            vals = shap_values[0, :]  # First sample
    else:
        vals = shap_values[0]
    
    # Get top positive and negative contributions
    top_pos_idx = np.argsort(vals)[::-1][:top_k]
    top_neg_idx = np.argsort(vals)[:top_k]
    
    pos_features = [(feature_names[i], float(vals[i])) for i in top_pos_idx if vals[i] > 0]
    neg_features = [(feature_names[i], float(vals[i])) for i in top_neg_idx if vals[i] < 0]
    
    return pos_features, neg_features


def generate_global_shap_summary(explainer, X_sample, feature_names, output_path):
    """Generate global SHAP summary plot."""
    print("Generating SHAP summary plot...")
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different shapes - need 2D array for summary_plot
    if isinstance(shap_values, list):
        # Multi-class list: stack them
        sv = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
        # Average across classes
        sv = np.mean(np.abs(sv), axis=0)  # (n_samples, n_features)
    elif shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        sv = np.mean(np.abs(shap_values), axis=2)  # (n_samples, n_features)
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
        # Shape: (n_samples, n_features, n_classes)
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
    print("SHAP EXPLAINABILITY")
    print("="*60)
    
    # Load model and data
    model, scaler, label_encoder = load_model()
    X, y, _, vocab, symptom_features = load_processed_data()
    
    # Use a small background sample for explainer
    n_background = min(200, len(X))
    bg_indices = np.random.choice(len(X), n_background, replace=False)
    X_background = X[bg_indices]
    
    # Create explainer
    explainer = create_shap_explainer(model, X_background, scaler)
    
    # Test on a sample
    test_idx = 0
    X_test = X[test_idx:test_idx+1]
    
    print(f"Test sample: {label_encoder.classes_[test_idx]}")
    
    # Get SHAP values
    shap_values = explain_prediction(explainer, X_test[0], scaler)
    
    # Get top features for predicted class
    pred_class = np.argmax(model.predict_proba(scaler.transform(X_test))[0])
    pos_feats, neg_feats = get_top_contributing_features(
        shap_values, symptom_features, pred_class, top_k=10
    )
    
    print(f"\nPredicted class: {label_encoder.classes_[pred_class]}")
    print("Top positive contributors:")
    for feat, val in pos_feats[:5]:
        print(f"  {feat}: {val:.4f}")
    print("Top negative contributors:")
    for feat, val in neg_feats[:5]:
        print(f"  {feat}: {val:.4f}")
    
    # Generate summary plot
    n_sample = min(500, len(X))
    sample_idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[sample_idx]
    
    if scaler is not None:
        X_sample_scaled = scaler.transform(X_sample)
    else:
        X_sample_scaled = X_sample
    
    generate_global_shap_summary(explainer, X_sample_scaled, symptom_features, 
                                FIGURES_DIR / "shap_summary.png")
    
    # Generate plot for a few top diseases
    for class_idx in [pred_class]:
        generate_class_shap_plot(explainer, X_sample_scaled, symptom_features, 
                                class_idx, label_encoder, 
                                FIGURES_DIR / f"shap_class_{class_idx}.png")
    
    print("="*60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
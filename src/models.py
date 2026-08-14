"""Model training with XGBoost."""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import xgboost as xgb


def load_processed_data():
    """Load processed data."""
    print("Loading processed data...")
    X = np.load(PROCESSED_DATA_DIR / "X.npy")
    y = np.load(PROCESSED_DATA_DIR / "y.npy")
    
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(SYMPTOM_VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(PROCESSED_DATA_DIR / "symptom_features.pkl", 'rb') as f:
        symptom_features = pickle.load(f)
    
    merged = pd.read_csv(PROCESSED_DISEASES_FILE)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    
    return X, y, label_encoder, vocab, symptom_features, merged


def create_augmented_data(X, y, n_augment=3, noise_level=0.05):
    """Create augmented training data."""
    print(f"Creating augmented data (n_augment={n_augment})...")
    
    X_aug = [X]
    y_aug = [y]
    
    for i in range(n_augment):
        X_noisy = X.copy()
        
        binary_mask = (X_noisy == 0) | (X_noisy == 1)
        flip_mask = np.random.random(X_noisy.shape) < noise_level
        flip_mask = flip_mask & binary_mask
        X_noisy[flip_mask] = 1 - X_noisy[flip_mask]
        
        for j in range(X.shape[1]):
            unique_vals = np.unique(X[:, j])
            if len(unique_vals) > 2:
                noise = np.random.normal(0, noise_level * X_noisy[:, j].std(), X_noisy.shape[0])
                X_noisy[:, j] = np.clip(X_noisy[:, j] + noise, 0, None)
        
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    X_aug = np.vstack(X_aug)
    y_aug = np.hstack(y_aug)
    print(f"  Augmented shape: {X_aug.shape}")
    return X_aug, y_aug


def train_test_split_samples(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Simple random split."""
    print("Splitting samples (random)...")
    
    n = len(y)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - test_n - val_n
    
    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:]
    
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    
    print(f"Train samples: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"Unique diseases in train: {len(np.unique(y[train_mask]))}")
    print(f"Unique diseases in val: {len(np.unique(y[val_mask]))}")
    print(f"Unique diseases in test: {len(np.unique(y[test_mask]))}")
    
    return train_mask, val_mask, test_mask


def evaluate_model(model, X_test, y_test, label_encoder, k_values=[1, 3, 5]):
    """Evaluate model with top-k accuracy."""
    print("Evaluating model...")
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = model.predict(X_test)
        if y_proba.ndim == 1:
            y_proba = y_proba.reshape(-1, 1)
        y_proba = softmax(y_proba, axis=1)
    
    # Handle XGBoost label mapping
    if hasattr(model, 'inv_label_map_'):
        inv_label_map = model.inv_label_map_
        model_classes = np.array([inv_label_map[i] for i in range(len(inv_label_map))])
    elif hasattr(model, 'classes_'):
        model_classes = model.classes_
    else:
        model_classes = label_encoder.classes_
    
    class_to_idx = {cls: idx for idx, cls in enumerate(model_classes)}
    y_test_mapped = np.array([class_to_idx.get(cls, -1) for cls in y_test])
    
    valid_mask = y_test_mapped >= 0
    if not valid_mask.any():
        print("  No test samples match training classes!")
        return {f'top_{k}_accuracy': 0.0 for k in k_values} | {
            'accuracy': 0, 'precision_macro': 0, 'recall_macro': 0, 'f1_macro': 0,
            'precision_weighted': 0, 'recall_weighted': 0, 'f1_weighted': 0
        }, y_proba
    
    y_test_valid = y_test_mapped[valid_mask]
    y_proba_valid = y_proba[valid_mask]
    
    labels = np.arange(len(model_classes))
    
    results = {}
    for k in k_values:
        if k <= y_proba_valid.shape[1]:
            top_k_acc = top_k_accuracy_score(y_test_valid, y_proba_valid, k=k, labels=labels)
            results[f'top_{k}_accuracy'] = top_k_acc
            print(f"  Top-{k} Accuracy: {top_k_acc:.4f}")
    
    y_pred = np.argmax(y_proba_valid, axis=1)
    results['accuracy'] = accuracy_score(y_test_valid, y_pred)
    results['precision_macro'] = precision_score(y_test_valid, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_test_valid, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_test_valid, y_pred, average='macro', zero_division=0)
    results['precision_weighted'] = precision_score(y_test_valid, y_pred, average='weighted', zero_division=0)
    results['recall_weighted'] = recall_score(y_test_valid, y_pred, average='weighted', zero_division=0)
    results['f1_weighted'] = f1_score(y_test_valid, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['f1_macro']:.4f}")
    print(f"  Weighted F1: {results['f1_weighted']:.4f}")
    print(f"  Valid test samples: {valid_mask.sum()} / {len(y_test)}")
    
    return results, y_proba


def train_xgboost(X_train, y_train, X_val=None, y_val=None, num_classes=None):
    """Train XGBoost model."""
    print("Training XGBoost...")
    
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Remap labels to contiguous 0...n_classes-1 for XGBoost
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[label] for label in y_train])
    
    eval_set = None
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        y_val_mapped = np.array([label_map.get(label, -1) for label in y_val])
        valid_mask = y_val_mapped >= 0
        if valid_mask.any():
            eval_set = [(X_val_scaled[valid_mask], y_val_mapped[valid_mask])]
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(unique_labels),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist',
        enable_categorical=False,
        early_stopping_rounds=5
    )
    
    model.fit(X_train_scaled, y_train_mapped, eval_set=eval_set, verbose=False)
    
    # Store label mapping for prediction
    model.label_map_ = label_map
    model.inv_label_map_ = {v: k for k, v in label_map.items()}
    
    return model, scaler


def save_model(model, scaler, label_encoder, model_name):
    """Save trained model."""
    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }, model_path)
    print(f"  Saved model to {model_path}")


def main():
    print("="*60)
    print("XGBOOST MODEL TRAINING")
    print("="*60)
    
    X, y, label_encoder, vocab, symptom_features, merged = load_processed_data()
    
    X_aug, y_aug = create_augmented_data(X, y, n_augment=3, noise_level=0.05)
    
    train_mask, val_mask, test_mask = train_test_split_samples(
        X_aug, y_aug, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    
    X_train, y_train = X_aug[train_mask], y_aug[train_mask]
    X_val, y_val = X_aug[val_mask], y_aug[val_mask]
    X_test, y_test = X_aug[test_mask], y_aug[test_mask]
    
    num_classes = len(label_encoder.classes_)
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    model, scaler = train_xgboost(X_train, y_train, X_val, y_val)
    results, _ = evaluate_model(model, X_test, y_test, label_encoder)
    save_model(model, scaler, label_encoder, 'xgboost')
    save_model(model, scaler, label_encoder, 'best_model')
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\nResults saved to {RESULTS_DIR / 'model_comparison.csv'}")
    print(results_df.to_string())
    
    print("="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
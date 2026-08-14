"""Hierarchical Model Training - Stage 1: Group/Type, Stage 2: Disease within Group+Type."""

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import softmax
import xgboost as xgb


class HierarchicalModelWrapper:
    """Wrapper for hierarchical model that can be pickled."""
    
    def __init__(self, group_model, type_model, disease_models, 
                 group_le, type_le, disease_le, scaler_group, scaler_type):
        self.group_model = group_model
        self.type_model = type_model
        self.disease_models = disease_models
        self.group_le = group_le
        self.type_le = type_le
        self.disease_le = disease_le
        self.scaler_group = scaler_group
        self.scaler_type = scaler_type
    
    def predict_proba(self, X):
        return hierarchical_predict(
            self.group_model, self.type_model, self.disease_models,
            X, self.group_le, self.type_le, None, self.disease_le,
            self.scaler_group, self.scaler_type, self.scaler_group
        )[0]
    
    @property
    def classes_(self):
        return self.disease_le.classes_


def load_processed_data():
    """Load processed data with hierarchical labels."""
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
    
    # Create hierarchical labels
    group_le = LabelEncoder()
    type_le = LabelEncoder()
    
    y_group = group_le.fit_transform(merged['DisorderGroup'].values)
    y_type = type_le.fit_transform(merged['DisorderType'].values)
    
    # Combined group+type label
    y_group_type = np.array([f"{g}_{t}" for g, t in zip(y_group, y_type)])
    group_type_le = LabelEncoder()
    y_group_type_encoded = group_type_le.fit_transform(y_group_type)
    
    print(f"  X shape: {X.shape}")
    print(f"  y (disease) shape: {y.shape}")
    print(f"  y_group classes: {len(group_le.classes_)} - {group_le.classes_}")
    print(f"  y_type classes: {len(type_le.classes_)} - {type_le.classes_}")
    print(f"  y_group_type classes: {len(group_type_le.classes_)}")
    
    return X, y, label_encoder, vocab, symptom_features, merged, \
           y_group, group_le, y_type, type_le, y_group_type_encoded, group_type_le


def create_augmented_data(X, y_dict, n_augment=3, noise_level=0.05):
    """Create augmented training data for all label types."""
    print(f"Creating augmented data (n_augment={n_augment})...")
    
    X_aug = [X]
    y_aug_dict = {k: [v] for k, v in y_dict.items()}
    
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
        for k in y_dict:
            y_aug_dict[k].append(y_dict[k])
    
    X_aug = np.vstack(X_aug)
    y_aug_dict = {k: np.hstack(v) for k, v in y_aug_dict.items()}
    print(f"  Augmented shape: {X_aug.shape}")
    return X_aug, y_aug_dict


def train_test_split_samples(X, y_dict, test_size=0.2, val_size=0.1, random_state=42):
    """Simple random split with same indices for all labels."""
    print("Splitting samples (random)...")
    
    n = len(list(y_dict.values())[0])
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
    for k, v in y_dict.items():
        print(f"  {k} unique in train: {len(np.unique(v[train_mask]))}")
    
    return train_mask, val_mask, test_mask


def train_xgboost(X_train, y_train, X_val=None, y_val=None, model_name="model"):
    """Train XGBoost model with label remapping."""
    print(f"Training XGBoost ({model_name})...")
    
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    
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
    
    n_classes = len(unique_labels)
    model = xgb.XGBClassifier(
        n_estimators=100 if n_classes > 20 else 50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss',
        tree_method='hist',
        enable_categorical=False,
        early_stopping_rounds=10
    )
    
    model.fit(X_train_scaled, y_train_mapped, eval_set=eval_set, verbose=False)
    
    model.label_map_ = label_map
    model.inv_label_map_ = {v: k for k, v in label_map.items()}
    
    return model, scaler


def evaluate_model(model, X_test, y_test, label_encoder, k_values=[1, 3, 5]):
    """Evaluate model with top-k accuracy."""
    print("Evaluating model...")
    
    y_proba = model.predict_proba(X_test)
    
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


def save_model(model, scaler, label_encoder, model_name):
    """Save trained model."""
    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }, model_path)
    print(f"  Saved model to {model_path}")


def hierarchical_predict(group_model, type_model, disease_models, 
                         X, group_le, type_le, group_type_le, disease_le,
                         scaler_group, scaler_type, scaler_disease):
    """
    Hierarchical prediction:
    1. Predict group
    2. Predict type
    3. Use group+type to select disease model
    """
    X_group_scaled = scaler_group.transform(X)
    X_type_scaled = scaler_type.transform(X)
    
    group_proba = group_model.predict_proba(X_group_scaled)
    type_proba = type_model.predict_proba(X_type_scaled)
    
    group_preds = np.argmax(group_proba, axis=1)
    type_preds = np.argmax(type_proba, axis=1)
    
    group_labels = [group_le.inverse_transform([g])[0] for g in group_preds]
    type_labels = [type_le.inverse_transform([t])[0] for t in type_preds]
    
    # For each sample, get disease predictions from appropriate model
    n_samples = X.shape[0]
    disease_proba_full = np.zeros((n_samples, len(disease_le.classes_)))
    
    for i in range(n_samples):
        gt_key = f"{group_labels[i]}_{type_labels[i]}"
        if gt_key in disease_models:
            model, scaler, gt_disease_le = disease_models[gt_key]
            X_scaled = scaler.transform(X[i:i+1])
            proba = model.predict_proba(X_scaled)[0]
            
            # Map back to global disease indices using per-group-type label encoder
            model_classes = [model.inv_label_map_[j] for j in range(len(model.inv_label_map_))]
            for j, cls in enumerate(model_classes):
                # cls is the local class index, map to global via gt_disease_le
                global_disease = gt_disease_le.inverse_transform([cls])[0]
                if global_disease in disease_le.classes_:
                    global_idx = np.where(disease_le.classes_ == global_disease)[0][0]
                    disease_proba_full[i, global_idx] = proba[j]
    
    return disease_proba_full, group_proba, type_proba


def main():
    print("="*60)
    print("HIERARCHICAL XGBOOST MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y, disease_le, vocab, symptom_features, merged, \
    y_group, group_le, y_type, type_le, y_group_type, group_type_le = load_processed_data()
    
    # Prepare label dictionaries
    y_dict = {
        'disease': y,
        'group': y_group,
        'type': y_type,
        'group_type': y_group_type
    }
    
    # Augment
    X_aug, y_aug_dict = create_augmented_data(X, y_dict, n_augment=3, noise_level=0.05)
    
    # Split
    train_mask, val_mask, test_mask = train_test_split_samples(
        X_aug, y_aug_dict, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    
    X_train, X_val, X_test = X_aug[train_mask], X_aug[val_mask], X_aug[test_mask]
    
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ========== STAGE 1: Group Classifier ==========
    print("\n" + "="*50)
    print("STAGE 1: DISORDER GROUP CLASSIFIER (3 classes)")
    print("="*50)
    group_model, scaler_group = train_xgboost(
        X_train, y_aug_dict['group'][train_mask],
        X_val, y_aug_dict['group'][val_mask],
        "group_classifier"
    )
    group_results, _ = evaluate_model(group_model, X_test, y_aug_dict['group'][test_mask], group_le)
    save_model(group_model, scaler_group, group_le, 'group_classifier')
    
    # ========== STAGE 1: Type Classifier ==========
    print("\n" + "="*50)
    print("STAGE 1: DISORDER TYPE CLASSIFIER (11 classes)")
    print("="*50)
    type_model, scaler_type = train_xgboost(
        X_train, y_aug_dict['type'][train_mask],
        X_val, y_aug_dict['type'][val_mask],
        "type_classifier"
    )
    type_results, _ = evaluate_model(type_model, X_test, y_aug_dict['type'][test_mask], type_le)
    save_model(type_model, scaler_type, type_le, 'type_classifier')
    
    # ========== STAGE 2: Disease Classifiers per Group+Type ==========
    print("\n" + "="*50)
    print("STAGE 2: DISEASE CLASSIFIERS PER GROUP+TYPE")
    print("="*50)
    
    disease_models = {}
    disease_results = {}
    
    # Get unique group+type combinations in training data
    gt_combinations = np.unique(y_aug_dict['group_type'][train_mask])
    print(f"Found {len(gt_combinations)} group+type combinations")
    
    for gt_encoded in gt_combinations:
        gt_label = group_type_le.inverse_transform([gt_encoded])[0]
        group_idx, type_idx = map(int, gt_label.split('_'))
        group_name = group_le.inverse_transform([group_idx])[0]
        type_name = type_le.inverse_transform([type_idx])[0]
        
        # Get disease labels for this group+type
        mask_train = (y_aug_dict['group_type'][train_mask] == gt_encoded)
        mask_val = (y_aug_dict['group_type'][val_mask] == gt_encoded)
        mask_test = (y_aug_dict['group_type'][test_mask] == gt_encoded)
        
        n_train = mask_train.sum()
        n_val = mask_val.sum()
        n_test = mask_test.sum()
        
        if n_train < 10:
            print(f"  Skipping {group_name}/{type_name}: only {n_train} train samples")
            continue
        
        y_disease_train = y_aug_dict['disease'][train_mask][mask_train]
        y_disease_val = y_aug_dict['disease'][val_mask][mask_val]
        y_disease_test = y_aug_dict['disease'][test_mask][mask_test]
        
        # Fit label encoder on ALL data for this group+type (train+val+test)
        all_diseases = np.concatenate([y_disease_train, y_disease_val, y_disease_test])
        unique_diseases = np.unique(all_diseases)
        print(f"  {group_name}/{type_name}: {len(unique_diseases)} diseases, "
              f"train={n_train}, val={n_val}, test={n_test}")
        
        if len(unique_diseases) < 2:
            print(f"    Only 1 disease class, skipping")
            continue
        
        # Train disease model for this group+type
        X_train_gt = X_train[mask_train]
        X_val_gt = X_val[mask_val]
        
        # Create per-group-type label encoder
        gt_disease_le = LabelEncoder()
        gt_disease_le.fit(unique_diseases)
        
        # Remap labels for training
        y_train_mapped = gt_disease_le.transform(y_disease_train)
        y_val_mapped = gt_disease_le.transform(y_disease_val)
        y_test_mapped = gt_disease_le.transform(y_disease_test)
        
        try:
            model, scaler = train_xgboost(
                X_train_gt, y_train_mapped,
                X_val_gt, y_val_mapped,
                f"disease_{group_name}_{type_name}"
            )
            
            # Evaluate with per-group-type label encoder
            results, _ = evaluate_model(model, X_test[mask_test], y_test_mapped, gt_disease_le)
            disease_results[f"{group_name}_{type_name}"] = results
            
            # Save model with per-group-type label encoder
            save_model(model, scaler, gt_disease_le, f'disease_{group_name}_{type_name}')
            
            disease_models[gt_label] = (model, scaler, gt_disease_le)
        except Exception as e:
            print(f"    ERROR training {group_name}/{type_name}: {e}")
            continue
    
    # ========== HIERARCHICAL EVALUATION ==========
    print("\n" + "="*50)
    print("HIERARCHICAL EVALUATION (Full Pipeline)")
    print("="*50)
    
    # Build group+type predictions
    gt_proba, group_proba, type_proba = hierarchical_predict(
        group_model, type_model, disease_models,
        X_test, group_le, type_le, group_type_le, disease_le,
        scaler_group, scaler_type, scaler_group
    )
    
    # Evaluate hierarchical predictions
    y_test_disease = y_aug_dict['disease'][test_mask]
    
    class HierarchicalModel:
        def __init__(self, gt_proba, disease_le):
            self.gt_proba = gt_proba
            self.classes_ = disease_le.classes_
        def predict_proba(self, X):
            return self.gt_proba
    
    eval_results, _ = evaluate_model(
        HierarchicalModel(gt_proba, disease_le),
        X_test, y_test_disease, disease_le
    )
    
    # Save combined results
    all_results = {
        'group': group_results,
        'type': type_results,
        'hierarchical': eval_results,
        'disease_models': {k: v for k, v in disease_results.items()}
    }
    
    results_df = pd.DataFrame([{
        'group_top1': group_results.get('top_1_accuracy', 0),
        'group_top3': group_results.get('top_3_accuracy', 0),
        'group_top5': group_results.get('top_5_accuracy', 0),
        'type_top1': type_results.get('top_1_accuracy', 0),
        'type_top3': type_results.get('top_3_accuracy', 0),
        'type_top5': type_results.get('top_5_accuracy', 0),
        'hierarchical_top1': eval_results.get('top_1_accuracy', 0),
        'hierarchical_top3': eval_results.get('top_3_accuracy', 0),
        'hierarchical_top5': eval_results.get('top_5_accuracy', 0),
        'n_disease_models': len(disease_models)
    }])
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\nResults saved to {RESULTS_DIR / 'model_comparison.csv'}")
    print(results_df.to_string())
    
    # Save best hierarchical model bundle
    joblib.dump({
        'group_model': group_model,
        'type_model': type_model,
        'disease_models': disease_models,
        'scaler_group': scaler_group,
        'scaler_type': scaler_type,
        'group_le': group_le,
        'type_le': type_le,
        'disease_le': disease_le,
        'group_type_le': group_type_le
    }, MODEL_DIR / 'hierarchical_model.pkl')
    
    # Also save as best_model for compatibility
    joblib.dump({
        'model': type('HierarchicalModel', (), {
            'predict_proba': lambda self, X: hierarchical_predict(
                group_model, type_model, disease_models,
                X, group_le, type_le, group_type_le, disease_le,
                scaler_group, scaler_type, scaler_group
            )[0],
            'classes_': disease_le.classes_
        })(),
        'scaler': scaler_group,  # Use group scaler as default
        'label_encoder': disease_le
    }, MODEL_DIR / 'best_model.pkl')
    
    print("="*60)
    print("HIERARCHICAL MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
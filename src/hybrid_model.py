"""Hybrid model combining ML and graph scores."""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

from src.graph_reasoning import load_graph, compute_graph_scores, map_symptoms_to_graph_nodes, get_top_k_diseases


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


def get_ml_scores(model, scaler, X, label_encoder):
    """Get ML prediction probabilities for all diseases."""
    if scaler is not None:
        X = scaler.transform(X)
    
    # Get probabilities for all classes
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
    else:
        proba = model.decision_function(X)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        from scipy.special import softmax
        proba = softmax(proba, axis=1)
    
    # Ensure output matches label_encoder classes
    if hasattr(model, 'classes_'):
        model_classes = model.classes_
        # Map to label_encoder order
        class_to_idx = {cls: i for i, cls in enumerate(model_classes)}
        full_proba = np.zeros((X.shape[0], len(label_encoder.classes_)))
        for i, cls in enumerate(label_encoder.classes_):
            if cls in class_to_idx:
                full_proba[:, i] = proba[:, class_to_idx[cls]]
        return full_proba
    
    return proba


def compute_hybrid_scores(ml_scores, graph_scores, alpha=0.5):
    """Combine ML and graph scores."""
    # Normalize both to [0, 1]
    ml_norm = ml_scores.copy()
    if ml_norm.max() > 0:
        ml_norm = ml_norm / ml_norm.max()
    
    graph_norm = graph_scores.copy()
    if graph_norm.max() > 0:
        graph_norm = graph_norm / graph_norm.max()
    
    # Weighted combination
    hybrid = alpha * ml_norm + (1 - alpha) * graph_norm
    return hybrid


def evaluate_hybrid(model, scaler, label_encoder, vocab, symptom_features, X_test, y_test, G, alphas=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """Evaluate hybrid model with different alpha values."""
    from sklearn.metrics import top_k_accuracy_score, accuracy_score
    from scipy.special import softmax
    
    print("Evaluating hybrid models...")
    
    # Get ML scores for test set
    ml_scores = get_ml_scores(model, scaler, X_test, label_encoder)
    
    # Pre-compute graph scores for all test samples
    print("  Pre-computing graph scores...")
    all_graph_scores = []
    for i in range(len(X_test)):
        active_features = np.where(X_test[i] > 0)[0]
        symptom_nodes = map_symptoms_to_graph_nodes(active_features, symptom_features, vocab)
        
        if len(symptom_nodes) > 0:
            graph_scores = compute_graph_scores(G, symptom_nodes, label_encoder)
        else:
            graph_scores = np.zeros(len(label_encoder.classes_))
        all_graph_scores.append(graph_scores)
    
    all_graph_scores = np.array(all_graph_scores)
    
    results = {}
    
    for alpha in alphas:
        # Combine
        all_hybrid_scores = []
        for i in range(len(X_test)):
            hybrid_scores = compute_hybrid_scores(ml_scores[i], all_graph_scores[i], alpha)
            all_hybrid_scores.append(hybrid_scores)
        
        all_hybrid_scores = np.array(all_hybrid_scores)
        
        # Evaluate
        class_to_idx = {cls: i for i, cls in enumerate(label_encoder.classes_)}
        y_mapped = np.array([class_to_idx.get(cls, -1) for cls in y_test])
        valid = y_mapped >= 0
        
        if valid.sum() > 0:
            y_valid = y_mapped[valid]
            scores_valid = all_hybrid_scores[valid]
            
            labels = np.arange(len(label_encoder.classes_))
            
            alpha_results = {}
            for k in [1, 3, 5]:
                if k <= scores_valid.shape[1]:
                    acc = top_k_accuracy_score(y_valid, scores_valid, k=k, labels=labels)
                    alpha_results[f'top_{k}_accuracy'] = acc
            
            y_pred = np.argmax(scores_valid, axis=1)
            alpha_results['accuracy'] = accuracy_score(y_valid, y_pred)
            
            results[alpha] = alpha_results
            print(f"  Alpha={alpha}: Top-1={alpha_results.get('top_1_accuracy', 0):.4f}, "
                  f"Top-3={alpha_results.get('top_3_accuracy', 0):.4f}, "
                  f"Top-5={alpha_results.get('top_5_accuracy', 0):.4f}")
    
    return results


def recommend_diseases(symptom_indices, model, scaler, label_encoder, vocab, symptom_features, G, alpha=0.5, top_k=5):
    """Full recommendation pipeline for user symptoms."""
    # Create feature vector from symptom indices
    X_user = np.zeros(len(symptom_features))
    X_user[symptom_indices] = 1
    X_user = X_user.reshape(1, -1)
    
    # ML scores
    ml_scores = get_ml_scores(model, scaler, X_user, label_encoder)[0]
    
    # Graph scores
    symptom_nodes = map_symptoms_to_graph_nodes(symptom_indices, symptom_features, vocab)
    if len(symptom_nodes) > 0:
        graph_scores = compute_graph_scores(G, symptom_nodes, label_encoder)
    else:
        graph_scores = np.zeros(len(label_encoder.classes_))
    
    # Hybrid scores
    hybrid_scores = compute_hybrid_scores(ml_scores, graph_scores, alpha)
    
    # Get top-k
    top_diseases = get_top_k_diseases(hybrid_scores, label_encoder, k=top_k)
    
    # Add individual scores
    for d in top_diseases:
        idx = np.where(label_encoder.classes_ == d['orpha_code'])[0][0]
        d['ml_score'] = float(ml_scores[idx])
        d['graph_score'] = float(graph_scores[idx])
        d['hybrid_score'] = d['score']
    
    return top_diseases, symptom_nodes


def main():
    print("="*60)
    print("HYBRID MODEL EVALUATION")
    print("="*60)
    
    # Load everything
    model, scaler, label_encoder = load_model()
    X, y, _, vocab, symptom_features = load_processed_data()
    G = load_graph()
    
    # Use a subset for evaluation (too slow on full test set)
    n_test = min(500, len(X))
    indices = np.random.choice(len(X), n_test, replace=False)
    X_test = X[indices]
    y_test = y[indices]
    
    print(f"Evaluating on {n_test} test samples...")
    
    # Evaluate different alphas
    results = evaluate_hybrid(model, scaler, label_encoder, vocab, symptom_features, 
                             X_test, y_test, G)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(RESULTS_DIR / "hybrid_evaluation.csv")
    print(f"\nResults saved to {RESULTS_DIR / 'hybrid_evaluation.csv'}")
    print(results_df.to_string())
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda a: results[a].get('top_3_accuracy', 0))
    print(f"\nBest alpha: {best_alpha}")
    
    print("="*60)
    print("HYBRID EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from scipy.special import softmax
from src.config import *
from sklearn.preprocessing import LabelEncoder

# Load data
merged = pd.read_csv(PROCESSED_DISEASES_FILE)
X = np.load(PROCESSED_DATA_DIR / 'X.npy')
y = np.load(PROCESSED_DATA_DIR / 'y.npy')

with open(LABEL_ENCODER_FILE, 'rb') as f:
    disease_le = joblib.load(f)

# Load models
group_data = joblib.load('models/group_classifier.pkl')
type_data = joblib.load('models/type_classifier.pkl')
group_model = group_data['model']
group_scaler = group_data['scaler']
group_le = group_data['label_encoder']
type_model = type_data['model']
type_scaler = type_data['scaler']
type_le = type_data['label_encoder']

# Load disease models
disease_models = {}
for fname in ['disease_Disorder_Biological anomaly.pkl', 'disease_Disorder_Clinical syndrome.pkl', 
              'disease_Disorder_Disease.pkl', 'disease_Disorder_Particular clinical situation in a disease or syndrome.pkl']:
    data = joblib.load('models/' + fname)
    gt_key = fname.replace('disease_', '').replace('.pkl', '')
    disease_models[gt_key] = (data['model'], data['scaler'], data['label_encoder'])

# Hierarchical predict function
def hierarchical_predict(group_model, type_model, disease_models, 
                         X, group_le, type_le, group_type_le, disease_le,
                         scaler_group, scaler_type, scaler_disease):
    X_group_scaled = scaler_group.transform(X)
    X_type_scaled = scaler_type.transform(X)
    
    group_proba = group_model.predict_proba(X_group_scaled)
    type_proba = type_model.predict_proba(X_type_scaled)
    
    group_preds = np.argmax(group_proba, axis=1)
    type_preds = np.argmax(type_proba, axis=1)
    
    group_labels = [group_le.inverse_transform([g])[0] for g in group_preds]
    type_labels = [type_le.inverse_transform([t])[0] for t in type_preds]
    
    n_samples = X.shape[0]
    disease_proba_full = np.zeros((n_samples, len(disease_le.classes_)))
    
    for i in range(n_samples):
        gt_key = '{}_{}'.format(group_labels[i], type_labels[i])
        if gt_key in disease_models:
            model, scaler, gt_disease_le = disease_models[gt_key]
            X_scaled = scaler.transform(X[i:i+1])
            proba = model.predict_proba(X_scaled)[0]
            
            model_classes = [model.inv_label_map_[j] for j in range(len(model.inv_label_map_))]
            for j, cls in enumerate(model_classes):
                global_disease = gt_disease_le.inverse_transform([cls])[0]
                if global_disease in disease_le.classes_:
                    global_idx = np.where(disease_le.classes_ == global_disease)[0][0]
                    disease_proba_full[i, global_idx] = proba[j]
    
    return disease_proba_full, group_proba, type_proba

# Get predictions on test set
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
X_test = X[test_idx]
y_test = y[test_idx]

# Evaluate group
X_test_group = group_scaler.transform(X_test)
group_proba = group_model.predict_proba(X_test_group)
group_preds = np.argmax(group_proba, axis=1)
group_true = [group_le.transform([merged.iloc[i]['DisorderGroup']])[0] for i in test_idx]
group_acc = accuracy_score(group_true, group_preds)
print('Group Accuracy: {:.4f}'.format(group_acc))

# Evaluate type
X_test_type = type_scaler.transform(X_test)
type_proba = type_model.predict_proba(X_test_type)
type_preds = np.argmax(type_proba, axis=1)
type_true = [type_le.transform([merged.iloc[i]['DisorderType']])[0] for i in test_idx]
type_acc = accuracy_score(type_true, type_preds)
print('Type Accuracy: {:.4f}'.format(type_acc))

# Evaluate hierarchical
gt_proba, _, _ = hierarchical_predict(
    group_model, type_model, disease_models,
    X_test, group_le, type_le, None, disease_le,
    group_scaler, type_scaler, group_scaler
)

# Top-k evaluation
def eval_topk(y_true, y_proba, k_values=[1,3,5]):
    results = {}
    for k in k_values:
        if k <= y_proba.shape[1]:
            try:
                acc = top_k_accuracy_score(y_true, y_proba, k=k)
                results['top_{}_accuracy'.format(k)] = acc
            except:
                results['top_{}_accuracy'.format(k)] = 0.0
    return results

hier_results = eval_topk(y_test, gt_proba)
print('Hierarchical Top-1: {:.4f}'.format(hier_results.get('top_1_accuracy', 0)))
print('Hierarchical Top-3: {:.4f}'.format(hier_results.get('top_3_accuracy', 0)))
print('Hierarchical Top-5: {:.4f}'.format(hier_results.get('top_5_accuracy', 0)))

# Save results
results_df = pd.DataFrame([{
    'group_top1': group_acc, 'group_top3': group_acc, 'group_top5': group_acc,
    'type_top1': type_acc, 'type_top3': type_acc, 'type_top5': type_acc,
    'hierarchical_top1': hier_results.get('top_1_accuracy', 0),
    'hierarchical_top3': hier_results.get('top_3_accuracy', 0),
    'hierarchical_top5': hier_results.get('top_5_accuracy', 0),
    'n_disease_models': len(disease_models)
}])
results_df.to_csv('reports/results/model_comparison.csv', index=False)
print('Saved model_comparison.csv')
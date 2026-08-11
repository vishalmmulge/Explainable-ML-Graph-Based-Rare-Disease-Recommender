"""Unit tests for Rare Disease Recommendation System."""

import numpy as np
import pandas as pd
import pickle
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.preprocessing import DataPreprocessor
from src.graph_reasoning import map_symptoms_to_graph_nodes, compute_graph_scores, get_top_k_diseases
from src.hybrid_model import load_model, load_processed_data, recommend_diseases, compute_hybrid_scores


class TestPreprocessing:
    """Test data preprocessing functions."""
    
    def test_load_raw_data(self):
        """Test loading all raw data files."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        assert preprocessor.complete is not None
        assert preprocessor.genes is not None
        assert preprocessor.natural is not None
        assert preprocessor.prevalence is not None
        
        assert len(preprocessor.complete) == 11456
        assert len(preprocessor.genes) == 8374
        assert len(preprocessor.natural) == 7374
        assert len(preprocessor.prevalence) == 16657
    
    def test_clean_complete(self):
        """Test cleaning complete dataset."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_complete()
        
        # Check columns dropped
        assert 'MeSH' not in preprocessor.complete.columns
        assert 'MedDRA' not in preprocessor.complete.columns
        
        # Check DiseaseName filled
        assert preprocessor.complete['DiseaseName'].notna().all()
    
    def test_create_gene_features(self):
        """Test gene feature creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_genes()
        
        gene_features = preprocessor.create_gene_features()
        
        assert 'gene_count' in gene_features.columns
        assert 'gene_symbols' in gene_features.columns
        assert len(gene_features) == preprocessor.genes['OrphaCode'].nunique()
        
        # Check top gene binary features
        gene_cols = [c for c in gene_features.columns if c.startswith('gene_') and c not in ['gene_count', 'gene_symbols', 'gene_names', 'association_types']]
        assert len(gene_cols) == 100  # Top 100 genes
    
    def test_create_onset_features(self):
        """Test onset feature creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_natural_history()
        
        onset_features = preprocessor.create_onset_features()
        
        onset_cols = [c for c in onset_features.columns if c.startswith('onset_')]
        assert len(onset_cols) == 8
        
        # Check binary values
        for col in onset_cols:
            assert set(onset_features[col].unique()).issubset({0, 1})
    
    def test_create_inheritance_features(self):
        """Test inheritance feature creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_natural_history()
        
        inherit_features = preprocessor.create_inheritance_features()
        
        inherit_cols = [c for c in inherit_features.columns if c.startswith('inherit_')]
        assert len(inherit_cols) == 11
    
    def test_create_prevalence_features(self):
        """Test prevalence feature creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_prevalence()
        
        prev_features = preprocessor.create_prevalence_features()
        
        assert 'prevalence_score' in prev_features.columns
        assert 'prevalence_class' in prev_features.columns
        assert len(prev_features) > 0
    
    def test_create_disease_type_features(self):
        """Test disease type feature creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_complete()
        
        type_features = preprocessor.create_disease_type_features()
        
        type_cols = [c for c in type_features.columns if c.startswith('type_')]
        group_cols = [c for c in type_features.columns if c.startswith('group_')]
        assert len(type_cols) == 11
        assert len(group_cols) == 3


class TestEncoding:
    """Test feature encoding."""
    
    def test_symptom_vocabulary(self):
        """Test vocabulary creation."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_complete()
        preprocessor.clean_genes()
        preprocessor.clean_natural_history()
        preprocessor.clean_prevalence()
        preprocessor.merge_all_features()
        
        vocab, symptom_features = preprocessor.create_symptom_vocabulary()
        
        assert len(vocab) == len(symptom_features)
        assert all(isinstance(v, int) for v in vocab.values())
        assert all(v >= 0 for v in vocab.values())
    
    def test_encode_features(self):
        """Test feature encoding."""
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        preprocessor.clean_complete()
        preprocessor.clean_genes()
        preprocessor.clean_natural_history()
        preprocessor.clean_prevalence()
        preprocessor.merge_all_features()
        
        vocab, symptom_features = preprocessor.create_symptom_vocabulary()
        X, y, label_encoder = preprocessor.encode_features(symptom_features)
        
        assert X.shape[0] == 11456
        assert X.shape[1] == len(symptom_features)
        assert len(label_encoder.classes_) == 11456
        assert y.min() == 0
        assert y.max() == 11455


class TestGraph:
    """Test knowledge graph functions."""
    
    @pytest.fixture(scope="class")
    def graph_resources(self):
        """Load graph and data once for all tests."""
        import joblib
        G = joblib.load(GRAPH_FILE)
        X, y, label_encoder, vocab, symptom_features = load_processed_data()
        return G, X, y, label_encoder, vocab, symptom_features
    
    def test_map_symptoms_to_graph_nodes(self, graph_resources):
        """Test symptom to graph node mapping."""
        G, X, y, label_encoder, vocab, symptom_features = graph_resources
        
        # Test with some active features
        active_indices = [i for i, f in enumerate(symptom_features) 
                         if f.startswith('onset_') or f.startswith('inherit_')][:5]
        
        graph_nodes = map_symptoms_to_graph_nodes(active_indices, symptom_features, vocab)
        
        assert isinstance(graph_nodes, list)
        assert len(graph_nodes) > 0
        assert all(isinstance(n, str) for n in graph_nodes)
    
    def test_compute_graph_scores(self, graph_resources):
        """Test graph score computation."""
        G, X, y, label_encoder, vocab, symptom_features = graph_resources
        
        symptom_nodes = ['onset_neonatal', 'inherit_autosomal_recessive', 'type_disease']
        scores = compute_graph_scores(G, symptom_nodes, label_encoder)
        
        assert scores.shape == (len(label_encoder.classes_),)
        assert scores.min() >= 0
        assert scores.max() <= 1
    
    def test_get_top_k_diseases(self, graph_resources):
        """Test top-k disease retrieval."""
        G, X, y, label_encoder, vocab, symptom_features = graph_resources
        
        scores = np.random.rand(len(label_encoder.classes_))
        top5 = get_top_k_diseases(scores, label_encoder, k=5)
        
        assert len(top5) == 5
        assert all('rank' in r and 'orpha_code' in r and 'score' in r for r in top5)
        assert top5[0]['rank'] == 1
        assert top5[0]['score'] >= top5[1]['score']


class TestRanking:
    """Test ranking functions."""
    
    def test_compute_hybrid_scores(self):
        """Test hybrid score computation."""
        ml_scores = np.array([0.8, 0.6, 0.4, 0.2])
        graph_scores = np.array([0.3, 0.7, 0.5, 0.9])
        
        hybrid = compute_hybrid_scores(ml_scores, graph_scores, alpha=0.5)
        
        assert hybrid.shape == ml_scores.shape
        assert hybrid.min() >= 0
        assert hybrid.max() <= 1
        
        # Test extreme alphas
        hybrid_ml = compute_hybrid_scores(ml_scores, graph_scores, alpha=1.0)
        hybrid_graph = compute_hybrid_scores(ml_scores, graph_scores, alpha=0.0)
        
        np.testing.assert_array_almost_equal(hybrid_ml / hybrid_ml.max(), ml_scores / ml_scores.max())
        np.testing.assert_array_almost_equal(hybrid_graph / hybrid_graph.max(), graph_scores / graph_scores.max())


class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    def test_recommend_diseases(self):
        """Test full recommendation pipeline."""
        model, scaler, label_encoder = load_model()
        X, y, _, vocab, symptom_features = load_processed_data()
        import joblib
        G = joblib.load(GRAPH_FILE)
        
        # Select some features
        selected_indices = [i for i, f in enumerate(symptom_features) 
                           if f.startswith('onset_')][:3]
        
        recommendations, symptom_nodes = recommend_diseases(
            selected_indices, model, scaler, label_encoder,
            vocab, symptom_features, G, alpha=0.5, top_k=3
        )
        
        assert len(recommendations) == 3
        assert all('rank' in r for r in recommendations)
        assert all('ml_score' in r and 'graph_score' in r and 'hybrid_score' in r for r in recommendations)
        
        # Check scores are in valid range
        for r in recommendations:
            assert 0 <= r['ml_score'] <= 1
            assert 0 <= r['graph_score'] <= 1
            assert 0 <= r['hybrid_score'] <= 1
    
    def test_empty_symptoms(self):
        """Test handling of empty symptom list."""
        model, scaler, label_encoder = load_model()
        X, y, _, vocab, symptom_features = load_processed_data()
        import joblib
        G = joblib.load(GRAPH_FILE)
        
        recommendations, symptom_nodes = recommend_diseases(
            [], model, scaler, label_encoder,
            vocab, symptom_features, G, alpha=0.5, top_k=3
        )
        
        # Should still return results (based on ML only)
        assert len(recommendations) == 3
    
    def test_model_loading(self):
        """Test model loading."""
        model, scaler, label_encoder = load_model()
        
        assert model is not None
        assert label_encoder is not None
        assert len(label_encoder.classes_) > 0


class TestDataValidation:
    """Data validation tests."""
    
    def test_processed_data_exists(self):
        """Test that processed data files exist."""
        assert (PROCESSED_DATA_DIR / "X.npy").exists()
        assert (PROCESSED_DATA_DIR / "y.npy").exists()
        assert LABEL_ENCODER_FILE.exists()
        assert SYMPTOM_VOCAB_FILE.exists()
        assert (PROCESSED_DATA_DIR / "symptom_features.pkl").exists()
        assert PROCESSED_DISEASES_FILE.exists()
    
    def test_model_exists(self):
        """Test that model files exist."""
        assert (MODEL_DIR / "best_model.pkl").exists()
        assert (MODEL_DIR / "logistic_regression.pkl").exists()
    
    def test_graph_exists(self):
        """Test that graph file exists."""
        assert GRAPH_FILE.exists()
    
    def test_feature_count(self):
        """Test feature count matches expectation."""
        with open(PROCESSED_DATA_DIR / "symptom_features.pkl", 'rb') as f:
            symptom_features = pickle.load(f)
        assert len(symptom_features) == 135
    
    def test_class_count(self):
        """Test class count."""
        with open(LABEL_ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        assert len(label_encoder.classes_) == 11456


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Configuration module for the Rare Disease Recommendation System."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
RESULTS_DIR = REPORTS_DIR / "results"

# Random state for reproducibility
RANDOM_STATE = 42

# Top-K recommendations
TOP_K = 5

# Hybrid model weights
ML_WEIGHT = 0.5
GRAPH_WEIGHT = 0.5

# Test split
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Data files
COMPLETE_FILE = RAW_DATA_DIR / "rare_diseases_complete.csv"
GENES_FILE = RAW_DATA_DIR / "rare_diseases_genes.csv"
NATURAL_HISTORY_FILE = RAW_DATA_DIR / "rare_diseases_natural_history.csv"
PREVALENCE_FILE = RAW_DATA_DIR / "rare_diseases_prevalence.csv"

# Processed files
PROCESSED_DISEASES_FILE = PROCESSED_DATA_DIR / "diseases_processed.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "features.csv"
PROCESSED_LABELS_FILE = PROCESSED_DATA_DIR / "labels.csv"
SYMPTOM_VOCAB_FILE = PROCESSED_DATA_DIR / "symptom_vocab.pkl"
LABEL_ENCODER_FILE = PROCESSED_DATA_DIR / "label_encoder.pkl"
GRAPH_FILE = PROCESSED_DATA_DIR / "knowledge_graph.gpickle"
MODEL_FILE = MODEL_DIR / "best_model.pkl"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 MODEL_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
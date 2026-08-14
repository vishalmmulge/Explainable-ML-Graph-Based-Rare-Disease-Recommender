# Explainable Graph-Based Rare Disease Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61+-red.svg)](https://streamlit.io)

A research prototype for rare disease recommendation combining machine learning with biomedical knowledge graph reasoning and explainable AI.

## 🎯 Overview

This system takes patient symptoms/phenotypes as input and produces:
- **Top-K rare disease recommendations** with calibrated confidence scores
- **Dual explanations**: SHAP-based ML feature importance + knowledge graph relational evidence
- **Interactive dashboard** for exploration and evaluation

> ⚠️ **Medical Disclaimer**: This system is a research and educational prototype and is not intended to provide medical diagnosis or treatment. Predictions should not be used as a substitute for professional medical evaluation.

## 🏗️ Architecture

```
Symptoms → Multi-hot Encoding → ML Model (XGBoost)
                ↓
        Knowledge Graph (NetworkX)
                ↓
        Hybrid Score = α × ML + (1-α) × Graph
                ↓
        Top-K Recommendations + Explanations
```

## 📊 Dataset

**Primary Dataset**: [Rare Diseases Orphadata 2026](https://www.kaggle.com/datasets/ahsanneural/rare-diseases-orphadata-2026/data) (Kaggle)

| Component | Records | Description |
|-----------|---------|-------------|
| Diseases | 11,456 | OrphaCode, name, type, group, identifiers |
| Gene Associations | 8,374 | Disease-gene links with association types |
| Natural History | 7,374 | Age of onset, inheritance patterns |
| Prevalence | 16,657 | Prevalence classes by geography |

**PrimeKG Integration** (optional): [PrimeKG](https://github.com/mims-harvard/PrimeKG) - Precision Medicine Knowledge Graph with 17,080 diseases, 100K+ nodes, 4M+ relationships

**Knowledge Graph**: 15,954 nodes, 57,951 edges
- Disease nodes: 11,456
- Gene nodes: 4,458
- Onset categories: 8
- Inheritance patterns: 11
- Prevalence classes: 7
- Disorder types: 11
- Disorder groups: 3

## 🚀 Quick Start

### Installation

```powershell
# Clone and navigate
cd rare-disease-recommender

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run Pipeline

```powershell
# 1. Data profiling
python scripts/profile_data.py

# 2. Preprocessing
python src/preprocessing.py

# 3. Model training (XGBoost)
python src/models.py

# 4. Knowledge graph construction
python src/knowledge_graph.py

# 5. Graph reasoning test
python src/graph_reasoning.py

# 6. Hybrid model evaluation
python src/hybrid_model.py

# 7. SHAP explainability
python src/explainability.py

# 8. (Optional) Process PrimeKG data
python scripts/process_primekg.py

# 9. Launch dashboard
streamlit run app/app.py
```

### Run Tests

```powershell
pytest tests/test_pipeline.py -v
```

## 📁 Project Structure

```
rare-disease-recommender/
│
├── data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Processed data, models, graph
│   └── README.md
│
├── notebooks/                  # Jupyter notebooks for exploration
│
├── src/
│   ├── config.py              # Configuration and paths
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── models.py              # ML model training and evaluation
│   ├── knowledge_graph.py     # NetworkX graph construction
│   ├── graph_reasoning.py     # Graph-based scoring
│   ├── hybrid_model.py        # ML + Graph combination
│   ├── explainability.py      # SHAP explanations
│   └── evaluation.py          # Evaluation utilities
│
├── app/
│   └── app.py                 # Streamlit dashboard
│
├── models/                    # Trained model artifacts
│
├── reports/
│   ├── figures/               # Generated plots
│   ├── tables/                # Result tables
│   └── results/               # Evaluation CSVs
│
├── tests/
│   └── test_pipeline.py       # Unit and integration tests
│
├── scripts/
│   ├── profile_data.py        # Data profiling script
│   ├── train.py               # Training script
│   ├── build_graph.py         # Graph construction script
│   └── evaluate.py            # Evaluation script
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🔬 Methodology

### Feature Engineering
Since the dataset lacks explicit HPO phenotype annotations, we derive "symptom-like" features from available biomedical data:

1. **Gene Features**: Binary indicators for top 100 most frequent disease-associated genes
2. **Onset Features**: 8 binary features for age of onset categories (Antenatal, Neonatal, Infancy, Childhood, Adolescent, Adult, Elderly, All ages)
3. **Inheritance Features**: 11 binary features for inheritance patterns
4. **Disease Type/Group**: One-hot encoded disorder types (11) and groups (3)
5. **Prevalence**: Ordinal prevalence score

Total: **135 features** (96.6% sparse)

### Machine Learning
- **Model**: XGBoost (gradient boosting) with 50 estimators, max_depth=6
- **Training**: Disease-level augmented data (3 augmentations with 5% noise)
- **Split**: Random sample split (70/15/15) with disease stratification
- **Early Stopping**: 5 rounds on validation set

**Baseline Results** (on test samples):
| Metric | Score |
|--------|-------|
| Top-1 Accuracy | TBD |
| Top-3 Accuracy | TBD |
| Top-5 Accuracy | TBD |
| Macro F1 | TBD |

### Knowledge Graph
Built with NetworkX `MultiDiGraph` preserving edge types:
- Disease → Gene (HAS_GENE)
- Disease → Onset (HAS_ONSET)
- Disease → Inheritance (HAS_INHERITANCE)
- Disease → Prevalence (HAS_PREVALENCE)
- Disease → Type (HAS_TYPE)
- Disease → Group (HAS_GROUP)

### Graph Reasoning
For a set of input symptoms mapped to graph nodes:
1. Direct connections: Count edges from disease to symptom nodes
2. 2-hop connections: Shared neighbors between disease and symptoms
3. Normalized by number of input symptoms

### Hybrid Scoring
```
HybridScore = α × MLScore + (1-α) × GraphScore
```
Evaluated α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

### Explainability
1. **SHAP (ML)**: TreeExplainer (XGBoost) / LinearExplainer (Logistic Regression) for feature contribution
2. **Graph (Relational)**: Subgraph visualization showing disease-feature connections

## 📈 Results

### Model Comparison
| Model | Top-1 | Top-3 | Top-5 | Macro F1 |
|-------|-------|-------|-------|----------|
| Logistic Regression | 1.21% | 3.09% | 5.03% | 0.41% |
| XGBoost | TBD | TBD | TBD | TBD |

### Hybrid Weight Sensitivity
The optimal α depends on the evaluation metric. Graph-only (α=0) provides complementary signals to ML-only (α=1).

### Key Findings
1. **Sparse features** (135 dimensions, 96.6% zeros) limit ML performance
2. **Graph reasoning** provides orthogonal evidence from biomedical relationships
3. **Hybrid approach** combines statistical patterns with domain knowledge
4. **Dual explanations** offer both model-centric and domain-centric interpretability

## 🧪 Evaluation Metrics

- **Ranking**: Top-1, Top-3, Top-5 Accuracy, Mean Reciprocal Rank
- **Classification**: Accuracy, Precision/Recall/F1 (Macro & Weighted)
- **Calibration**: Reliability diagrams, Brier score (when applicable)
- **Graph**: Node/edge counts, degree distribution, connected components

## ⚙️ Configuration

Key parameters in `src/config.py`:
```python
RANDOM_STATE = 42
TOP_K = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.1
ML_WEIGHT = 0.5
GRAPH_WEIGHT = 0.5
```

## 📝 Limitations

1. **No explicit phenotypes**: Uses proxy features (genes, onset, inheritance) instead of HPO terms
2. **Single sample per disease**: Requires augmentation for training
3. **Class imbalance**: 11,456 classes with extreme sparsity
4. **No external validation**: Evaluated only on held-out augmented samples
5. **Research prototype**: Not clinically validated

## 🔮 Future Work

- [ ] Integrate HPO phenotype annotations from Orphadata
- [ ] **Integrate PrimeKG** for expanded disease-gene-phenotype relationships
- [ ] Implement GNN for graph-based learning (PyTorch Geometric)
- [ ] Add patient-level evaluation with synthetic cohorts
- [ ] Improve feature representations (TF-IDF, embeddings)
- [ ] Cross-validation with disease-level splits
- [ ] Deploy as REST API with FastAPI

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Orphanet/Orphadata** for rare disease data
- **Kaggle** for dataset hosting
- **PrimeKG** (mims-harvard) for precision medicine knowledge graph
- **NetworkX**, **scikit-learn**, **XGBoost**, **SHAP**, **Streamlit** communities

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.
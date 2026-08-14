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
Hierarchical ML + Graph Reasoning:

Symptoms → Multi-hot Encoding → Stage 1: Group Classifier (3 classes)
                                      ↓
                                 Stage 1: Type Classifier (11 classes)
                                      ↓
                                 Stage 2: Disease Classifier (per Group+Type)
                                      ↓
                 Knowledge Graph (NetworkX, 46K nodes, 519K edges)
                 - Orphadata + PrimeKG HPO phenotypes
                 ↓
                 Hybrid Score = α × ML + (1-α) × Graph
                 ↓
                 Top-K Recommendations + Explanations
```

**Stage 1 (Coarse):** Predicts Disorder Group (3) + Type (11) → ~100% accuracy
**Stage 2 (Fine):** Predicts specific disease within Group+Type → ~15% Top-1, ~50% Top-5

## 📊 Datasets

### Primary Dataset: Rare Diseases Orphadata 2026
**Source**: [Kaggle - Rare Diseases Orphadata 2026](https://www.kaggle.com/datasets/ahsanneural/rare-diseases-orphadata-2026/data)  
**Original**: [Orphanet/Orphadata](https://www.orphadata.com/) - Rare disease knowledge base

| File | Records | Description |
|------|---------|-------------|
| `rare_diseases_complete.csv` | 11,456 | Diseases: OrphaCode, name, type, group, identifiers (ICD-10, OMIM, MONDO, UMLS, MeSH, MedDRA, GARD) |
| `rare_diseases_genes.csv` | 8,374 | Disease-gene associations with association types and status |
| `rare_diseases_natural_history.csv` | 7,374 | Age of onset, inheritance patterns |
| `rare_diseases_prevalence.csv` | 16,657 | Prevalence classes by geography (point prevalence, cases/families, incidence) |

### Knowledge Graph Dataset: PrimeKG (Precision Medicine Knowledge Graph)
**Source**: [GitHub - mims-harvard/PrimeKG](https://github.com/mims-harvard/PrimeKG)  
**Publication**: [Nature Scientific Data (2023)](https://www.nature.com/articles/s41597-023-01960-3)  
**Download**: [Harvard Dataverse - DOI:10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM)  
**Direct Download**: [kg.csv (4M+ edges)](https://dataverse.harvard.edu/api/access/datafile/6180620)

| Statistic | Value |
|-----------|-------|
| Diseases | 17,080 |
| Total Nodes | 100,000+ |
| Total Edges | 4,050,249 |
| Edge Types | 29 |
| Biological Scales | 10 |

**PrimeKG Node Types Used in This Project**:
| Node Type | Count (in KG) | Description |
|-----------|---------------|-------------|
| `disease` | 17,080 | MONDO/OMIM/OrphaCode mapped diseases |
| `gene/protein` | ~20,000 | NCBI Gene / UniProt |
| `effect/phenotype` | 15,311 | HPO (Human Phenotype Ontology) terms |
| `drug` | ~10,000 | DrugBank, DrugCentral |
| `pathway` | ~2,000 | Reactome |
| `anatomy` | ~5,000 | UBERON |
| `biological_process` | ~7,000 | GO Biological Process |
| `molecular_function` | ~4,000 | GO Molecular Function |
| `cellular_component` | ~1,000 | GO Cellular Component |
| `exposure` | ~1,000 | CTD environmental exposures |

**PrimeKG Edge Types Used**:
| Relation | Display | Count | Source |
|----------|---------|-------|--------|
| `disease_protein` | associated with | 160,822 | DisGeNET, OMIM, etc. |
| `disease_phenotype_positive` | associated with | 300,634 | HPO annotations |
| `disease_phenotype_negative` | not associated with | 2,386 | HPO negative annotations |

### Integrated Knowledge Graph (Orphadata + PrimeKG)
Built by merging Orphadata rare diseases with PrimeKG HPO phenotypes and gene associations.

| Metric | Value |
|--------|-------|
| **Total Nodes** | 46,540 |
| **Total Edges** | 519,407 |
| Disease nodes | 28,536 (Orphadata 11,456 + PrimeKG MONDO 17,080) |
| Gene/Protein nodes | 12,150 |
| HPO Phenotype nodes | 5,814 |
| Onset categories | 8 |
| Inheritance patterns | 11 |
| Prevalence classes | 7 |
| Disorder types | 11 |
| Disorder groups | 3 |

**Edge Types in Final Graph**:
| Relation Type | Count | Source |
|---------------|-------|--------|
| HAS_GENE | 168,820 | Orphadata + PrimeKG |
| HAS_PHENOTYPE | 300,634 | PrimeKG (HPO positive) |
| HAS_ONSET | 11,739 | Orphadata |
| HAS_INHERITANCE | 7,285 | Orphadata |
| HAS_PREVALENCE | 8,017 | Orphadata |
| HAS_TYPE | 11,456 | Orphadata |
| HAS_GROUP | 11,456 | Orphadata |

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

# 3. Model training (Hierarchical XGBoost)
python src/models.py

# 4. Knowledge graph construction (with PrimeKG HPO)
python src/knowledge_graph.py

# 5. Graph reasoning test
python src/graph_reasoning.py

# 6. Hybrid model evaluation
python src/hybrid_model.py

# 7. SHAP explainability (Group/Type classifiers)
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
- **Model**: Hierarchical XGBoost (2-stage)
  - Stage 1: Disorder Group Classifier (3 classes) → 100% accuracy
  - Stage 1: Disorder Type Classifier (11 classes) → 100% accuracy  
  - Stage 2: Disease Classifiers per Group+Type combination (4 models)
- **Training**: Disease-level augmented data (3 augmentations with 5% noise)
- **Split**: Random sample split (70/15/15) with disease stratification
- **Early Stopping**: 5-10 rounds on validation set

**Baseline Results** (on test samples):
| Metric | Score |
|--------|-------|
| Group Top-1 Accuracy | 100% |
| Type Top-1 Accuracy | 100% |
| Hierarchical Top-1 Accuracy | ~15% |
| Hierarchical Top-3 Accuracy | ~35% |
| Hierarchical Top-5 Accuracy | ~50% |

### Knowledge Graph
Built with NetworkX `MultiDiGraph` preserving edge types:
- Disease → Gene (HAS_GENE) - 168,820 edges (Orphadata + PrimeKG)
- Disease → Onset (HAS_ONSET) - 11,739 edges
- Disease → Inheritance (HAS_INHERITANCE) - 7,285 edges
- Disease → Prevalence (HAS_PREVALENCE) - 8,017 edges
- Disease → Type (HAS_TYPE) - 11,456 edges
- Disease → Group (HAS_GROUP) - 11,456 edges
- Disease → Phenotype (HAS_PHENOTYPE) - 300,634 edges (PrimeKG HPO)

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
1. **SHAP (ML)**: TreeExplainer for Group/Type classifiers (fast, interpretable)
2. **Graph (Relational)**: Subgraph visualization showing disease-feature connections

## 📈 Results

### Model Comparison
| Model | Top-1 | Top-3 | Top-5 | Macro F1 |
|-------|-------|-------|-------|----------|
| Logistic Regression | 1.21% | 3.09% | 5.03% | 0.41% |
| XGBoost (flat) | ~0% | ~0% | ~0% | ~0% |
| **Hierarchical XGBoost** | **~15%** | **~35%** | **~50%** | - |

| Component | Top-1 Accuracy |
|-----------|----------------|
| Disorder Group (3 classes) | 100% |
| Disorder Type (11 classes) | 100% |
| Disease | Group+Type (4 models) | ~15% |

### Hybrid Weight Sensitivity
The optimal α depends on the evaluation metric. Graph-only (α=0) provides complementary signals to ML-only (α=1). With hierarchical model, α=0.3-0.5 typically works best.

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
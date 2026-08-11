"""Data preprocessing module for the Rare Disease Recommendation System."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

class DataPreprocessor:
    """Preprocess and merge all rare disease datasets."""
    
    def __init__(self):
        self.complete = None
        self.genes = None
        self.natural = None
        self.prevalence = None
        self.merged = None
        
    def load_raw_data(self):
        """Load all raw data files."""
        print("Loading raw data...")
        self.complete = pd.read_csv(COMPLETE_FILE)
        self.genes = pd.read_csv(GENES_FILE)
        self.natural = pd.read_csv(NATURAL_HISTORY_FILE)
        self.prevalence = pd.read_csv(PREVALENCE_FILE)
        print(f"  Complete: {len(self.complete)} rows")
        print(f"  Genes: {len(self.genes)} rows")
        print(f"  Natural history: {len(self.natural)} rows")
        print(f"  Prevalence: {len(self.prevalence)} rows")
        
    def clean_complete(self):
        """Clean the complete diseases dataset."""
        print("Cleaning complete dataset...")
        df = self.complete.copy()
        
        # Drop columns with >70% missing
        missing_pct = df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > 0.7].index.tolist()
        if cols_to_drop:
            print(f"  Dropping columns with >70% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Fill missing DiseaseName with Name
        df['DiseaseName'] = df['DiseaseName'].fillna(df['Name'])
        
        # Normalize text columns
        text_cols = ['Name', 'DiseaseName', 'DisorderType', 'DisorderGroup']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        self.complete = df
        print(f"  Cleaned complete: {len(df)} rows, {len(df.columns)} columns")
        
    def clean_genes(self):
        """Clean the genes dataset."""
        print("Cleaning genes dataset...")
        df = self.genes.copy()
        
        # Keep only assessed associations
        df = df[df['AssociationStatus'] == 'Assessed'].copy()
        print(f"  After filtering assessed: {len(df)} rows")
        
        # Normalize gene symbols
        df['GeneSymbol'] = df['GeneSymbol'].astype(str).str.strip().str.upper()
        df['GeneName'] = df['GeneName'].astype(str).str.strip()
        
        self.genes = df
        
    def clean_natural_history(self):
        """Clean the natural history dataset."""
        print("Cleaning natural history dataset...")
        df = self.natural.copy()
        
        # Normalize text columns
        for col in ['AgeOfOnset', 'TypeOfInheritance']:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
        
        self.natural = df
        
    def clean_prevalence(self):
        """Clean the prevalence dataset."""
        print("Cleaning prevalence dataset...")
        df = self.prevalence.copy()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"  After removing duplicates: {len(df)} rows")
        
        # Normalize text columns
        for col in ['PrevalenceType', 'PrevalenceQualification', 'PrevalenceClass', 'PrevalenceGeographic']:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
        
        self.prevalence = df
        
    def create_gene_features(self):
        """Create gene-based features for each disease."""
        print("Creating gene features...")
        
        # Get unique genes per disease
        gene_features = self.genes.groupby('OrphaCode').agg(
            gene_count=('GeneSymbol', 'nunique'),
            gene_symbols=('GeneSymbol', lambda x: '|'.join(sorted(x.unique()))),
            gene_names=('GeneName', lambda x: '|'.join(sorted(x.unique()))),
            association_types=('AssociationType', lambda x: '|'.join(sorted(x.unique()))),
        ).reset_index()
        
        # Create binary features for top genes
        all_genes = self.genes['GeneSymbol'].value_counts()
        top_genes = all_genes.head(100).index.tolist()  # Top 100 genes
        
        for gene in top_genes:
            gene_diseases = set(self.genes[self.genes['GeneSymbol'] == gene]['OrphaCode'].unique())
            gene_features[f'gene_{gene}'] = gene_features['OrphaCode'].apply(
                lambda x: 1 if x in gene_diseases else 0
            )
        
        print(f"  Created gene features for {len(top_genes)} top genes")
        return gene_features
    
    def create_onset_features(self):
        """Create age of onset features."""
        print("Creating age of onset features...")
        
        df = self.natural[['OrphaCode', 'AgeOfOnset']].copy()
        df = df.dropna(subset=['AgeOfOnset'])
        
        # Parse onset categories
        onset_categories = [
            'Antenatal', 'Neonatal', 'Infancy', 'Childhood', 
            'Adolescent', 'Adult', 'Elderly', 'All ages'
        ]
        
        for cat in onset_categories:
            df[f'onset_{cat.lower()}'] = df['AgeOfOnset'].str.contains(cat, case=False, na=False).astype(int)
        
        # Aggregate by disease (take max since a disease can have multiple onsets)
        onset_features = df.groupby('OrphaCode').agg({
            f'onset_{cat.lower()}': 'max' for cat in onset_categories
        }).reset_index()
        
        print(f"  Created {len(onset_categories)} onset features")
        return onset_features
    
    def create_inheritance_features(self):
        """Create inheritance pattern features."""
        print("Creating inheritance features...")
        
        df = self.natural[['OrphaCode', 'TypeOfInheritance']].copy()
        df = df.dropna(subset=['TypeOfInheritance'])
        
        # Parse inheritance categories
        inherit_categories = [
            'Autosomal dominant', 'Autosomal recessive', 'X-linked dominant',
            'X-linked recessive', 'Mitochondrial inheritance', 'Multigenic/multifactorial',
            'Not applicable', 'Unknown', 'Semi-dominant', 'Oligogenic', 'Y-linked'
        ]
        
        for cat in inherit_categories:
            # Clean category name for column
            col_name = cat.lower().replace('/', '_').replace(' ', '_').replace('-', '_')
            df[f'inherit_{col_name}'] = df['TypeOfInheritance'].str.contains(
                cat.replace('(', r'\(').replace(')', r'\)'), case=False, na=False, regex=True
            ).astype(int)
        
        # Aggregate by disease
        inherit_features = df.groupby('OrphaCode').agg({
            f'inherit_{cat.lower().replace("/", "_").replace(" ", "_").replace("-", "_")}': 'max' 
            for cat in inherit_categories
        }).reset_index()
        
        print(f"  Created {len(inherit_categories)} inheritance features")
        return inherit_features
    
    def create_prevalence_features(self):
        """Create prevalence features."""
        print("Creating prevalence features...")
        
        df = self.prevalence.copy()
        
        # Use point prevalence class as primary
        point_prev = df[df['PrevalenceType'] == 'Point prevalence'].copy()
        
        # Map prevalence classes to ordinal values
        class_map = {
            '>1 / 1000': 7,
            '6-9 / 10 000': 6,
            '1-5 / 10 000': 5,
            '1-9 / 100 000': 4,
            '1-9 / 1 000 000': 3,
            '<1 / 1 000 000': 2,
            'Unknown': 1,
            'Not yet documented': 0
        }
        
        point_prev['prevalence_score'] = point_prev['PrevalenceClass'].map(class_map).fillna(0)
        
        # Take max prevalence score per disease
        prev_features = point_prev.groupby('OrphaCode').agg(
            prevalence_score=('prevalence_score', 'max'),
            prevalence_class=('PrevalenceClass', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        ).reset_index()
        
        print(f"  Created prevalence features for {len(prev_features)} diseases")
        return prev_features
    
    def create_disease_type_features(self):
        """Create disease type/category features."""
        print("Creating disease type features...")
        
        df = self.complete[['OrphaCode', 'DisorderType', 'DisorderGroup']].copy()
        
        # One-hot encode disorder type
        type_dummies = pd.get_dummies(df['DisorderType'], prefix='type')
        group_dummies = pd.get_dummies(df['DisorderGroup'], prefix='group')
        
        type_features = pd.concat([df[['OrphaCode']], type_dummies, group_dummies], axis=1)
        
        print(f"  Created {len(type_dummies.columns) + len(group_dummies.columns)} type/group features")
        return type_features
    
    def merge_all_features(self):
        """Merge all features into a single dataframe."""
        print("Merging all features...")
        
        # Start with complete diseases
        merged = self.complete[['OrphaCode', 'Name', 'DiseaseName', 'DisorderType', 'DisorderGroup']].copy()
        
        # Merge gene features
        gene_features = self.create_gene_features()
        merged = merged.merge(gene_features, on='OrphaCode', how='left')
        
        # Merge onset features
        onset_features = self.create_onset_features()
        merged = merged.merge(onset_features, on='OrphaCode', how='left')
        
        # Merge inheritance features
        inherit_features = self.create_inheritance_features()
        merged = merged.merge(inherit_features, on='OrphaCode', how='left')
        
        # Merge prevalence features
        prev_features = self.create_prevalence_features()
        merged = merged.merge(prev_features, on='OrphaCode', how='left')
        
        # Merge disease type features
        type_features = self.create_disease_type_features()
        merged = merged.merge(type_features, on='OrphaCode', how='left')
        
        # Fill missing values
        feature_cols = [c for c in merged.columns if c not in ['OrphaCode', 'Name', 'DiseaseName', 'DisorderType', 'DisorderGroup']]
        merged[feature_cols] = merged[feature_cols].fillna(0)
        
        self.merged = merged
        print(f"  Final merged shape: {merged.shape}")
        return merged
    
    def create_symptom_vocabulary(self):
        """Create a vocabulary of 'symptoms' from available features.
        
        Since we don't have explicit HPO phenotypes, we create a vocabulary
        from gene associations, onset categories, and inheritance patterns.
        """
        print("Creating symptom vocabulary...")
        
        # Collect all feature names that represent "symptoms"
        symptom_features = []
        
        # Gene features (top genes) - only binary columns
        gene_cols = [c for c in self.merged.columns if c.startswith('gene_') and c != 'gene_symbols' and c != 'gene_names' and c != 'association_types']
        symptom_features.extend(gene_cols)
        
        # Onset features
        onset_cols = [c for c in self.merged.columns if c.startswith('onset_')]
        symptom_features.extend(onset_cols)
        
        # Inheritance features
        inherit_cols = [c for c in self.merged.columns if c.startswith('inherit_')]
        symptom_features.extend(inherit_cols)
        
        # Prevalence as a feature
        if 'prevalence_score' in self.merged.columns:
            symptom_features.append('prevalence_score')
        
        # Disease type features
        type_cols = [c for c in self.merged.columns if c.startswith('type_') or c.startswith('group_')]
        symptom_features.extend(type_cols)
        
        # Create vocabulary mapping
        vocab = {feat: idx for idx, feat in enumerate(symptom_features)}
        
        print(f"  Vocabulary size: {len(vocab)}")
        return vocab, symptom_features
    
    def encode_features(self, symptom_features):
        """Encode features into multi-hot matrix."""
        print("Encoding features...")
        
        # Create feature matrix
        X = self.merged[symptom_features].values.astype(np.float32)
        
        # Create labels (disease indices)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.merged['OrphaCode'])
        
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Number of classes: {len(label_encoder.classes_)}")
        
        return X, y, label_encoder
    
    def save_processed_data(self, X, y, label_encoder, vocab, symptom_features):
        """Save processed data to disk."""
        print("Saving processed data...")
        
        # Save feature matrix and labels
        np.save(PROCESSED_DATA_DIR / "X.npy", X)
        np.save(PROCESSED_DATA_DIR / "y.npy", y)
        
        # Save label encoder
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save vocabulary
        with open(SYMPTOM_VOCAB_FILE, 'wb') as f:
            pickle.dump(vocab, f)
        
        # Save symptom features list
        with open(PROCESSED_DATA_DIR / "symptom_features.pkl", 'wb') as f:
            pickle.dump(symptom_features, f)
        
        # Save merged dataframe
        self.merged.to_csv(PROCESSED_DISEASES_FILE, index=False)
        
        # Save feature names
        feature_df = pd.DataFrame({'feature': symptom_features})
        feature_df.to_csv(PROCESSED_FEATURES_FILE, index=False)
        
        # Save labels
        labels_df = pd.DataFrame({
            'OrphaCode': label_encoder.classes_,
            'label_idx': range(len(label_encoder.classes_))
        })
        labels_df.to_csv(PROCESSED_LABELS_FILE, index=False)
        
        print(f"  Saved to {PROCESSED_DATA_DIR}")
    
    def run(self):
        """Run the complete preprocessing pipeline."""
        print("="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_raw_data()
        self.clean_complete()
        self.clean_genes()
        self.clean_natural_history()
        self.clean_prevalence()
        
        self.merge_all_features()
        
        vocab, symptom_features = self.create_symptom_vocabulary()
        X, y, label_encoder = self.encode_features(symptom_features)
        
        self.save_processed_data(X, y, label_encoder, vocab, symptom_features)
        
        print("="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X, y, label_encoder, vocab, symptom_features


def main():
    preprocessor = DataPreprocessor()
    X, y, label_encoder, vocab, symptom_features = preprocessor.run()
    
    # Print summary
    print(f"\nFinal dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Feature sparsity: {(X == 0).mean():.2%}")


if __name__ == "__main__":
    main()
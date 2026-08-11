"""Data profiling script for the Rare Disease Recommendation System."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

def profile_dataset(name, df, id_col=None):
    """Profile a single dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    
    if id_col and id_col in df.columns:
        print(f"Unique {id_col}: {df[id_col].nunique():,}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Missing'] > 0].to_string())
    
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")
    
    # Show sample
    print(f"\nSample (first 3 rows):")
    print(df.head(3).to_string())
    
    return missing_df

def main():
    print("="*60)
    print("DATA PROFILING REPORT")
    print("="*60)
    
    # Load all datasets
    complete = pd.read_csv(COMPLETE_FILE)
    genes = pd.read_csv(GENES_FILE)
    natural = pd.read_csv(NATURAL_HISTORY_FILE)
    prevalence = pd.read_csv(PREVALENCE_FILE)
    
    # Profile each dataset
    profile_dataset("COMPLETE (Diseases)", complete, "OrphaCode")
    profile_dataset("GENES (Disease-Gene Associations)", genes, "OrphaCode")
    profile_dataset("NATURAL HISTORY", natural, "OrphaCode")
    profile_dataset("PREVALENCE", prevalence, "OrphaCode")
    
    # Cross-dataset analysis
    print(f"\n{'='*60}")
    print("CROSS-DATASET ANALYSIS")
    print(f"{'='*60}")
    
    complete_codes = set(complete['OrphaCode'].unique())
    genes_codes = set(genes['OrphaCode'].unique())
    natural_codes = set(natural['OrphaCode'].unique())
    prevalence_codes = set(prevalence['OrphaCode'].unique())
    
    print(f"Diseases in complete: {len(complete_codes):,}")
    print(f"Diseases with genes: {len(genes_codes):,}")
    print(f"Diseases with natural history: {len(natural_codes):,}")
    print(f"Diseases with prevalence: {len(prevalence_codes):,}")
    
    # Overlap
    all_codes = complete_codes | genes_codes | natural_codes | prevalence_codes
    print(f"\nTotal unique diseases across all datasets: {len(all_codes):,}")
    
    # Diseases with all info
    diseases_all = complete_codes & genes_codes & natural_codes & prevalence_codes
    print(f"Diseases with ALL information: {len(diseases_all):,}")
    
    # Diseases with at least genes
    diseases_with_genes = complete_codes & genes_codes
    print(f"Diseases with gene info: {len(diseases_with_genes):,}")
    
    # Disease type distribution
    print(f"\n{'='*60}")
    print("DISEASE TYPE DISTRIBUTION")
    print(f"{'='*60}")
    print(complete['DisorderType'].value_counts().to_string())
    
    print(f"\n{'='*60}")
    print("DISEASE GROUP DISTRIBUTION")
    print(f"{'='*60}")
    print(complete['DisorderGroup'].value_counts().to_string())
    
    # Gene association types
    print(f"\n{'='*60}")
    print("GENE ASSOCIATION TYPES")
    print(f"{'='*60}")
    print(genes['AssociationType'].value_counts().to_string())
    
    print(f"\n{'='*60}")
    print("GENE ASSOCIATION STATUS")
    print(f"{'='*60}")
    print(genes['AssociationStatus'].value_counts().to_string())
    
    # Age of onset distribution
    print(f"\n{'='*60}")
    print("AGE OF ONSET DISTRIBUTION (top 20)")
    print(f"{'='*60}")
    print(natural['AgeOfOnset'].value_counts().head(20).to_string())
    
    # Inheritance distribution
    print(f"\n{'='*60}")
    print("TYPE OF INHERITANCE DISTRIBUTION (top 20)")
    print(f"{'='*60}")
    print(natural['TypeOfInheritance'].value_counts().head(20).to_string())
    
    # Prevalence distribution
    print(f"\n{'='*60}")
    print("PREVALENCE TYPE DISTRIBUTION")
    print(f"{'='*60}")
    print(prevalence['PrevalenceType'].value_counts().to_string())
    
    print(f"\n{'='*60}")
    print("PREVALENCE CLASS DISTRIBUTION")
    print(f"{'='*60}")
    print(prevalence['PrevalenceClass'].value_counts().to_string())
    
    # Save summary to file
    summary = {
        'total_diseases': len(complete_codes),
        'diseases_with_genes': len(diseases_with_genes),
        'diseases_with_natural_history': len(complete_codes & natural_codes),
        'diseases_with_prevalence': len(complete_codes & prevalence_codes),
        'diseases_with_all': len(diseases_all),
        'total_genes': genes['GeneSymbol'].nunique(),
        'total_gene_associations': len(genes),
        'disorder_types': complete['DisorderType'].nunique(),
        'disorder_groups': complete['DisorderGroup'].nunique(),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RESULTS_DIR / "data_profile_summary.csv", index=False)
    print(f"\nSummary saved to {RESULTS_DIR / 'data_profile_summary.csv'}")
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
"""PrimeKG data processing and integration with Orphadata."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *


def load_primekg():
    """Load PrimeKG dataset."""
    if not PRIMEKG_FILE.exists():
        print(f"PrimeKG file not found at {PRIMEKG_FILE}")
        print("Download from: https://dataverse.harvard.edu/api/access/datafile/6180620")
        return None
    
    print("Loading PrimeKG...")
    primekg = pd.read_csv(PRIMEKG_FILE, low_memory=False)
    print(f"  Shape: {primekg.shape}")
    print(f"  Columns: {primekg.columns.tolist()}")
    print(f"  x_type: {primekg['x_type'].unique()}")
    print(f"  y_type: {primekg['y_type'].unique()}")
    print(f"  relation: {primekg['relation'].unique()}")
    return primekg


def extract_disease_info(primekg):
    """Extract disease nodes and their features from PrimeKG."""
    if primekg is None:
        return None
    
    # Get disease nodes
    diseases_x = primekg[primekg['x_type'] == 'disease'][['x_id', 'x_name']].drop_duplicates()
    diseases_y = primekg[primekg['y_type'] == 'disease'][['y_id', 'y_name']].drop_duplicates()
    
    diseases = pd.concat([diseases_x.rename(columns={'x_id': 'id', 'x_name': 'name'}),
                          diseases_y.rename(columns={'y_id': 'id', 'y_name': 'name'})]).drop_duplicates()
    
    print(f"Unique diseases in PrimeKG: {len(diseases)}")
    return diseases


def extract_disease_gene_edges(primekg):
    """Extract disease-gene/protein associations from PrimeKG."""
    if primekg is None:
        return None
    
    # disease -> gene/protein edges (PrimeKG uses 'gene/protein')
    dg = primekg[(primekg['x_type'] == 'disease') & (primekg['y_type'] == 'gene/protein') |
                 (primekg['y_type'] == 'disease') & (primekg['x_type'] == 'gene/protein')]
    
    # Standardize: disease as source, gene/protein as target
    mask = dg['x_type'] == 'gene/protein'
    dg.loc[mask, ['x_id', 'x_name', 'x_type', 'y_id', 'y_name', 'y_type']] = \
        dg.loc[mask, ['y_id', 'y_name', 'y_type', 'x_id', 'x_name', 'x_type']].values
    
    dg = dg.rename(columns={'x_id': 'disease_id', 'x_name': 'disease_name',
                            'y_id': 'gene_id', 'y_name': 'gene_name',
                            'relation': 'association_type'})
    
    print(f"Disease-gene/protein edges: {len(dg)}")
    return dg


def extract_disease_phenotype_edges(primekg):
    """Extract disease-phenotype (HPO) associations from PrimeKG."""
    if primekg is None:
        return None
    
    # disease -> phenotype edges (PrimeKG uses 'effect/phenotype')
    dp = primekg[(primekg['x_type'] == 'disease') & (primekg['y_type'] == 'effect/phenotype') |
                 (primekg['y_type'] == 'disease') & (primekg['x_type'] == 'effect/phenotype')]
    
    mask = dp['x_type'] == 'effect/phenotype'
    dp.loc[mask, ['x_id', 'x_name', 'x_type', 'y_id', 'y_name', 'y_type']] = \
        dp.loc[mask, ['y_id', 'y_name', 'y_type', 'x_id', 'x_name', 'x_type']].values
    
    dp = dp.rename(columns={'x_id': 'disease_id', 'x_name': 'disease_name',
                            'y_id': 'phenotype_id', 'y_name': 'phenotype_name',
                            'relation': 'association_type'})
    
    print(f"Disease-phenotype edges: {len(dp)}")
    return dp


def extract_phenotype_info(primekg):
    """Extract phenotype (HPO) info from PrimeKG."""
    if primekg is None:
        return None
    
    pheno_x = primekg[primekg['x_type'] == 'effect/phenotype'][['x_id', 'x_name']].drop_duplicates()
    pheno_y = primekg[primekg['y_type'] == 'effect/phenotype'][['y_id', 'y_name']].drop_duplicates()
    
    phenotypes = pd.concat([pheno_x.rename(columns={'x_id': 'id', 'x_name': 'name'}),
                            pheno_y.rename(columns={'y_id': 'id', 'y_name': 'name'})]).drop_duplicates()
    
    print(f"Unique phenotypes (HPO) in PrimeKG: {len(phenotypes)}")
    return phenotypes


def extract_drug_edges(primekg):
    """Extract drug-disease and drug-target edges."""
    if primekg is None:
        return None
    
    # drug -> disease
    dd = primekg[(primekg['x_type'] == 'drug') & (primekg['y_type'] == 'disease') |
                 (primekg['y_type'] == 'drug') & (primekg['x_type'] == 'disease')]
    
    # drug -> gene/protein
    dg = primekg[(primekg['x_type'] == 'drug') & (primekg['y_type'] == 'gene/protein') |
                 (primekg['y_type'] == 'drug') & (primekg['x_type'] == 'gene/protein')]
    
    return dd, dg


def main():
    print("="*60)
    print("PRIMEKG DATA PROCESSING")
    print("="*60)
    
    primekg = load_primekg()
    if primekg is None:
        return
    
    # Extract various components
    diseases = extract_disease_info(primekg)
    disease_genes = extract_disease_gene_edges(primekg)
    disease_phenotypes = extract_disease_phenotype_edges(primekg)
    phenotypes = extract_phenotype_info(primekg)
    drug_disease, drug_target = extract_drug_edges(primekg)
    
    # Save extracted data
    output_dir = PRIMEKG_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if diseases is not None:
        diseases.to_csv(output_dir / "diseases.csv", index=False)
    if disease_genes is not None:
        disease_genes.to_csv(output_dir / "disease_genes.csv", index=False)
    if disease_phenotypes is not None:
        disease_phenotypes.to_csv(output_dir / "disease_phenotypes.csv", index=False)
    if phenotypes is not None:
        phenotypes.to_csv(output_dir / "phenotypes.csv", index=False)
    if drug_disease is not None:
        drug_disease.to_csv(output_dir / "drug_disease.csv", index=False)
    if drug_target is not None:
        drug_target.to_csv(output_dir / "drug_targets.csv", index=False)
    
    print(f"\nProcessed data saved to {output_dir}")
    print("="*60)
    print("PRIMEKG PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
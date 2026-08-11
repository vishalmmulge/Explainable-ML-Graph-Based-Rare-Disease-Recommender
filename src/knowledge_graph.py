"""Knowledge graph construction for Rare Disease Recommendation System."""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *


def load_data():
    """Load raw data for graph construction."""
    print("Loading data for knowledge graph...")
    complete = pd.read_csv(COMPLETE_FILE)
    genes = pd.read_csv(GENES_FILE)
    natural = pd.read_csv(NATURAL_HISTORY_FILE)
    prevalence = pd.read_csv(PREVALENCE_FILE)
    
    print(f"  Complete: {len(complete)} diseases")
    print(f"  Genes: {len(genes)} associations")
    print(f"  Natural history: {len(natural)} records")
    print(f"  Prevalence: {len(prevalence)} records")
    
    return complete, genes, natural, prevalence


def build_knowledge_graph(complete, genes, natural, prevalence):
    """Build a MultiDiGraph with diseases, genes, onset, inheritance, prevalence."""
    print("Building knowledge graph...")
    
    G = nx.MultiDiGraph()
    
    # Add disease nodes
    for _, row in complete.iterrows():
        orpha = row['OrphaCode']
        G.add_node(orpha, 
                   node_type='disease',
                   name=row['Name'],
                   disorder_type=row['DisorderType'],
                   disorder_group=row['DisorderGroup'])
    
    # Add gene nodes and disease-gene edges
    genes_assessed = genes[genes['AssociationStatus'] == 'Assessed']
    for _, row in genes_assessed.iterrows():
        orpha = row['OrphaCode']
        gene_symbol = row['GeneSymbol']
        gene_name = row['GeneName']
        assoc_type = row['AssociationType']
        
        # Add gene node if not exists
        if not G.has_node(gene_symbol):
            G.add_node(gene_symbol, 
                       node_type='gene',
                       name=gene_name)
        
        # Add edge
        G.add_edge(orpha, gene_symbol, 
                   relation_type='HAS_GENE',
                   association_type=assoc_type)
    
    # Add onset nodes and edges
    onset_categories = ['Antenatal', 'Neonatal', 'Infancy', 'Childhood', 
                       'Adolescent', 'Adult', 'Elderly', 'All ages']
    
    for onset in onset_categories:
        onset_id = f'onset_{onset.lower().replace(" ", "_")}'
        G.add_node(onset_id, node_type='onset', name=onset)
    
    for _, row in natural.dropna(subset=['AgeOfOnset']).iterrows():
        orpha = row['OrphaCode']
        onset_str = row['AgeOfOnset']
        for onset in onset_categories:
            if onset.lower() in onset_str.lower():
                onset_id = f'onset_{onset.lower().replace(" ", "_")}'
                G.add_edge(orpha, onset_id, relation_type='HAS_ONSET')
    
    # Add inheritance nodes and edges
    inherit_categories = [
        'Autosomal dominant', 'Autosomal recessive', 'X-linked dominant',
        'X-linked recessive', 'Mitochondrial inheritance', 'Multigenic/multifactorial',
        'Not applicable', 'Unknown', 'Semi-dominant', 'Oligogenic', 'Y-linked'
    ]
    
    for inherit in inherit_categories:
        inherit_id = f'inherit_{inherit.lower().replace(" ", "_").replace("/", "_").replace("-", "_")}'
        G.add_node(inherit_id, node_type='inheritance', name=inherit)
    
    for _, row in natural.dropna(subset=['TypeOfInheritance']).iterrows():
        orpha = row['OrphaCode']
        inherit_str = row['TypeOfInheritance']
        for inherit in inherit_categories:
            if inherit.lower() in inherit_str.lower():
                inherit_id = f'inherit_{inherit.lower().replace(" ", "_").replace("/", "_").replace("-", "_")}'
                G.add_edge(orpha, inherit_id, relation_type='HAS_INHERITANCE')
    
    # Add prevalence class nodes
    prev_classes = ['>1 / 1000', '6-9 / 10 000', '1-5 / 10 000', 
                   '1-9 / 100 000', '1-9 / 1 000 000', '<1 / 1 000 000', 'Unknown']
    
    for pc in prev_classes:
        pc_id = f'prev_{pc.lower().replace(" ", "_").replace("/", "_").replace(">", "gt").replace("<", "lt")}'
        G.add_node(pc_id, node_type='prevalence_class', name=pc)
    
    point_prev = prevalence[prevalence['PrevalenceType'] == 'Point prevalence']
    for _, row in point_prev.dropna(subset=['PrevalenceClass']).iterrows():
        orpha = row['OrphaCode']
        pc = row['PrevalenceClass']
        if pc in prev_classes:
            pc_id = f'prev_{pc.lower().replace(" ", "_").replace("/", "_").replace(">", "gt").replace("<", "lt")}'
            G.add_edge(orpha, pc_id, relation_type='HAS_PREVALENCE')
    
    # Add disorder type nodes
    for dtype in complete['DisorderType'].unique():
        dtype_id = f'type_{dtype.lower().replace(" ", "_").replace("/", "_")}'
        G.add_node(dtype_id, node_type='disorder_type', name=dtype)
    
    for _, row in complete.iterrows():
        orpha = row['OrphaCode']
        dtype_id = f'type_{row["DisorderType"].lower().replace(" ", "_").replace("/", "_")}'
        G.add_edge(orpha, dtype_id, relation_type='HAS_TYPE')
    
    # Add disorder group nodes
    for dgroup in complete['DisorderGroup'].unique():
        dgroup_id = f'group_{dgroup.lower().replace(" ", "_").replace("/", "_")}'
        G.add_node(dgroup_id, node_type='disorder_group', name=dgroup)
    
    for _, row in complete.iterrows():
        orpha = row['OrphaCode']
        dgroup_id = f'group_{row["DisorderGroup"].lower().replace(" ", "_").replace("/", "_")}'
        G.add_edge(orpha, dgroup_id, relation_type='HAS_GROUP')
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def analyze_graph(G):
    """Analyze graph statistics."""
    print("\nGraph Analysis:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('node_type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
    
    print("  Node types:")
    for ntype, count in sorted(node_types.items()):
        print(f"    {ntype}: {count}")
    
    relation_types = {}
    for u, v, data in G.edges(data=True):
        rtype = data.get('relation_type', 'unknown')
        relation_types[rtype] = relation_types.get(rtype, 0) + 1
    
    print("  Relation types:")
    for rtype, count in sorted(relation_types.items()):
        print(f"    {rtype}: {count}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    print(f"  Avg degree: {np.mean(degrees):.2f}")
    print(f"  Max degree: {max(degrees)}")
    
    # Connected components
    if nx.is_connected(G.to_undirected()):
        print("  Graph is connected")
    else:
        components = list(nx.connected_components(G.to_undirected()))
        print(f"  Connected components: {len(components)}")
        print(f"  Largest component: {max(len(c) for c in components)} nodes")
    
    return node_types, relation_types


def save_graph(G):
    """Save graph to disk."""
    print(f"Saving graph to {GRAPH_FILE}...")
    # Use pickle for NetworkX 3.x compatibility
    import pickle
    with open(GRAPH_FILE, 'wb') as f:
        pickle.dump(G, f)
    print("  Done")


def main():
    print("="*60)
    print("KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*60)
    
    complete, genes, natural, prevalence = load_data()
    G = build_knowledge_graph(complete, genes, natural, prevalence)
    analyze_graph(G)
    save_graph(G)
    
    print("="*60)
    print("KNOWLEDGE GRAPH COMPLETE")
    print("="*60)
    
    return G


if __name__ == "__main__":
    main()
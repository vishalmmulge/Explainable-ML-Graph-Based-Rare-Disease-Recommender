"""Graph-based reasoning for Rare Disease Recommendation System."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

import networkx as nx


def load_graph():
    """Load the knowledge graph."""
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_processed_data():
    """Load processed data for feature mapping."""
    X = np.load(PROCESSED_DATA_DIR / "X.npy")
    y = np.load(PROCESSED_DATA_DIR / "y.npy")
    
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(SYMPTOM_VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(PROCESSED_DATA_DIR / "symptom_features.pkl", 'rb') as f:
        symptom_features = pickle.load(f)
    
    merged = pd.read_csv(PROCESSED_DISEASES_FILE)
    
    return X, y, label_encoder, vocab, symptom_features, merged


def map_symptoms_to_graph_nodes(symptom_indices, symptom_features, vocab):
    """Map user-selected symptom indices to graph nodes."""
    graph_nodes = []
    for idx in symptom_indices:
        if idx < len(symptom_features):
            feature = symptom_features[idx]
            # Map feature to graph node
            if feature.startswith('gene_') and feature != 'gene_count':
                gene = feature.replace('gene_', '')
                graph_nodes.append(gene)
            elif feature.startswith('onset_'):
                onset = feature.replace('onset_', '').replace('_', ' ').title()
                graph_nodes.append(f'onset_{onset.lower().replace(" ", "_")}')
            elif feature.startswith('inherit_'):
                inherit = feature.replace('inherit_', '').replace('_', ' ').title()
                graph_nodes.append(f'inherit_{inherit.lower().replace(" ", "_").replace("/", "_").replace("-", "_")}')
            elif feature.startswith('type_'):
                dtype = feature.replace('type_', '').replace('_', ' ').title()
                graph_nodes.append(f'type_{dtype.lower().replace(" ", "_").replace("/", "_")}')
            elif feature.startswith('group_'):
                dgroup = feature.replace('group_', '').replace('_', ' ').title()
                graph_nodes.append(f'group_{dgroup.lower().replace(" ", "_").replace("/", "_")}')
            elif feature in ['prevalence_score', 'gene_count']:
                # Skip continuous/count features
                pass
            else:
                # Skip unknown features
                pass
    
    return graph_nodes


def compute_graph_scores(G, symptom_nodes, label_encoder):
    """Compute graph-based scores for all diseases given symptom nodes."""
    print(f"Computing graph scores for {len(symptom_nodes)} symptom nodes...")
    
    # Get all disease nodes
    disease_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'disease']
    disease_scores = {}
    
    # For each disease, compute connectivity to symptom nodes
    for disease in disease_nodes:
        score = 0.0
        
        # Count direct connections to symptom nodes
        for symptom in symptom_nodes:
            if G.has_edge(disease, symptom):
                # Weight by edge type
                edge_data = G.get_edge_data(disease, symptom)
                if edge_data:
                    # Simple weighting
                    score += 1.0
        
        # Also consider 2-hop connections (disease -> intermediate -> symptom)
        for symptom in symptom_nodes:
            if not G.has_edge(disease, symptom):
                # Find common neighbors
                disease_neighbors = set(G.successors(disease)) | set(G.predecessors(disease))
                symptom_neighbors = set(G.successors(symptom)) | set(G.predecessors(symptom))
                common = disease_neighbors & symptom_neighbors
                if common:
                    score += 0.5 * len(common)
        
        # Normalize by number of symptom nodes
        if len(symptom_nodes) > 0:
            score = score / len(symptom_nodes)
        
        disease_scores[disease] = score
    
    # Convert to array matching label_encoder order
    scores_array = np.zeros(len(label_encoder.classes_))
    for i, disease in enumerate(label_encoder.classes_):
        scores_array[i] = disease_scores.get(disease, 0.0)
    
    # Normalize to [0, 1]
    if scores_array.max() > 0:
        scores_array = scores_array / scores_array.max()
    
    print(f"  Score range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
    return scores_array


def get_top_k_diseases(scores, label_encoder, k=5):
    """Get top-k diseases by score."""
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'orpha_code': label_encoder.classes_[idx],
            'score': float(scores[idx])
        })
    return results


def get_disease_subgraph(G, disease_orpha, max_nodes=20):
    """Get subgraph around a disease for visualization."""
    if not G.has_node(disease_orpha):
        return None
    
    # Get 1-hop neighborhood
    neighbors = list(G.successors(disease_orpha)) + list(G.predecessors(disease_orpha))
    subgraph_nodes = [disease_orpha] + neighbors[:max_nodes-1]
    
    subgraph = G.subgraph(subgraph_nodes).copy()
    return subgraph


def format_subgraph_for_display(subgraph):
    """Format subgraph for display in dashboard."""
    if subgraph is None:
        return None
    
    nodes = []
    edges = []
    
    for node, data in subgraph.nodes(data=True):
        nodes.append({
            'id': node,
            'label': data.get('name', node),
            'type': data.get('node_type', 'unknown')
        })
    
    for u, v, data in subgraph.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'relation': data.get('relation_type', 'unknown')
        })
    
    return {'nodes': nodes, 'edges': edges}


def main():
    print("="*60)
    print("GRAPH REASONING TEST")
    print("="*60)
    
    G = load_graph()
    X, y, label_encoder, vocab, symptom_features, merged = load_processed_data()
    
    # Test with a sample disease's features
    test_idx = 0
    test_features = X[test_idx]
    active_features = np.where(test_features > 0)[0]
    
    print(f"Test disease: {label_encoder.classes_[test_idx]}")
    print(f"Active features: {len(active_features)}")
    
    symptom_nodes = map_symptoms_to_graph_nodes(active_features, symptom_features, vocab)
    print(f"Mapped to graph nodes: {symptom_nodes[:10]}...")
    
    scores = compute_graph_scores(G, symptom_nodes, label_encoder)
    top5 = get_top_k_diseases(scores, label_encoder, k=5)
    
    print("\nTop 5 by graph score:")
    for r in top5:
        print(f"  {r['rank']}. {r['orpha_code']}: {r['score']:.4f}")
    
    # Test subgraph
    subgraph = get_disease_subgraph(G, label_encoder.classes_[test_idx])
    formatted = format_subgraph_for_display(subgraph)
    print(f"\nSubgraph: {len(formatted['nodes'])} nodes, {len(formatted['edges'])} edges")
    
    print("="*60)
    print("GRAPH REASONING TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
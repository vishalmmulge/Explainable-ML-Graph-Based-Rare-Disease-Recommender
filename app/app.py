"""Streamlit dashboard for Rare Disease Recommendation System."""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

from src.graph_reasoning import load_graph, map_symptoms_to_graph_nodes, compute_graph_scores, get_top_k_diseases, get_disease_subgraph, format_subgraph_for_display
from src.hybrid_model import load_model, load_processed_data, recommend_diseases


# Page config
st.set_page_config(
    page_title="Rare Disease Recommendation System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_all_resources():
    """Load all models and data."""
    model, scaler, label_encoder = load_model()
    X, y, _, vocab, symptom_features = load_processed_data()
    G = load_graph()
    merged = pd.read_csv(PROCESSED_DISEASES_FILE)
    return model, scaler, label_encoder, vocab, symptom_features, G, merged


def get_feature_display_name(feature):
    """Convert feature name to human-readable display."""
    if feature.startswith('gene_') and feature != 'gene_count':
        return f"Gene: {feature.replace('gene_', '')}"
    elif feature.startswith('onset_'):
        onset = feature.replace('onset_', '').replace('_', ' ').title()
        return f"Onset: {onset}"
    elif feature.startswith('inherit_'):
        inherit = feature.replace('inherit_', '').replace('_', ' ').title()
        return f"Inheritance: {inherit}"
    elif feature.startswith('type_'):
        dtype = feature.replace('type_', '').replace('_', ' ').title()
        return f"Type: {dtype}"
    elif feature.startswith('group_'):
        dgroup = feature.replace('group_', '').replace('_', ' ').title()
        return f"Group: {dgroup}"
    elif feature == 'prevalence_score':
        return "Prevalence Score"
    elif feature == 'gene_count':
        return "Gene Count"
    return feature


def main():
    # Load resources
    model, scaler, label_encoder, vocab, symptom_features, G, merged = load_all_resources()
    
    # Sidebar
    st.sidebar.title("🧬 Rare Disease Recommender")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Disease Recommendation", "Explainability", "Knowledge Graph", "Model Evaluation"]
    )
    
    # Medical disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Medical Disclaimer**: This system is a research and educational prototype "
        "and is not intended to provide medical diagnosis or treatment. Predictions should "
        "not be used as a substitute for professional medical evaluation."
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Disease Recommendation":
        show_recommendation_page(model, scaler, label_encoder, vocab, symptom_features, G, merged)
    elif page == "Explainability":
        show_explainability_page(model, scaler, label_encoder, vocab, symptom_features, G, merged)
    elif page == "Knowledge Graph":
        show_graph_page(G, merged)
    elif page == "Model Evaluation":
        show_evaluation_page()


def show_home_page():
    st.title("🧬 Rare Disease Recommendation System")
    st.markdown("""
    ### Explainable Graph-Based Rare Disease Recommendation Using Symptom-Phenotype Mapping
    
    This research prototype combines **machine learning** with **biomedical knowledge graph reasoning** 
    to recommend rare diseases based on patient symptoms/phenotypes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - **Top-K Disease Recommendations**: Rank rare diseases by relevance to input symptoms
        - **Calibrated Confidence Scores**: Probability estimates for each recommendation
        - **Symptom-Level Explanations**: SHAP-based feature importance
        - **Graph-Based Evidence**: Knowledge graph relationships between symptoms, diseases, and genes
        - **Hybrid Approach**: Combine ML predictions with graph reasoning
        """)
    
    with col2:
        st.subheader("📊 Dataset")
        st.markdown("""
        - **Source**: Orphanet/Orphadata rare disease information (Kaggle)
        - **Diseases**: 11,456 rare diseases
        - **Gene Associations**: 8,374 disease-gene links
        - **Natural History**: 7,374 records (onset, inheritance)
        - **Prevalence**: 16,657 prevalence records
        - **Knowledge Graph**: 15,954 nodes, 57,951 edges
        """)
    
    st.subheader("🔬 Research Motivation")
    st.markdown("""
    Rare disease diagnosis is challenging due to:
    - Overlapping symptoms across diseases
    - Limited training examples per disease
    - Conventional ML treats symptoms as independent features
    - Black-box predictions lack interpretability
    
    **Our Approach**: Hybrid ML + Knowledge Graph + Explainable AI
    """)
    
    st.subheader("🏗️ Architecture")
    st.markdown("""
    ```
    Symptoms → Multi-hot Encoding → ML Model (Logistic Regression)
                    ↓
            Knowledge Graph (NetworkX)
                    ↓
            Hybrid Score = α × ML + (1-α) × Graph
                    ↓
            Top-K Recommendations + Explanations
    ```
    """)
    
    st.info("👈 Use the sidebar to navigate to different pages.")


def show_recommendation_page(model, scaler, label_encoder, vocab, symptom_features, G, merged):
    st.title("🔍 Disease Recommendation")
    
    st.markdown("Select symptoms/phenotypes to get disease recommendations.")
    
    # Group features by type for UI
    feature_groups = {
        'Genes': [f for f in symptom_features if f.startswith('gene_') and f != 'gene_count'],
        'Age of Onset': [f for f in symptom_features if f.startswith('onset_')],
        'Inheritance': [f for f in symptom_features if f.startswith('inherit_')],
        'Disease Type': [f for f in symptom_features if f.startswith('type_')],
        'Disease Group': [f for f in symptom_features if f.startswith('group_')],
    }
    
    # Symptom selection
    st.subheader("Select Features")
    
    selected_indices = []
    
    for group_name, features in feature_groups.items():
        with st.expander(group_name, expanded=True):
            cols = st.columns(3)
            for i, feat in enumerate(features):
                col_idx = i % 3
                display_name = get_feature_display_name(feat)
                feat_idx = symptom_features.index(feat)
                if cols[col_idx].checkbox(display_name, key=f"feat_{feat_idx}"):
                    selected_indices.append(feat_idx)
    
    # Additional continuous features
    with st.expander("Additional Features", expanded=False):
        if st.checkbox("Include Prevalence Score", key="feat_prev"):
            feat_idx = symptom_features.index('prevalence_score')
            selected_indices.append(feat_idx)
        if st.checkbox("Include Gene Count", key="feat_genecount"):
            feat_idx = symptom_features.index('gene_count')
            selected_indices.append(feat_idx)
    
    # Alpha slider
    alpha = st.slider(
        "Hybrid Weight (α): ML vs Graph", 
        0.0, 1.0, 0.5, 0.05,
        help="α=1.0: ML only, α=0.0: Graph only, α=0.5: Equal weight"
    )
    
    # Top-K selector
    top_k = st.selectbox("Number of Recommendations", [1, 3, 5, 10], index=2)
    
    if st.button("🔍 Get Recommendations", type="primary"):
        if len(selected_indices) == 0:
            st.warning("Please select at least one feature.")
        else:
            with st.spinner("Computing recommendations..."):
                recommendations, symptom_nodes = recommend_diseases(
                    selected_indices, model, scaler, label_encoder, 
                    vocab, symptom_features, G, alpha=alpha, top_k=top_k
                )
            
            st.success(f"Found {len(recommendations)} recommendations!")
            
            # Display results
            st.subheader(f"Top-{top_k} Recommendations")
            
            results_df = pd.DataFrame([
                {
                    'Rank': r['rank'],
                    'OrphaCode': r['orpha_code'],
                    'ML Score': f"{r['ml_score']:.4f}",
                    'Graph Score': f"{r['graph_score']:.4f}",
                    'Hybrid Score': f"{r['hybrid_score']:.4f}"
                }
                for r in recommendations
            ])
            
st.dataframe(results_df, use_container_width=True, width='stretch')
            
            # Bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='ML Score',
                x=[r['orpha_code'] for r in recommendations],
                y=[r['ml_score'] for r in recommendations],
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                name='Graph Score',
                x=[r['orpha_code'] for r in recommendations],
                y=[r['graph_score'] for r in recommendations],
                marker_color='#ff7f0e'
            ))
            fig.add_trace(go.Bar(
                name='Hybrid Score',
                x=[r['orpha_code'] for r in recommendations],
                y=[r['hybrid_score'] for r in recommendations],
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                barmode='group',
                title='Score Comparison',
                xaxis_title='OrphaCode',
                yaxis_title='Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show selected features
            st.subheader("Selected Features")
            selected_names = [get_feature_display_name(symptom_features[i]) for i in selected_indices]
            st.write(", ".join(selected_names))
            
            # Show mapped graph nodes
            st.subheader("Mapped Graph Nodes")
            st.write(symptom_nodes)


def show_explainability_page(model, scaler, label_encoder, vocab, symptom_features, G, merged):
    st.title("📊 Explainability")
    
    st.markdown("""
    This page provides two complementary explanations:
    1. **ML Explanation (SHAP)**: Which features contributed most to the model's prediction
    2. **Graph Explanation**: Knowledge graph relationships supporting the recommendation
    """)
    
    # Feature selection (same as recommendation page)
    feature_groups = {
        'Genes': [f for f in symptom_features if f.startswith('gene_') and f != 'gene_count'],
        'Age of Onset': [f for f in symptom_features if f.startswith('onset_')],
        'Inheritance': [f for f in symptom_features if f.startswith('inherit_')],
        'Disease Type': [f for f in symptom_features if f.startswith('type_')],
        'Disease Group': [f for f in symptom_features if f.startswith('group_')],
    }
    
    st.subheader("Select Features for Explanation")
    
    selected_indices = []
    
    for group_name, features in feature_groups.items():
        with st.expander(group_name, expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(features):
                col_idx = i % 3
                display_name = get_feature_display_name(feat)
                feat_idx = symptom_features.index(feat)
                if cols[col_idx].checkbox(display_name, key=f"expl_feat_{feat_idx}"):
                    selected_indices.append(feat_idx)
    
    with st.expander("Additional Features", expanded=False):
        if st.checkbox("Include Prevalence Score", key="expl_feat_prev"):
            feat_idx = symptom_features.index('prevalence_score')
            selected_indices.append(feat_idx)
        if st.checkbox("Include Gene Count", key="expl_feat_genecount"):
            feat_idx = symptom_features.index('gene_count')
            selected_indices.append(feat_idx)
    
    alpha = st.slider("Hybrid Weight (α)", 0.0, 1.0, 0.5, 0.05, key="expl_alpha")
    
    if st.button("🔍 Generate Explanation", type="primary"):
        if len(selected_indices) == 0:
            st.warning("Please select at least one feature.")
        else:
            # Get recommendations
            recommendations, symptom_nodes = recommend_diseases(
                selected_indices, model, scaler, label_encoder, 
                vocab, symptom_features, G, alpha=alpha, top_k=5
            )
            
            if recommendations:
                top_disease = recommendations[0]
                orpha_code = top_disease['orpha_code']
                
                st.subheader(f"Explanation for: {orpha_code}")
                
                # ML Explanation (SHAP)
                st.markdown("### 🤖 ML Explanation (SHAP)")
                st.markdown("*Features that increased/decreased the model's prediction score*")
                
                # Create feature vector
                X_user = np.zeros(len(symptom_features))
                X_user[selected_indices] = 1
                X_user = X_user.reshape(1, -1)
                
                if scaler is not None:
                    X_scaled = scaler.transform(X_user)
                else:
                    X_scaled = X_user
                
                # Get SHAP values using the model's coef_ (for linear model)
                if hasattr(model, 'coef_'):
                    # For logistic regression, SHAP values ≈ coef_ * feature_value
                    coef = model.coef_
                    pred_class_idx = np.where(label_encoder.classes_ == orpha_code)[0][0]
                    
                    if coef.ndim > 1:
                        shap_vals = coef[pred_class_idx] * X_user[0]
                    else:
                        shap_vals = coef * X_user[0]
                    
                    # Get top contributing features
                    feature_importance = []
                    for i, feat in enumerate(symptom_features):
                        if X_user[0, i] > 0:
                            feature_importance.append({
                                'Feature': get_feature_display_name(feat),
                                'SHAP Value': float(shap_vals[i]),
                                'Direction': 'Supports' if shap_vals[i] > 0 else 'Reduces'
                            })
                    
                    importance_df = pd.DataFrame(feature_importance).sort_values('SHAP Value', key=abs, ascending=False)
                    
                    # Plot
                    fig = px.bar(
                        importance_df.head(15),
                        x='SHAP Value',
                        y='Feature',
                        color='Direction',
                        color_discrete_map={'Supports': '#2ca02c', 'Reduces': '#d62728'},
                        orientation='h',
                        title='Feature Contributions (SHAP Values)'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.dataframe(importance_df, use_container_width=True)
                
                # Graph Explanation
                st.markdown("### 🕸️ Graph Explanation")
                st.markdown("*Knowledge graph connections between selected features and the recommended disease*")
                
                subgraph = get_disease_subgraph(G, orpha_code)
                if subgraph:
                    formatted = format_subgraph_for_display(subgraph)
                    
                    # Filter to show only selected symptom nodes and their connections
                    selected_nodes = set(symptom_nodes)
                    selected_nodes.add(orpha_code)
                    
                    # Show connections
                    st.write(f"**Disease**: {orpha_code}")
                    st.write(f"**Connected Features**: {', '.join(symptom_nodes) if symptom_nodes else 'None directly connected'}")
                    
                    # Show subgraph stats
                    st.write(f"**Subgraph**: {len(formatted['nodes'])} nodes, {len(formatted['edges'])} edges")
                    
                    # Display relevant edges
                    relevant_edges = []
                    for edge in formatted['edges']:
                        if edge['source'] == orpha_code and edge['target'] in symptom_nodes:
                            relevant_edges.append(f"{edge['source']} --{edge['relation']}--> {edge['target']}")
                        elif edge['target'] == orpha_code and edge['source'] in symptom_nodes:
                            relevant_edges.append(f"{edge['source']} --{edge['relation']}--> {edge['target']}")
                    
                    if relevant_edges:
                        st.write("**Direct Connections:**")
                        for e in relevant_edges[:20]:
                            st.code(e)
                    else:
                        st.info("No direct connections found between selected features and this disease in the graph.")
                    
                    # Show all subgraph nodes by type
                    node_types = {}
                    for node in formatted['nodes']:
                        ntype = node['type']
                        if ntype not in node_types:
                            node_types[ntype] = []
                        node_types[ntype].append(node['label'])
                    
                    for ntype, nodes in node_types.items():
                        with st.expander(f"{ntype} ({len(nodes)})"):
                            st.write(", ".join(nodes[:50]))
                else:
                    st.warning("Disease not found in knowledge graph.")


def show_graph_page(G, merged):
    st.title("🕸️ Knowledge Graph Explorer")
    
    st.markdown("Explore the biomedical knowledge graph structure.")
    
    # Graph statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", G.number_of_nodes())
    with col2:
        st.metric("Total Edges", G.number_of_edges())
    with col3:
        disease_nodes = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'disease')
        st.metric("Disease Nodes", disease_nodes)
    with col4:
        gene_nodes = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'gene')
        st.metric("Gene Nodes", gene_nodes)
    
    # Node type distribution
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('node_type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
    
    fig = px.pie(
        values=list(node_types.values()),
        names=list(node_types.keys()),
        title="Node Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Relation type distribution
    rel_types = {}
    for u, v, data in G.edges(data=True):
        rtype = data.get('relation_type', 'unknown')
        rel_types[rtype] = rel_types.get(rtype, 0) + 1
    
    fig = px.bar(
        x=list(rel_types.keys()),
        y=list(rel_types.values()),
        title="Relation Type Distribution",
        labels={'x': 'Relation Type', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Disease search
    st.subheader("Search Disease Subgraph")
    disease_search = st.text_input("Enter OrphaCode", placeholder="e.g., 58")
    
    if disease_search:
        try:
            orpha = int(disease_search)
            if G.has_node(orpha):
                subgraph = get_disease_subgraph(G, orpha)
                if subgraph:
                    formatted = format_subgraph_for_display(subgraph)
                    
                    st.write(f"**Subgraph for {orpha}**: {len(formatted['nodes'])} nodes, {len(formatted['edges'])} edges")
                    
                    # Group by type
                    node_types = {}
                    for node in formatted['nodes']:
                        ntype = node['type']
                        if ntype not in node_types:
                            node_types[ntype] = []
                        node_types[ntype].append(node['label'])
                    
                    for ntype, nodes in node_types.items():
                        with st.expander(f"{ntype} ({len(nodes)})"):
                            st.write(", ".join(nodes))
                    
                    # Show edges
                    with st.expander("Edges"):
                        for edge in formatted['edges'][:50]:
                            st.code(f"{edge['source']} --{edge['relation']}--> {edge['target']}")
            else:
                st.error(f"Disease {orpha} not found in graph")
        except ValueError:
            st.error("Please enter a valid numeric OrphaCode")


def show_evaluation_page():
    st.title("📈 Model Evaluation")
    
    st.markdown("### Model Performance Comparison")
    
    # Load results if available
    try:
        results_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
        st.dataframe(results_df, use_container_width=True)
        
        # Plot comparison
        if 'top_1_accuracy' in results_df.columns:
            fig = go.Figure()
            for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']:
                if metric in results_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=results_df.index,
                        y=results_df[metric]
                    ))
            fig.update_layout(
                barmode='group',
                title='Top-K Accuracy Comparison',
                xaxis_title='Model',
                yaxis_title='Accuracy'
            )
            st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("No evaluation results found. Run model training first.")
    
    # Hybrid evaluation
    try:
        hybrid_df = pd.read_csv(RESULTS_DIR / "hybrid_evaluation.csv")
        st.subheader("Hybrid Model Evaluation (α sensitivity)")
        st.dataframe(hybrid_df, use_container_width=True)
        
        # Plot alpha sensitivity
        fig = go.Figure()
        for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']:
            if metric in hybrid_df.columns:
                fig.add_trace(go.Scatter(
                    x=hybrid_df.index,
                    y=hybrid_df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title()
                ))
        fig.update_layout(
            title='Hybrid Weight (α) Sensitivity',
            xaxis_title='Alpha (ML Weight)',
            yaxis_title='Accuracy'
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("No hybrid evaluation results found.")
    
    # Data statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Total Diseases | 11,456 |
        | Diseases with Genes | 4,128 |
        | Diseases with Natural History | 7,374 |
        | Diseases with Prevalence | 6,443 |
        | Unique Genes | 4,552 |
        """)
    
    with col2:
        st.markdown("""
        | Feature Type | Count |
        |--------------|-------|
        | Gene Features | 100 |
        | Onset Features | 8 |
        | Inheritance Features | 11 |
        | Disease Type Features | 11 |
        | Disease Group Features | 3 |
        | **Total Features** | **135** |
        """)


if __name__ == "__main__":
    main()
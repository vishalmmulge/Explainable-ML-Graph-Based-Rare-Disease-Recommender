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
    # --- Custom animated styling ---
    st.markdown("""
    <style>
    .home-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.4);
    }
    .home-hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: heroFadeIn 1.2s ease-out both;
    }
    .home-hero p {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin: 0;
        animation: heroFadeIn 1.4s ease-out both;
    }
    @keyframes heroFadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: cardFadeIn 0.8s ease-out both;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 12px 40px -12px rgba(56, 189, 248, 0.15);
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .term-block {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-family: 'Menlo', 'Consolas', 'monospace';
        font-size: 0.85rem;
        line-height: 1.55;
        color: #c9d1d9;
        overflow-x: auto;
        margin-top: 1rem;
        animation: termFadeIn 1s ease-out both;
        box-shadow: 0 8px 32px -12px rgba(0,0,0,0.5);
    }
    .term-prompt { color: #58a6ff; font-weight: 700; margin-right: 0.4rem; }
    .term-cmd { color: #79c0ff; }
    .term-cursor {
        display: inline-block;
        width: 8px;
        height: 1em;
        background: #58a6ff;
        animation: blink 1s step-end infinite;
        vertical-align: middle;
        margin-left: 2px;
    }
    @keyframes blink { 50% { opacity: 0; } }
    @keyframes termFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .arch-desc {
        background: #f8fafc;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        animation: descFadeIn 1s ease-out both;
        color: #0f172a;
    }
    .arch-desc b,
    .arch-desc strong {
        color: #0f172a !important;
    }
    .arch-desc .stage-label {
        color: #475569;
        font-weight: 600;
    }
    .arch-desc .stage-desc {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0.25rem;
    }
    @keyframes descFadeIn {
        from { opacity: 0; transform: translateX(-8px); }
        to { opacity: 1; transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Animated hero section ---
    st.markdown("""
    <div class="home-hero">
        <h1>🧬 Rare Disease Recommendation System</h1>
        <p>Hierarchical ML + Knowledge Graph + Explainable AI for Rare Disease Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Interactive metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("11,456", "Rare Diseases", "Orphadata 2026"),
        ("519K", "Graph Edges", "Orphadata + PrimeKG HPO"),
        ("100%", "Group Accuracy", "Stage 1 Classifier"),
        ("~50%", "Top-5 Accuracy", "Stage 2 Disease"),
    ]
    for col, (value, label, sub) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # --- Interactive architecture diagram ---
    st.subheader("🏗️ System Architecture")
    st.markdown("Select a pipeline stage below to explore its role — the diagram highlights the active stage.")

    stages = [
        ("Symptoms", "Input", "Patient symptoms/phenotypes encoded as a multi-hot feature vector (135 features: genes, onset, inheritance, type, group, prevalence).", "#38bdf8"),
        ("Group Classifier", "Stage 1 • 3 classes", "Hierarchical XGBoost predicts the Disorder Group with 100% accuracy.", "#22d3ee"),
        ("Type Classifier", "Stage 1 • 11 classes", "Predicts the Disorder Type within the group with 100% accuracy.", "#06b6d4"),
        ("Disease Classifiers", "Stage 2 • per Group+Type", "Fine-grained disease prediction per Group+Type (~15% Top-1, ~50% Top-5).", "#0891b2"),
        ("Knowledge Graph", "Graph Reasoning", "NetworkX MultiDiGraph: 46,540 nodes, 519,407 edges (Orphadata + PrimeKG HPO phenotypes).", "#0f766e"),
        ("Hybrid Score", "α × ML + (1−α) × Graph", "Combines ML and graph evidence with tunable weight α (default 0.5).", "#115e59"),
        ("Top-K Recommendations", "Output", "Ranked rare disease recommendations with confidence + dual explanations (SHAP + Graph).", "#14532d"),
    ]

    stage_names = [s[0] for s in stages]
    if "home_active_stage" not in st.session_state:
        st.session_state.home_active_stage = "Symptoms"

    selected = st.segmented_control("Architecture Stage", stage_names, default="Symptoms", key="home_arch_control")
    if selected:
        st.session_state.home_active_stage = selected
    active = st.session_state.home_active_stage

    # Build Plotly architecture figure
    fig = go.Figure()

    # Wider spacing so labels sit above nodes and arrows don't collide with text
    pos = {
        "Symptoms": (0, 0),
        "Group Classifier": (1.25, 1),
        "Type Classifier": (2.5, 1),
        "Disease Classifiers": (3.75, 1),
        "Knowledge Graph": (5, 0),
        "Hybrid Score": (6.25, 0),
        "Top-K Recommendations": (7.5, 0),
    }

    order = ["Symptoms", "Group Classifier", "Type Classifier",
             "Disease Classifiers", "Knowledge Graph", "Hybrid Score", "Top-K Recommendations"]
    for i in range(len(order) - 1):
        x0, y0 = pos[order[i]]
        x1, y1 = pos[order[i + 1]]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=3, arrowsize=1.6, arrowwidth=2.5,
            arrowcolor="#334155",
            standoff=30,   # keep arrow tip away from node edge
        )

    for name, label, desc, color in stages:
        x, y = pos[name]
        is_active = (name == active)
        size = 44 if is_active else 30
        line_width = 4 if is_active else 2
        line_color = color if is_active else "#475569"
        fill_color = color if is_active else "#0f172a"
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=size,
                color=fill_color,
                line=dict(width=line_width, color=line_color),
                symbol="circle",
            ),
            text=[name],
            textposition="top center",   # label floats above the node
            textfont=dict(
                color="white" if is_active else "#94a3b8",
                size=10.5 if is_active else 9,
                family="Arial Black",
            ),
            hoverinfo="text",
            hovertext=f"<b>{name}</b><br>{label}<br>{desc}",
            showlegend=False,
        ))

    fig.update_layout(
        title="Interactive Architecture Pipeline",
        xaxis=dict(visible=False, range=[-0.9, 8.4]),
        yaxis=dict(visible=False, range=[-1.9, 1.5]),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=430,
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    active_info = next(s for s in stages if s[0] == active)
    st.markdown(f"""
    <div class="arch-desc">
        <b style="color:{active_info[3]};">▸ {active_info[0]}</b>
        <span class="stage-label"> — {active_info[1]}</span>
        <div class="stage-desc">{active_info[2]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Terminal commands section ---
    st.subheader("💻 Run in Terminal")
    st.markdown("Setup and run the project from your terminal (PowerShell):")

    term = (
        "# 1. Create and activate virtual environment\n"
        "$cmd python -m venv projectenv$cursor\n"
        "$cmd .\\projectenv\\Scripts\\Activate.ps1$cursor\n\n"
        "# 2. Install dependencies\n"
        "$cmd pip install -r requirements.txt$cursor\n\n"
        "# 3. Run the pipeline\n"
        "$cmd python scripts/profile_data.py$cursor\n"
        "$cmd python src/preprocessing.py$cursor\n"
        "$cmd python src/models.py$cursor\n"
        "$cmd python src/knowledge_graph.py$cursor\n"
        "$cmd python src/graph_reasoning.py$cursor\n"
        "$cmd python src/hybrid_model.py$cursor\n"
        "$cmd python src/explainability.py$cursor\n\n"
        "# 4. Launch the dashboard\n"
        "$cmd streamlit run app/app.py$cursor"
    )
    term_html = term.replace("$cmd", '<span class="term-prompt">PS C:\\> </span><span class="term-cmd">')
    term_html = term_html.replace("$cursor", '</span><span class="term-cursor"></span>')
    st.markdown(f'<div class="term-block">{term_html}</div>', unsafe_allow_html=True)

    st.info("👈 Use the sidebar to navigate: **Disease Recommendation** for predictions, **Explainability** for SHAP/Graph analysis, **Knowledge Graph** for graph exploration, **Model Evaluation** for metrics.")


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
            
            # Get feature importance based on model type
            st.markdown("**Feature Importance Analysis**")
            
            # Check if it's the hierarchical model wrapper
            is_hierarchical = hasattr(model, 'group_model') and hasattr(model, 'type_model')
            
            if is_hierarchical:
                # Hierarchical XGBoost model - show group/type feature importance
                st.info("📊 **Hierarchical Model**: Showing Stage 1 (Group & Type) feature importance. Disease-level uses separate models per Group+Type.")
                
                # Get group prediction and feature importance
                hier_data = joblib.load(MODEL_DIR / "hierarchical_model.pkl")
                group_model = hier_data['group_model']
                group_scaler = hier_data['scaler_group']
                group_le = hier_data['group_le']
                type_model = hier_data['type_model']
                type_scaler = hier_data['scaler_type']
                type_le = hier_data['type_le']
                
                # Group features
                X_group_scaled = group_scaler.transform(X_user)
                group_proba = group_model.predict_proba(X_group_scaled)
                group_pred = np.argmax(group_proba, axis=1)[0]
                group_name = group_le.inverse_transform([group_pred])[0]
                
                # Type features
                X_type_scaled = type_scaler.transform(X_user)
                type_proba = type_model.predict_proba(X_type_scaled)
                type_pred = np.argmax(type_proba, axis=1)[0]
                type_name = type_le.inverse_transform([type_pred])[0]
                
                st.write(f"**Predicted Group**: {group_name} | **Predicted Type**: {type_name}")
                
                # Feature importance for group model
                group_importance = group_model.feature_importances_
                group_feat_imp = [(symptom_features[i], float(group_importance[i])) for i in range(len(symptom_features)) if X_user[0, i] > 0]
                group_feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)
                
                if group_feat_imp:
                    group_df = pd.DataFrame(group_feat_imp, columns=['Feature', 'Importance'])
                    group_df['Feature'] = group_df['Feature'].apply(get_feature_display_name)
                    group_df['Direction'] = group_df['Importance'].apply(lambda x: 'Supports' if x > 0 else 'Reduces')
                    
                    fig = px.bar(
                        group_df.head(15),
                        x='Importance', y='Feature', color='Direction',
                        color_discrete_map={'Supports': '#2ca02c', 'Reduces': '#d62728'},
                        orientation='h',
                        title=f'Group Classifier Feature Importance (Predicted: {group_name})'
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for type model
                type_importance = type_model.feature_importances_
                type_feat_imp = [(symptom_features[i], float(type_importance[i])) for i in range(len(symptom_features)) if X_user[0, i] > 0]
                type_feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)
                
                if type_feat_imp:
                    type_df = pd.DataFrame(type_feat_imp, columns=['Feature', 'Importance'])
                    type_df['Feature'] = type_df['Feature'].apply(get_feature_display_name)
                    type_df['Direction'] = type_df['Importance'].apply(lambda x: 'Supports' if x > 0 else 'Reduces')
                    
                    fig = px.bar(
                        type_df.head(15),
                        x='Importance', y='Feature', color='Direction',
                        color_discrete_map={'Supports': '#2ca02c', 'Reduces': '#d62728'},
                        orientation='h',
                        title=f'Type Classifier Feature Importance (Predicted: {type_name})'
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            elif hasattr(model, 'feature_importances_'):
                # XGBoost model
                importance = model.feature_importances_
                feat_imp = [(symptom_features[i], float(importance[i])) for i in range(len(symptom_features)) if X_user[0, i] > 0]
                feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)
                
                if feat_imp:
                    imp_df = pd.DataFrame(feat_imp, columns=['Feature', 'Importance'])
                    imp_df['Feature'] = imp_df['Feature'].apply(get_feature_display_name)
                    imp_df['Direction'] = imp_df['Importance'].apply(lambda x: 'Supports' if x > 0 else 'Reduces')
                    
                    fig = px.bar(
                        imp_df.head(15),
                        x='Importance', y='Feature', color='Direction',
                        color_discrete_map={'Supports': '#2ca02c', 'Reduces': '#d62728'},
                        orientation='h',
                        title='Feature Importance (XGBoost)'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(imp_df, use_container_width=True)
            
            elif hasattr(model, 'coef_'):
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
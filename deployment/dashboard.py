import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Bitcoin Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black professional styling
st.markdown("""
<style>
    /* Black professional styling with navy blue */
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        color: white;
        padding: 1rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    
    .nav-container {
        background: #1a1a1a;
        border-bottom: 2px solid #333;
        padding: 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .nav-tabs {
        display: flex;
        justify-content: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .nav-tab {
        padding: 1rem 2rem;
        cursor: pointer;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        color: #ccc;
        font-weight: 500;
        text-decoration: none;
    }
    
    .nav-tab.active {
        border-bottom-color: #1e3a8a;
        color: #3b82f6;
        font-weight: bold;
    }
    
    .nav-tab:hover {
        background-color: #333;
        color: #3b82f6;
    }
    
    .stock-header {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-left: 4px solid #1e3a8a;
        color: white;
    }
    
    .metric-card {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #333;
        transition: all 0.3s ease;
        color: white;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #ccc;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .tab-container {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: white;
    }
    
    .chart-container {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: white;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .positive-change {
        color: #3b82f6;
    }
    
    .negative-change {
        color: #ef4444;
    }
    
    .neutral-change {
        color: #3b82f6;
    }
    
    .section-title {
        color: #3b82f6;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .achievement-card {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        color: white;
    }
    
    .achievement-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        border-color: #3b82f6;
    }
    
    .achievement-title {
        color: #3b82f6;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .achievement-desc {
        color: #ccc;
        line-height: 1.6;
    }
    
    /* Hide selectbox completely */
    .stSelectbox {
        display: none !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make all text white by default */
    body {
        background-color: #000000;
        color: white;
    }
    
    /* Style Streamlit elements for dark theme */
    .stMarkdown {
        color: white;
    }
    
    .stDataFrame {
        background-color: #1a1a1a;
        color: white;
    }
    
    /* Custom scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e3a8a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }
    
    /* Style navigation buttons for dark theme */
    .stButton > button {
        background-color: #1a1a1a !important;
        color: #ccc !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #333 !important;
        color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:focus {
        background-color: #333 !important;
        color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load all the saved data and models."""
    try:
        # Define the correct path to the models directory
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Load predictions
        predictions_path = os.path.join(models_path, 'enhanced_predictions.csv')
        predictions_df = pd.read_csv(predictions_path)
        
        # Load metrics
        metrics_path = os.path.join(models_path, 'enhanced_model_metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        
        # Load ensemble weights
        weights_path = os.path.join(models_path, 'ensemble_weights.csv')
        weights_df = pd.read_csv(weights_path)
        
        # Load feature columns
        feature_path = os.path.join(models_path, 'enhanced_feature_columns.txt')
        with open(feature_path, 'r') as f:
            feature_columns = f.read().splitlines()
        
        # Load models (if available)
        try:
            xgb_path = os.path.join(models_path, 'enhanced_xgboost_model.joblib')
            scaler_path = os.path.join(models_path, 'enhanced_scaler.joblib')
            ensemble_path = os.path.join(models_path, 'ensemble_models.joblib')
            
            xgb_model = joblib.load(xgb_path)
            scaler = joblib.load(scaler_path)
            ensemble_models = joblib.load(ensemble_path)
        except Exception as e:
            st.warning(f"Could not load some model files: {str(e)}")
            xgb_model, scaler, ensemble_models = None, None, None
            
        return predictions_df, metrics_df, weights_df, feature_columns, xgb_model, scaler, ensemble_models
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure you're running the dashboard from the correct directory and all model files exist.")
        return None, None, None, None, None, None, None

def main():
    # Initialize session state for navigation
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Overview"
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Bitcoin Prediction Dashboard</h1>
        <p style="margin: 0; font-size: 1.2rem;">High-Performance ML Model with R² = 0.8158</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    predictions_df, metrics_df, weights_df, feature_columns, xgb_model, scaler, ensemble_models = load_data()
    
    if predictions_df is None:
        st.error("Could not load data. Please ensure all model files are in the models/ directory.")
        st.info("""
        **Troubleshooting:**
        1. Make sure you're in the correct directory: `bitcoin-timeseries-predictor/`
        2. Ensure the `models/` folder contains all required files
        3. Run: `streamlit run deployment/dashboard.py`
        """)
        return
    
    # Stock-style header with key metrics
    if not metrics_df.empty:
        metrics = metrics_df.iloc[0]
        
        st.markdown(f"""
        <div class="stock-header">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h2 style="margin: 0; color: #3b82f6;">BTC-USD</h2>
                    <p style="margin: 0; color: #ccc;">Bitcoin Prediction Model</p>
                </div>
                <div style="text-align: right;">
                    <div class="metric-value positive-change">0.8158</div>
                    <div class="metric-label">R² Score</div>
                </div>
            </div>
            <div class="stats-grid">
                <div class="metric-card">
                    <div class="metric-value">${metrics.get('MAE', 0):.0f}</div>
                    <div class="metric-label">Mean Absolute Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.get('RMSE', 0):.0f}</div>
                    <div class="metric-label">Root Mean Square Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('Directional_Accuracy', 0):.1%}</div>
                    <div class="metric-label">Directional Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('Pearson_Correlation', 0):.3f}</div>
                    <div class="metric-label">Correlation</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Updated navigation with only main sections
    nav_sections = [
        "Overview", "Performance", "Architecture", "Features", "Technical"
    ]
    nav_cols = st.columns(len(nav_sections))
    for i, section in enumerate(nav_sections):
        if nav_cols[i].button(section, key=f"nav_{section.lower()}", use_container_width=True):
            st.session_state.current_tab = section
    selected_tab = st.session_state.current_tab
    
    if selected_tab == "Overview":
        show_overview(metrics_df, weights_df)
    elif selected_tab == "Performance":
        show_model_performance(metrics_df, predictions_df)
    elif selected_tab == "Architecture":
        show_model_architecture(weights_df, ensemble_models)
    elif selected_tab == "Features":
        show_feature_analysis(feature_columns, xgb_model)
    elif selected_tab == "Technical":
        show_technical_details(metrics_df, feature_columns)

def show_overview(metrics_df, weights_df):
    """Show project overview in CNN Markets style."""
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.header("📊 Model Overview")
    
    if not metrics_df.empty:
        metrics = metrics_df.iloc[0]
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value positive-change">{:.4f}</div>
                <div class="metric-label">R² Score</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">81.58% variance explained</div>
            </div>
            """.format(metrics.get('R2', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">${:.0f}</div>
                <div class="metric-label">Mean Absolute Error</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Average prediction error</div>
            </div>
            """.format(metrics.get('MAE', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Directional Accuracy</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Correct direction predictions</div>
            </div>
            """.format(metrics.get('Directional_Accuracy', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.3f}</div>
                <div class="metric-label">Pearson Correlation</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Linear correlation with actual</div>
            </div>
            """.format(metrics.get('Pearson_Correlation', 0)), unsafe_allow_html=True)
    
    # Ensemble composition
    if not weights_df.empty:
        st.subheader("🏗️ Ensemble Model Composition")
        
        weights = weights_df.iloc[0].to_dict()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create ensemble weights visualization
            models = list(weights.keys())
            weight_values = list(weights.values())
            
            fig = go.Figure(data=[go.Pie(labels=models, values=weight_values, hole=0.3)])
            fig.update_layout(
                title="Model Weights Distribution",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Model Breakdown:**")
            for model, weight in weights.items():
                st.markdown(f"- **{model}**: {weight:.1%}")
    
    # Project highlights
    st.subheader("🎯 Key Achievements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="achievement-card">
            <div class="achievement-title">🔧 150+ Features</div>
            <div class="achievement-desc">Comprehensive feature engineering including technical indicators, moving averages, and time-based features.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="achievement-card">
            <div class="achievement-title">🤝 4 Model Ensemble</div>
            <div class="achievement-desc">XGBoost, LightGBM, Random Forest, and Gradient Boosting combined for superior performance.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="achievement-card">
            <div class="achievement-title">📊 Production Ready</div>
            <div class="achievement-desc">Comprehensive evaluation with 20+ metrics including financial and trading-specific measures.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance(metrics_df, predictions_df):
    """Show model performance in CNN Markets style."""
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.header("📈 Model Performance")
    
    if metrics_df.empty:
        st.error("No metrics data available.")
        return
    
    metrics = metrics_df.iloc[0]
    
    # Performance metrics comparison
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value positive-change">{:.4f}</div>
            <div class="metric-label">R² Score</div>
        </div>
        """.format(metrics.get('R2', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">${:.0f}</div>
            <div class="metric-label">MAE</div>
        </div>
        """.format(metrics.get('MAE', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">${:.0f}</div>
            <div class="metric-label">RMSE</div>
        </div>
        """.format(metrics.get('RMSE', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}%</div>
            <div class="metric-label">MAPE</div>
        </div>
        """.format(metrics.get('MAPE', 0) * 100), unsafe_allow_html=True)
    
    # Time series metrics
    st.subheader("⏰ Time Series Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.4f}</div>
            <div class="metric-label">MASE</div>
        </div>
        """.format(metrics.get('MASE', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}%</div>
            <div class="metric-label">RMSPE</div>
        </div>
        """.format(metrics.get('RMSPE', 0) * 100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">Directional Accuracy</div>
        </div>
        """.format(metrics.get('Directional_Accuracy', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.4f}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """.format(metrics.get('Direction_F1', 0)), unsafe_allow_html=True)
    
    # Performance visualization
    if not predictions_df.empty:
        st.subheader("📈 Predictions vs Actual")
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create time series plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=predictions_df['Actual'],
            name='Actual Price',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=predictions_df['Predicted'],
            name='Predicted Price',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Bitcoin Price Predictions vs Actual Values",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=500,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_architecture(weights_df, ensemble_models):
    """Show model architecture in CNN Markets style."""
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.header("⚙️ Model Architecture")
    
    st.subheader("🏗️ Ensemble Model Composition")
    
    if not weights_df.empty:
        weights = weights_df.iloc[0].to_dict()
        
        # Model details
        models_info = {
            'XGBoost': {
                'type': 'Gradient Boosting',
                'strengths': 'High performance, regularization, feature importance',
                'parameters': 'n_estimators=1000, learning_rate=0.05, max_depth=6'
            },
            'LightGBM': {
                'type': 'Gradient Boosting',
                'strengths': 'Fast training, memory efficient, categorical support',
                'parameters': 'n_estimators=1000, learning_rate=0.05, max_depth=6'
            },
            'RandomForest': {
                'type': 'Bagging',
                'strengths': 'Robust to outliers, parallelizable, feature importance',
                'parameters': 'n_estimators=200, max_depth=10'
            },
            'GradientBoosting': {
                'type': 'Gradient Boosting',
                'strengths': 'Proven method, stable, interpretable',
                'parameters': 'n_estimators=200, learning_rate=0.1'
            }
        }
        
        for model_name, weight in weights.items():
            if model_name in models_info:
                info = models_info[model_name]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{model_name} ({weight:.1%})</h4>
                    <p><strong>Type:</strong> {info['type']}</p>
                    <p><strong>Strengths:</strong> {info['strengths']}</p>
                    <p><strong>Key Parameters:</strong> {info['parameters']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.subheader("🔄 Ensemble Combination Strategy")
    
    st.markdown("""
    <div class="metric-card">
        <h4>Weighted Averaging Approach</h4>
        <p><strong>Performance-Based Weights:</strong> Each model's weight is proportional to its R² score</p>
        <p><strong>Normalization:</strong> Weights are normalized to sum to 1</p>
        <p><strong>Fallback:</strong> Equal weights if all models perform poorly</p>
        <p><strong>Final Prediction:</strong> Weighted average of all model predictions</p>
        
        <h5>Mathematical Formula:</h5>
        <code>Ensemble_Prediction = Σ(w_i × y_i)</code><br>
        <code>Where: w_i = weight of model i, y_i = prediction of model i</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎯 Why Ensemble Learning Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Bias-Variance Tradeoff</h4>
            <ul>
                <li>Reduces variance through averaging</li>
                <li>Maintains bias at reasonable levels</li>
                <li>Improves generalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Diversity Principle</h4>
            <ul>
                <li>Different algorithms capture different patterns</li>
                <li>Errors don't correlate perfectly</li>
                <li>Complementary strengths</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_feature_analysis(feature_columns, xgb_model):
    """Show feature analysis in CNN Markets style."""
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.header("📊 Feature Analysis")
    
    if not feature_columns:
        st.error("No feature columns data available.")
        return
    
    st.subheader("🔍 Feature Categories")
    
    # Categorize features
    feature_categories = {
        'Moving Averages': [f for f in feature_columns if 'MA_' in f or 'EMA_' in f],
        'Technical Indicators': [f for f in feature_columns if any(indicator in f for indicator in ['RSI', 'MACD', 'BB_'])],
        'Time Features': [f for f in feature_columns if any(time_feat in f for time_feat in ['Hour', 'Day', 'Month', 'sin', 'cos'])],
        'Lag Features': [f for f in feature_columns if 'Lag_' in f],
        'Volatility': [f for f in feature_columns if 'Volatility' in f],
        'Rolling Statistics': [f for f in feature_columns if 'Rolling_' in f],
        'Patterns': [f for f in feature_columns if any(pattern in f for pattern in ['Higher', 'Lower', 'Double', 'Support', 'Resistance'])],
        'Other': [f for f in feature_columns if not any(cat in f for cat in ['MA_', 'EMA_', 'RSI', 'MACD', 'BB_', 'Hour', 'Day', 'Month', 'sin', 'cos', 'Lag_', 'Volatility', 'Rolling_', 'Higher', 'Lower', 'Double', 'Support', 'Resistance'])]
    }
    
    # Display feature counts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Distribution:**")
        for category, features in feature_categories.items():
            if features:
                st.markdown(f"- **{category}**: {len(features)} features")
    
    with col2:
        # Create pie chart of feature categories
        categories = [cat for cat, features in feature_categories.items() if features]
        counts = [len(features) for cat, features in feature_categories.items() if features]
        
        fig = go.Figure(data=[go.Pie(labels=categories, values=counts)])
        fig.update_layout(title="Feature Categories Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🎯 Top Features by Category")
    
    # Show top features from each category
    for category, features in feature_categories.items():
        if features:
            st.markdown(f"**{category}:**")
            # Show first 10 features from each category
            for feature in features[:10]:
                st.markdown(f"- {feature}")
            if len(features) > 10:
                st.markdown(f"- ... and {len(features) - 10} more")
            st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_technical_details(metrics_df, feature_columns):
    """Show technical details in CNN Markets style."""
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.header("🔧 Technical Details")
    
    st.subheader("📊 Complete Metrics Summary")
    
    if not metrics_df.empty:
        metrics = metrics_df.iloc[0]
        
        # Display all metrics in a table
        metrics_data = []
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'R2' in metric or 'Accuracy' in metric or 'Correlation' in metric:
                    formatted_value = f"{value:.4f}"
                elif 'MAE' in metric or 'RMSE' in metric:
                    formatted_value = f"${value:.2f}"
                elif 'MAPE' in metric or 'RMSPE' in metric:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
                metrics_data.append([metric, formatted_value])
        
        metrics_df_display = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
        st.dataframe(metrics_df_display, use_container_width=True)
    
    st.subheader("🏗️ Technical Architecture")
    
    st.markdown("""
    <div class="metric-card">
        <h4>Data Pipeline</h4>
        <ol>
            <li><strong>Data Loading:</strong> Kaggle Bitcoin dataset (hourly prices)</li>
            <li><strong>Feature Engineering:</strong> 150+ features including technical indicators</li>
            <li><strong>Preprocessing:</strong> RobustScaler, time series split</li>
            <li><strong>Modeling:</strong> Ensemble of 4 algorithms</li>
            <li><strong>Evaluation:</strong> 20+ comprehensive metrics</li>
            <li><strong>Deployment:</strong> Model persistence and monitoring</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4>Key Technologies</h4>
        <ul>
            <li><strong>Python:</strong> pandas, numpy, scikit-learn</li>
            <li><strong>ML Libraries:</strong> XGBoost, LightGBM</li>
            <li><strong>Optimization:</strong> Optuna for hyperparameter tuning</li>
            <li><strong>Visualization:</strong> Plotly for interactive charts</li>
            <li><strong>Deployment:</strong> Streamlit for dashboard</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📈 Performance Improvements")
    
    improvement_data = {
        'Metric': ['R² Score', 'MAE', 'RMSE'],
        'Before': [0.5836, 9228, 16774],
        'After': [0.8158, 384.93, 593.92],
        'Improvement': ['+39.8%', '-95.8%', '-96.5%']
    }
    
    improvement_df = pd.DataFrame(improvement_data)
    st.dataframe(improvement_df, use_container_width=True)
    
    st.subheader("🔍 Feature Engineering Details")
    
    if feature_columns:
        st.markdown(f"**Total Features Created: {len(feature_columns)}**")
        
        # Show feature engineering breakdown
        feature_breakdown = {
            'Moving Averages': len([f for f in feature_columns if 'MA_' in f or 'EMA_' in f]),
            'Technical Indicators': len([f for f in feature_columns if any(indicator in f for indicator in ['RSI', 'MACD', 'BB_'])]),
            'Time Features': len([f for f in feature_columns if any(time_feat in f for time_feat in ['Hour', 'Day', 'Month', 'sin', 'cos'])]),
            'Lag Features': len([f for f in feature_columns if 'Lag_' in f]),
            'Volatility': len([f for f in feature_columns if 'Volatility' in f]),
            'Rolling Statistics': len([f for f in feature_columns if 'Rolling_' in f]),
            'Patterns': len([f for f in feature_columns if any(pattern in f for pattern in ['Higher', 'Lower', 'Double', 'Support', 'Resistance'])]),
            'Other': len([f for f in feature_columns if not any(cat in f for cat in ['MA_', 'EMA_', 'RSI', 'MACD', 'BB_', 'Hour', 'Day', 'Month', 'sin', 'cos', 'Lag_', 'Volatility', 'Rolling_', 'Higher', 'Lower', 'Double', 'Support', 'Resistance'])])
        }
        
        breakdown_df = pd.DataFrame(list(feature_breakdown.items()), columns=['Category', 'Count'])
        st.dataframe(breakdown_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans                    # NEW: ML Clustering
from sklearn.ensemble import RandomForestClassifier   # NEW: ML Prediction
from sklearn.decomposition import PCA                 # NEW: Dimensionality
from sklearn.ensemble import IsolationForest          # NEW: Anomaly Detection
from xgboost import XGBRegressor                      # NEW: Advanced ML
import warnings
import sys
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Load and clean data (same as before)
df = pd.read_csv('World_Top_Companies_Master_Dataset.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
numeric_cols = ['price_gbp', 'revenue_ttm', 'marketcap', 'earnings_ttm', 'pe_ratio_ttm', 'dividend_yield_ttm']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("ML Analysis Starting...")

# =============================================================================
# NEW: ML ALGORITHM 1 - K-MEANS CLUSTERING (Company Groups)
# =============================================================================
features = df[['marketcap', 'revenue_ttm', 'pe_ratio_ttm', 'earnings_ttm']].fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Visualize clusters
fig_cluster = px.scatter(df.nlargest(1000, 'marketcap'), 
                        x='marketcap', y='revenue_ttm',
                        color='cluster', size='price_gbp',
                        hover_name='name', 
                        title="🤖 ML CLUSTERING: 5 Company Groups",
                        log_x=True, log_y=True)
fig_cluster.show()

# =============================================================================
# NEW: ML ALGORITHM 2 - HIGH-GROWTH PREDICTION (Random Forest)
# =============================================================================
# Target: Is this a high-growth company? (Top 20% revenue/marketcap)
df['high_growth'] = ((df['revenue_ttm'] > df['revenue_ttm'].quantile(0.8)) & 
                    (df['marketcap'] > df['marketcap'].quantile(0.8))).astype(int)

X = df[['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm']].fillna(0)
y = df['high_growth']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

df['growth_probability'] = rf_model.predict_proba(X)[:, 1]  # Probability 0-1

# Top predicted growth stocks
top_growth = df.nlargest(10, 'growth_probability')[['name', 'growth_probability', 'country']]
print("\n🏆 TOP 10 PREDICTED GROWTH STOCKS:")
print(top_growth)

# =============================================================================
# NEW: ML ALGORITHM 3 - OUTLIER DETECTION (Isolation Forest)
# =============================================================================
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['is_outlier'] = iso_forest.fit_predict(features)
outliers = df[df['is_outlier'] == -1].nlargest(5, 'marketcap')

print("\n🚨 TOP 5 OUTLIER COMPANIES (Unusual patterns):")
print(outliers[['name', 'country', 'marketcap']].round(2))

# =============================================================================
# NEW: ML ALGORITHM 4 - PCA (2D Visualization)
# =============================================================================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features.fillna(0))
df['pca_x'] = pca_result[:, 0]
df['pca_y'] = pca_result[:, 1]

fig_pca = px.scatter(df.nlargest(2000, 'marketcap'), 
                    x='pca_x', y='pca_y', color='cluster',
                    hover_name='name', size='marketcap',
                    title="🔍 PCA: Companies in 2D Space (ML Dimensionality Reduction)")
fig_pca.show()

# =============================================================================
# NEW: ML ALGORITHM 5 - XGBoost (Growth Ranking)
# =============================================================================
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X, df['revenue_ttm'])
df['xgb_growth_score'] = xgb_model.predict(X)

top_xgb = df.nlargest(10, 'xgb_growth_score')[['name', 'xgb_growth_score']]
print("\n⚡ XGBoost TOP 10 GROWTH RANKINGS:")

# =============================================================================
# ORIGINAL VISUALIZATIONS (Enhanced with ML)
# =============================================================================
fig1 = make_subplots(rows=1, cols=2)
top_market = df.nlargest(10, 'marketcap')
fig1.add_trace(go.Bar(y=[name[:25] for name in top_market['name']], 
                      x=top_market['marketcap']/1e12, orientation='h'), row=1, col=1)
fig1.show()

# ML Results Table
ml_results = df.nlargest(15, 'growth_probability')[['name', 'country', 'growth_probability', 
                                                  'cluster', 'is_outlier', 'xgb_growth_score']].round(3)

fig_ml_table = go.Figure(data=[go.Table(
    header=dict(values=['Company', 'Country', 'Growth Prob', 'Cluster', 'Outlier', 'XGB Score'],
                fill_color='#1f77b4', font_color='white'),
    cells=dict(values=[ml_results['name'].str[:25], ml_results['country'],
                    [f"{x:.1%}" for x in ml_results['growth_probability']],
                    ml_results['cluster'], ml_results['is_outlier'],
                    ml_results['xgb_growth_score'].round(1)]))
])
fig_ml_table.update_layout(title="🎯 ML RESULTS: Top Growth Predictions", height=500)
fig_ml_table.show()

print("\n" + "="*80)
print("✅ ML ANALYSIS COMPLETE!")
print(f"📊 5 ML Algorithms Applied:")
print("1. K-Means Clustering → 5 company groups")
print("2. Random Forest → Growth probability predictions") 
print("3. Isolation Forest → Outlier detection")
print("4. PCA → 2D visualization")
print("5. XGBoost → Growth ranking")
print(f"🚀 Top Growth Prediction: {top_growth.iloc[0]['name']} ({top_growth.iloc[0]['growth_probability']:.1%})")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import shap  # pip install shap
import joblib # pip install joblib
import warnings
import sys
import os

# =============================================================================
# 🎨 THEME: FORCE DARK MODE (TradingView Look)
# =============================================================================
pio.templates.default = "plotly_dark"

TRADINGVIEW_BG = '#131722'  
TRADINGVIEW_BLUE = '#2962FF'
TRADINGVIEW_GREEN = '#00E676'
TRADINGVIEW_RED = '#FF0055'

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================
def load_data():
    print("⏳ [SYSTEM] Loading Financial Data...", end="\r")
    try:
        df = pd.read_csv('World_Top_Companies_Master_Dataset.csv')
    except FileNotFoundError:
        print("❌ Error: 'World_Top_Companies_Master_Dataset.csv' not found.")
        sys.exit()

    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
    
    numeric_cols = ['price_gbp', 'revenue_ttm', 'marketcap', 'earnings_ttm', 'pe_ratio_ttm', 'dividend_yield_ttm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Robust Cleaning
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Feature Engineering
    df['profit_margin'] = np.where(df['revenue_ttm'] > 0, df['earnings_ttm'] / df['revenue_ttm'], 0)
    df['price_to_sales'] = np.where(df['revenue_ttm'] > 0, df['marketcap'] / df['revenue_ttm'], 0)
    
    # Units
    df['marketcap_trillion'] = df['marketcap'] / 1e12
    df['revenue_billions'] = df['revenue_ttm'] / 1e9
    df['earnings_billions'] = df['earnings_ttm'] / 1e9
    
    print("✅ [SYSTEM] Data Loaded & Engineered.   ")
    return df

# =============================================================================
# 2. ADVANCED ML PIPELINE
# =============================================================================
def train_and_evaluate(df):
    print("⚙️  [SYSTEM] Training & Evaluating Models...", end="\r")
    
    features = ['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm', 'profit_margin', 'price_to_sales']
    X = df[features]
    
    # Target: High Growth (Top 20%)
    df['high_growth'] = ((df['revenue_ttm'] > df['revenue_ttm'].quantile(0.8)) & 
                        (df['marketcap'] > df['marketcap'].quantile(0.8))).astype(int)
    y = df['high_growth']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'financial_brain.pkl')
    
    # Predictions
    y_pred_test = rf_model.predict(X_test)
    y_probs_test = rf_model.predict_proba(X_test)[:, 1]
    
    # Apply to Full Dataset
    df['growth_probability'] = rf_model.predict_proba(X)[:, 1]

    # Unsupervised
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    df['cluster'] = kmeans.labels_
    df['cluster_label'] = "Group " + (df['cluster'] + 1).astype(str)
    
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    df['is_outlier'] = iso_forest.fit_predict(X)
    df['outlier_status'] = df['is_outlier'].map({1: 'Normal', -1: '⚠️ ANOMALY'})

    # PCA
    pca = PCA(n_components=2).fit_transform(X)
    df['pca_x'] = pca[:, 0]
    df['pca_y'] = pca[:, 1]

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, df['revenue_ttm'])
    df['xgb_growth_score'] = xgb_model.predict(X)

    acc = accuracy_score(y_test, y_pred_test)
    print(f"✅ [SYSTEM] Training Complete. Test Accuracy: {acc:.1%} | Model Saved.")
    
    return df, rf_model, X_test, y_test, y_probs_test

# =============================================================================
# 3. VISUALIZATION FUNCTIONS
# =============================================================================

def apply_tradingview_style(fig):
    fig.update_layout(
        paper_bgcolor=TRADINGVIEW_BG, plot_bgcolor=TRADINGVIEW_BG,
        font={'color': '#D9D9D9', 'family': 'Roboto, monospace'},
        title={'font': {'size': 24, 'color': TRADINGVIEW_BLUE}},
        xaxis=dict(gridcolor='#363c4e'), yaxis=dict(gridcolor='#363c4e')
    )
    return fig

# --- BROWSER-BASED ANALYST REPORT (Split Layout) ---
def get_analyst_report_fig(df, rf_model, X, target_country=None):
    """Generates a Dashboard Report: Table on Top, Narrative Text at Bottom"""
    
    # 1. Filter Logic
    if target_country:
        subset = df[df['country'].str.lower() == target_country.lower()]
        if subset.empty:
            print(f"❌ No data for {target_country}")
            return None
        top_pick_idx = subset['growth_probability'].idxmax()
        title_text = f"Analyst Report: Top Pick in {target_country.title()}"
    else:
        top_pick_idx = df['growth_probability'].idxmax()
        title_text = "Analyst Report: Global #1 Top Pick"

    top_pick = df.loc[top_pick_idx]
    
    # 2. SHAP Calculation
    explainer = shap.TreeExplainer(rf_model)
    features_list = ['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm', 'profit_margin', 'price_to_sales']
    company_features = df.loc[[top_pick_idx], features_list]
    shap_values = explainer.shap_values(company_features)
    
    if isinstance(shap_values, list): vals = shap_values[1]
    else: vals = np.array(shap_values)
    if len(vals.shape) == 3: vals = vals[:, :, 1]
    vals = np.array(vals).flatten()
    
    explanation = pd.DataFrame({'Feature': features_list, 'Impact': vals, 'Value': company_features.iloc[0].values})
    explanation['Abs_Impact'] = explanation['Impact'].abs()
    explanation = explanation.sort_values('Abs_Impact', ascending=False)
    
    # 3. Data Preparation
    name_map = {'price_gbp': 'Share Price', 'pe_ratio_ttm': 'P/E Ratio', 'earnings_ttm': 'Annual Earnings', 
                'dividend_yield_ttm': 'Dividend Yield', 'profit_margin': 'Profit Margin', 'price_to_sales': 'P/S Ratio'}

    def fmt_money(v):
        if v > 1e12: return f"${v/1e12:.2f}T"
        if v > 1e9: return f"${v/1e9:.2f}B"
        return f"${v:,.0f}"

    # Build Table Rows (Only Stats, No Long Text)
    rows = []
    rows.append(["🏆 ASSET NAME", top_pick['name']])
    rows.append(["🎯 GROWTH SCORE", f"{top_pick['growth_probability']:.1%}"])
    rows.append(["📍 SECTOR/GROUP", top_pick['cluster_label']])
    
    for i in range(3):
        row = explanation.iloc[i]
        feat_name = name_map.get(row['Feature'], row['Feature'])
        sentiment = "✅ POSITIVE (+)" if row['Impact'] > 0 else "⚠️ NEGATIVE (-)"
        val = row['Value']
        val_str = f"{val*100:.1f}%" if 'margin' in row['Feature'] or 'yield' in row['Feature'] else fmt_money(val)
        if 'ratio' in row['Feature'] or 'sales' in row['Feature']: val_str = f"{val:.2f}x"
        rows.append([f"DRIVER #{i+1}: {feat_name}", f"Value: {val_str} | Impact: {sentiment}"])

    # Build Narrative Text (HTML Formatted)
    feature_1 = name_map.get(explanation.iloc[0]['Feature'])
    val_1 = fmt_money(explanation.iloc[0]['Value']) if explanation.iloc[0]['Value'] > 1000 else f"{explanation.iloc[0]['Value']:.2f}"
    feature_2 = name_map.get(explanation.iloc[1]['Feature'])
    
    narrative_html = (
        f"<b>🤖 AI VERDICT & INVESTMENT THESIS:</b><br>"
        f"<span style='font-size: 14px; color: #CCCCCC;'>"
        f"The AI model has identified <b>{top_pick['name']}</b> as the standout performer in this region with a Growth Confidence Score of "
        f"<span style='color: #00E676;'><b>{top_pick['growth_probability']:.1%}</b></span>. "
        f"<br><br>"
        f"This high conviction is primarily driven by its exceptional <b>{feature_1}</b> of {val_1}, which the model weights heavily as a signal of financial dominance. "
        f"Additionally, the company's <b>{feature_2}</b> acts as a secondary catalyst, reinforcing the 'High Growth' classification. "
        f"Unlike its peers in {top_pick['cluster_label']}, this asset demonstrates a unique combination of scale and efficiency that aligns perfectly with the model's learned criteria for future appreciation."
        f"</span>"
    )

    # 4. Create Layout with Table + Text Annotation
    fig = go.Figure()

    # Add Table (Top 60% of screen)
    fig.add_trace(go.Table(
        domain=dict(x=[0, 1], y=[0.4, 1]), # Takes top 60%
        columnorder=[1, 2], columnwidth=[80, 200],
        header=dict(values=[title_text, "Analysis Details"],
                    line_color='#333', fill_color=TRADINGVIEW_BLUE,
                    align='left', font=dict(color='white', size=16)),
        cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows]],
                   line_color='#333', fill_color=TRADINGVIEW_BG,
                   align='left', font=dict(color='#D9D9D9', size=14),
                   height=35)
    ))

    # Add Narrative Text as an Annotation (Bottom 40% of screen)
    fig.update_layout(
        paper_bgcolor=TRADINGVIEW_BG,
        margin=dict(l=20,r=20,t=40,b=20),
        annotations=[
            dict(
                x=0.01, y=0.35, # Position below table
                xref="paper", yref="paper",
                text=narrative_html,
                showarrow=False,
                align="left",
                font=dict(family="Roboto", size=14, color="white"),
                bordercolor="#333", borderwidth=1,
                bgcolor="#1E222D",
                width=1400 # Max width to force wrapping if needed, but HTML handles it
            )
        ]
    )
    
    return fig

# --- CHART FUNCTIONS ---
def get_market_overview(df):
    fig = px.bar(df.nlargest(10, 'marketcap'), y='name', x='marketcap_trillion', orientation='h',
                 title="🌍 GLOBAL TITANS: Top 10 by Market Cap", text_auto='$.2fT',
                 color='marketcap_trillion', color_continuous_scale=['#2962FF', '#00E676'])
    return apply_tradingview_style(fig)

def get_clustering(df):
    fig = px.scatter(df.nlargest(500, 'marketcap'), x='marketcap_trillion', y='revenue_billions',
                     color='cluster_label', size='price_gbp', hover_name='name',
                     title="🧬 MARKET SEGMENTATION (K-Means)", log_x=True, log_y=True)
    return apply_tradingview_style(fig)

def get_3d_scatter(df):
    fig = px.scatter_3d(df.nlargest(200, 'marketcap'), 
                        x='marketcap_trillion', y='revenue_billions', z='earnings_billions',
                        color='cluster_label', hover_name='name', size='price_gbp',
                        title="🧊 3D MARKET CUBE")
    fig.update_layout(scene = dict(xaxis=dict(backgroundcolor=TRADINGVIEW_BG),
                                   yaxis=dict(backgroundcolor=TRADINGVIEW_BG),
                                   zaxis=dict(backgroundcolor=TRADINGVIEW_BG)),
                      paper_bgcolor=TRADINGVIEW_BG, font={'color': 'white'})
    return fig

def get_growth_table(df):
    results = df.nlargest(20, 'growth_probability')
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Company', 'Country', 'Growth Prob', 'Mkt Cap ($T)', 'Status'],
                    fill_color=TRADINGVIEW_BLUE, font=dict(color='white', size=14)),
        cells=dict(values=[results['name'].str[:20], results['country'], 
                           [f"🔥 {x:.1%}" for x in results['growth_probability']],
                           [f"${x:.2f}T" for x in results['marketcap_trillion']],
                           results['outlier_status']],
                   fill_color='#1E222D', font=dict(color='lightgrey'), height=30))
    ])
    fig.update_layout(title="🏆 AI PREDICTIONS: Top Growth Stocks", paper_bgcolor=TRADINGVIEW_BG)
    return fig

def get_shap_summary(rf_model, X_test):
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    vals = None
    if isinstance(shap_values, list): vals = shap_values[1]
    else: vals = np.array(shap_values)
    if len(vals.shape) == 3: vals = vals[:, :, 1]
    
    if len(vals.shape) != 2: print("⚠️ SHAP Shape Issue."); return go.Figure()

    shap_sum = np.abs(vals).mean(axis=0)
    df_shap = pd.DataFrame({'Feature': X_test.columns, 'Impact': shap_sum})
    df_shap = df_shap.sort_values('Impact', ascending=True)
    
    fig = px.bar(df_shap, x='Impact', y='Feature', orientation='h',
                 title="🧠 SHAP VALUES: Feature Impact (Global Logic)",
                 color='Impact', color_continuous_scale='Plasma')
    return apply_tradingview_style(fig)

def get_roc_curve(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f"📈 ROC CURVE (AUC = {roc_auc:.2f})",
                  labels=dict(x='False Positive Rate', y='True Positive Rate'))
    fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
    return apply_tradingview_style(fig)

def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, title=f"✅ CONFUSION MATRIX (Acc: {acc:.1%})",
                    labels=dict(x="Predicted Class", y="Actual Class"), color_continuous_scale='Viridis',
                    x=['Low Growth', 'High Growth'], y=['Low Growth', 'High Growth'])
    return apply_tradingview_style(fig)

def get_rotating_globe(df):
    country_stats = df.groupby('country')[['marketcap_trillion', 'revenue_billions']].sum().reset_index()
    fig = px.scatter_geo(country_stats, locations="country", locationmode='country names',
                         size="marketcap_trillion", hover_name="country",
                         hover_data=["revenue_billions"], color="marketcap_trillion",
                         projection="orthographic", title="🌐 3D INTERACTIVE GLOBE",
                         color_continuous_scale='Plasma')
    fig.update_layout(geo=dict(bgcolor=TRADINGVIEW_BG, showframe=False, projection_type='orthographic', 
                               landcolor='#1E222D', oceancolor=TRADINGVIEW_BG, lakecolor=TRADINGVIEW_BG),
                      paper_bgcolor=TRADINGVIEW_BG, font={'color': 'white'})
    return fig

def get_outliers(df):
    fig = px.scatter(df.nlargest(1000, 'marketcap'), x='earnings_billions', y='revenue_billions',
                     color='outlier_status', size='marketcap', hover_name='name',
                     title="🚨 ANOMALY DETECTION", color_discrete_map={'Normal': '#00E676', '⚠️ ANOMALY': '#FF0055'},
                     log_x=True, log_y=True)
    return apply_tradingview_style(fig)

def get_pca(df):
    fig = px.scatter(df.nlargest(2000, 'marketcap'), x='pca_x', y='pca_y', color='cluster_label', 
                     hover_name='name', title="🔍 PCA VISUALIZATION")
    return apply_tradingview_style(fig)

def get_feature_importance(rf_model, X): 
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    fig = px.bar(importances, orientation='h', title="🧠 STANDARD FEATURE IMPORTANCE",
                 color=importances.values, color_continuous_scale='Plasma')
    return apply_tradingview_style(fig)

def get_correlation(df):
    corr = df[['marketcap', 'revenue_ttm', 'price_gbp', 'earnings_ttm', 'pe_ratio_ttm']].corr()
    fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="🔥 CORRELATION MATRIX",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    return apply_tradingview_style(fig)

def get_elbow(features):
    inertia = []
    for k in range(1, 10):
        inertia.append(KMeans(n_clusters=k, random_state=42).fit(features).inertia_)
    fig = px.line(x=list(range(1, 10)), y=inertia, markers=True, title="💪 ELBOW METHOD",
                  labels={'x': 'Clusters', 'y': 'Error'})
    fig.update_traces(line_color=TRADINGVIEW_BLUE)
    return apply_tradingview_style(fig)

def get_geo_map(df):
    country_group = df.groupby('country')[['marketcap_trillion', 'name']].agg({'marketcap_trillion': 'sum', 'name': 'count'}).reset_index()
    fig = px.choropleth(country_group, locations="country", locationmode='country names',
                        color="marketcap_trillion", title="🌍 WORLD ECONOMY (FLAT MAP)",
                        color_continuous_scale="Plasma")
    fig.update_geos(bgcolor=TRADINGVIEW_BG, lakecolor=TRADINGVIEW_BG)
    fig.update_layout(paper_bgcolor=TRADINGVIEW_BG, font={'color': 'white'})
    return fig

def get_xgboost_perf(df):
    fig = px.scatter(x=df['revenue_billions'], y=df['xgb_growth_score']/1e9,
                     labels={'x': 'Actual Revenue', 'y': 'Predicted'},
                     title="📈 XGBoost Regression Accuracy", trendline="ols", trendline_color_override='#00E676')
    return apply_tradingview_style(fig)

# --- 99. HTML Report ---
def generate_full_report(figs):
    print("⏳ Generating HTML Report...")
    with open("ML_Advanced_Report.html", 'w', encoding="utf-8") as f:
        f.write("<html><head><title>Advanced ML Analysis</title></head><body style='background-color:#131722; color:white;'>")
        f.write("<h1 style='text-align:center;'>🚀 Advanced AI Financial Analysis</h1>")
        for fig in figs:
            if fig: f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")
    print("✅ Report Saved as 'ML_Advanced_Report.html'")
    try: os.system("start ML_Advanced_Report.html")
    except: pass

# =============================================================================
# 4. MAIN MENU
# =============================================================================
def main():
    try: import shap
    except ImportError: print("⚠️ Warning: SHAP library not found. Install via 'pip install shap'.")

    df = load_data()
    features = ['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm', 'profit_margin', 'price_to_sales']
    X_full = df[features]
    
    df, rf_model, X_test, y_test, y_probs_test = train_and_evaluate(df)
    y_pred_test = rf_model.predict(X_test)
    
    while True:
        print("\n" + "="*50)
        print("   🚀 ADVANCED ANALYTICS SUITE (FULL)")
        print("="*50)
        print(" 1. 🌍 Market Overview ($T)")
        print(" 2. 🧬 Clustering Groups (2D)")
        print(" 3. 🧊 3D Clustering Cube")
        print(" 4. 🧠 SHAP Value Analysis [Global Logic]")
        print(" 5. 📈 ROC-AUC Curve")
        print(" 6. ✅ Confusion Matrix")
        print(" 7. 🏆 Top Growth Predictions [Table]")
        print(" 8. 🎤 AI ANALYST REPORT (Top Pick)")
        print(" 9. 🧠 Standard Feature Importance")
        print(" 10.🚨 Outlier Detection")
        print(" 11.🔥 Correlation Heatmap")
        print(" 12.💪 Elbow Method")
        print(" 13.🗺️  World Map (Flat)")
        print(" 14.🌐 3D ROTATING GLOBE")
        print("-" * 50)
        print(" 100. 💥 LAUNCH ALL DASHBOARDS")
        print(" 99. 📄 EXPORT FULL HTML REPORT")
        print(" 0.  ❌ Exit")
        print("="*50)
        
        choice = input("👉 Select Analysis: ")

        if choice == '1': get_market_overview(df).show()
        elif choice == '2': get_clustering(df).show()
        elif choice == '3': get_3d_scatter(df).show()
        elif choice == '4': get_shap_summary(rf_model, X_test).show()
        elif choice == '5': get_roc_curve(y_test, y_probs_test).show()
        elif choice == '6': get_confusion_matrix(y_test, y_pred_test).show()
        elif choice == '7': get_growth_table(df).show()
        elif choice == '8': 
            # Ask for country input, then generate BROWSER report
            country_input = input("   🌎 Enter Country Name (or Press Enter for Global): ").strip()
            c_target = country_input if country_input else None
            fig = get_analyst_report_fig(df, rf_model, X_full, target_country=c_target)
            if fig: fig.show()
        elif choice == '9': get_feature_importance(rf_model, X_test).show()
        elif choice == '10': get_outliers(df).show()
        elif choice == '11': get_correlation(df).show()
        elif choice == '12': get_elbow(X_test).show()
        elif choice == '13': get_geo_map(df).show()
        elif choice == '14': get_rotating_globe(df).show()
        elif choice == '99': 
            figs = [get_market_overview(df), get_shap_summary(rf_model, X_test), 
                    get_roc_curve(y_test, y_probs_test), get_confusion_matrix(y_test, y_pred_test),
                    get_growth_table(df), get_rotating_globe(df), get_analyst_report_fig(df, rf_model, X_full)]
            generate_full_report(figs)
        elif choice == '100':
            print("💥 LAUNCHING ALL DASHBOARDS...")
            get_market_overview(df).show()
            get_clustering(df).show()
            get_3d_scatter(df).show()
            get_shap_summary(rf_model, X_test).show()
            get_roc_curve(y_test, y_probs_test).show()
            get_confusion_matrix(y_test, y_pred_test).show()
            get_growth_table(df).show()
            get_analyst_report_fig(df, rf_model, X_full, target_country=None).show()
            get_feature_importance(rf_model, X_test).show()
            get_outliers(df).show()
            get_correlation(df).show()
            get_elbow(X_test).show()
            get_geo_map(df).show()
            get_rotating_globe(df).show()
        elif choice == '0': break
        else: print("❌ Invalid Option.")

if __name__ == "__main__":
    main()
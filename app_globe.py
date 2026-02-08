import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================================================
# 1. LOAD DATA & ML ENGINE
# =============================================================================
print("⏳ Loading Data & Training AI...")
try:
    df = pd.read_csv('World_Top_Companies_Master_Dataset.csv')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
    
    # Clean numeric columns
    numeric_cols = ['price_gbp', 'revenue_ttm', 'marketcap', 'earnings_ttm', 'pe_ratio_ttm', 'dividend_yield_ttm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Safe Cleaning
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Fill text columns
    text_cols = df.select_dtypes(include=['object']).columns
    df[text_cols] = df[text_cols].fillna("Unknown")

    # Feature Engineering
    # Safe division using numpy to avoid crashes
    df['profit_margin'] = np.where(df['revenue_ttm'] > 0, df['earnings_ttm'] / df['revenue_ttm'], 0)
    df['price_to_sales'] = np.where(df['revenue_ttm'] > 0, df['marketcap'] / df['revenue_ttm'], 0)
    
    shares_outstanding = np.where(df['price_gbp'] > 0, df['marketcap'] / df['price_gbp'], 1)
    df['revenue_per_share'] = df['revenue_ttm'] / shares_outstanding

    # Units
    df['marketcap_trillion'] = df['marketcap'] / 1e12
    df['revenue_billions'] = df['revenue_ttm'] / 1e9
    
    # Max values for bars (Used for scaling graphs)
    MAX_CAP = df['marketcap'].max()
    MAX_REV = df['revenue_ttm'].max()
    MAX_PE = 60         # Cap visual at 60x
    MAX_EPS = df['revenue_per_share'].max()

    # --- ML: TRAIN/TEST SPLIT (Scientific Rigor) ---
    features = df[['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm', 'profit_margin', 'price_to_sales']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features) 
    
    # Define Target: High Growth (Top 20%)
    df['high_growth'] = ((df['revenue_ttm'] > df['revenue_ttm'].quantile(0.8)) & 
                        (df['marketcap'] > df['marketcap'].quantile(0.8))).astype(int)
    
    # Split Data: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(features, df['high_growth'], test_size=0.2, random_state=42)
    
    # Train Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Calculate Validation Score
    y_pred = rf.predict(X_test)
    MODEL_ACCURACY = accuracy_score(y_test, y_pred) # Save this for the dashboard
    
    # Predict on FULL dataset for the dashboard view
    df['growth_prob'] = rf.predict_proba(features)[:, 1]

    # Other Models
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['status'] = pd.Series(iso.fit_predict(features)).map({1: 'Normal', -1: '⚠️ Anomaly'})
    
    kmeans = KMeans(n_clusters=5, random_state=42).fit(features)
    df['group'] = "Group " + (kmeans.labels_ + 1).astype(str)
    
    similarity_matrix = cosine_similarity(scaled_features)
    country_stats = df.groupby('country')[['marketcap_trillion', 'revenue_billions']].sum().reset_index()

except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# =============================================================================
# 2. DASH APP SETUP
# =============================================================================
app = dash.Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Global Market AI</title>
        <style>
            body {margin: 0; background-color: #0b0e11; color: #e0e0e0; font-family: 'Roboto', sans-serif;}
            .container {display: flex; height: 100vh;}
            .globe-container {width: 65%; padding: 20px; border-right: 1px solid #2a2e39; position: relative;}
            .list-container {
                width: 35%; padding: 20px; background: #131722; overflow-y: auto;
                box-shadow: -5px 0 15px rgba(0,0,0,0.5); display: flex; flex-direction: column;
            }
            h1 {color: #2962FF; font-weight: 300; letter-spacing: 2px; margin-top: 0;}
            
            /* Status Bar */
            .status-bar {
                position: absolute; top: 20px; right: 20px; 
                background: rgba(0,0,0,0.7); padding: 10px 20px; 
                border-radius: 20px; border: 1px solid #333;
                color: #00E676; font-weight: bold; font-size: 0.9em;
            }

            /* Info Card */
            .info-card {
                background: #1e222d; padding: 15px; margin-bottom: 20px; 
                border-radius: 8px; border-left: 4px solid #00E676;
                transition: transform 0.2s;
            }
            .info-card:hover {transform: translateX(5px);}
            .info-card.anomaly {border-left: 4px solid #FF0055;} 
            
            .card-header {display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;}
            .company-name {margin: 0; color: #2962FF; font-size: 1.4em;}
            .badge {padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: bold; color: #0b0e11;}
            .badge-growth {background: #00E676;}
            .badge-normal {background: #888;}
            
            /* Simulator Section */
            .simulator-box {
                margin-top: auto; /* Push to bottom */
                background: #1A1E29; padding: 15px; border-radius: 8px; border-top: 2px solid #2962FF;
            }
            .sim-title {color: #2962FF; font-weight: bold; margin-bottom: 10px;}
            .sim-label {font-size: 0.8em; color: #888; margin-bottom: 5px;}
            
            .similar-companies {margin-top: 15px; padding-top: 10px; border-top: 1px solid #333; font-size: 0.85em;}
            .similar-label {color: #8b949e; margin-bottom: 3px;}
            .similar-val {color: #00E676;}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# Reusable Metric Function (Creates the Bar Graph underneath)
def create_visual_metric(label, value_text, desc, raw_value, max_val, color_theme='blue'):
    percent = (raw_value / max_val) * 100
    if percent > 100: percent = 100
    if percent < 1: percent = 1
    
    bar_color = '#2962FF' # Blue
    if color_theme == 'green': bar_color = '#00E676'
    if color_theme == 'red_inverse': 
        bar_color = '#FF0055' if percent > 80 else '#00E676'
    
    return html.Div(style={'marginBottom': '12px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}, children=[
            html.Div(children=[
                html.Span(label, style={'color': '#8b949e', 'fontSize': '0.9em'}), 
                html.Span(f" {desc}", style={'color': '#555', 'fontSize': '0.75em'})
            ]),
            html.Div(value_text, style={'color': 'white', 'fontWeight': 'bold'})
        ]),
        html.Div(style={'height': '6px', 'backgroundColor': '#333', 'borderRadius': '3px', 'width': '100%'}, children=[
            html.Div(style={'width': f'{percent}%', 'height': '100%', 'backgroundColor': bar_color, 'borderRadius': '3px', 'transition': 'width 0.5s ease'})
        ])
    ])

app.layout = html.Div(className='container', children=[
    # LEFT: Globe
    html.Div(className='globe-container', children=[
        html.H1("🌐 GLOBAL AI COMMAND"),
        html.Div(f"✅ Model Accuracy: {MODEL_ACCURACY:.1%}", className='status-bar'),
        html.P("Click a country to view AI Analysis.", style={'color': '#8b949e'}),
        dcc.Graph(id='globe-graph', style={'height': '85vh'}, config={'scrollZoom': True})
    ]),
    
    # RIGHT: Sidebar with Scrollable List + Simulator
    html.Div(className='list-container', children=[
        # Dynamic Content (Scrollable)
        html.Div(id='company-list-content', style={'flex': '1', 'overflowY': 'auto'}, children=[
            html.H2("Waiting for Data...", style={'color': '#444'})
        ]),
        
        # FIXED SIMULATOR AT BOTTOM
        html.Div(className='simulator-box', children=[
            html.Div("🔮 AI Prediction Simulator", className='sim-title'),
            
            html.Div("Adjust Profit Margin:", className='sim-label'),
            dcc.Slider(id='sim-margin', min=0, max=1, step=0.05, value=0.2, 
                       marks={0:'0%', 0.5:'50%', 1:'100%'}, tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Div("Adjust P/E Ratio:", className='sim-label', style={'marginTop': '10px'}),
            dcc.Slider(id='sim-pe', min=0, max=100, step=5, value=25, 
                       marks={0:'0', 50:'50', 100:'100+'}, tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Div(id='sim-output', style={'textAlign': 'center', 'marginTop': '15px', 'fontWeight': 'bold', 'color': 'white'})
        ])
    ])
])

# CALLBACK 1: Update Globe & Cards
@app.callback(
    [Output('globe-graph', 'figure'), Output('company-list-content', 'children')],
    [Input('globe-graph', 'clickData')]
)
def update_view(clickData):
    scale = 1.3 if clickData else 1.1
    fig = px.scatter_geo(country_stats, locations="country", locationmode='country names',
                         size="marketcap_trillion", hover_name="country",
                         color="marketcap_trillion", projection="orthographic", color_continuous_scale='Plasma')
    fig.update_layout(paper_bgcolor='#0b0e11', geo=dict(bgcolor='#0b0e11', showframe=False, projection_scale=scale, landcolor='#1E222D', oceancolor='#0b0e11'), margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, font={'color':'white'})

    if not clickData: return fig, [html.H2("Select a Region", style={'color': '#2962FF'})]
    
    try: country = clickData['points'][0]['location']
    except: country = clickData['points'][0]['hovertext']

    data = df[df['country'] == country].sort_values(by='marketcap', ascending=False)
    cards = [html.H2(f"{country} Analysis 📊", style={'color': '#fff', 'borderBottom': '1px solid #333'})]
    if data.empty: cards.append(html.P("No data available."))

    for idx, row in data.iterrows():
        comp_idx = df.index[df['name'] == row['name']].tolist()[0]
        sim_scores = sorted(list(enumerate(similarity_matrix[comp_idx])), key=lambda x: x[1], reverse=True)
        similar_text = ", ".join(df.iloc[[i[0] for i in sim_scores[1:4]]]['name'].tolist())

        is_anomaly = row['status'] == '⚠️ Anomaly'
        growth_badge = html.Span("🚀 HIGH GROWTH", className='badge badge-growth') if row['growth_prob'] > 0.7 else html.Span(row['group'], className='badge badge-normal')
        
        cards.append(html.Div(className='info-card anomaly' if is_anomaly else 'info-card', children=[
            html.Div(className='card-header', children=[html.H3(row['name'], className='company-name'), growth_badge]),
            
            # VISUAL METRICS GRID
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}, children=[
                html.Div([
                    create_visual_metric("Market Cap", f"${row['marketcap_trillion']:.2f} T", "", row['marketcap'], MAX_CAP),
                    create_visual_metric("Revenue", f"${row['revenue_billions']:.2f} B", "", row['revenue_ttm'], MAX_REV),
                    create_visual_metric("Profit Margin", f"{row['profit_margin']*100:.1f}%", "", row['profit_margin'], 0.40, 'green'),
                ]),
                html.Div([
                    create_visual_metric("P/E Ratio", f"{row['pe_ratio_ttm']:.1f}x", "", row['pe_ratio_ttm'], MAX_PE, 'red_inverse'),
                    create_visual_metric("Rev/Share", f"${row['revenue_per_share']:.2f}", "", row['revenue_per_share'], MAX_EPS),
                    create_visual_metric("P/S Ratio", f"{row['price_to_sales']:.2f}x", "", row['price_to_sales'], 15, 'red_inverse'),
                ])
            ]),
            
            html.Div(style={'marginTop': '10px', 'borderTop': '1px solid #333', 'paddingTop': '5px', 'fontSize': '0.85em'}, children=[
                html.Span("👀 Similar: ", style={'color': '#888'}), html.Span(similar_text, style={'color': '#00E676'})
            ]),

            html.Div(style={'marginTop': '15px'}, children=[
                html.Div(f"AI Growth Score: {row['growth_prob']:.1%}", style={'fontSize': '0.8em', 'color': '#00E676'}),
                html.Div(style={'height': '6px', 'background': '#333', 'borderRadius': '3px', 'marginTop': '5px'}, children=[
                    html.Div(style={'width': f"{row['growth_prob']*100}%", 'height': '100%', 'background': 'linear-gradient(90deg, #2962FF, #00E676)', 'borderRadius': '3px'})
                ])
            ])
        ]))
    return fig, cards

# CALLBACK 2: SIMULATOR LOGIC
@app.callback(
    Output('sim-output', 'children'),
    [Input('sim-margin', 'value'), Input('sim-pe', 'value')]
)
def run_simulation(margin, pe):
    # Dummy values for simulator
    dummy_price = df['price_gbp'].median()
    dummy_div = df['dividend_yield_ttm'].median()
    dummy_ps = df['price_to_sales'].median()
    dummy_earnings = 1000000000 
    
    input_data = pd.DataFrame([[dummy_price, pe, dummy_earnings, dummy_div, margin, dummy_ps]], 
                              columns=['price_gbp', 'pe_ratio_ttm', 'earnings_ttm', 'dividend_yield_ttm', 'profit_margin', 'price_to_sales'])
    
    prob = rf.predict_proba(input_data)[0][1]
    color = "#00E676" if prob > 0.5 else "#FF0055"
    text = "HIGH GROWTH 🚀" if prob > 0.5 else "LOW GROWTH 📉"
    
    return html.Div([
        html.Div(f"Predicted Score: {prob:.1%}", style={'fontSize': '1.2em'}),
        html.Div(text, style={'color': color, 'fontSize': '1.5em', 'marginTop': '5px'})
    ])

if __name__ == '__main__':
    print("🚀 ML DASHBOARD LIVE: http://127.0.0.1:8050/")
    app.run(debug=True)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Filipino Food Price Forecasting and Predictive Modeling using Time Series Regression",
    page_icon="🌾",
    layout="wide"
)

# ==========================================
# 2. ENHANCED CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* 1. Main Background & Font */
    .stApp {
        background-color: #0e1117; /* Deep Charcoal/Black */
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    
    /* 2. Header Styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    p, span, div {
        color: #cfd8dc;
    }
    
    /* 3. Metric Card Styling (Dark Cards) */
    .metric-container {
        background-color: #1f2937; /* Dark Slate */
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .metric-container:hover {
        transform: translateY(-3px);
        border-color: #60a5fa; /* Blue glow on hover */
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af; /* Muted gray text */
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f3f4f6; /* Bright White */
    }
    .metric-delta {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 5px;
    }
    
    /* 4. Risk Indicators (Vibrant for Dark Mode) */
    .risk-high { 
        color: #ff6b6b; 
        background-color: rgba(255, 107, 107, 0.1); 
        border: 1px solid #ff6b6b; 
        padding: 8px 12px; 
        border-radius: 6px; 
    }
    .risk-moderate { 
        color: #feca57; 
        background-color: rgba(254, 202, 87, 0.1); 
        border: 1px solid #feca57; 
        padding: 8px 12px; 
        border-radius: 6px; 
    }
    .risk-low { 
        color: #1dd1a1; 
        background-color: rgba(29, 209, 161, 0.1); 
        border: 1px solid #1dd1a1; 
        padding: 8px 12px; 
        border-radius: 6px; 
    }
    
    /* 5. Insight/Interpretation Boxes */
    .interpretation-box {
        background-color: #1e293b;
        border-left: 4px solid #10b981; /* Emerald Green */
        padding: 15px;
        margin-top: 15px;
        border-radius: 4px;
        color: #e2e8f0;
    }
    .warning-box {
        background-color: #1e293b;
        border-left: 4px solid #f59e0b; /* Amber */
        padding: 15px;
        margin-top: 15px;
        border-radius: 4px;
        color: #e2e8f0;
    }
    
    /* 6. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1f2937;
        border-radius: 8px 8px 0 0;
        border: 1px solid #374151;
        border-bottom: none;
        padding: 0 20px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: #111827; /* Darker active tab */
        color: #60a5fa; /* Blue text */
        border-top: 3px solid #60a5fa;
    }
    
    /* 7. Plot Container */
    .plot-container {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid #374151;
        margin-bottom: 20px;
    }
    
    /* 8. Fix Streamlit Default Elements for Dark Mode */
    div[data-testid="stExpander"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & SETUP
# ==========================================
DATA_PATH = "wfp_food_prices_phl (main).csv"
EXOG_PATH = "sarimax_final_training_data_complete_updated.csv"
JSON_PATH = "dashboard_data.json"
MODELS_DIR = "models"

LINEAR_TREND_COLS = ['GWPI_Index', 'YoY_Inflation_Rate', 'Brent_Crude_USD', 'USGC_Diesel_USD']
EXOG_COLS = [
    'GWPI_Index', 'YoY_Inflation_Rate', 'Brent_Crude_USD', 'USGC_Diesel_USD',
    'GWPI_Index_Lag_1', 'GWPI_Index_Lag_3', 'GWPI_Index_Lag_12',
    'YoY_Inflation_Rate_Lag_1', 'YoY_Inflation_Rate_Lag_3',
    'GWPI_Index_MA3', 'YoY_Inflation_Rate_MA3',
    'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
    'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12'
]

COMMODITIES = [
    "Rice (regular, milled)", "Rice (milled, superior)", 
    "Maize (white)", "Maize (yellow)", "Sugar",
    "Meat (pork)", "Meat (beef)", "Meat (chicken, whole)", 
    "Fish (tilapia)", "Fish (fresh)", "Eggs", 
    "Onions (red)", "Cabbage", "Potatoes (Irish)", 
    "Sweet potatoes", "Garlic", "Tomatoes", 
    "Bananas (lakatan)", "Bananas (saba)", 
    "Coconut", "Mangoes (carabao)"
]

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

@st.cache_data
def load_static_data():
    try:
        df_p = pd.read_csv(DATA_PATH)
        df_e = pd.read_csv(EXOG_PATH)
        
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r') as f:
                meta_data = json.load(f)
        else:
            meta_data = {}
            
        df_p = df_p[df_p['date'] != '#date']
        df_p['date'] = pd.to_datetime(df_p['date'], errors='coerce')
        df_p = df_p.sort_values("date")
        
        df_e['Date'] = pd.to_datetime(df_e['Date'])
        df_e = df_e.set_index('Date')
        
        return df_p, df_e, meta_data
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None, None, None

def get_historical_data(df_prices, df_exog, commodity, region=None):
    d = df_prices[df_prices['commodity'].str.contains(commodity, case=False, na=False, regex=False)].copy()
    
    if region and region != "National Average":
        d = d[d['admin1'] == region]
    
    if d.empty: return None, None
    
    d = d.set_index('date')
    d['price'] = pd.to_numeric(d['price'], errors='coerce')
    d = d.dropna(subset=['price'])
    
    y_series = d.groupby(pd.Grouper(freq='MS'))['price'].mean()
    y_series = y_series.interpolate(method='linear').dropna()
    
    df_aligned = pd.merge(y_series.rename('price'), df_exog, left_index=True, right_index=True, how='inner')
    return df_aligned['price'], df_aligned[EXOG_COLS]

def run_live_forecast(model, X_hist, steps):
    last_exog = X_hist.iloc[-1].to_dict()
    X_future = pd.DataFrame([last_exog] * steps, columns=EXOG_COLS)
    
    X_trend = X_hist.copy()
    X_trend['Time'] = np.arange(len(X_trend))
    future_time = pd.DataFrame(np.arange(len(X_trend), len(X_trend) + steps), columns=['Time'])
    
    for col in LINEAR_TREND_COLS:
        lr = LinearRegression()
        lr.fit(X_trend[['Time']], X_trend[col])
        X_future[col] = lr.predict(future_time)
        
    last_date = X_hist.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
    
    try:
        forecast, conf_int = model.predict(n_periods=steps, exogenous=X_future, return_conf_int=True, alpha=0.05)
    except:
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        
    return forecast, conf_int, future_dates

# ==========================================
# 4. FRONTEND LAYOUT
# ==========================================

def main():
    # --- HEADER ---
    st.markdown("""
        <div>
            <h1>Filipino Food Price Forecasting and Predictive Modeling using Time Series Regression</h1>
            <p style='color:#7f8c8d; font-size:1.1rem; margin-top:5px;'>
                <b>System Overview: An AI-powered forecasting tool for monitoring food security. 
    Select a commodity and a time horizon to generate predictive insights based on historical trends and economic indicators.
            </p>
        </div>
        <hr style='margin: 10px 0px 20px 0px; border-top: 1px solid #eee;'>
    """, unsafe_allow_html=True)
    
    df_prices, df_exog, meta_data = load_static_data()
    if df_prices is None: return

    # --- SIDEBAR: SYSTEM OVERVIEW & SETTINGS ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Target Commodity")
        selected_c = st.selectbox("Select Commodity", COMMODITIES, index=0)
        
        subset = df_prices[df_prices['commodity'].str.contains(selected_c, case=False, na=False, regex=False)]
        available_regions = subset['admin1'].unique()
        region_options = ["National Average"] + list(available_regions)
        selected_region = st.selectbox("Region/Market", region_options)
        
        st.subheader("Forecast Settings")
        horizon_map = {
            "Next Month (Immediate)": 1,
            "Next Quarter (3 Months)": 3, 
            "Next Season (6 Months)": 6, 
            "Next Year (12 Months)": 12, 
            "Long Term (18 Months)": 18
        }
        selected_horizon_label = st.selectbox("Time Horizon", list(horizon_map.keys()), index=2)
        steps = horizon_map[selected_horizon_label]
        
        st.markdown("---")
        
        # SYSTEM OVERVIEW SECTION
        with st.expander("ℹ️ About this System", expanded=True):
            st.markdown("""
            <div style='font-size:0.9rem; color:#ffffff;'>
            <b>What does this system do?</b><br>
            It predicts future food prices using a <b>SARIMAX Model</b>, analyzing history, seasonality, and economic factors.<br><br>
            <b>Who is it for?</b><br>
            Policymakers needing early warnings on price spikes to trigger subsidies or imports.
            </div>
            """, unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Dashboard", "⚖️ Comparative Analysis", "📅 Seasonal Trends"])

    # 1. FETCH DATA
    y_hist, X_hist = get_historical_data(df_prices, df_exog, selected_c, selected_region)
    
    if y_hist is None:
        st.error("No data available for this selection.")
        return

    # LOAD MODEL
    model_file = f"{selected_c.replace(' ', '_').replace('/', '_')}_SARIMAX_model.joblib"
    model_path = os.path.join(MODELS_DIR, model_file)
    model_loaded = False
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model_loaded = True
        except:
            st.error("Model file corrupted.")
    
    st.title("Commodity Price Forecasting App")

    # 🔍 DEBUG: Check what’s inside the models folder
    st.write("Files inside /models directory:")
    st.write(os.listdir("models"))

    # ==========================================
    # TAB 1: MAIN FORECAST DASHBOARD
    # ==========================================
    with tab1:
        if not model_loaded:
            st.warning(f"⚠️ Model not found at `{model_path}`. Showing history only.")
            st.line_chart(y_hist)
        else:
            # Run Forecast
            preds, conf, dates = run_live_forecast(model, X_hist, steps)
            
            # --- METRICS CALCULATION ---
            last_recorded_price = y_hist.iloc[-1]
            last_recorded_date = y_hist.index[-1].strftime('%b %Y')
            
            target_price = preds[-1] 
            target_date = dates[-1].strftime('%b %Y')
            
            pct_change = ((target_price - last_recorded_price) / last_recorded_price) * 100
            
            # Risk Logic & Interpretation
            if pct_change > 5:
                risk_class = "risk-high"
                risk_title = "CRITICAL ALERT"
                risk_msg = f"Prices expected to SPIKE (+{pct_change:.1f}%)"
                risk_desc = f"⚠️ **Action Required:** Prices are rising significantly. Monitor supply chain and consider buffer stock release."
            elif pct_change < -2:
                risk_class = "risk-low"
                risk_title = "FAVORABLE OUTLOOK"
                risk_msg = f"Prices expected to DROP ({pct_change:.1f}%)"
                risk_desc = f"✅ **Positive Trend:** Prices are cooling down, likely due to seasonal supply. No intervention needed."
            else:
                risk_class = "risk-moderate"
                risk_title = "STABLE MARKET"
                risk_msg = f"Prices expected to be STABLE ({pct_change:.1f}%)"
                risk_desc = f"⚖️ **Stability:** Minor fluctuations predicted. Routine monitoring recommended."

            # Custom Metric Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Last Recorded Price</div>
                    <div class="metric-value">₱{last_recorded_price:.2f}</div>
                    <div class="metric-label">{last_recorded_date}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Forecast ({steps} Months)</div>
                    <div class="metric-value">₱{target_price:.2f}</div>
                    <div class="metric-delta" style="color: {'red' if pct_change > 0 else 'green'}">
                        {'+' if pct_change > 0 else ''}{pct_change:.1f}% Change
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            mape = meta_data.get(selected_c, {}).get('metrics', {}).get('MAPE', 0)
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Model Confidence (MAPE)</div>
                    <div class="metric-value">{mape:.2f}%</div>
                    <div class="metric-label" style="color: #7f8c8d;">Lower is Better</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="{risk_class}" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-weight:bold; font-size:0.8rem; margin-bottom:5px;">{risk_title}</div>
                    <div style="font-weight:900; font-size:1.1rem;">{risk_msg}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- MAIN CHART ---
            st.markdown(f"<h3 style='text-align:center; margin-bottom:15px;'>📈 Price Trajectory: {selected_c} ({selected_region})</h3>", unsafe_allow_html=True)
            
            # Wrap chart in a white container
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = go.Figure()
            # Historical Line
            fig.add_trace(go.Scatter(x=y_hist.index, y=y_hist, mode='lines', name='Historical (Actual)', line=dict(color='#2980b9', width=2.5)))
            # Forecast Line
            fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines+markers', name='Forecast (Predicted)', line=dict(color='#e74c3c', width=2.5, dash='dash')))
            # Confidence Band
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates)[::-1],
                y=list(conf[:, 1]) + list(conf[:, 0])[::-1],
                fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Band'
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=450, 
                hovermode="x unified",
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Price (PHP)"),
                plot_bgcolor='#081112',
                paper_bgcolor='#081112',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- LEGEND & SUGGESTIVE INTERPRETATION ---
            col_explain, col_suggest = st.columns([1, 1])
            
            with col_explain:
                with st.expander("ℹ️ How to read this chart & Legends", expanded=False):
                    st.markdown("""
                    * **🔵 Blue Line (Historical):** Validated past data from WFP.
                    * **🔴 Red Dashed Line (Forecast):** The AI's predicted trend for the future.
                    * **🌸 Shaded Area (95% Confidence):** The "Margin of Error". 
                        * *Why 95%?* It means there is a 95% probability the real future price will fall within this range.
                    """)

            with col_suggest:
                # Dynamic Suggestion Logic
                box_class = "warning-box" if pct_change > 5 else "interpretation-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <strong style="color:#ffffff;">🤖 AI Interpretation & Recommendation:</strong><br>
                    {risk_desc}
                </div>
                """, unsafe_allow_html=True)

            # --- DATA TABLE ---
            with st.expander("📋 View Detailed Data Table"):
                forecast_df = pd.DataFrame({"Date": dates, "Forecast": preds, "Lower": conf[:, 0], "Upper": conf[:, 1]})
                st.dataframe(forecast_df, use_container_width=True)
                st.download_button("Download Report (CSV)", forecast_df.to_csv().encode('utf-8'), f"{selected_c}_forecast.csv", "text/csv")

    # ==========================================
    # TAB 2: COMPARATIVE ANALYSIS
    # ==========================================
    with tab2:
        st.subheader("⚖️ Comparative Analysis Tool")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            c1 = st.selectbox("Commodity A", COMMODITIES, index=COMMODITIES.index(selected_c), key="comp1")
            y1, _ = get_historical_data(df_prices, df_exog, c1, "National Average")
        with col_c2:
            default_idx = 1 if COMMODITIES.index(selected_c) == 0 else 0
            c2 = st.selectbox("Commodity B", COMMODITIES, index=default_idx, key="comp2")
            y2, _ = get_historical_data(df_prices, df_exog, c2, "National Average")

        if y1 is not None and y2 is not None:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=y1.index, y=y1, name=c1, line=dict(color='#2980b9')))
            fig_comp.add_trace(go.Scatter(x=y2.index, y=y2, name=c2, line=dict(color='#e67e22')))
            fig_comp.update_layout(
                title=f"Price Comparison: {c1} vs {c2}", 
                margin=dict(l=20, r=20, t=40, b=20),
                height=450, 
                hovermode="x unified",
                plot_bgcolor='#081112',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Suggestive Interpretation for Correlation
            common_idx = y1.index.intersection(y2.index)
            corr = y1[common_idx].corr(y2[common_idx])
            
            corr_text = "No significant relationship."
            if corr > 0.7: corr_text = "Strong positive correlation. When one rises, the other usually follows."
            elif corr < -0.5: corr_text = "Inverse relationship. When one rises, the other tends to fall."
            
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>🤖 Comparative Insight:</strong><br>
                Correlation Coefficient: <strong>{corr:.2f}</strong>. <br>
                {corr_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Data missing for one of the selected commodities.")

    # ==========================================
    # TAB 3: SEASONAL TRENDS
    # ==========================================
    with tab3:
        st.subheader(f"📅 Seasonal Price Patterns: {selected_c}")
        
        if y_hist is not None:
            df_season = y_hist.to_frame()
            df_season['Month_Num'] = df_season.index.month
            seasonal_avg = df_season.groupby('Month_Num')['price'].mean()
            seasonal_std = df_season.groupby('Month_Num')['price'].std()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(x=month_names, y=seasonal_avg, mode='lines+markers', name='Avg Price', line=dict(color='#27ae60', width=3)))
            fig_season.add_trace(go.Scatter(
                x=month_names + month_names[::-1],
                y=(seasonal_avg + seasonal_std).tolist() + (seasonal_avg - seasonal_std).tolist()[::-1],
                fill='toself', fillcolor='rgba(39, 174, 96, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Volatility Range'
            ))
            fig_season.update_layout(
                title="Typical Monthly Price Cycle", 
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='#081112',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig_season, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            min_month = month_names[seasonal_avg.idxmin() - 1]
            max_month = month_names[seasonal_avg.idxmax() - 1]
            
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>🤖 Seasonal Insight:</strong><br>
                Historically, {selected_c} prices tend to be <strong>lowest in {min_month}</strong> (likely due to harvest supply) and <strong>highest in {max_month}</strong> (lean months).
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ How to read this chart"):
                st.write("""
                This chart collapses 20+ years of data into a single "Average Year".
                * **Green Line:** The expected price for that month.
                * **Shaded Area:** The range of volatility. A wide shaded area means the price in that month is very unpredictable (sometimes high, sometimes low).
                """)

if __name__ == "__main__":
    main()

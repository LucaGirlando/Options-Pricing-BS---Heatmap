import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
    <p style="font-size: 12px; text-align: center;">
        Created by: <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank">Luca Girlando</a>
    </p>
""", unsafe_allow_html=True)

# Premium scientific CSS
st.markdown("""
    <style>
        /* Global Styles */
        body {
            font-family: 'Arial', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: var(--sidebar-background-color);
        }

        /* Sidebar Title - BLACK-SCHOLES */
        .css-1d391kg h1 {
            color: var(--sidebar-title-color);
            font-size: 26px;
            font-weight: bold;
            text-transform: uppercase;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: var(--primary-color);
        }

        /* Metric Boxes */
        .metric-container {
            border-radius: 8px;
            padding: 10px;
            background-color: var(--card-background);
            margin: 10px;
            text-align: center;
        }

        .metric-container .metric-label {
            font-size: 16px;
            color: var(--text-color-light);
        }

        .metric-container .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }

        .greek-values {
            font-size: 16px;
            font-weight: bold;
            color: var(--greek-values-color);
        }

        .section-divider {
            border: 1px solid var(--divider-color);
            margin: 20px 0;
        }

        .footer {
            text-align: center;
            font-size: 14px;
            color: var(--footer-color);
        }

        /* Dark Theme */
        :root {
            --background-color: #ffffff;
            --text-color: #000000;
            --text-color-light: #777777;
            --primary-color: #2f87c0;
            --secondary-color: #555555;
            --card-background: #f7f7f7;
            --sidebar-background-color: #f7f7f7;
            --divider-color: #e0e0e0;
            --footer-color: #777777;
            --sidebar-title-color: #000000;
            --greek-values-color: #444444;
        }

        /* Light Theme */
        [data-theme="dark"] {
            --background-color: #121212;
            --text-color: #ffffff;
            --text-color-light: #bbbbbb;
            --primary-color: #2f87c0;
            --secondary-color: #999999;
            --card-background: #2a2a2a;
            --sidebar-background-color: #2a2a2a;
            --divider-color: #444444;
            --footer-color: #bbbbbb;
            --sidebar-title-color: #ffffff;
            --greek-values-color: #bbbbbb;
        }

        /* Custom styles for options (Call & Put) */
        .option-container {
            font-size: 22px;
            font-weight: bold;
            color: var(--primary-color);
            text-align: center;
            padding: 20px;
            background-color: var(--card-background);
            border-radius: 8px;
            margin-top: 15px;
        }

        .option-title {
            font-size: 32px;
            color: var(--primary-color);
        }

        /* Call Option Styling */
        .call-option {
            background-color: #b2ebf2;
            font-size: 26px;
            color: #00796b;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border: 2px solid #00796b;
        }

        /* Put Option Styling */
        .put-option {
            background-color: #f8bbd0;
            font-size: 26px;
            color: #d32f2f;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border: 2px solid #d32f2f;
        }

        .greek-values-call, .greek-values-put {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }

        /* Greek values for Call and Put */
        .greek-values-call span, .greek-values-put span {
            font-weight: normal;
            color: var(--secondary-color);
        }

    </style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        T = self.time_to_maturity
        K = self.strike
        S = self.current_price
        sigma = self.volatility
        r = self.interest_rate

        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        call_price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        put_price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1)/(S*sigma*sqrt(T))
        self.vega = S*norm.pdf(d1)*sqrt(T)
        self.call_theta = (-S*norm.pdf(d1)*sigma/(2*sqrt(T))) - r*K*exp(-r*T)*norm.cdf(d2)
        self.put_theta = (-S*norm.pdf(d1)*sigma/(2*sqrt(T))) + r*K*exp(-r*T)*norm.cdf(-d2)
        self.rho = K*T*exp(-r*T)*norm.cdf(d2) if call_price else -K*T*exp(-r*T)*norm.cdf(-d2)

        return call_price, put_price

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    theme = st.get_option("theme.base")
    cmap = cm.get_cmap('plasma').copy()
    cmap.set_bad(color='white')
    
    fig_call, ax_call = plt.subplots(figsize=(10, 7), dpi=120)
    sns.heatmap(call_prices, 
                xticklabels=np.round(spot_range, 1), 
                yticklabels=np.round(vol_range, 3), 
                annot=True, 
                fmt=".2f", 
                cmap=cmap,
                ax=ax_call,
                cbar_kws={'label': 'Price ($)', 'shrink': 0.75},
                linewidths=0.5,
                linecolor='#f0f0f0',
                annot_kws={'size': 9, 'color': 'black'})
    
    ax_call.set_title('CALL OPTION PRICE SENSITIVITY', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_call.set_xlabel('Underlying Price ($)', fontsize=11, labelpad=10)
    ax_call.set_ylabel('Volatility (σ)', fontsize=11, labelpad=10)
    ax_call.tick_params(axis='both', which='major', labelsize=9)
    
    fig_put, ax_put = plt.subplots(figsize=(10, 7), dpi=120)
    sns.heatmap(put_prices, 
                xticklabels=np.round(spot_range, 1), 
                yticklabels=np.round(vol_range, 3), 
                annot=True, 
                fmt=".2f", 
                cmap=cmap,
                ax=ax_put,
                cbar_kws={'label': 'Price ($)', 'shrink': 0.75},
                linewidths=0.5,
                linecolor='#f0f0f0',
                annot_kws={'size': 9, 'color': 'black'})
    
    ax_put.set_title('PUT OPTION PRICE SENSITIVITY', 
                    fontsize=14, fontweight='bold', pad=20)
    ax_put.set_xlabel('Underlying Price ($)', fontsize=11, labelpad=10)
    ax_put.set_ylabel('Volatility (σ)', fontsize=11, labelpad=10)
    ax_put.tick_params(axis='both', which='major', labelsize=9)
    
    return fig_call, fig_put

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 1.8rem; margin-bottom: 5px; color: white;">BLACK-SCHOLES</h1>
        <p style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Options Pricing Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Parameters")
    current_price = st.number_input("Underlying Price ($)", value=100.0, min_value=0.01, step=1.0)
    strike = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0)
    time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, step=0.1)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    interest_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=0.2, value=0.05, step=0.001, format="%.3f")
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        spot_min = st.number_input('Min Price', value=current_price*0.7, step=1.0)
        spot_max = st.number_input('Max Price', value=current_price*1.3, step=1.0)
    with col2:
        vol_min = st.number_input('Min Vol', value=max(0.01, volatility*0.5), step=0.01)
        vol_max = st.number_input('Max Vol', value=min(1.0, volatility*1.5), step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 style="font-size: 2.3rem; margin-bottom: 10px;">Options Pricing Analytics</h1>
    <p style="font-size: 1.05rem; color: var(--primary-medium); max-width: 700px; margin: 0 auto;">
        Advanced Black-Scholes calculator with sensitivity visualization. 
        Theoretical prices for European-style options.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Model Configuration")
param_data = {
    "Parameter": ["Underlying Price (S)", "Strike Price (K)", "Time to Maturity (T)", 
                 "Volatility (σ)", "Risk-Free Rate (r)"],
    "Value": [f"${current_price:.2f}", f"${strike:.2f}", f"{time_to_maturity:.2f} years", 
              f"{volatility:.3f}", f"{interest_rate:.3%}"],
    "Symbol": ["S", "K", "T", "σ", "r"]
}
param_df = pd.DataFrame(param_data)
st.dataframe(param_df, use_container_width=True, hide_index=True)

bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs.calculate_prices()

st.markdown("### Option Premiums")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-container metric-call">
        <div class="metric-label">Call Option</div>
        <div class="metric-value">${call_price:.4f}</div>
        <div class="greek-values">
            Δ: {bs.call_delta:.4f} | Γ: {bs.gamma:.6f}<br>
            ν: {bs.vega:.4f} | θ: {bs.call_theta:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="greek-explanation">
        <h4>Call Option Greeks:</h4>
        <p><strong>Δ (Delta):</strong> Sensitivity to underlying price change (~probability of ending ITM). Range: 0 to 1.</p>
        <p><strong>Γ (Gamma):</strong> Rate of change of Delta. Measures convexity.</p>
        <p><strong>ν (Vega):</strong> Sensitivity to volatility change ($ per 1% vol change).</p>
        <p><strong>θ (Theta):</strong> Time decay ($ lost per day). Negative for long options.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container metric-put">
        <div class="metric-label">Put Option</div>
        <div class="metric-value">${put_price:.4f}</div>
        <div class="greek-values">
            Δ: {bs.put_delta:.4f} | Γ: {bs.gamma:.6f}<br>
            ν: {bs.vega:.4f} | θ: {bs.put_theta:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="greek-explanation">
        <h4>Put Option Greeks:</h4>
        <p><strong>Δ (Delta):</strong> Sensitivity to underlying price change (~probability of ending ITM). Range: -1 to 0.</p>
        <p><strong>Γ (Gamma):</strong> Rate of change of Delta. Same as call for same strike.</p>
        <p><strong>ν (Vega):</strong> Sensitivity to volatility change. Always positive.</p>
        <p><strong>θ (Theta):</strong> Time decay. Typically less negative than calls.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 30px;">
    <h3>Price Sensitivity Analysis</h3>
    <p style="color: var(--primary-medium);">
        Interactive visualization of option price sensitivity to underlying parameters.
        Heatmaps show theoretical prices across a range of spot prices and volatilities.
    </p>
</div>
""", unsafe_allow_html=True)

fig_call, fig_put = plot_heatmap(bs, spot_range, vol_range, strike)

tab1, tab2 = st.tabs(["Call Option", "Put Option"])
with tab1:
    st.pyplot(fig_call)
with tab2:
    st.pyplot(fig_put)

st.markdown("""
<div class="interpretation-box">
    <h4>How to Interpret the Heatmaps:</h4>
    <p><strong>Color Gradient:</strong> The color intensity represents the option's theoretical price, with darker colors indicating higher values.</p>
    <p><strong>X-Axis (Underlying Price):</strong> Shows how option prices change with the asset price. Calls increase with higher underlying prices, while puts decrease.</p>
    <p><strong>Y-Axis (Volatility):</strong> Demonstrates the positive relationship between volatility and option prices for both calls and puts.</p>
    <p><strong>Key Observations:</strong></p>
    <ul>
        <li>ATM (At-the-Money) options show the highest sensitivity to volatility changes</li>
        <li>ITM (In-the-Money) options move nearly 1:1 with the underlying</li>
        <li>OTM (Out-of-the-Money) options are most sensitive to volatility changes</li>
        <li>The "volatility smile" can be observed in the curvature of price contours</li>
    </ul>
    <p><strong>Trading Implications:</strong> These visualizations help identify opportunities where volatility may be mispriced relative to historical ranges.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>BLACK-SCHOLES OPTION PRICING MODEL | Created by <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank" style="color: var(--accent-teal); text-decoration: none;">Luca Girlando</a></p>
</div>
""", unsafe_allow_html=True)

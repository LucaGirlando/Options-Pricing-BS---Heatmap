#App Options Pricing BS - Heatmap
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

# Inject custom CSS for both light and dark mode
st.markdown("""
<style>

:root {
    --primary-dark: #1a2639;
    --primary-medium: #3e4a61;
    --primary-light: #d9dad7;
    --accent-blue: #4a6fa5;
    --accent-teal: #166088;
    --call-green: #2e8b57;
    --put-red: #c04e4e;
    --highlight: #f0f4f8;
}

* {
    font-family: 'Lato', 'Segoe UI', Roboto, sans-serif;
}

/* Light theme defaults */
body, .stApp {
    background-color: #f8f9fa;
    color: #1a2639;
}

h1, h2, h3, h4 {
    color: var(--primary-dark);
    font-weight: 700;
    letter-spacing: -0.015em;
}

/* Dark theme overrides */
@media (prefers-color-scheme: dark) {
    body, .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    .metric-container {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }

    .metric-value {
        color: #ffffff !important;
    }

    .metric-label {
        color: #bbbbbb !important;
    }

    .greek-values {
        color: #dddddd !important;
    }

    .interpretation-box,
    .greek-explanation {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    .footer {
        color: #888888 !important;
        border-top: 1px solid #444444 !important;
    }

    .stDataFrame {
        background: #111111 !important;
        color: white !important;
        border: 1px solid #333333 !important;
    }

    .stSidebar {
        background: #111111 !important;
    }

    .stSidebar .sidebar-content,
    .stSidebar label,
    .stSidebar .stMarkdown h3 {
        color: white !important;
    }

    .st-emotion-cache-1qg05tj {
        color: #cccccc !important;
    }
}

.metric-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 1.8rem 2rem;
    border-radius: 12px;
    background: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(0,0,0,0.05);
}

.metric-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.12);
}

.metric-call {
    border-top: 4px solid var(--call-green);
}

.metric-put {
    border-top: 4px solid var(--put-red);
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    font-family: 'Roboto Mono', monospace;
    margin: 0.7rem 0;
    letter-spacing: -0.03em;
}

.metric-label {
    font-size: 1.05rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.greek-values {
    font-size: 0.85rem;
    font-family: 'Roboto Mono', monospace;
    margin-top: 0.5rem;
    letter-spacing: -0.01em;
}

/* Enhanced tables */
.stDataFrame {
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    border: 1px solid rgba(0,0,0,0.03) !important;
}

/* Heatmap title */
.heatmap-title {
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1rem !important;
}

/* Footer */
.footer {
    font-size: 0.78rem;
    text-align: center;
    margin-top: 3rem;
    padding: 1.2rem;
    letter-spacing: 0.03em;
}

/* Custom divider */
.section-divider {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent);
    margin: 2rem 0;
}

/* Explanation boxes */
.interpretation-box {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-left: 4px solid var(--accent-teal);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.greek-explanation {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
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

        # Greeks calculation
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
    
    # Premium colormap
    cmap = cm.get_cmap('plasma').copy()
    cmap.set_bad(color='white')
    
    # Create figure with constrained layout
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
                annot_kws={'size': 9})
    
    ax_call.set_title('CALL OPTION PRICE SENSITIVITY', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_call.set_xlabel('Underlying Price ($)', fontsize=11, labelpad=10)
    ax_call.set_ylabel('Volatility (σ)', fontsize=11, labelpad=10)
    ax_call.tick_params(axis='both', which='major', labelsize=9)
    
    # Put option heatmap
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
                annot_kws={'size': 9})
    
    ax_put.set_title('PUT OPTION PRICE SENSITIVITY', 
                    fontsize=14, fontweight='bold', pad=20)
    ax_put.set_xlabel('Underlying Price ($)', fontsize=11, labelpad=10)
    ax_put.set_ylabel('Volatility (σ)', fontsize=11, labelpad=10)
    ax_put.tick_params(axis='both', which='major', labelsize=9)
    
    return fig_call, fig_put

# Sidebar Configuration
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

# Main Content
st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 style="font-size: 2.3rem; margin-bottom: 10px;">Options Pricing Analytics</h1>
    <p style="font-size: 1.05rem; color: var(--primary-medium); max-width: 700px; margin: 0 auto;">
        Advanced Black-Scholes calculator with sensitivity visualization. 
        Theoretical prices for European-style options.
    </p>
</div>
""", unsafe_allow_html=True)

# Model Parameters Display
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

# Calculate prices
bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs.calculate_prices()

# Option Prices Display
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
    
    # Greek explanations for Call
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
    
    # Greek explanations for Put
    st.markdown("""
    <div class="greek-explanation">
        <h4>Put Option Greeks:</h4>
        <p><strong>Δ (Delta):</strong> Sensitivity to underlying price change (~probability of ending ITM). Range: -1 to 0.</p>
        <p><strong>Γ (Gamma):</strong> Rate of change of Delta. Same as call for same strike.</p>
        <p><strong>ν (Vega):</strong> Sensitivity to volatility change. Always positive.</p>
        <p><strong>θ (Theta):</strong> Time decay. Typically less negative than calls.</p>
    </div>
    """, unsafe_allow_html=True)

# Sensitivity Analysis
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

# Heatmap Interpretation
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

# Footer
st.markdown("""
<div class="footer">
    <p>BLACK-SCHOLES OPTION PRICING MODEL | Created by <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank" style="color: var(--accent-teal); text-decoration: none;">Luca Girlando</a></p>
</div>
""", unsafe_allow_html=True)

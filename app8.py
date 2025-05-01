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
:root {
    --primary-dark: #1a2639;
    --primary-medium: #3e4a61;
    --primary-light: #d9dad7;
    --accent-blue: #4a6fa5;
    --accent-teal: #166088;
    --call-green: #2e8b57;
    --put-red: #c04e4e;
    --highlight: #f0f4f8;
    --text-light: #333333;
    --text-dark: #f0f2f6;
    --bg-light: #f8f9fa;
    --bg-dark: #0e1117;
    --card-light: white;
    --card-dark: #1a2639;
    --border-light: rgba(0,0,0,0.05);
    --border-dark: #3e4a61;
    --greek-light: #333333;
    --greek-dark: #333333;
}

* {
    font-family: 'Lato', 'Segoe UI', Roboto, sans-serif;
}

h1, h2, h3, h4 {
    color: var(--primary-dark);
    font-weight: 700;
    letter-spacing: -0.015em;
}

body {
    background-color: var(--bg-light);
}

.stSidebar {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary-medium)) !important;
}

.stSidebar .sidebar-content {
    color: white !important;
}

.stSidebar label {
    color: white !important;
    font-weight: 500 !important;
}

.stSidebar .stSlider label {
    color: white !important;
}

.stSidebar .stNumberInput label {
    color: white !important;
}

.stSidebar .stMarkdown h3 {
    color: white !important;
}

.stNumberInput, .stSlider {
    margin-bottom: 1.2rem;
}

.metric-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 1.8rem 2rem;
    border-radius: 12px;
    background: var(--card-light);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    border: 1px solid var(--border-light);
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
    color: var(--text-light);
    letter-spacing: -0.03em;
}

.metric-label {
    font-size: 1.05rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--primary-medium);
    margin-bottom: 0.5rem;
}

.greek-values {
    font-size: 0.85rem;
    font-family: 'Roboto Mono', monospace;
    margin-top: 0.5rem;
    letter-spacing: -0.01em;
    color: var(--greek-light);
}

.stDataFrame {
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    border: 1px solid rgba(0,0,0,0.03) !important;
}

.heatmap-title {
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1rem !important;
}

.footer {
    font-size: 0.78rem;
    text-align: center;
    margin-top: 3rem;
    color: #6c757d;
    padding: 1.2rem;
    border-top: 1px solid #e9ecef;
    letter-spacing: 0.03em;
}

.st-emotion-cache-1qg05tj {
    font-weight: 500 !important;
    color: var(--primary-medium) !important;
}

.section-divider {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent);
    margin: 2rem 0;
}

.interpretation-box {
    background-color: var(--card-light);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-left: 4px solid var(--accent-teal);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    color: var(--text-light);
}

.greek-explanation {
    background-color: var(--card-light);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    color: var(--text-light);
}

@media (prefers-color-scheme: dark) {
    :root {
        --primary-dark: #f0f2f6;
        --primary-medium: #a1a9b8;
        --highlight: #1a2639;
    }
    
    body {
        background-color: var(--bg-dark) !important;
    }
    
    h1, h2, h3, h4 {
        color: var(--text-dark) !important;
    }
    
    .metric-container {
        background: var(--card-dark) !important;
        border: 1px solid var(--border-dark) !important;
    }
    
    .metric-value {
        color: var(--text-dark) !important;
    }
    
    .stDataFrame {
        background-color: var(--card-dark) !important;
    }
    
    .interpretation-box, .greek-explanation {
        background-color: var(--card-dark) !important;
        border-color: var(--border-dark) !important;
        color: var(--text-dark) !important;
    }
    
    .footer {
        color: var(--primary-medium) !important;
        border-top: 1px solid var(--border-dark) !important;
    }
    
    .metric-label {
        color: var(--primary-medium) !important;
    }
    
    .stMarkdown, .stDataFrame, .stNumberInput label {
        color: var(--text-dark) !important;
    }
    
    .greek-values {
        color: var(--greek-dark) !important;
    }
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
        self.rho_call = K*T*exp(-r*T)*norm.cdf(d2)
        self.rho_put = -K*T*exp(-r*T)*norm.cdf(-d2)

# User Inputs for Option
st.title("Black-Scholes Option Pricing Model 📊")

# Create sidebar inputs
col1, col2 = st.columns(2)
with col1:
    strike = st.number_input("Strike Price (K)", min_value=1, value=100, step=1)
    current_price = st.number_input("Current Price (S)", min_value=1, value=100, step=1)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
with col2:
    time_to_maturity = st.number_input("Time to Maturity (T) in years", min_value=0.01, value=1.0, step=0.01)
    interest_rate = st.slider("Risk-free Interest Rate (r)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

# Calculate Option Prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
bs_model.calculate_prices()

# Display results
st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

st.header("Option Price Calculations")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Price")
    st.metric(label="Call Price", value=f"${bs_model.call_price:.2f}", delta=f"{bs_model.call_delta:.2f}")
with col2:
    st.subheader("Put Option Price")
    st.metric(label="Put Price", value=f"${bs_model.put_price:.2f}", delta=f"{bs_model.put_delta:.2f}")

# Greeks (Options Sensitivity)
st.subheader("Option Greeks")
st.markdown("""
    <div class="interpretation-box">
    <p><strong>Delta</strong>: Measures the rate of change of option value with respect to changes in the underlying asset's price.</p>
    <p><strong>Gamma</strong>: Measures the rate of change of delta with respect to changes in the underlying asset's price.</p>
    <p><strong>Vega</strong>: Measures sensitivity to volatility.</p>
    <p><strong>Theta</strong>: Measures sensitivity to the passage of time (time decay).</p>
    <p><strong>Rho</strong>: Measures sensitivity to interest rates.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Call Delta", value=f"{bs_model.call_delta:.2f}")
    st.metric(label="Put Delta", value=f"{bs_model.put_delta:.2f}")
    st.metric(label="Gamma", value=f"{bs_model.gamma:.2f}")
with col2:
    st.metric(label="Vega", value=f"{bs_model.vega:.2f}")
    st.metric(label="Call Theta", value=f"{bs_model.call_theta:.2f}")
    st.metric(label="Put Theta", value=f"{bs_model.put_theta:.2f}")

# Visualization of Option Pricing Sensitivity to Volatility
fig, ax = plt.subplots(figsize=(8, 5))
volatility_range = np.linspace(0.01, 1.0, 100)
call_prices = [BlackScholes(time_to_maturity, strike, current_price, vol, interest_rate).calculate_prices().call_price for vol in volatility_range]
put_prices = [BlackScholes(time_to_maturity, strike, current_price, vol, interest_rate).calculate_prices().put_price for vol in volatility_range]

ax.plot(volatility_range, call_prices, label="Call Option Price", color=var(--call-green))
ax.plot(volatility_range, put_prices, label="Put Option Price", color=var(--put-red))

ax.set_title("Option Prices vs Volatility", fontsize=16)
ax.set_xlabel("Volatility (σ)", fontsize=12)
ax.set_ylabel("Option Price ($)", fontsize=12)
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("""
    <div class="footer">
        Option Pricing Model using the Black-Scholes Formula. © 2025 by Luca Girlando.
    </div>
""", unsafe_allow_html=True)

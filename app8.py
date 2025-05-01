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

        /* Add text in Sidebar for the Light Theme */
        [data-theme="light"] .css-1d391kg::after {
            content: "BLACK-SCHOLES Options Pricing Model";
            font-size: 18px;
            font-weight: bold;
            color: #000000;
            position: absolute;
            top: 10px;
            left: 10px;
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
            font-size: 12px;
            color: var(--secondary-color);
        }

        .section-divider {
            border: 1px solid var(--divider-color);
            margin: 20px 0;
        }

        /* Heatmap */
        .stMarkdown {
            margin-bottom: 10px;
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
        }

        /* Adjust size and contrast for Call and Put options in both themes */
        .call-option, .put-option {
            font-size: 20px;
            font-weight: bold;
            color: var(--primary-color);
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }

        /* Highlight Call and Put options with higher contrast */
        .call-option {
            color: #28a745; /* Green for Call */
        }

        .put-option {
            color: #dc3545; /* Red for Put */
        }
    </style>
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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import io

# 1. Streamlit Page Configuration
st.set_page_config(page_title="Ion Cloud Boundaries", layout="wide")
st.title("Interactive Library Ion Cloud Boundaries")
st.markdown("Upload your proteomics library `.tsv` file to calculate and adjust Ion Mobility boundaries interactively.")

# --- CACHED FUNCTIONS ---
# This ensures the heavy math (KDE density) only runs ONCE per file upload.
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file, sep="\t")
    precursors = df[['PrecursorMz', 'PrecursorIonMobility']].drop_duplicates().dropna()
    mz_vals = precursors['PrecursorMz'].values
    im_vals = precursors['PrecursorIonMobility'].values

    # Compute Kernel Density
    xy = np.vstack([mz_vals, im_vals])
    if len(mz_vals) > 5000:
        idx = np.random.choice(len(mz_vals), 5000, replace=False)
        xy_sample = np.vstack([mz_vals[idx], im_vals[idx]])
        density = gaussian_kde(xy_sample)(xy)
    else:
        density = gaussian_kde(xy)(xy)

    return mz_vals, im_vals, density

@st.cache_data
def calculate_initial_boundaries(mz_vals, im_vals):
    # Bin the data to find extreme slopes
    mz_bins = np.linspace(min(mz_vals), max(mz_vals), 20)
    top_im_points, bot_im_points, valid_mz_points = [], [], []

    for i in range(len(mz_bins)-1):
        mask = (mz_vals >= mz_bins[i]) & (mz_vals <= mz_bins[i+1])
        if np.sum(mask) > 0:
            bot_im_points.append(np.min(im_vals[mask]))
            top_im_points.append(np.max(im_vals[mask]))
            valid_mz_points.append((mz_bins[i] + mz_bins[i+1]) / 2)

    m_top, _ = np.polyfit(valid_mz_points, top_im_points, 1)
    m_bot, _ = np.polyfit(valid_mz_points, bot_im_points, 1)

    c_top = np.max(im_vals - (m_top * mz_vals))
    c_bot = np.min(im_vals - (m_bot * mz_vals))

    x_start = min(mz_vals) - 20
    x_end = max(mz_vals) + 20
    
    y_top_start = m_top * x_start + c_top
    y_top_end = m_top * x_end + c_top
    y_bot_start = m_bot * x_start + c_bot
    y_bot_end = m_bot * x_end + c_bot
    
    return x_start, x_end, y_top_start, y_top_end, y_bot_start, y_bot_end

# --- MAIN APP UI ---
# Sidebar for file upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Library .tsv", type=['tsv', 'txt', 'csv'])

if uploaded_file is not None:
    with st.spinner("Processing data and calculating density (this takes a few seconds)..."):
        mz_vals, im_vals, density = load_and_process_data(uploaded_file)
        x_start, x_end, init_y_ts, init_y_te, init_y_bs, init_y_be = calculate_initial_boundaries(mz_vals, im_vals)

    # 2. UI Controls (Sidebar)
    st.sidebar.header("2. Adjust Boundaries")
    st.sidebar.markdown("*Tip: You can drag the slider or click the number to type exact values.*")
    
    y_min_limit = float(min(im_vals) - 0.2)
    y_max_limit = float(max(im_vals) + 0.2)

    # Sliders
    top_y_start = st.sidebar.slider(f"Top Start (x={x_start:.0f})", y_min_limit, y_max_limit, float(init_y_ts), 0.001, format="%.4f")
    top_y_end = st.sidebar.slider(f"Top End (x={x_end:.0f})", y_min_limit, y_max_limit, float(init_y_te), 0.001, format="%.4f")
    
    st.sidebar.divider()
    
    bot_y_start = st.sidebar.slider(f"Bottom Start (x={x_start:.0f})", y_min_limit, y_max_limit, float(init_y_bs), 0.001, format="%.4f")
    bot_y_end = st.sidebar.slider(f"Bottom End (x={x_end:.0f})", y_min_limit, y_max_limit, float(init_y_be), 0.001, format="%.4f")

    # 3. Plotting Area
    col_plot, col_data = st.columns([3, 1])

    with col_plot:
        # Build Plotly Figure
        fig = go.Figure()

        # Add Density Scatter using WebGL for high performance with large datasets
        fig.add_trace(go.Scattergl(
            x=mz_vals,
            y=im_vals,
            mode='markers',
            marker=dict(
                color=density,
                colorscale='Jet',
                opacity=0.6,
                size=4
            ),
            name='Precursors',
            # This enables the free hover data functionality
            hovertemplate='<b>m/z:</b> %{x:.2f}<br><b>1/K0:</b> %{y:.4f}<extra></extra>' 
        ))

        # Add Upper Boundary Line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[top_y_start, top_y_end],
            mode='lines',
            line=dict(color='red', width=3),
            name='Upper Boundary',
            hoverinfo='skip' # Don't clutter hover box with line data
        ))

        # Add Lower Boundary Line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[bot_y_start, bot_y_end],
            mode='lines',
            line=dict(color='red', width=3),
            name='Lower Boundary',
            hoverinfo='skip'
        ))

        # Format Layout
        fig.update_layout(
            xaxis_title="Mass (m/z)",
            yaxis_title="Mobility (1/K0)",
            xaxis=dict(range=[min(mz_vals)-50, max(mz_vals)+50]),
            yaxis=dict(range=[min(im_vals)-0.05, max(im_vals)+0.05]),
            height=700,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="closest"
        )
        
        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📸 *Tip: Hover over the top right corner of the plot and click the camera icon to download this graph as a PNG.*")

    # 4. Results & Downloads Area
    with col_data:
        st.subheader("Current Coordinates")
        st.markdown(f"**Top Line:**\n* Start: ({x_start:.2f}, {top_y_start:.4f})\n* End: ({x_end:.2f}, {top_y_end:.4f})")
        st.markdown(f"**Bottom Line:**\n* Start: ({x_start:.2f}, {bot_y_start:.4f})\n* End: ({x_end:.2f}, {bot_y_end:.4f})")
        
        st.divider()
        st.subheader("Export Data")

        # Create TXT string for coordinates
        txt_content = (
            "Line,X_Start,Y_Start,X_End,Y_End\n"
            f"Top,{x_start:.2f},{top_y_start:.4f},{x_end:.2f},{top_y_end:.4f}\n"
            f"Bottom,{x_start:.2f},{bot_y_start:.4f},{x_end:.2f},{bot_y_end:.4f}\n"
        )
        
        st.download_button(
            "📄 Download Coordinates (.txt)", 
            data=txt_content, 
            file_name="Custom_Boundaries.txt", 
            mime="text/plain", 
            use_container_width=True
        )

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

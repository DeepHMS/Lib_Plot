import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import io

# 1. Streamlit Page Configuration
st.set_page_config(page_title="Adaptive Method Optimization", layout="wide")
st.title("Adaptive Density Optimization for dia-PASEF")

# --- INITIALIZE SESSION STATE ---
# This keeps track of what phase of the app the user is currently in.
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'locked_boundaries' not in st.session_state:
    st.session_state.locked_boundaries = {}
if 'method_params' not in st.session_state:
    st.session_state.method_params = {}

def reset_app():
    st.session_state.phase = 1
    st.session_state.locked_boundaries = {}
    st.session_state.method_params = {}

# --- CACHED DATA PROCESSING ---
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file, sep="\t")
    precursors = df[['PrecursorMz', 'PrecursorIonMobility']].drop_duplicates().dropna()
    mz_vals = precursors['PrecursorMz'].values
    im_vals = precursors['PrecursorIonMobility'].values

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

    x_start, x_end = min(mz_vals) - 20, max(mz_vals) + 20
    
    return x_start, x_end, m_top, c_top, m_bot, c_bot

# --- APP LAYOUT ---
st.sidebar.header("1. Upload Library")
uploaded_file = st.sidebar.file_uploader("Upload .tsv file", type=['tsv', 'txt', 'csv'], on_change=reset_app)

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        mz_vals, im_vals, density = load_and_process_data(uploaded_file)
        x_start, x_end, m_top_init, c_top_init, m_bot_init, c_bot_init = calculate_initial_boundaries(mz_vals, im_vals)

    # -------------------------------------------------------------------------
    # PHASE 1: BOUNDARY SELECTION
    # -------------------------------------------------------------------------
    if st.session_state.phase == 1:
        st.sidebar.header("2. Set Polygon Boundaries")
        
        init_y_ts = m_top_init * x_start + c_top_init
        init_y_te = m_top_init * x_end + c_top_init
        init_y_bs = m_bot_init * x_start + c_bot_init
        init_y_be = m_bot_init * x_end + c_bot_init

        y_min_limit, y_max_limit = float(min(im_vals) - 0.2), float(max(im_vals) + 0.2)

        top_y_start = st.sidebar.slider(f"Top Start (x={x_start:.0f})", y_min_limit, y_max_limit, float(init_y_ts), 0.001)
        top_y_end = st.sidebar.slider(f"Top End (x={x_end:.0f})", y_min_limit, y_max_limit, float(init_y_te), 0.001)
        st.sidebar.divider()
        bot_y_start = st.sidebar.slider(f"Bottom Start (x={x_start:.0f})", y_min_limit, y_max_limit, float(init_y_bs), 0.001)
        bot_y_end = st.sidebar.slider(f"Bottom End (x={x_end:.0f})", y_min_limit, y_max_limit, float(init_y_be), 0.001)

        # Calculate live slopes
        m_top = (top_y_end - top_y_start) / (x_end - x_start)
        c_top = top_y_start - (m_top * x_start)
        m_bot = (bot_y_end - bot_y_start) / (x_end - x_start)
        c_bot = bot_y_start - (m_bot * x_start)

        # Plot Phase 1
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.6, size=4), name='Precursors'))
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[top_y_start, top_y_end], mode='lines', line=dict(color='red', width=3), name='Upper Bound'))
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[bot_y_start, bot_y_end], mode='lines', line=dict(color='red', width=3), name='Lower Bound'))
            fig.update_layout(xaxis_title="Mass (m/z)", yaxis_title="Mobility (1/K0)", height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.success("Adjust the red boundaries to perfectly encompass your ion cloud.")
            txt_content = f"Line,X_Start,Y_Start,X_End,Y_End\nTop,{x_start:.2f},{top_y_start:.4f},{x_end:.2f},{top_y_end:.4f}\nBottom,{x_start:.2f},{bot_y_start:.4f},{x_end:.2f},{bot_y_end:.4f}\n"
            st.download_button("📥 Download Initial Coordinates", data=txt_content, file_name="Initial_Boundaries.txt", use_container_width=True)
            
            st.divider()
            if st.button("✅ Done (Lock Boundaries)", type="primary", use_container_width=True):
                st.session_state.locked_boundaries = {
                    'x_start': x_start, 'x_end': x_end,
                    'm_top': m_top, 'c_top': c_top,
                    'm_bot': m_bot, 'c_bot': c_bot
                }
                st.session_state.phase = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 2: METHOD LIMITS
    # -------------------------------------------------------------------------
    elif st.session_state.phase == 2:
        st.sidebar.header("3. Method Development Limits")
        st.sidebar.success("🔒 Boundaries Locked.")

        mz_data_min, mz_data_max = float(min(mz_vals)), float(max(mz_vals))
        
        # User sets constraints
        mz_min = st.sidebar.number_input("Min m/z", value=mz_data_min, min_value=mz_data_min)
        mz_max = st.sidebar.number_input("Max m/z", value=mz_data_max, max_value=mz_data_max)
        num_windows = st.sidebar.slider("Number of Vertical Bins", min_value=10, max_value=100, value=30)

        # Draw the locked polygon
        b = st.session_state.locked_boundaries
        y_top_start = b['m_top'] * b['x_start'] + b['c_top']
        y_top_end = b['m_top'] * b['x_end'] + b['c_top']
        y_bot_start = b['m_bot'] * b['x_start'] + b['c_bot']
        y_bot_end = b['m_bot'] * b['x_end'] + b['c_bot']

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.4, size=3), name='Precursors'))
            fig.add_trace(go.Scatter(x=[b['x_start'], b['x_end']], y=[y_top_start, y_top_end], mode='lines', line=dict(color='red', width=2), name='Locked Upper Bound'))
            fig.add_trace(go.Scatter(x=[b['x_start'], b['x_end']], y=[y_bot_start, y_bot_end], mode='lines', line=dict(color='red', width=2), name='Locked Lower Bound'))
            
            # Highlight chosen m/z limits with vertical dashed lines
            fig.add_vline(x=mz_min, line_dash="dash", line_color="black")
            fig.add_vline(x=mz_max, line_dash="dash", line_color="black")
            
            fig.update_layout(xaxis_title="Mass (m/z)", yaxis_title="Mobility (1/K0)", height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.info("Set your m/z limits for the method. The dashed black lines show your selected area.")
            if st.button("🚀 Start Method Development using Adaptive Density Optimization", type="primary", use_container_width=True):
                st.session_state.method_params = {'mz_min': mz_min, 'mz_max': mz_max, 'num_windows': num_windows}
                st.session_state.phase = 3
                st.rerun()
            
            if st.button("🔄 Unlock & Go Back"):
                st.session_state.phase = 1
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 3: ADAPTIVE DENSITY OPTIMIZATION
    # -------------------------------------------------------------------------
    elif st.session_state.phase == 3:
        st.sidebar.header("4. Results")
        st.sidebar.success("✅ Method Generated.")
        
        if st.sidebar.button("🔄 Start Over"):
            reset_app()
            st.rerun()

        # Extract variables
        b = st.session_state.locked_boundaries
        p = st.session_state.method_params
        
        # --- CORE ADAPTIVE DENSITY LOGIC ---
        # 1. Filter precursors within the chosen m/z range
        mask = (mz_vals >= p['mz_min']) & (mz_vals <= p['mz_max'])
        filtered_mz = np.sort(mz_vals[mask])
        
        # 2. Chop into equal precursor chunks (Density Adaptive)
        quantiles = np.linspace(0, 1, p['num_windows'] + 1)
        mz_edges = np.quantile(filtered_mz, quantiles)
        
        # 3. Generate Vertical Bins restricted by Polygon
        method_export = []
        rectangles = []

        for i in range(len(mz_edges) - 1):
            x_left = mz_edges[i]
            x_right = mz_edges[i+1]
            
            # To ensure the rectangle encompasses the sloped polygon inside this bin:
            # Evaluate the top/bottom lines at both edges and take the absolute max/min
            y_top_left = b['m_top'] * x_left + b['c_top']
            y_top_right = b['m_top'] * x_right + b['c_top']
            y_bot_left = b['m_bot'] * x_left + b['c_bot']
            y_bot_right = b['m_bot'] * x_right + b['c_bot']
            
            rect_top = max(y_top_left, y_top_right)
            rect_bot = min(y_bot_left, y_bot_right)
            
            rectangles.append((x_left, x_right, rect_bot, rect_top))
            
            method_export.append({
                "Window_No": i+1,
                "Start_mz": round(x_left, 2), "End_mz": round(x_right, 2),
                "Width_Da": round(x_right - x_left, 2),
                "Start_1/K0": round(rect_bot, 3), "End_1/K0": round(rect_top, 3)
            })
            
        method_df = pd.DataFrame(method_export)

        # Plot Phase 3
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            # 1. Cloud
            fig.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color='gray', opacity=0.2, size=3), name='Precursors', hoverinfo='skip'))
            
            # 2. Red Polygon (Constrained to user m/z min/max)
            poly_y_top1 = b['m_top'] * p['mz_min'] + b['c_top']
            poly_y_top2 = b['m_top'] * p['mz_max'] + b['c_top']
            poly_y_bot1 = b['m_bot'] * p['mz_min'] + b['c_bot']
            poly_y_bot2 = b['m_bot'] * p['mz_max'] + b['c_bot']
            
            fig.add_trace(go.Scatter(
                x=[p['mz_min'], p['mz_max'], p['mz_max'], p['mz_min'], p['mz_min']],
                y=[poly_y_top1, poly_y_top2, poly_y_bot2, poly_y_bot1, poly_y_top1],
                mode='lines', line=dict(color='red', width=3), name='User Polygon'
            ))

            # 3. Light Purple Bins
            for x1, x2, y1, y2 in rectangles:
                fig.add_trace(go.Scatter(
                    x=[x1, x2, x2, x1, x1], y=[y1, y1, y2, y2, y1],
                    mode='lines', line=dict(color='purple', width=1),
                    fill='toself', fillcolor='rgba(177, 156, 217, 0.4)', # Light Purple
                    hoverinfo='skip', showlegend=False
                ))

            fig.update_layout(xaxis_title="Mass (m/z)", yaxis_title="Mobility (1/K0)", height=700,
                              title="Adaptive Density Rectangles (Narrow = High Density, Wide = Low Density)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.success("Optimization Complete!")
            st.markdown("Notice how the light purple bins shrink tightly in the dense core, but expand widely in sparse areas—all while perfectly respecting your red polygon.")
            
            # Download Final Method
            csv_buffer = io.StringIO()
            method_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Final dia-PASEF Windows", data=csv_buffer.getvalue(), file_name="Adaptive_diaPASEF_Method.csv", mime="text/csv", use_container_width=True)
            
            st.dataframe(method_df[['Start_mz', 'End_mz', 'Width_Da']], height=400)

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

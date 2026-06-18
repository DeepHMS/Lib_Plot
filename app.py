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
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'b_state' not in st.session_state:
    st.session_state.b_state = {}
if 'p_state' not in st.session_state:
    st.session_state.p_state = {}

def reset_app():
    st.session_state.phase = 1
    st.session_state.b_state = {}
    st.session_state.p_state = {}

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

# --- APP LAYOUT & SIDEBAR ---
st.sidebar.header("1. Upload Library")
uploaded_file = st.sidebar.file_uploader("Upload .tsv file", type=['tsv', 'txt', 'csv'], on_change=reset_app)

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        mz_vals, im_vals, density = load_and_process_data(uploaded_file)
        x_start, x_end, m_top_init, c_top_init, m_bot_init, c_bot_init = calculate_initial_boundaries(mz_vals, im_vals)

    st.sidebar.header("2. Plot Appearance & Axis Limits")
    st.sidebar.markdown("*These settings apply to all plots globally.*")
    marker_size = st.sidebar.slider("Precursor Marker Size", min_value=1, max_value=15, value=4, step=1)
    
    st.sidebar.subheader("Axis Limits")
    c_x1, c_x2 = st.sidebar.columns(2)
    x_axis_min = c_x1.number_input("X Min (m/z)", value=float(min(mz_vals) - 50))
    x_axis_max = c_x2.number_input("X Max (m/z)", value=float(max(mz_vals) + 50))

    c_y1, c_y2 = st.sidebar.columns(2)
    y_axis_min = c_y1.number_input("Y Min (1/K0)", value=float(min(im_vals) - 0.05), format="%.3f")
    y_axis_max = c_y2.number_input("Y Max (1/K0)", value=float(max(im_vals) + 0.05), format="%.3f")

    # -------------------------------------------------------------------------
    # PHASE 1: BOUNDARY SELECTION
    # -------------------------------------------------------------------------
    st.markdown("### Step 1: Set Polygon Boundaries")
    
    init_y_ts = m_top_init * x_start + c_top_init
    init_y_te = m_top_init * x_end + c_top_init
    init_y_bs = m_bot_init * x_start + c_bot_init
    init_y_be = m_bot_init * x_end + c_bot_init

    val_ts = float(np.clip(init_y_ts, y_axis_min, y_axis_max))
    val_te = float(np.clip(init_y_te, y_axis_min, y_axis_max))
    val_bs = float(np.clip(init_y_bs, y_axis_min, y_axis_max))
    val_be = float(np.clip(init_y_be, y_axis_min, y_axis_max))

    p1_disabled = st.session_state.phase > 1

    # SAFETY NET 1
    if p1_disabled and 'm_top' not in st.session_state.b_state:
        st.session_state.phase = 1
        st.rerun()

    c1, c2 = st.columns(2)
    top_y_start = c1.slider(f"Top Start (x={x_start:.0f})", y_axis_min, y_axis_max, val_ts, 0.001, format="%.4f", disabled=p1_disabled)
    top_y_end = c2.slider(f"Top End (x={x_end:.0f})", y_axis_min, y_axis_max, val_te, 0.001, format="%.4f", disabled=p1_disabled)
    
    c3, c4 = st.columns(2)
    bot_y_start = c3.slider(f"Bottom Start (x={x_start:.0f})", y_axis_min, y_axis_max, val_bs, 0.001, format="%.4f", disabled=p1_disabled)
    bot_y_end = c4.slider(f"Bottom End (x={x_end:.0f})", y_axis_min, y_axis_max, val_be, 0.001, format="%.4f", disabled=p1_disabled)

    if p1_disabled:
        m_top = st.session_state.b_state['m_top']
        c_top = st.session_state.b_state['c_top']
        m_bot = st.session_state.b_state['m_bot']
        c_bot = st.session_state.b_state['c_bot']
    else:
        m_top = (top_y_end - top_y_start) / (x_end - x_start)
        c_top = top_y_start - (m_top * x_start)
        m_bot = (bot_y_end - bot_y_start) / (x_end - x_start)
        c_bot = bot_y_start - (m_bot * x_start)

    view_top_start, view_top_end = (m_top * x_axis_min) + c_top, (m_top * x_axis_max) + c_top
    view_bot_start, view_bot_end = (m_bot * x_axis_min) + c_bot, (m_bot * x_axis_max) + c_bot

    fig1 = go.Figure()
    fig1.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.6, size=marker_size), name='Precursors', hovertemplate='<b>m/z:</b> %{x:.2f}<br><b>1/K0:</b> %{y:.4f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=[x_axis_min, x_axis_max], y=[view_top_start, view_top_end], mode='lines', line=dict(color='red', width=3), name='Upper Bound', hoverinfo='skip'))
    fig1.add_trace(go.Scatter(x=[x_axis_min, x_axis_max], y=[view_bot_start, view_bot_end], mode='lines', line=dict(color='red', width=3), name='Lower Bound', hoverinfo='skip'))
    fig1.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    c_btn1, c_btn2 = st.columns([1, 4])
    if st.session_state.phase == 1:
        if c_btn1.button("✅ Done (Lock Boundaries)", type="primary"):
            st.session_state.b_state = {'x_start': x_start, 'x_end': x_end, 'm_top': m_top, 'c_top': c_top, 'm_bot': m_bot, 'c_bot': c_bot}
            st.session_state.phase = 2
            st.rerun()
    else:
        if c_btn1.button("🔓 Unlock & Edit Step 1"):
            st.session_state.phase = 1
            st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 2: METHOD LIMITS
    # -------------------------------------------------------------------------
    if st.session_state.phase >= 2:
        # SAFETY NET 2
        if 'm_top' not in st.session_state.b_state:
            st.session_state.phase = 1
            st.rerun()

        st.markdown("---")
        st.markdown("### Step 2: Method Development Limits")
        
        p2_disabled = st.session_state.phase > 2
        
        c1, c2, c3 = st.columns(3)
        mz_min = c1.number_input("Min m/z for Method", value=float(min(mz_vals)), disabled=p2_disabled)
        mz_max = c2.number_input("Max m/z for Method", value=float(max(mz_vals)), disabled=p2_disabled)
        num_windows = c3.slider("Number of Vertical Bins", min_value=10, max_value=100, value=30, disabled=p2_disabled)

        if p2_disabled:
            mz_min = st.session_state.p_state['mz_min']
            mz_max = st.session_state.p_state['mz_max']
            num_windows = st.session_state.p_state['num_windows']

        fig2 = go.Figure()
        fig2.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.3, size=marker_size), name='Precursors', hoverinfo='skip'))
        fig2.add_trace(go.Scatter(x=[x_axis_min, x_axis_max], y=[view_top_start, view_top_end], mode='lines', line=dict(color='red', width=2), name='Locked Upper Bound', hoverinfo='skip'))
        fig2.add_trace(go.Scatter(x=[x_axis_min, x_axis_max], y=[view_bot_start, view_bot_end], mode='lines', line=dict(color='red', width=2), name='Locked Lower Bound', hoverinfo='skip'))
        fig2.add_vline(x=mz_min, line_dash="dash", line_color="black")
        fig2.add_vline(x=mz_max, line_dash="dash", line_color="black")
        fig2.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        c_btn1, c_btn2 = st.columns([2, 4])
        if st.session_state.phase == 2:
            if c_btn1.button("🚀 Start Method Development (Adaptive Density)", type="primary"):
                st.session_state.p_state = {'mz_min': mz_min, 'mz_max': mz_max, 'num_windows': num_windows}
                st.session_state.phase = 3
                st.rerun()
        else:
            if c_btn1.button("🔓 Unlock & Edit Step 2"):
                st.session_state.phase = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 3: ADAPTIVE DENSITY OPTIMIZATION
    # -------------------------------------------------------------------------
    if st.session_state.phase == 3:
        # SAFETY NET 3
        if 'mz_min' not in st.session_state.p_state or 'm_top' not in st.session_state.b_state:
            st.session_state.phase = 1
            st.rerun()

        st.markdown("---")
        st.markdown("### Step 3: Adaptive Density Results")
        
        b = st.session_state.b_state
        p = st.session_state.p_state
        
        mask = (mz_vals >= p['mz_min']) & (mz_vals <= p['mz_max'])
        filtered_mz = np.sort(mz_vals[mask])
        
        quantiles = np.linspace(0, 1, p['num_windows'] + 1)
        mz_edges = np.quantile(filtered_mz, quantiles)
        
        method_export = []
        rectangles = []

        for i in range(len(mz_edges) - 1):
            x_left = mz_edges[i]
            x_right = mz_edges[i+1]
            
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

        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color='gray', opacity=0.2, size=marker_size), name='Precursors', hoverinfo='skip'))
        
        poly_y_top1 = b['m_top'] * p['mz_min'] + b['c_top']
        poly_y_top2 = b['m_top'] * p['mz_max'] + b['c_top']
        poly_y_bot1 = b['m_bot'] * p['mz_min'] + b['c_bot']
        poly_y_bot2 = b['m_bot'] * p['mz_max'] + b['c_bot']
        
        fig3.add_trace(go.Scatter(
            x=[p['mz_min'], p['mz_max'], p['mz_max'], p['mz_min'], p['mz_min']],
            y=[poly_y_top1, poly_y_top2, poly_y_bot2, poly_y_bot1, poly_y_top1],
            mode='lines', line=dict(color='red', width=3), name='User Polygon', hoverinfo='skip'
        ))

        # --- FIX: PROPER HOVERTEXT FOR SHAPES ---
        for i, (x1, x2, y1, y2) in enumerate(rectangles):
            bin_mask = (mz_vals >= x1) & (mz_vals <= x2) & (im_vals >= y1) & (im_vals <= y2)
            prec_count = np.sum(bin_mask)
            
            hover_text = (f"<b>Bin {i+1}</b><br>"
                          f"Precursors: {prec_count}<br>"
                          f"m/z: {x1:.2f} - {x2:.2f}<br>"
                          f"1/K0: {y1:.3f} - {y2:.3f}")

            fig3.add_trace(go.Scatter(
                x=[x1, x2, x2, x1, x1], y=[y1, y1, y2, y2, y1],
                mode='lines', 
                line=dict(color='purple', width=1),
                fill='toself', 
                fillcolor='rgba(177, 156, 217, 0.4)',
                text=hover_text, 
                hovertemplate="%{text}<extra></extra>", # `<extra></extra>` removes the "trace X" box
                hoveron='fills', # Forces the hover to trigger anywhere inside the polygon
                showlegend=False
            ))

        fig3.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=600, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        # --- SUMMARY TABLE ---
        st.markdown("#### Method Summary")
        summary_data = [{
            "1/K0 Start": f"{min([r[2] for r in rectangles]):.4f}",
            "1/K0 End": f"{max([r[3] for r in rectangles]):.4f}",
            "MS1 Ramps": 1,
            "MS/MS Ramps": int(np.ceil(p['num_windows'] / 3)),
            "Total Windows": p['num_windows'],
            "Mass Range (m/z)": f"{p['mz_min']:.2f} - {p['mz_max']:.2f}"
        }]
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # --- EXPORT BUTTONS ---
        c1, c2, c3 = st.columns(3)
        csv_buffer = io.StringIO()
        method_df.to_csv(csv_buffer, index=False)
        c1.download_button("📥 Download Final Windows (.csv)", data=csv_buffer.getvalue(), file_name="Adaptive_diaPASEF_Method.csv", mime="text/csv", use_container_width=True)

        txt_content = f"Line,X_Start,Y_Start,X_End,Y_End\nTop,{x_start:.2f},{view_top_start:.4f},{x_end:.2f},{view_top_end:.4f}\nBottom,{x_start:.2f},{view_bot_start:.4f},{x_end:.2f},{view_bot_end:.4f}\n"
        c2.download_button("📄 Download Polygon Coordinates (.txt)", data=txt_content, file_name="Custom_Boundaries.txt", mime="text/plain", use_container_width=True)

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

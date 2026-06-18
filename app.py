import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import io
import zipfile

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
if 'generated_methods' not in st.session_state:
    st.session_state.generated_methods = []

def reset_app():
    st.session_state.phase = 1
    st.session_state.b_state = {}
    st.session_state.p_state = {}
    st.session_state.generated_methods = []

# Helper function to convert Hex + Opacity into RGBA for Plotly fills
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# --- CACHED DATA PROCESSING ---
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file, sep="\t")
    
    # Fallback for different search engine export formats
    mz_col = 'PrecursorMz' if 'PrecursorMz' in df.columns else 'Precursor.Mz'
    im_col = 'PrecursorIonMobility' if 'PrecursorIonMobility' in df.columns else 'IonMobility'
    
    precursors = df[[mz_col, im_col]].drop_duplicates().dropna()
    mz_vals = precursors[mz_col].values
    im_vals = precursors[im_col].values

    # Performance Fix: Downsample for plotting and KDE if dataset is massive
    if len(mz_vals) > 10000:
        idx = np.random.choice(len(mz_vals), 10000, replace=False)
        mz_vals = mz_vals[idx]
        im_vals = im_vals[idx]

    xy = np.vstack([mz_vals, im_vals])
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
    marker_size = st.sidebar.slider("Precursor Marker Size", min_value=1, max_value=15, value=4, step=1)
    
    st.sidebar.subheader("Axis Limits")
    c_x1, c_x2 = st.sidebar.columns(2)
    x_axis_min = c_x1.number_input("X Min (m/z)", value=float(min(mz_vals) - 50))
    x_axis_max = c_x2.number_input("X Max (m/z)", value=float(max(mz_vals) + 50))
    c_y1, c_y2 = st.sidebar.columns(2)
    y_axis_min = c_y1.number_input("Y Min (1/K0)", value=float(min(im_vals) - 0.05), format="%.3f")
    y_axis_max = c_y2.number_input("Y Max (1/K0)", value=float(max(im_vals) + 0.05), format="%.3f")

    st.sidebar.subheader("Bin Appearance (Steps 3 & 4)")
    bin_color_hex = st.sidebar.color_picker("Bin Edge & Fill Color", value="#9370DB")
    bin_opacity = st.sidebar.slider("Bin Fill Opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    bin_fill_rgba = hex_to_rgba(bin_color_hex, bin_opacity)

    st.sidebar.subheader("Method Parameters")
    mz_overlap = st.sidebar.number_input("Window Overlap (m/z)", min_value=0.0, max_value=5.0, value=1.0, step=0.5, help="Added to boundaries to account for quadrupole isolation efficiency.")

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
        m_top, c_top = st.session_state.b_state['m_top'], st.session_state.b_state['c_top']
        m_bot, c_bot = st.session_state.b_state['m_bot'], st.session_state.b_state['c_bot']
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
        if 'm_top' not in st.session_state.b_state:
            st.session_state.phase = 1
            st.rerun()

        st.markdown("---")
        st.markdown("### Step 2: Method Development Limits")
        p2_disabled = st.session_state.phase > 2
        
        c1, c2, c3 = st.columns(3)
        mz_min = c1.number_input("Min m/z for Method", value=float(min(mz_vals)), disabled=p2_disabled)
        mz_max = c2.number_input("Max m/z for Method", value=float(max(mz_vals)), disabled=p2_disabled)
        num_windows = c3.slider("Number of Vertical Bins (Base Method)", min_value=10, max_value=100, value=30, disabled=p2_disabled)

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
            if c_btn1.button("🚀 View Base Method", type="primary"):
                st.session_state.p_state = {'mz_min': mz_min, 'mz_max': mz_max, 'num_windows': num_windows}
                st.session_state.phase = 3
                st.rerun()
        else:
            if c_btn1.button("🔓 Unlock & Edit Step 2"):
                st.session_state.phase = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 3: BASE METHOD (METHOD 1)
    # -------------------------------------------------------------------------
    if st.session_state.phase >= 3:
        if 'mz_min' not in st.session_state.p_state or 'm_top' not in st.session_state.b_state:
            st.session_state.phase = 1
            st.rerun()

        st.markdown("---")
        st.markdown("### Step 3: Base Method Generation")
        
        b = st.session_state.b_state
        p = st.session_state.p_state
        mask = (mz_vals >= p['mz_min']) & (mz_vals <= p['mz_max'])
        filtered_mz = np.sort(mz_vals[mask])
        
        def generate_method_logic(cycles, mz_arr, b_params, overlap):
            w_count = cycles * 3 
            quants = np.linspace(0, 1, w_count + 1)
            edges = np.quantile(mz_arr, quants)
            
            rects, m_export, bruker_export = [], [], []
            for i in range(len(edges) - 1):
                # Apply logical overlap for isolation efficiency
                raw_x1, raw_x2 = edges[i], edges[i+1]
                x1 = max(mz_arr.min(), raw_x1 - overlap) if i > 0 else raw_x1
                x2 = min(mz_arr.max(), raw_x2 + overlap) if i < (len(edges) - 2) else raw_x2

                y_tl = b_params['m_top'] * x1 + b_params['c_top']
                y_tr = b_params['m_top'] * x2 + b_params['c_top']
                y_bl = b_params['m_bot'] * x1 + b_params['c_bot']
                y_br = b_params['m_bot'] * x2 + b_params['c_bot']
                
                rect_top, rect_bot = max(y_tl, y_tr), min(y_bl, y_br)
                rects.append((x1, x2, rect_bot, rect_top))
                
                cycle_id = (i // 3) + 1
                bruker_export.append({
                    "#MS Type": "PASEF", "Cycle Id": cycle_id,
                    "Start IM [1/K0]": f"{rect_bot:.4f}", "End IM [1/K0]": f"{rect_top:.4f}",
                    "Start Mass [m/z]": f"{x1:.2f}", "End Mass [m/z]": f"{x2:.2f}", "CE [eV]": "-"
                })
            
            bruker_df = pd.DataFrame(bruker_export)
            ms1_row = pd.DataFrame([{"#MS Type": "MS1", "Cycle Id": 0, "Start IM [1/K0]": "-", "End IM [1/K0]": "-", "Start Mass [m/z]": "-", "End Mass [m/z]": "-", "CE [eV]": "-"}])
            bruker_df = pd.concat([ms1_row, bruker_df], ignore_index=True)
            
            summary = {
                "1/K0 Start": f"{min([r[2] for r in rects]):.4f}",
                "1/K0 End": f"{max([r[3] for r in rects]):.4f}",
                "MS1 Ramps": 1,
                "MS/MS Ramps": cycles,
                "Total Windows": w_count,
                "Mass Range (m/z)": f"{edges[0]:.2f} - {edges[-1]:.2f}"
            }
            return rects, bruker_df, summary

        base_cycles = int(np.ceil(p['num_windows'] / 3))
        base_rects, _, _ = generate_method_logic(base_cycles, filtered_mz, b, mz_overlap)

        fig3 = go.Figure()
        
        fig3.add_trace(go.Scattergl(x=mz_vals, y=im_vals, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.5, size=marker_size), hoverinfo='skip', showlegend=False))
        
        poly_y_top1, poly_y_top2 = b['m_top'] * p['mz_min'] + b['c_top'], b['m_top'] * p['mz_max'] + b['c_top']
        poly_y_bot1, poly_y_bot2 = b['m_bot'] * p['mz_min'] + b['c_bot'], b['m_bot'] * p['mz_max'] + b['c_bot']
        fig3.add_trace(go.Scatter(x=[p['mz_min'], p['mz_max'], p['mz_max'], p['mz_min'], p['mz_min']], y=[poly_y_top1, poly_y_top2, poly_y_bot2, poly_y_bot1, poly_y_top1], mode='lines', line=dict(color='red', width=3), hoverinfo='skip', showlegend=False))
        
        for i, (x1, x2, y1, y2) in enumerate(base_rects):
            prec_count = np.sum((mz_vals >= x1) & (mz_vals <= x2) & (im_vals >= y1) & (im_vals <= y2))
            
            hover_text = (f"<b>Bin {i+1}</b><br>"
                          f"Precursors: {prec_count}<br>"
                          f"m/z: {x1:.2f} - {x2:.2f}<br>"
                          f"1/K0: {y1:.3f} - {y2:.3f}")
            
            fig3.add_trace(go.Scatter(
                x=[x1, x2, x2, x1, x1], y=[y1, y1, y2, y2, y1], 
                mode='lines', line=dict(color=bin_color_hex, width=1), 
                fill='toself', fillcolor=bin_fill_rgba, 
                text=hover_text, hovertemplate="%{text}<extra></extra>", hoveron='fills', name=f"Bin {i+1}", showlegend=False
            ))
        
        fig3.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        if st.session_state.phase == 3:
            if st.button("Proceed to Multi-Method Generation", type="primary"):
                st.session_state.phase = 4
                st.session_state.generated_methods = [] 
                st.rerun()
        else:
            if st.button("🔓 Unlock & Edit Step 3"):
                st.session_state.phase = 3
                st.session_state.generated_methods = []
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 4: ITERATIVE METHOD DEVELOPMENT (MULTI-PLOT)
    # -------------------------------------------------------------------------
    if st.session_state.phase == 4:
        st.markdown("---")
        st.markdown("### Step 4: Iterative High-Throughput Generation")
        
        c1, c2 = st.columns([1, 3])
        num_methods = c1.slider("How many methods to develop?", min_value=1, max_value=25, value=10, step=1)
        c2.warning("⚠️ High number of methods involves intensive computation. The plots below are optimized to prevent browser crashes.")

        if st.button("⚡ Generate Adaptive Methods", type="primary"):
            with st.spinner("Calculating methods..."):
                generated_data = []
                summary_table = []
                
                for m_idx in range(num_methods):
                    c_cycles = base_cycles + m_idx
                    rects, b_df, summary = generate_method_logic(c_cycles, filtered_mz, b, mz_overlap)
                    
                    summary["Method"] = f"Method {m_idx + 1}"
                    summary_table.append(summary)
                    
                    generated_data.append({
                        "id": m_idx + 1,
                        "name": f"Method_{m_idx + 1}",
                        "df": b_df,
                        "rects": rects,
                        "cycles": c_cycles
                    })
                
                st.session_state.generated_methods = generated_data
                st.session_state.summary_df = pd.DataFrame(summary_table)[["Method", "1/K0 Start", "1/K0 End", "MS1 Ramps", "MS/MS Ramps", "Total Windows", "Mass Range (m/z)"]]

        if st.session_state.generated_methods:
            st.success("✅ Iterative Generation Complete!")
            
            # Optimized sub-plot downsampling
            if len(mz_vals) > 2500:
                np.random.seed(42) 
                sample_idx = np.random.choice(len(mz_vals), 2500, replace=False)
                mini_mz = mz_vals[sample_idx]
                mini_im = im_vals[sample_idx]
                mini_density = density[sample_idx]
            else:
                mini_mz = mz_vals
                mini_im = im_vals
                mini_density = density

            cols = st.columns(4)
            selected_methods = []
            
            for idx, m_data in enumerate(st.session_state.generated_methods):
                if 'rects' not in m_data:
                    st.error("Stale data detected. Please click 'Generate Adaptive Methods' above to refresh.")
                    break

                with cols[idx % 4]:
                    fig_m = go.Figure()
                    
                    fig_m.add_trace(go.Scatter(
                        x=mini_mz, y=mini_im, mode='markers', 
                        marker=dict(color=mini_density, colorscale='Jet', opacity=0.5, size=marker_size), 
                        hoverinfo='skip', showlegend=False
                    ))
                    
                    for i, (x1, x2, y1, y2) in enumerate(m_data['rects']):
                        bin_mask = (mz_vals >= x1) & (mz_vals <= x2) & (im_vals >= y1) & (im_vals <= y2)
                        prec_count = np.sum(bin_mask)
                        
                        hover_text = (f"<b>Bin {i+1}</b><br>"
                                      f"Precursors: {prec_count}<br>"
                                      f"m/z: {x1:.2f} - {x2:.2f}<br>"
                                      f"1/K0: {y1:.3f} - {y2:.3f}")

                        fig_m.add_trace(go.Scatter(
                            x=[x1, x2, x2, x1, x1], y=[y1, y1, y2, y2, y1],
                            mode='lines', line=dict(color=bin_color_hex, width=1),
                            fill='toself', fillcolor=bin_fill_rgba,
                            text=hover_text, hovertemplate="%{text}<extra></extra>", hoveron='fills', name=f"Bin {i+1}", showlegend=False
                        ))
                    
                    fig_m.update_layout(title=f"{m_data['name']} ({m_data['cycles']} Cycles)", xaxis_title="m/z", yaxis_title="1/K0", xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), showlegend=False, height=350, margin=dict(l=10, r=10, t=30, b=10))
                    
                    st.plotly_chart(fig_m, use_container_width=True)
                    if st.checkbox(f"Select {m_data['name']}", value=True, key=f"chk_{idx}"):
                        m_data_copy = m_data.copy()
                        m_data_copy['fig'] = fig_m 
                        selected_methods.append(m_data_copy)

            st.markdown("#### Methods Summary Table")
            st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Export Section")
            c_d1, c_d2 = st.columns(2)
            
            csv_table = st.session_state.summary_df.to_csv(index=False)
            c_d1.download_button("📊 Download Summary Table (.csv)", data=csv_table, file_name="Methods_Summary.csv", mime="text/csv", use_container_width=True)

            if c_d2.button("🗜️ Prepare ZIP of Selected Methods", use_container_width=True):
                with st.spinner("Compiling ZIP file..."):
                    zip_buffer = io.BytesIO()
                    image_export_failed = False
                    
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for m in selected_methods:
                            csv_str = m['df'].to_csv(index=False)
                            zf.writestr(f"{m['name']}.txt", csv_str)
                            
                            # Kaleido Safety Net
                            try:
                                img_bytes = m['fig'].to_image(format="png", width=800, height=600)
                                zf.writestr(f"{m['name']}_Plot.png", img_bytes)
                            except Exception:
                                image_export_failed = True
                    
                    if image_export_failed:
                        st.warning("⚠️ Text files exported successfully, but image export failed. Please ensure the `kaleido` package is installed in your environment to generate PNG plots.")
                    
                    st.download_button("📥 Click Here to Download Final ZIP", data=zip_buffer.getvalue(), file_name="Iterated_Methods.zip", mime="application/zip")

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

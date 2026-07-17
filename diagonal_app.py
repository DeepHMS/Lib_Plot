import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import io
import zipfile

# 1. Streamlit Page Configuration
st.set_page_config(page_title="Diagonal Adaptive Method Optimization", layout="wide")
st.title("Diagonal DIA-PASEF Method Development")
st.markdown("Optimize quadrupole isolation windows based on pure precursor spatial density within a diagonal corridor.")

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

def to_bruker_format(method_df, im_min, im_max):
    """Converts the dataframe to the strict Bruker txt format."""
    header = "type, mobility pos.1 [1/K0], mass pos.1 start [m/z], mass pos.1 end [m/z], mobility pos.2 [1/K0], mass pos.2 start [m/z]\n"
    init_row = "ms,-,-,-,-,-\n"
    lines = [header, init_row]
    for _, row in method_df.iterrows():
        line = f"diagonal,{im_min:.2f},{row['mz_start']:.1f},{row['mz_end']:.1f},{im_max:.2f},{row['mz_pos2_start']:.1f}\n"
        lines.append(line)
    return "".join(lines)

# --- CACHED DATA PROCESSING ---
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file, sep="\t")
    
    # Fallback for different search engine export formats
    mz_col = 'PrecursorMz' if 'PrecursorMz' in df.columns else 'Precursor.Mz'
    if mz_col not in df.columns:
        mz_col = next((col for col in df.columns if col.lower() in ['m/z', 'mz']), df.columns[0])
        
    im_col = 'PrecursorIonMobility' if 'PrecursorIonMobility' in df.columns else 'IonMobility'
    if im_col not in df.columns:
        im_col = next((col for col in df.columns if col.lower() in ['1/k0', 'mobility', 'im']), df.columns[min(1, len(df.columns)-1)])
    
    precursors = df[[mz_col, im_col]].drop_duplicates().dropna()
    mz_vals = precursors[mz_col].values
    im_vals = precursors[im_col].values

    # Performance Fix: Downsample for plotting and KDE if dataset is massive
    if len(mz_vals) > 10000:
        np.random.seed(42)
        idx = np.random.choice(len(mz_vals), 10000, replace=False)
        plot_mz = mz_vals[idx]
        plot_im = im_vals[idx]
    else:
        plot_mz = mz_vals
        plot_im = im_vals

    xy = np.vstack([plot_mz, plot_im])
    density = gaussian_kde(xy)(xy)
    
    return mz_vals, im_vals, plot_mz, plot_im, density

# --- APP LAYOUT & SIDEBAR ---
st.sidebar.header("1. Upload Library")
uploaded_file = st.sidebar.file_uploader("Upload .tsv file", type=['tsv', 'txt', 'csv'], on_change=reset_app)

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        mz_vals, im_vals, plot_mz, plot_im, density = load_and_process_data(uploaded_file)

    st.sidebar.header("2. Plot Appearance & Axis Limits")
    marker_size = st.sidebar.slider("Precursor Marker Size", min_value=1, max_value=15, value=4, step=1)
    
    st.sidebar.subheader("Axis Limits")
    c_x1, c_x2 = st.sidebar.columns(2)
    x_axis_min = c_x1.number_input("X Min (m/z)", value=float(min(mz_vals) - 50))
    x_axis_max = c_x2.number_input("X Max (m/z)", value=float(max(mz_vals) + 50))
    c_y1, c_y2 = st.sidebar.columns(2)
    y_axis_min = c_y1.number_input("Y Min (1/K0)", value=float(min(im_vals) - 0.05), format="%.3f")
    y_axis_max = c_y2.number_input("Y Max (1/K0)", value=float(max(im_vals) + 0.05), format="%.3f")

    st.sidebar.subheader("Bin Appearance")
    bin_color_hex = st.sidebar.color_picker("Bin Edge & Fill Color", value="#9370DB")
    bin_opacity = st.sidebar.slider("Bin Fill Opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    bin_fill_rgba = hex_to_rgba(bin_color_hex, bin_opacity)

    # -------------------------------------------------------------------------
    # PHASE 1: MERGED DIAGONAL & METHOD BOUNDARY SELECTION
    # -------------------------------------------------------------------------
    st.markdown("### Step 1: Set Diagonal Boundaries (The Whole Box)")
    p1_disabled = st.session_state.phase > 1
    
    # Merged Parameters
    c1, c2, c3 = st.columns(3)
    im_min = c1.number_input("1/K0 Min (pos.1)", value=0.60, step=0.05, format="%.2f", disabled=p1_disabled)
    im_max = c2.number_input("1/K0 Max (pos.2)", value=1.50, step=0.05, format="%.2f", disabled=p1_disabled)
    slope_offset = c3.slider(f"Angle / Slope (Δ m/z to {im_max:.2f} 1/K0)", min_value=100, max_value=1500, value=1215, step=5, disabled=p1_disabled)

    c4, c5 = st.columns(2)
    mz_min = c4.number_input("m/z Min (Left edge at pos.1)", value=400.0, step=10.0, disabled=p1_disabled)
    mz_max = c5.number_input("m/z Max (Right edge at pos.1)", value=900.0, step=10.0, disabled=p1_disabled)

    # Calculate Box Corners
    box_x = [mz_min, mz_max, mz_max + slope_offset, mz_min + slope_offset, mz_min]
    box_y = [im_min, im_min, im_max, im_max, im_min]

    fig1 = go.Figure()
    fig1.add_trace(go.Scattergl(x=plot_mz, y=plot_im, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.6, size=marker_size), name='Precursors', hovertemplate='<b>m/z:</b> %{x:.2f}<br><b>1/K0:</b> %{y:.4f}<extra></extra>'))
    
    # Draw the merged "Whole Box" Parallelogram
    fig1.add_trace(go.Scatter(
        x=box_x, y=box_y, 
        mode='lines', line=dict(color='red', width=3), 
        fill='toself', fillcolor='rgba(255, 0, 0, 0.1)',
        name='Boundary Box', hoverinfo='skip'
    ))
    
    fig1.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    c_btn1, c_btn2 = st.columns([1, 4])
    if st.session_state.phase == 1:
        if c_btn1.button("✅ Lock Boundaries & Proceed", type="primary"):
            st.session_state.b_state = {'im_min': im_min, 'im_max': im_max, 'mz_min': mz_min, 'mz_max': mz_max, 'slope': slope_offset}
            st.session_state.phase = 2
            st.rerun()
    else:
        if c_btn1.button("🔓 Unlock & Edit Boundaries"):
            st.session_state.phase = 1
            st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 2: OPTIMIZATION STRATEGY
    # -------------------------------------------------------------------------
    if st.session_state.phase >= 2:
        st.markdown("---")
        st.markdown("### Step 2: Optimization Strategy")
        p2_disabled = st.session_state.phase > 2
        
        c1, c2 = st.columns(2)
        num_windows = c1.slider("Number of MS/MS Scans", min_value=1, max_value=20, value=8, step=1, disabled=p2_disabled)
        method_type = c2.radio("Optimization Strategy", ["Fixed", "Variable (Density-Based)"], disabled=p2_disabled)

        if p2_disabled:
            num_windows = st.session_state.p_state['num_windows']
            method_type = st.session_state.p_state['method_type']

        b = st.session_state.b_state

        c_btn1, c_btn2 = st.columns([2, 4])
        if st.session_state.phase == 2:
            if c_btn1.button("🚀 View Base Method", type="primary"):
                st.session_state.p_state = {'num_windows': num_windows, 'method_type': method_type}
                st.session_state.phase = 3
                st.rerun()
        else:
            if c_btn1.button("🔓 Unlock & Edit Strategy"):
                st.session_state.phase = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # PHASE 3: BASE METHOD GENERATION
    # -------------------------------------------------------------------------
    if st.session_state.phase >= 3:
        st.markdown("---")
        st.markdown("### Step 3: Base Method Generation")
        
        b = st.session_state.b_state
        p = st.session_state.p_state
        
        # Exact mask: Filter precursors falling inside the "Whole Box" Parallelogram
        im_ratio = (im_vals - b['im_min']) / (b['im_max'] - b['im_min'])
        expected_shift = im_ratio * b['slope']
        
        corridor_mask = (im_vals >= b['im_min']) & (im_vals <= b['im_max']) & \
                        (mz_vals >= (b['mz_min'] + expected_shift)) & \
                        (mz_vals <= (b['mz_max'] + expected_shift))
                        
        filtered_mz = mz_vals[corridor_mask]
        # Project all filtered masses down to the pos.1 line to density map them properly
        projected_mz_at_pos1 = filtered_mz - expected_shift[corridor_mask]

        def generate_diagonal_logic(mz_arr_projected, scans, m_type, m_min, m_max, b_state, iteration=0):
            windows = []
            widths = []
            rects = []
            
            if m_type == "Fixed":
                step_size = (m_max - m_min) / scans
                for i in range(scans):
                    start = m_min + (i * step_size)
                    end = start + step_size
                    pos2_start = start + b_state['slope']
                    
                    windows.append({
                        'Scan': i + 1, 'mz_start': start, 'mz_end': end, 'mz_pos2_start': pos2_start
                    })
                    widths.append(f"{step_size:.1f}")
                    rects.append((start, end, pos2_start, pos2_start + step_size))

            else:
                sorted_mz = np.sort(mz_arr_projected)
                total_precursors = len(sorted_mz)
                
                if total_precursors == 0:
                    st.warning("No precursors found within the Box Limits!")
                    return [], pd.DataFrame(), {}

                precursors_per_scan = total_precursors / scans
                shift = (iteration * 0.05 * precursors_per_scan)
                
                target_indices = [int(shift + (i * precursors_per_scan)) for i in range(scans + 1)]
                target_indices = [min(idx, total_precursors - 1) for idx in target_indices]
                
                mz_boundaries = [sorted_mz[idx] for idx in target_indices]
                mz_boundaries[0] = m_min
                mz_boundaries[-1] = m_max
                
                for i in range(scans):
                    start = mz_boundaries[i]
                    end = mz_boundaries[i+1]
                    pos2_start = start + b_state['slope']
                    
                    windows.append({
                        'Scan': i + 1, 'mz_start': start, 'mz_end': end, 'mz_pos2_start': pos2_start
                    })
                    widths.append(f"{(end - start):.2f}")
                    rects.append((start, end, pos2_start, pos2_start + (end - start)))

            method_df = pd.DataFrame(windows)
            summary = {
                "Scans": scans,
                "Mass Range (m/z)": f"{m_min:.1f} - {m_max:.1f}",
                "Isolation Widths (Th)": ", ".join(widths)
            }
            return rects, method_df, summary

        base_rects, base_df, base_summary = generate_diagonal_logic(projected_mz_at_pos1, p['num_windows'], p['method_type'], b['mz_min'], b['mz_max'], b)

        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=plot_mz, y=plot_im, mode='markers', marker=dict(color=density, colorscale='Jet', opacity=0.5, size=marker_size), hoverinfo='skip', showlegend=False))
        
        for i, (m1_start, m1_end, m2_start, m2_end) in enumerate(base_rects):
            x_pts = [m1_start, m1_end, m2_end, m2_start, m1_start]
            y_pts = [b['im_min'], b['im_min'], b['im_max'], b['im_max'], b['im_min']]
            
            # Precise precursor count for this slice
            bin_mask = (im_vals >= b['im_min']) & (im_vals <= b['im_max']) & \
                       (mz_vals >= (m1_start + expected_shift)) & \
                       (mz_vals <= (m1_end + expected_shift))
            prec_count = np.sum(bin_mask)
            
            hover_text = (f"<b>Scan {i+1}</b><br>"
                          f"Precursors: {prec_count}<br>"
                          f"m/z pos.1: {m1_start:.2f} - {m1_end:.2f}<br>"
                          f"Width: {(m1_end - m1_start):.2f} Th")
            
            fig3.add_trace(go.Scatter(
                x=x_pts, y=y_pts, 
                mode='lines', line=dict(color=bin_color_hex, width=1), 
                fill='toself', fillcolor=bin_fill_rgba, 
                text=hover_text, hovertemplate="%{text}<extra></extra>", hoveron='fills', name=f"Scan {i+1}", showlegend=False
            ))
        
        fig3.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(pd.DataFrame([base_summary]), use_container_width=True, hide_index=True)

        if p['method_type'] == "Fixed":
            st.success("Fixed Method Generated! Proceed to export below.")
            st.session_state.phase = 4
            st.session_state.generated_methods = [{"id": 1, "name": "Fixed_Method", "df": base_df, "rects": base_rects, "summary": base_summary}]
            st.session_state.summary_df = pd.DataFrame([base_summary])
        else:
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
    # PHASE 4: ITERATIVE METHOD DEVELOPMENT / EXPORT
    # -------------------------------------------------------------------------
    if st.session_state.phase == 4:
        st.markdown("---")
        
        if p['method_type'] == "Variable (Density-Based)":
            st.markdown("### Step 4: Iterative High-Throughput Generation")
            c1, c2 = st.columns([1, 3])
            num_methods = c1.slider("How many methods to develop?", min_value=1, max_value=25, value=3, step=1)
            c2.warning("⚠️ High number of methods involves intensive computation. The plots below are optimized to prevent browser crashes.")

            if st.button("⚡ Generate Adaptive Methods", type="primary"):
                with st.spinner("Calculating methods..."):
                    generated_data = []
                    summary_table = []
                    
                    for m_idx in range(num_methods):
                        rects, b_df, summary = generate_diagonal_logic(projected_mz_at_pos1, p['num_windows'], p['method_type'], b['mz_min'], b['mz_max'], b, iteration=m_idx)
                        
                        summary["Method"] = f"Variable_Method_{m_idx + 1}"
                        summary_table.append(summary)
                        
                        generated_data.append({
                            "id": m_idx + 1,
                            "name": summary["Method"],
                            "df": b_df,
                            "rects": rects
                        })
                    
                    st.session_state.generated_methods = generated_data
                    cols = ["Method"] + [c for c in summary_table[0] if c != "Method"]
                    st.session_state.summary_df = pd.DataFrame(summary_table)[cols]

        if st.session_state.generated_methods:
            if p['method_type'] == "Variable (Density-Based)":
                st.success("✅ Iterative Generation Complete!")
                
                cols = st.columns(4)
                selected_methods = []
                
                for idx, m_data in enumerate(st.session_state.generated_methods):
                    with cols[idx % 4]:
                        fig_m = go.Figure()
                        fig_m.add_trace(go.Scatter(
                            x=plot_mz, y=plot_im, mode='markers', 
                            marker=dict(color=density, colorscale='Jet', opacity=0.5, size=marker_size), 
                            hoverinfo='skip', showlegend=False
                        ))
                        
                        for i, (m1_start, m1_end, m2_start, m2_end) in enumerate(m_data['rects']):
                            x_pts = [m1_start, m1_end, m2_end, m2_start, m1_start]
                            y_pts = [b['im_min'], b['im_min'], b['im_max'], b['im_max'], b['im_min']]
                            
                            fig_m.add_trace(go.Scatter(
                                x=x_pts, y=y_pts,
                                mode='lines', line=dict(color=bin_color_hex, width=1),
                                fill='toself', fillcolor=bin_fill_rgba,
                                hoverinfo='skip', showlegend=False
                            ))
                        
                        fig_m.update_layout(title=m_data['name'], xaxis_title="m/z", yaxis_title="1/K0", xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), showlegend=False, height=350, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_m, use_container_width=True)
                        
                        if st.checkbox(f"Select {m_data['name']}", value=True, key=f"chk_{idx}"):
                            m_data_copy = m_data.copy()
                            m_data_copy['fig'] = fig_m 
                            selected_methods.append(m_data_copy)

                st.markdown("#### Methods Summary Table")
                st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)
            else:
                selected_methods = [{"name": m['name'], "df": m['df'], "fig": fig3} for m in st.session_state.generated_methods]

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
                            bruker_str = to_bruker_format(m['df'], b['im_min'], b['im_max'])
                            zf.writestr(f"{m['name']}.txt", bruker_str)
                            
                            try:
                                img_bytes = m['fig'].to_image(format="png", width=800, height=600)
                                zf.writestr(f"{m['name']}_Plot.png", img_bytes)
                            except Exception:
                                image_export_failed = True
                    
                    if image_export_failed:
                        st.warning("⚠️ Text files exported successfully, but image export failed. Please ensure the `kaleido` package is installed in your environment to generate PNG plots.")
                    
                    st.download_button("📥 Click Here to Download Final ZIP", data=zip_buffer.getvalue(), file_name="Iterated_Diagonal_Methods.zip", mime="application/zip")

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import io
import zipfile

# ==========================================
# 1. Streamlit Page Configuration
# ==========================================
st.set_page_config(page_title="Slice DIA-PASEF Optimization", layout="wide")
st.title("Slice DIA-PASEF Method Development")
st.markdown("Generate fixed, vertical rectangular quadrupole isolation slices for DIA-PASEF.")

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

def hex_to_rgba(hex_color, opacity):
    """Convert Hex + Opacity into RGBA for Plotly fills."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

def to_bruker_slice_format(method_df, im_min, im_max):
    """
    Converts the dataframe to the strict Bruker txt format.
    For Slice-PASEF (vertical rectangles), pos.1 m/z and pos.2 m/z are identical.
    """
    header = "type, mobility pos.1 [1/K0], mass pos.1 start [m/z], mass pos.1 end [m/z], mobility pos.2 [1/K0], mass pos.2 start [m/z]\n"
    init_row = "ms,-,-,-,-,-\n"
    lines = [header, init_row]
    for _, row in method_df.iterrows():
        # mz_pos2_start is identical to mz_start for a perfect vertical slice
        line = f"diagonal,{im_min:.2f},{row['mz_start']:.1f},{row['mz_end']:.1f},{im_max:.2f},{row['mz_start']:.1f}\n"
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

# ==========================================
# 2. APP LAYOUT & SIDEBAR
# ==========================================
st.sidebar.header("1. Upload Library")
uploaded_file = st.sidebar.file_uploader("Upload .tsv file", type=['tsv', 'txt', 'csv'], on_change=reset_app)

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        mz_vals, im_vals, plot_mz, plot_im, density = load_and_process_data(uploaded_file)

    st.sidebar.header("2. Plot Appearance & Axis Limits")
    marker_size = st.sidebar.slider("Precursor Marker Size", min_value=1, max_value=15, value=8, step=1)
    
    st.sidebar.subheader("Axis Limits")
    c_x1, c_x2 = st.sidebar.columns(2)
    x_axis_min = c_x1.number_input("X Min (m/z)", value=100.0)
    x_axis_max = c_x2.number_input("X Max (m/z)", value=1800.0)
    c_y1, c_y2 = st.sidebar.columns(2)
    y_axis_min = c_y1.number_input("Y Min (1/K0)", value=0.200, format="%.3f")
    y_axis_max = c_y2.number_input("Y Max (1/K0)", value=2.000, format="%.3f")

    st.sidebar.subheader("Bin Appearance")
    bin_color_hex = st.sidebar.color_picker("Bin Edge & Fill Color", value="#2E8B57") # Changed default to SeaGreen for differentiation
    bin_opacity = st.sidebar.slider("Bin Fill Opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    bin_fill_rgba = hex_to_rgba(bin_color_hex, bin_opacity)

    # ==========================================
    # PHASE 1: SET RECTANGULAR BOUNDARIES
    # ==========================================
    st.markdown("### Step 1: Set Slice Boundaries (The Bounding Box)")
    p1_disabled = st.session_state.phase > 1
    
    c1, c2, c3, c4 = st.columns(4)
    im_min = c1.number_input("1/K0 Min (Bottom Edge)", value=0.60, step=0.05, format="%.2f", disabled=p1_disabled)
    im_max = c2.number_input("1/K0 Max (Top Edge)", value=1.50, step=0.05, format="%.2f", disabled=p1_disabled)
    mz_min = c3.number_input("m/z Min (Left Edge)", value=400.0, step=10.0, disabled=p1_disabled)
    mz_max = c4.number_input("m/z Max (Right Edge)", value=1000.0, step=10.0, disabled=p1_disabled)

    # Calculate Rectangular Box Corners
    box_x = [mz_min, mz_max, mz_max, mz_min, mz_min]
    box_y = [im_min, im_min, im_max, im_max, im_min]

    fig1 = go.Figure()
    fig1.add_trace(go.Scattergl(
        x=plot_mz, y=plot_im, mode='markers', 
        marker=dict(color=density, colorscale='Jet', opacity=0.6, size=marker_size), 
        name='Precursors', hovertemplate='<b>m/z:</b> %{x:.2f}<br><b>1/K0:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Draw the Bounding Rectangle
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
            st.session_state.b_state = {'im_min': im_min, 'im_max': im_max, 'mz_min': mz_min, 'mz_max': mz_max}
            st.session_state.phase = 2
            st.rerun()
    else:
        if c_btn1.button("🔓 Unlock & Edit Boundaries"):
            st.session_state.phase = 1
            st.rerun()

    # ==========================================
    # PHASE 2: SLICING STRATEGY
    # ==========================================
    if st.session_state.phase >= 2:
        st.markdown("---")
        st.markdown("### Step 2: Slicing Strategy")
        p2_disabled = st.session_state.phase > 2
        
        c1, c2 = st.columns([1, 3])
        num_windows = c1.slider("Number of Vertical Slices (MS/MS Scans)", min_value=1, max_value=30, value=8, step=1, disabled=p2_disabled)
        
        if p2_disabled:
            num_windows = st.session_state.p_state['num_windows']

        c_btn1, c_btn2 = st.columns([2, 4])
        if st.session_state.phase == 2:
            if c_btn1.button("🚀 Generate Fixed Slices", type="primary"):
                st.session_state.p_state = {'num_windows': num_windows}
                st.session_state.phase = 3
                st.rerun()
        else:
            if c_btn1.button("🔓 Unlock & Edit Strategy"):
                st.session_state.phase = 2
                st.rerun()

    # ==========================================
    # PHASE 3: METHOD GENERATION & EXPORT
    # ==========================================
    if st.session_state.phase >= 3:
        st.markdown("---")
        st.markdown("### Step 3: Base Method Generation & Export")
        
        b = st.session_state.b_state
        p = st.session_state.p_state
        
        # Mathematical Logic for Fixed Slicing
        step_size = (b['mz_max'] - b['mz_min']) / p['num_windows']
        
        windows = []
        widths = []
        rects = []
        
        for i in range(p['num_windows']):
            start = b['mz_min'] + (i * step_size)
            end = start + step_size
            
            windows.append({
                'Scan': i + 1, 
                'mz_start': start, 
                'mz_end': end
            })
            widths.append(f"{step_size:.1f}")
            rects.append((start, end))

        method_df = pd.DataFrame(windows)
        summary = {
            "Method": "Fixed_Slice_PASEF",
            "Scans": p['num_windows'],
            "Mass Range (m/z)": f"{b['mz_min']:.1f} - {b['mz_max']:.1f}",
            "Isolation Widths (Th)": ", ".join(widths)
        }
        
        # Visualization
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(
            x=plot_mz, y=plot_im, mode='markers', 
            marker=dict(color=density, colorscale='Jet', opacity=0.5, size=marker_size), 
            hoverinfo='skip', showlegend=False
        ))
        
        for i, (m1_start, m1_end) in enumerate(rects):
            x_pts = [m1_start, m1_end, m1_end, m1_start, m1_start]
            y_pts = [b['im_min'], b['im_min'], b['im_max'], b['im_max'], b['im_min']]
            
            # Count precise precursors in this purely vertical slice
            bin_mask = (im_vals >= b['im_min']) & (im_vals <= b['im_max']) & (mz_vals >= m1_start) & (mz_vals <= m1_end)
            prec_count = np.sum(bin_mask)
            
            hover_text = (f"<b>Scan {i+1}</b><br>"
                          f"Precursors: {prec_count}<br>"
                          f"m/z Limits: {m1_start:.2f} - {m1_end:.2f}<br>"
                          f"Width: {(m1_end - m1_start):.2f} Th")
            
            fig3.add_trace(go.Scatter(
                x=x_pts, y=y_pts, 
                mode='lines', line=dict(color=bin_color_hex, width=1), 
                fill='toself', fillcolor=bin_fill_rgba, 
                text=hover_text, hovertemplate="%{text}<extra></extra>", hoveron='fills', name=f"Scan {i+1}", showlegend=False
            ))
        
        fig3.update_layout(xaxis=dict(range=[x_axis_min, x_axis_max]), yaxis=dict(range=[y_axis_min, y_axis_max]), height=500, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Method Summary")
        summary_df = pd.DataFrame([summary])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Export Block
        st.divider()
        c_d1, c_d2 = st.columns(2)
        
        csv_table = summary_df.to_csv(index=False)
        c_d1.download_button("📊 Download Summary Table (.csv)", data=csv_table, file_name="Slice_Method_Summary.csv", mime="text/csv", use_container_width=True)

        if c_d2.button("🗜️ Prepare Method & Plot for Download", use_container_width=True):
            with st.spinner("Compiling ZIP file..."):
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Write Bruker text file
                    bruker_str = to_bruker_slice_format(method_df, b['im_min'], b['im_max'])
                    zf.writestr("Fixed_Slice_Method.txt", bruker_str)
                    
                    # Write HTML Plot
                    html_bytes = fig3.to_html(include_plotlyjs='cdn').encode('utf-8')
                    zf.writestr("Fixed_Slice_Plot.html", html_bytes)
                
                st.download_button("📥 Click Here to Download Final ZIP", data=zip_buffer.getvalue(), file_name="Slice_PASEF_Method.zip", mime="application/zip")

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

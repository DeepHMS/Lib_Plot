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
st.set_page_config(page_title="py_SLICER PASEF Optimization", layout="wide")
st.markdown("<h2 style='text-align: center; margin-bottom: 40px;'>py_SLICER</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate fixed, stepped rectangular quadrupole isolation slices for DIA-PASEF.</p>", unsafe_allow_html=True)

# Helper function to convert Hex + Opacity into RGBA for Plotly fills
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

def to_slice_format(df_rows):
    """Converts the generated dictionary rows to the strict text format."""
    lines = ["#MS Type,Cycle Id,Start IM [1/K0],End IM [1/K0],Start Mass [m/z],End Mass [m/z],CE [eV]"]
    lines.append("MS1,0,-,-,-,-,-")
    for row in df_rows:
        lines.append(f"{row['MS Type']},{row['Cycle Id']},{row['Start IM [1/K0]']},{row['End IM [1/K0]']},"
                     f"{row['Start Mass [m/z]']},{row['End Mass [m/z]']},{row['CE [eV]']}")
    return "\n".join(lines)

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

    # Downsample for plotting and KDE if dataset is massive to prevent lag
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
uploaded_file = st.sidebar.file_uploader("Upload .tsv file", type=['tsv', 'txt', 'csv'])

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        mz_vals, im_vals, plot_mz, plot_im, density = load_and_process_data(uploaded_file)

    st.sidebar.header("2. Plot Appearance & Axis Limits")
    marker_size = st.sidebar.slider("Precursor Marker Size", min_value=1, max_value=15, value=8, step=1)
    
    c_x1, c_x2 = st.sidebar.columns(2)
    x_axis_min = c_x1.number_input("X Min (m/z)", value=100.0)
    x_axis_max = c_x2.number_input("X Max (m/z)", value=1800.0)
    c_y1, c_y2 = st.sidebar.columns(2)
    y_axis_min = c_y1.number_input("Y Min (1/K0)", value=0.200, format="%.3f")
    y_axis_max = c_y2.number_input("Y Max (1/K0)", value=2.000, format="%.3f")

    bin_color_hex = st.sidebar.color_picker("Bin Edge & Fill Color", value="#FF4B4B")
    bin_opacity = st.sidebar.slider("Bin Fill Opacity", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    bin_fill_rgba = hex_to_rgba(bin_color_hex, bin_opacity)

    st.sidebar.markdown("---")
    st.sidebar.header("3. py_SLICER Parameters")
    
    # Exact mapping of user's requested widgets
    cycles = st.sidebar.slider("Cycles", min_value=1, max_value=10, value=5, step=1)
    im_start = st.sidebar.slider("IM Start [1/K0]", min_value=0.60, max_value=1.60, value=0.60, step=0.05)
    im_end = st.sidebar.slider("IM End [1/K0]", min_value=0.60, max_value=1.60, value=1.40, step=0.05)
    im_step_vertical = st.sidebar.number_input("IM Step Vertical", value=0.05, step=0.01)
    im_step_horizontal = st.sidebar.number_input("IM Step Horizontal", value=0.05, step=0.01)
    
    mass_start = st.sidebar.slider("Mass Start [m/z]", min_value=100, max_value=1700, value=250, step=10)
    mass_end = st.sidebar.slider("Mass End [m/z]", min_value=100, max_value=1700, value=1320, step=10)
    mass_step_vertical = st.sidebar.number_input("Mass Step Vertical", value=55.0, step=1.0)
    mass_step_horizontal = st.sidebar.number_input("Mass Step Horizontal", value=300.0, step=10.0)

    # ==========================================
    # 3. METHOD GENERATION LOGIC
    # ==========================================
    df_rows = []
    plot_rects = []
    
    # Strict implementation of the provided stepwise algorithm
    num_steps = int(round((im_end - im_start) / im_step_vertical, 4)) + 1

    for cycle in range(1, cycles + 1):
        current_im = im_start
        current_mass = mass_start
        
        for _ in range(num_steps):
            if current_im >= im_end:
                break
                
            im_end_current = min(im_end, current_im + im_step_vertical)
            mass_end_current = min(mass_end, current_mass + mass_step_horizontal)
            
            # Export payload
            df_rows.append({
                'MS Type': 'PASEF',
                'Cycle Id': cycle,
                'Start IM [1/K0]': f"{current_im:.4f}",
                'End IM [1/K0]': f"{im_end_current:.4f}",
                'Start Mass [m/z]': f"{current_mass:.2f}",
                'End Mass [m/z]': f"{mass_end_current:.2f}",
                'CE [eV]': '-'
            })
            
            # Record coordinates for the Plotly visualizer (only need cycle 1 to avoid overdrawing identical shapes)
            if cycle == 1:
                plot_rects.append((current_mass, mass_end_current, current_im, im_end_current))
                
            current_im = im_end_current
            current_mass += mass_step_vertical
            if current_mass > mass_end:
                current_mass = mass_end

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    fig = go.Figure()
    
    # Base precursor cloud
    fig.add_trace(go.Scattergl(
        x=plot_mz, y=plot_im, mode='markers', 
        marker=dict(color=density, colorscale='Jet', opacity=0.5, size=marker_size), 
        hoverinfo='skip', showlegend=False
    ))
    
    # Draw Stepped Slices
    for i, (m1_start, m1_end, im1_start, im1_end) in enumerate(plot_rects):
        x_pts = [m1_start, m1_end, m1_end, m1_start, m1_start]
        y_pts = [im1_start, im1_start, im1_end, im1_end, im1_start]
        
        # Count precise library precursors captured in this specific rectangular slice
        bin_mask = (im_vals >= im1_start) & (im_vals <= im1_end) & (mz_vals >= m1_start) & (mz_vals <= m1_end)
        prec_count = np.sum(bin_mask)
        
        hover_text = (f"<b>Slice {i+1}</b><br>"
                      f"Precursors: {prec_count}<br>"
                      f"m/z Limits: {m1_start:.2f} - {m1_end:.2f}<br>"
                      f"IM Limits: {im1_start:.4f} - {im1_end:.4f}")
        
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts, 
            mode='lines', line=dict(color=bin_color_hex, width=1.5), 
            fill='toself', fillcolor=bin_fill_rgba, 
            text=hover_text, hovertemplate="%{text}<extra></extra>", hoveron='fills', name=f"Slice {i+1}", showlegend=False
        ))
    
    # Apply standard view constraints
    fig.update_layout(
        xaxis=dict(range=[x_axis_min, x_axis_max], title="m/z"), 
        yaxis=dict(range=[y_axis_min, y_axis_max], title="1/K0"), 
        height=650, margin=dict(t=30, b=30),
        annotations=[
            dict(x=0.02, y=0.98, xref='paper', yref='paper', text=f"Number of Cycles = {cycles}", showarrow=False, align='left', bgcolor="rgba(255,255,255,0.7)"),
            dict(x=0.98, y=0.02, xref='paper', yref='paper', text=f"Total Entries = {1 + len(df_rows)} (1 MS1 + {len(df_rows)} MS2)", showarrow=False, align='right', bgcolor="rgba(255,255,255,0.7)")
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 5. EXPORT & DOWNLOAD
    # ==========================================
    st.divider()
    c_d1, c_d2 = st.columns([1, 1])
    
    # Preview generated data
    method_df = pd.DataFrame(df_rows)
    with st.expander("Preview Generated Method Table"):
        st.dataframe(method_df, use_container_width=True, hide_index=True)

    if c_d1.button("🗜️ Download Method & Plot (ZIP)", use_container_width=True, type="primary"):
        with st.spinner("Compiling ZIP file..."):
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Write formatted text file
                bruker_str = to_slice_format(df_rows)
                zf.writestr("pasef_method_output.txt", bruker_str)
                
                # Write HTML Plot
                html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
                zf.writestr("slice-pasef_plot.html", html_bytes)
            
            st.download_button("📥 Click Here to Download Final ZIP", data=zip_buffer.getvalue(), file_name="py_SLICER_Method.zip", mime="application/zip")

else:
    st.info("👈 Please upload a proteomics library (.tsv) file in the sidebar to begin.")

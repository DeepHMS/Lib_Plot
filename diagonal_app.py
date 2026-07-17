import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# ==========================================
# UI & Page Configuration
# ==========================================
st.set_page_config(page_title="Diagonal DIA-PASEF Optimizer", layout="wide")
st.title("Diagonal DIA-PASEF Method Development")
st.markdown("Optimize quadrupole isolation windows based on pure precursor spatial density.")

# ==========================================
# Helper Functions
# ==========================================
def generate_mock_data():
    """Generates a realistic peptide cloud if no library is uploaded."""
    np.random.seed(42)
    n_points = 20000
    # Create a diagonal-leaning cloud
    mz = np.random.normal(loc=700, scale=250, size=n_points)
    mz = np.clip(mz, 400, 1200)
    # 1/K0 correlates with m/z
    k0 = 0.5 + (mz * 0.0006) + np.random.normal(0, 0.08, n_points)
    return pd.DataFrame({'m/z': mz, '1/K0': k0})

def to_bruker_format(method_df):
    """Converts the dataframe to the strict Bruker txt format."""
    # Enforce formatting based on reference files[cite: 1, 2]
    header = "type, mobility pos.1 [1/K0], mass pos.1 start [m/z], mass pos.1 end [m/z], mobility pos.2 [1/K0], mass pos.2 start [m/z]\n"
    init_row = "ms,-,-,-,-,-\n"
    
    lines = [header, init_row]
    for _, row in method_df.iterrows():
        line = f"diagonal,0.60,{row['mz_start']:.1f},{row['mz_end']:.1f},1.50,{row['mz_pos2_start']:.1f}\n"
        lines.append(line)
    return "".join(lines)

# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Library (CSV)", type=["csv"])

st.sidebar.header("2. Boundary Limits")
mz_min = st.sidebar.number_input("m/z Min", value=400.0, step=10.0)
mz_max = st.sidebar.number_input("m/z Max", value=1200.0, step=10.0)

st.sidebar.header("3. Diagonal Parameters")
st.sidebar.markdown("Define the boundaries at 1/K0 = 0.60")
top_line_mz = st.sidebar.number_input("Top Line (m/z start)", value=400.0, step=10.0)
bottom_line_mz = st.sidebar.number_input("Bottom Line (m/z end)", value=900.0, step=10.0)
slope_offset = st.sidebar.slider("Angle / Slope (Δ m/z to 1.50 1/K0)", min_value=500, max_value=1500, value=1215, step=5)

st.sidebar.header("4. Method Generation")
method_type = st.sidebar.radio("Optimization Strategy", ["Fixed", "Variable (Density-Based)"])
num_scans = st.sidebar.number_input("Number of MS/MS Scans", min_value=1, max_value=20, value=8, step=1)

num_iterations = 1
if method_type == "Variable (Density-Based)":
    num_iterations = st.sidebar.number_input("Number of Iterations (High-Throughput)", min_value=1, max_value=5, value=1, step=1)

# ==========================================
# Main Pipeline Logic
# ==========================================
# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Assume columns are 'm/z' and '1/K0' for this template
else:
    df = generate_mock_data()
    st.info("No file uploaded. Using simulated peptide library.")

# Filter by standard limits
df_filtered = df[(df['m/z'] >= mz_min) & (df['m/z'] <= mz_max)].copy()

# Filter by Diagonal Boundaries (Math implementation of the visual corridor)
# Calculate expected 1/K0 for a given m/z based on user slope
# For simplicity in this demo, we just use the raw filtered cloud for generation.
# In production, a strict point-in-polygon or linear constraint mask goes here.

# ==========================================
# Method Calculations
# ==========================================
generated_methods = []
summary_data = []

if method_type == "Fixed":
    # Step 3: Fixed Arithmetic Slicing[cite: 2]
    step_size = (mz_max - mz_min) / num_scans
    
    windows = []
    widths = []
    for i in range(num_scans):
        start = mz_min + (i * step_size)
        end = start + step_size
        pos2_start = start + slope_offset
        
        windows.append({
            'Scan': i + 1,
            'mz_start': start,
            'mz_end': end,
            'mz_pos2_start': pos2_start
        })
        widths.append(f"{step_size:.1f}")
        
    method_df = pd.DataFrame(windows)
    generated_methods.append(("Fixed_Method.txt", method_df))
    summary_data.append({"Method": "Fixed", "Isolation Widths (Th)": ", ".join(widths)})

else:
    # Step 3 & 4: Variable Precursor Density Slicing[cite: 1]
    df_sorted = df_filtered.sort_values(by='m/z').reset_index(drop=True)
    total_precursors = len(df_sorted)
    precursors_per_scan = total_precursors / num_scans
    
    for iteration in range(num_iterations):
        # Calculate offset for High-Throughput Iteration
        # Shift targets by 5% of the bucket size per iteration
        shift = (iteration * 0.05 * precursors_per_scan)
        
        target_indices = [int(shift + (i * precursors_per_scan)) for i in range(num_scans + 1)]
        # Cap indices to dataframe length to avoid out-of-bounds
        target_indices = [min(idx, total_precursors - 1) for idx in target_indices]
        
        mz_boundaries = [df_sorted.iloc[idx]['m/z'] for idx in target_indices]
        
        # Override first and last to ensure full limit coverage
        mz_boundaries[0] = mz_min
        mz_boundaries[-1] = mz_max
        
        windows = []
        widths = []
        for i in range(num_scans):
            start = mz_boundaries[i]
            end = mz_boundaries[i+1]
            pos2_start = start + slope_offset
            
            windows.append({
                'Scan': i + 1,
                'mz_start': start,
                'mz_end': end,
                'mz_pos2_start': pos2_start
            })
            widths.append(f"{(end - start):.2f}")
            
        method_df = pd.DataFrame(windows)
        method_name = f"Variable_Method_Iter_{iteration+1}.txt"
        generated_methods.append((method_name, method_df))
        summary_data.append({"Method": method_name, "Isolation Widths (Th)": ", ".join(widths)})

# ==========================================
# Visualization & Output
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Peptide Cloud Density & Isolation Windows")
    
    # Plotly 2D Histogram/Scatter
    fig = px.density_heatmap(df_filtered, x='m/z', y='1/K0', nbinsx=100, nbinsy=100, 
                             color_continuous_scale="Viridis", 
                             title="Precursor Spatial Density")
    
    # Draw the windows of the FIRST generated method to the plot
    if generated_methods:
        first_method = generated_methods[0][1]
        for _, row in first_method.iterrows():
            # Add vertical isolation lines mapped to the 0.6 to 1.5 slope geometry
            fig.add_shape(type="line",
                x0=row['mz_start'], y0=0.60, x1=row['mz_pos2_start'], y1=1.50,
                line=dict(color="red", width=1, dash="dash")
            )
            # End boundary
            fig.add_shape(type="line",
                x0=row['mz_end'], y0=0.60, x1=row['mz_pos2_start'] + (row['mz_end']-row['mz_start']), y1=1.50,
                line=dict(color="red", width=1, dash="dash")
            )

    fig.update_layout(yaxis_range=[0.6, 1.5], xaxis_range=[mz_min, mz_max])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Generated Summary")
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True)
    
    st.subheader("Download Methods")
    st.markdown("Files are formatted strictly for Bruker timsControl import.")
    
    for filename, m_df in generated_methods:
        bruker_string = to_bruker_format(m_df)
        st.download_button(
            label=f"Download {filename}",
            data=bruker_string,
            file_name=filename,
            mime="text/csv"
        )

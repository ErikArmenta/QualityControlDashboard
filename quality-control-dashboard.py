import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical for Streamlit Cloud
import matplotlib.pyplot as plt
import math
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set page configuration
st.set_page_config(
    page_title="Quality Control Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2e86ab;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    .success { color: #28a745; font-weight: bold; }
    .warning { color: #ffc107; font-weight: bold; }
    .danger { color: #dc3545; font-weight: bold; }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample manufacturing data
def generate_manufacturing_data():
    np.random.seed(42)
    n = 300  # Reduced for faster loading
    
    data = {
        'part_id': range(1, n+1),
        'length': np.random.normal(10.0, 0.15, n),
        'diameter': np.random.normal(5.0, 0.08, n),
        'weight': np.random.normal(100.0, 3.0, n),
        'defect': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'operator': np.random.choice(['Op1', 'Op2', 'Op3'], n),
        'machine': np.random.choice(['M1', 'M2'], n),
        'shift': np.random.choice(['A', 'B'], n),
        'material_batch': np.random.choice(['B1', 'B2', 'B3'], n)
    }
    
    return pd.DataFrame(data)

# Quality metrics calculation functions
def calculate_cp(upper_spec, lower_spec, std_dev):
    """Calculate Process Capability Index (Cp)"""
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Capability Index (Cpk)"""
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    """Calculate Process Performance Index (Pp)"""
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Performance Index (Ppk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Machine Capability Index (Cmk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    """Calculate Defects Per Million Opportunities (DPMO)"""
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    """Calculate Sigma Level from DPMO"""
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        # Simple approximation without scipy
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

# Sampling recommendation functions
def recommend_sampling_method(data_type, data_nature, application):
    """Recommend sampling method based on data characteristics"""
    recommendations = []
    
    if data_type == "Variable":
        recommendations.append("📏 Variables Sampling: Use measurement data")
        recommendations.append("✅ Recommended: SPC control charts, Acceptance sampling by variables")
    elif data_type == "Attribute":
        recommendations.append("🔢 Attributes Sampling: Use count data (pass/fail)")
        recommendations.append("✅ Recommended: Acceptance sampling by attributes, p-charts, np-charts")
    
    if data_nature == "Continuous":
        recommendations.append("⏰ Data is continuous - Consider time-based sampling")
    elif data_nature == "Discrete":
        recommendations.append("📦 Data is discrete - Consider lot-based sampling")
    
    if "Normal" in data_nature:
        recommendations.append("📊 Normal distribution - Parametric methods can be used")
    elif "Non-normal" in data_nature:
        recommendations.append("📈 Non-normal distribution - Non-parametric methods recommended")
    
    if application == "Process Control":
        recommendations.append("🎯 For process control: Use SPC control charts, Regular sampling intervals")
    elif application == "Lot Acceptance":
        recommendations.append("📋 For lot acceptance: Use acceptance sampling plans")
    elif application == "Capability Analysis":
        recommendations.append("📐 For capability analysis: Ensure random sampling, Adequate sample size")
    elif application == "Defect Analysis":
        recommendations.append("🔍 For defect analysis: Use stratified sampling by defect type")
    
    return recommendations

# Main application
def main():
    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data()
    
    # Header
    st.markdown('<div class="main-header">🏭 Quality Control Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose Module", 
        ["🏠 Home", "📁 Data Import", "📊 Data Overview", "📐 Quality Metrics", 
         "🎯 Sampling Recommender", "📈 SPC Analysis", "🔍 Defect Analysis", "🔬 Advanced Analytics"])
    
    # Home Page
    if app_mode == "🏠 Home":
        st.markdown('<div class="section-header">🚀 Welcome to Quality Control Dashboard</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Free Web-Based Quality Control Analysis Tool</h3>
        <p>This application provides comprehensive quality control analysis capabilities for manufacturing and process industries.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📐 Quality Metrics")
            st.write("• Cp, Cpk, Pp, Ppk calculations")
            st.write("• DPMO and Sigma Level")
            st.write("• Process capability analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📈 SPC Charts")
            st.write("• X-bar and R control charts")
            st.write("• Real-time process monitoring")
            st.write("• Control limit calculations")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("🔍 Defect Analysis")
            st.write("• Pareto analysis")
            st.write("• Defect rate calculations")
            st.write("• Categorical analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📖 How to Use")
        st.write("1. **Start with Data Import** - Upload your data or use sample data")
        st.write("2. **Explore Data Overview** - Understand your dataset")
        st.write("3. **Calculate Quality Metrics** - Analyze process capability")
        st.write("4. **Generate SPC Charts** - Monitor process stability")
        st.write("5. **Analyze Defects** - Identify improvement opportunities")
        
        st.markdown("---")
        st.subheader("🌐 Share This App")
        st.write("This app is completely free to use and share! Send the link to your colleagues:")
        st.code("https://your-username-quality-control-dashboard.streamlit.app", language="text")
        
        # Dependency status
        st.markdown("---")
        st.subheader("🔧 System Status")
        col1, col2 = st.columns(2)
        with col1:
            if HAS_SCIPY:
                st.success("✅ SciPy: Available")
            else:
                st.warning("⚠️ SciPy: Limited functionality")
            
            if HAS_SEABORN:
                st.success("✅ Seaborn: Available")
            else:
                st.info("ℹ️ Seaborn: Using matplotlib")
        
        with col2:
            st.success("✅ Pandas: Available")
            st.success("✅ NumPy: Available")
            st.success("✅ Matplotlib: Available")
    
    # Data Import Module
    elif app_mode == "📁 Data Import":
        st.markdown('<div class="section-header">📁 Data Import</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Your Data")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                    
                    # Show preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(8), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.subheader("🌐 Google Sheets Import")
            st.info("Coming soon - currently supports CSV uploads")
            
            st.subheader("🔬 Sample Data")
            if st.button("🎲 Use Sample Manufacturing Data"):
                st.session_state.df = generate_manufacturing_data()
                st.success("✅ Sample data loaded successfully!")
                st.dataframe(st.session_state.df.head(8), use_container_width=True)
        
        # Data quality check
        if 'df' in st.session_state:
            df = st.session_state.df
            st.markdown("---")
            st.subheader("📊 Data Quality Check")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
    
    # Data Overview Module
    elif app_mode == "📊 Data Overview":
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            numeric_cols = len(df.select_dtypes(include=np.number).columns)
            st.metric("🔢 Numeric Columns", numeric_cols)
        with col3:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Categorical Columns", categorical_cols)
        with col4:
            if 'defect' in df.columns:
                defect_rate = df['defect'].mean()
                st.metric("⚠️ Defect Rate", f"{defect_rate:.2%}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Column information
        st.subheader("🗂️ Column Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔢 Numeric Columns:**")
            numeric_cols = list(df.select_dtypes(include=np.number).columns)
            for col in numeric_cols:
                st.write(f"• {col}")
        
        with col2:
            st.write("**📝 Categorical Columns:**")
            cat_cols = list(df.select_dtypes(include=['object']).columns)
            for col in cat_cols:
                st.write(f"• {col}")
        
        # Visualizations
        st.subheader("📊 Data Distributions")
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            if selected_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                data = df[selected_col].dropna()
                
                ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution of {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1 Std Dev')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                ax.legend()
                
                st.pyplot(fig)
    
    # Quality Metrics Module
    elif app_mode == "📐 Quality Metrics":
        st.markdown('<div class="section-header">📐 Quality Metrics Calculator</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Select Variable", numeric_cols)
            data = df[variable].dropna()
            
            if len(data) > 0:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sample Size", len(data))
                st.metric("Mean", f"{np.mean(data):.4f}")
                st.metric("Standard Deviation", f"{np.std(data, ddof=1):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            lsl = st.number_input("Lower Specification Limit (LSL)", value=float(data_mean - 3*np.std(data)))
            usl = st.number_input("Upper Specification Limit (USL)", value=float(data_mean + 3*np.std(data)))
            subgroup_size = st.slider("Subgroup Size for Cmk", min_value=2, max_value=10, value=5)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Calculate Quality Metrics", type="primary"):
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            
            # Calculate metrics
            cp = calculate_cp(usl, lsl, std_val)
            cpk = calculate_cpk(usl, lsl, mean_val, std_val)
            pp = calculate_pp(usl, lsl, std_val)
            ppk = calculate_ppk(usl, lsl, mean_val, std_val)
            
            # Calculate Cmk
            subgroup_means = []
            subgroup_stds = []
            for i in range(0, min(len(data), subgroup_size*5), subgroup_size):
                subgroup = data[i:i+subgroup_size]
                subgroup_means.append(np.mean(subgroup))
                subgroup_stds.append(np.std(subgroup, ddof=1))
            
            short_term_std = np.mean(subgroup_stds) if subgroup_stds else std_val
            cmk = calculate_cmk(usl, lsl, mean_val, short_term_std)
            
            # Calculate DPMO and Sigma Level
            if variable == 'defect':
                defect_count = np.sum(data)
            else:
                defect_count = np.sum((data < lsl) | (data > usl))
            
            total_units = len(data)
            dpmo = calculate_dpmo(defect_count, total_units)
            sigma_level = calculate_sigma_level(dpmo)
            
            # Display results
            st.subheader("📊 Quality Metrics Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Cp", f"{cp:.3f}")
                st.metric("Cpk", f"{cpk:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Pp", f"{pp:.3f}")
                st.metric("Ppk", f"{ppk:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Cmk", f"{cmk:.3f}")
                st.metric("DPMO", f"{dpmo:,.0f}")
                st.metric("Sigma Level", f"{sigma_level:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interpretation
            st.subheader("🎯 Interpretation")
            capability_metrics = [
                ("Cp", cp, 1.33, 1.0),
                ("Cpk", cpk, 1.33, 1.0),
                ("Pp", pp, 1.33, 1.0),
                ("Ppk", ppk, 1.33, 1.0),
                ("Cmk", cmk, 1.67, 1.33)
            ]
            
            for name, value, good_threshold, marginal_threshold in capability_metrics:
                if value >= good_threshold:
                    st.success(f"✅ **{name}: {value:.3f}** - Good (≥ {good_threshold})")
                elif value >= marginal_threshold:
                    st.warning(f"⚠️ **{name}: {value:.3f}** - Marginal (≥ {marginal_threshold})")
                else:
                    st.error(f"❌ **{name}: {value:.3f}** - Poor (< {marginal_threshold})")
            
            # Visualization
            st.subheader("📈 Distribution Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with specification limits
            ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax1.axvline(usl, color='green', linestyle='dashed', linewidth=2, label=f'USL: {usl}')
            ax1.axvline(lsl, color='green', linestyle='dashed', linewidth=2, label=f'LSL: {lsl}')
            
            # Add normal distribution curve
            x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
            if HAS_SCIPY:
                y = stats.norm.pdf(x, mean_val, std_val)
                ax1.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
            
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Density')
            ax1.set_title(f'Distribution of {variable}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Capability indices bar chart
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk', 'Cmk']
            values = [cp, cpk, pp, ppk, cmk]
            colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values]
            
            bars = ax2.bar(indices, values, color=colors, alpha=0.7)
            ax2.axhline(y=1.33, color='red', linestyle='--', alpha=0.7, label='Minimum Recommended (1.33)')
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Minimum Acceptable (1.0)')
            ax2.set_ylabel('Index Value')
            ax2.set_title('Process Capability Indices')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Sampling Recommender Module
    elif app_mode == "🎯 Sampling Recommender":
        st.markdown('<div class="section-header">🎯 Sampling Method Recommender</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            data_type = st.radio("**Data Type**", ["Variable", "Attribute"])
            data_nature = st.selectbox("**Data Nature**", 
                ["Continuous", "Discrete", "Continuous - Normal", "Continuous - Non-normal"])
            application = st.selectbox("**Application**", 
                ["Process Control", "Lot Acceptance", "Capability Analysis", "Defect Analysis"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            sample_size = st.slider("**Sample Size**", min_value=5, max_value=200, value=30)
            population_size = st.slider("**Population Size**", min_value=100, max_value=10000, value=1000)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Get Sampling Recommendations", type="primary"):
            recommendations = recommend_sampling_method(data_type, data_nature, application)
            
            st.subheader("📋 Sampling Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            st.subheader("📚 General Guidelines")
            guidelines = [
                "For variables data: Consider SPC control charts (X-bar R, X-bar S)",
                "For attributes data: Consider p-charts, np-charts, c-charts, u-charts",
                "For lot acceptance: Use ANSI/ASQ Z1.4 (MIL-STD-105E) for attributes",
                "For variables acceptance: Use ANSI/ASQ Z1.9 (MIL-STD-414)",
                "For process capability: Ensure random sampling, minimum 25-30 subgroups",
                "For machine capability (Cmk): Use 50 consecutive parts, 1 machine, 1 operator"
            ]
            
            for guideline in guidelines:
                st.write(f"• {guideline}")
            
            st.subheader("📊 Sample Size Justification")
            sampling_fraction = sample_size / population_size
            
            if sample_size < 30:
                st.warning(f"⚠️ Sample size of {sample_size} may be too small for reliable estimates")
            elif sample_size < 100:
                st.info(f"ℹ️ Sample size of {sample_size} is adequate for most applications")
            else:
                st.success(f"✅ Sample size of {sample_size} is excellent for precise estimates")
            
            st.write(f"**Sampling Fraction:** {sampling_fraction:.2%}")
            
            if sampling_fraction > 0.1:
                st.info(f"📈 Consider finite population correction for sampling fraction > 10%")
    
    # SPC Analysis Module
    elif app_mode == "📈 SPC Analysis":
        st.markdown('<div class="section-header">📈 Statistical Process Control (SPC)</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for SPC analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Select Variable for SPC", numeric_cols)
            subgroup_size = st.slider("👥 Subgroup Size", min_value=2, max_value=10, value=5)
        
        with col2:
            data = df[variable].dropna()
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            lsl = st.number_input("📏 LSL for SPC", value=float(data_mean - 3*data_std))
            usl = st.number_input("📏 USL for SPC", value=float(data_mean + 3*data_std))
        
        if st.button("📊 Generate Control Charts", type="primary"):
            data_values = data.values
            n_subgroups = len(data_values) // subgroup_size
            
            if n_subgroups < 2:
                st.error(f"❌ Not enough data for subgrouping. Need at least {2*subgroup_size} data points.")
                return
            
            subgrouped_data = data_values[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
            
            # Calculate subgroup statistics
            subgroup_means = np.mean(subgrouped_data, axis=1)
            subgroup_ranges = np.ptp(subgrouped_data, axis=1)  # Peak-to-peak (range)
            
            # Calculate control limits
            overall_mean = np.mean(subgroup_means)
            mean_range = np.mean(subgroup_ranges)
            
            # Constants for control limits
            a2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
            d3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
            d4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
            
            # Control limits
            ucl_x = overall_mean + a2.get(subgroup_size, 0.577) * mean_range
            lcl_x = overall_mean - a2.get(subgroup_size, 0.577) * mean_range
            ucl_r = d4.get(subgroup_size, 2.114) * mean_range
            lcl_r = d3.get(subgroup_size, 0) * mean_range
            
            # Create SPC charts
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # X-bar chart
            ax1.plot(subgroup_means, 'bo-', label='Subgroup Means', markersize=4)
            ax1.axhline(overall_mean, color='green', linestyle='-', label=f'CL: {overall_mean:.3f}')
            ax1.axhline(ucl_x, color='red', linestyle='--', label=f'UCL: {ucl_x:.3f}')
            ax1.axhline(lcl_x, color='red', linestyle='--', label=f'LCL: {lcl_x:.3f}')
            ax1.axhline(usl, color='purple', linestyle='-.', label=f'USL: {usl}')
            ax1.axhline(lsl, color='purple', linestyle='-.', label=f'LSL: {lsl}')
            ax1.set_title(f'X-bar Chart for {variable}')
            ax1.set_ylabel('Mean Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # R chart
            ax2.plot(subgroup_ranges, 'go-', label='Subgroup Ranges', markersize=4)
            ax2.axhline(mean_range, color='green', linestyle='-', label=f'CL: {mean_range:.3f}')
            ax2.axhline(ucl_r, color='red', linestyle='--', label=f'UCL: {ucl_r:.3f}')
            ax2.axhline(lcl_r, color='red', linestyle='--', label=f'LCL: {lcl_r:.3f}')
            ax2.set_title(f'R Chart for {variable}')
            ax2.set_ylabel('Range')
            ax2.set_xlabel('Subgroup Number')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Process capability from control chart
            d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
            within_std = mean_range / d2.get(subgroup_size, 2.326)
            cp = calculate_cp(usl, lsl, within_std)
            cpk = calculate_cpk(usl, lsl, overall_mean, within_std)
            
            st.subheader("📐 Process Capability from Control Chart")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Within-subgroup Std Dev", f"{within_std:.4f}")
            with col2:
                st.metric("Cp", f"{cp:.3f}")
            with col3:
                st.metric("Cpk", f"{cpk:.3f}")
            
            # Check for out-of-control points
            out_of_control_x = np.sum((subgroup_means > ucl_x) | (subgroup_means < lcl_x))
            out_of_control_r = np.sum(subgroup_ranges > ucl_r)
            
            if out_of_control_x == 0 and out_of_control_r == 0:
                st.success("✅ Process appears to be in statistical control")
            else:
                st.warning(f"⚠️ Process shows signs of being out of control")
                st.write(f"• Out-of-control points in X-bar chart: {out_of_control_x}")
                st.write(f"• Out-of-control points in R chart: {out_of_control_r}")
    
    # Defect Analysis Module
    elif app_mode == "🔍 Defect Analysis":
        st.markdown('<div class="section-header">🔍 Defect Analysis</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        if 'defect' not in df.columns:
            st.error("❌ No 'defect' column found in the dataset")
            st.info("💡 Use sample data or upload a dataset with a 'defect' column (0/1 values)")
            return
        
        cat_cols = list(df.select_dtypes(include=['object']).columns)
        
        if not cat_cols:
            st.error("❌ No categorical variables available for defect analysis")
            return
        
        category = st.selectbox("📂 Stratify Defects By", cat_cols)
        
        # Calculate defect rates
        defect_rates = df.groupby(category)['defect'].mean()
        defect_counts = df.groupby(category)['defect'].sum()
        total_counts = df.groupby(category).size()
        
        # Calculate DPMO and Sigma Level
        dpmo_values = defect_counts / total_counts * 1000000
        sigma_levels = [calculate_sigma_level(dpmo) for dpmo in dpmo_values]
        
        # Create summary table
        summary_df = pd.DataFrame({
            'Total Units': total_counts,
            'Defect Count': defect_counts,
            'Defect Rate': defect_rates,
            'DPMO': dpmo_values,
            'Sigma Level': sigma_levels
        })
        
        st.subheader("📋 Defect Analysis Summary")
        st.dataframe(summary_df.style.format({
            'Defect Rate': '{:.2%}',
            'DPMO': '{:,.0f}',
            'Sigma Level': '{:.2f}'
        }), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Defect rate bar chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars1 = ax1.bar(defect_rates.index, defect_rates.values, color='lightcoral', alpha=0.7)
            ax1.set_title(f'Defect Rate by {category}')
            ax1.set_xlabel(category)
            ax1.set_ylabel('Defect Rate')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, defect_rates.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # DPMO bar chart
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.bar(dpmo_values.index, dpmo_values.values, color='lightblue', alpha=0.7)
            ax2.set_title(f'DPMO by {category}')
            ax2.set_xlabel(category)
            ax2.set_ylabel('DPMO')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, dpmo_values.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(dpmo_values.values)*0.01,
                        f'{value:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Pareto analysis
        st.subheader("📊 Pareto Analysis")
        pareto_data = defect_counts.sort_values(ascending=False)
        cumulative_percent = pareto_data.cumsum() / pareto_data.sum() * 100
        
        fig3, ax1 = plt.subplots(figsize=(12, 6))
        
        bars = ax1.bar(range(len(pareto_data)), pareto_data.values, color='lightcoral', alpha=0.7)
        ax1.set_xlabel(category)
        ax1.set_ylabel('Defect Count', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_xticks(range(len(pareto_data)))
        ax1.set_xticklabels(pareto_data.index, rotation=45)
        
        ax2 = ax1.twinx()
        ax2.plot(range(len(pareto_data)), cumulative_percent.values, 'bo-', linewidth=2, markersize=4)
        ax2.set_ylabel('Cumulative Percentage', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.title(f'Pareto Chart of Defects by {category}')
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Pareto analysis results
        st.subheader("📈 Pareto Analysis Results")
        for i, (category_name, count) in enumerate(pareto_data.items(), 1):
            percent = count / pareto_data.sum() * 100
            cum_percent = cumulative_percent.iloc[i-1]
            st.write(f"**{i}. {category_name}:** {count} defects ({percent:.1f}%), Cumulative: {cum_percent:.1f}%")
        
        # Identify vital few
        vital_few = pareto_data[cumulative_percent <= 80]
        if len(vital_few) > 0:
            st.success(f"🎯 **Vital Few** ({(len(vital_few)/len(pareto_data)*100):.1f}% of categories cause 80% of defects):")
            for cat in vital_few.index:
                st.write(f"• **{cat}** - {pareto_data[cat]} defects ({pareto_data[cat]/pareto_data.sum()*100:.1f}%)")
    
    # Advanced Analytics Module
    elif app_mode == "🔬 Advanced Analytics":
        st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for advanced analytics")
            return
        
        analysis_type = st.selectbox("🔧 Select Analysis Type", 
            ["Normality Test", "Q-Q Plot", "Correlation Analysis"])
        
        if analysis_type == "Normality Test":
            if not HAS_SCIPY:
                st.warning("⚠️ SciPy not available. Using basic normality assessment.")
                variable = st.selectbox("📊 Select Variable", numeric_cols)
                data = df[variable].dropna()
                
                # Basic normality assessment
                skewness = (data.mean() - data.median()) / data.std() if data.std() > 0 else 0
                cv = data.std() / data.mean() if data.mean() != 0 else 0
                
                st.subheader("📊 Basic Normality Assessment")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skewness", f"{skewness:.3f}")
                with col2:
                    st.metric("Coefficient of Variation", f"{cv:.3f}")
                
                if abs(skewness) < 0.5:
                    st.success("✅ Data appears approximately symmetric")
                else:
                    st.warning("⚠️ Data shows significant skewness")
            else:
                variable = st.selectbox("📊 Select Variable", numeric_cols)
                data = df[variable].dropna()
                
                if st.button("📊 Run Normality Test"):
                    stat, p_value = stats.shapiro(data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Shapiro-Wilk Statistic", f"{stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    
                    if p_value > 0.05:
                        st.success("✅ Data appears to be normally distributed (fail to reject H0)")
                    else:
                        st.error("❌ Data does not appear to be normally distributed (reject H0)")
            
            # Visualization
            variable_viz = st.selectbox("📈 Select Variable for Visualization", numeric_cols)
            data_viz = df[variable_viz].dropna()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with normal curve
            ax1.hist(data_viz, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            if HAS_SCIPY:
                xmin, xmax = ax1.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, np.mean(data_viz), np.std(data_viz))
                ax1.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
            ax1.set_title(f'Distribution of {variable_viz}')
            ax1.set_xlabel(variable_viz)
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(data_viz)
            ax2.set_title(f'Box Plot of {variable_viz}')
            ax2.set_ylabel(variable_viz)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif analysis_type == "Q-Q Plot":
            if not HAS_SCIPY:
                st.error("❌ Q-Q Plot requires SciPy. Please install scipy or use another analysis.")
            else:
                variable = st.selectbox("📊 Select Variable for Q-Q Plot", numeric_cols)
                data = df[variable].dropna()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Q-Q plot
                stats.probplot(data, dist="norm", plot=ax1)
                ax1.set_title(f'Q-Q Plot of {variable}')
                ax1.grid(True, alpha=0.3)
                
                # Histogram
                ax2.hist(data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_title(f'Distribution of {variable}')
                ax2.set_xlabel(variable)
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_type == "Correlation Analysis":
            selected_vars = st.multiselect("📊 Select Variables for Correlation", numeric_cols, default=numeric_cols[:3])
            
            if len(selected_vars) >= 2:
                corr_matrix = df[selected_vars].corr()
                
                st.subheader("📈 Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), 
                           use_container_width=True)
                
                # Correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(selected_vars)))
                ax.set_yticks(np.arange(len(selected_vars)))
                ax.set_xticklabels(selected_vars, rotation=45, ha='right')
                ax.set_yticklabels(selected_vars)
                ax.set_title('Correlation Heatmap')
                
                # Add correlation values to heatmap
                for i in range(len(selected_vars)):
                    for j in range(len(selected_vars)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=12)
                
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Simple scatter plots
                if len(selected_vars) >= 2:
                    st.subheader("📊 Scatter Plots")
                    var1 = st.selectbox("Select X-axis variable", selected_vars, index=0)
                    var2 = st.selectbox("Select Y-axis variable", selected_vars, index=1)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[var1], df[var2], alpha=0.5)
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    ax.set_title(f'Scatter Plot: {var1} vs {var2}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add correlation coefficient
                    corr_coef = np.corrcoef(df[var1].dropna(), df[var2].dropna())[0, 1]
                    ax.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', 
                           transform=ax.transAxes, fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig)

# Run the application
if __name__ == "__main__":
    main()
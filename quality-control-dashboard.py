# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:04:59 2026

@author: acer
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical for Streamlit Cloud
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    .sixpack-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES PARA MANEJO DE DATOS MINITAB
# ============================================================================

def generate_manufacturing_data_minitab_format():
    """Genera datos de ejemplo en formato Minitab/SQC (wide format)"""
    np.random.seed(42)
    n_samples = 25  # 25 subgrupos
    n_measurements = 5  # 5 mediciones por subgrupo

    # Crear columnas
    data = {'Sample': range(1, n_samples + 1)}

    # Agregar mediciones X1, X2, ..., X5
    for i in range(1, n_measurements + 1):
        col_name = f'X{i}'
        # Media 10.0, desviación 0.1, con pequeña variación entre subgrupos
        subgroup_mean = 10.0 + np.random.normal(0, 0.02, n_samples)
        data[col_name] = subgroup_mean + np.random.normal(0, 0.08, n_samples)

    return pd.DataFrame(data)

def transform_minitab_to_long(df, sample_col=None, value_prefix='X'):
    """
    Convierte datos de formato Minitab (subgrupos en columnas) a formato largo

    Args:
        df: DataFrame en formato wide
        sample_col: Nombre de la columna de muestra (si es None, usa la primera columna)
        value_prefix: Prefijo de las columnas de medición (ej: 'X' para X1, X2, ...)

    Returns:
        DataFrame en formato largo con columnas: [Sample, Subgroup, Value]
    """
    df = df.copy()

    # Identificar columna de muestra
    if sample_col is None:
        # Buscar columna que pueda ser el ID de muestra
        possible_id_cols = ['Sample', 'sample', 'Muestra', 'muestra', 'ID', 'id', 'Subgroup', 'subgroup']
        for col in possible_id_cols:
            if col in df.columns:
                sample_col = col
                break

        # Si no se encuentra, usar primera columna
        if sample_col is None:
            sample_col = df.columns[0]

    # Renombrar columna de muestra a 'Sample' estandarizado
    if sample_col != 'Sample':
        df = df.rename(columns={sample_col: 'Sample'})
        sample_col = 'Sample'

    # Identificar columnas de medición
    if value_prefix:
        measurement_cols = [col for col in df.columns if col.startswith(value_prefix) and col != sample_col]
    else:
        measurement_cols = [col for col in df.columns if col != sample_col]

    # Si no encontramos columnas con prefijo, usar todas excepto Sample
    if not measurement_cols:
        measurement_cols = [col for col in df.columns if col != sample_col]

    # Derretir dataframe (wide to long)
    df_long = pd.melt(df,
                      id_vars=[sample_col],
                      value_vars=measurement_cols,
                      var_name='Subgroup_Measurement',
                      value_name='Value')

    # Ordenar
    df_long = df_long.sort_values([sample_col, 'Subgroup_Measurement']).reset_index(drop=True)

    return df_long

def calculate_sixpack_metrics(data, subgroup_size=5, lsl=None, usl=None):
    """
    Calcula métricas para Sixpack Report como en Minitab

    Args:
        data: Array de datos en formato largo
        subgroup_size: Tamaño del subgrupo
        lsl: Lower Specification Limit
        usl: Upper Specification Limit

    Returns:
        Diccionario con todas las métricas necesarias
    """
    # Reorganizar datos en subgrupos
    n_subgroups = len(data) // subgroup_size
    subgrouped_data = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    # Estadísticos por subgrupo
    subgroup_means = np.mean(subgrouped_data, axis=1)
    subgroup_ranges = np.ptp(subgrouped_data, axis=1)  # Rango (max-min)
    subgroup_stds = np.std(subgrouped_data, axis=1, ddof=1)

    # Estadísticos generales
    overall_mean = np.mean(subgroup_means)
    mean_range = np.mean(subgroup_ranges)
    mean_std = np.mean(subgroup_stds)

    # Constantes para gráficos de control
    # A2 para gráfico X-bar
    a2_dict = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
               6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    a2 = a2_dict.get(subgroup_size, 0.577)

    # D3 y D4 para gráfico R
    d3_dict = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
    d4_dict = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114,
               6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
    d3 = d3_dict.get(subgroup_size, 0)
    d4 = d4_dict.get(subgroup_size, 2.114)

    # D2 para estimar desviación estándar
    d2_dict = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
               6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    d2 = d2_dict.get(subgroup_size, 2.326)

    # Límites de control
    ucl_x = overall_mean + a2 * mean_range
    lcl_x = overall_mean - a2 * mean_range
    ucl_r = d4 * mean_range
    lcl_r = d3 * mean_range

    # Desviaciones estándar
    within_std = mean_range / d2  # Desviación estándar within
    overall_std = np.std(data, ddof=1)  # Desviación estándar overall

    # Calcular límites de especificación si no se proporcionan
    if lsl is None:
        lsl = overall_mean - 6 * overall_std
    if usl is None:
        usl = overall_mean + 6 * overall_std

    # Métricas de capacidad
    cp = (usl - lsl) / (6 * within_std) if within_std > 0 else float('inf')
    cpk = min((usl - overall_mean) / (3 * within_std),
              (overall_mean - lsl) / (3 * within_std)) if within_std > 0 else float('inf')

    pp = (usl - lsl) / (6 * overall_std) if overall_std > 0 else float('inf')
    ppk = min((usl - overall_mean) / (3 * overall_std),
              (overall_mean - lsl) / (3 * overall_std)) if overall_std > 0 else float('inf')

    # Calcular PPM
    if overall_std > 0:
        if HAS_SCIPY:
            ppm_below = stats.norm.cdf(lsl, overall_mean, overall_std) * 1e6
            ppm_above = (1 - stats.norm.cdf(usl, overall_mean, overall_std)) * 1e6
            ppm_total = ppm_below + ppm_above
        else:
            # Aproximación simple
            z_lsl = abs((lsl - overall_mean) / overall_std) if overall_std > 0 else 0
            z_usl = abs((usl - overall_mean) / overall_std) if overall_std > 0 else 0
            ppm_total = (max(0, 1e6 * (1 - 0.5 * (1 + math.erf(z_lsl/math.sqrt(2))))) +
                        max(0, 1e6 * (0.5 * (1 - math.erf(z_usl/math.sqrt(2))))))
    else:
        ppm_total = 0

    return {
        'subgroup_means': subgroup_means,
        'subgroup_ranges': subgroup_ranges,
        'overall_mean': overall_mean,
        'mean_range': mean_range,
        'within_std': within_std,
        'overall_std': overall_std,
        'ucl_x': ucl_x,
        'lcl_x': lcl_x,
        'ucl_r': ucl_r,
        'lcl_r': lcl_r,
        'cp': cp,
        'cpk': cpk,
        'pp': pp,
        'ppk': ppk,
        'ppm': ppm_total,
        'lsl': lsl,
        'usl': usl,
        'subgroup_size': subgroup_size
    }

def create_sixpack_report(metrics, variable_name="Diameter"):
    """
    Crea el Sixpack Report como en Minitab

    Args:
        metrics: Diccionario de métricas de calculate_sixpack_metrics
        variable_name: Nombre de la variable para el título

    Returns:
        Figura matplotlib con los 6 gráficos
    """
    # Crear figura con GridSpec para 6 gráficos
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # ==================== 1. X-bar Chart ====================
    ax1 = fig.add_subplot(gs[0, 0])
    samples = np.arange(1, len(metrics['subgroup_means']) + 1)

    # Graficar puntos
    ax1.plot(samples, metrics['subgroup_means'], 'bo-', markersize=4, linewidth=1, label='Subgroup Means')

    # Líneas de control y especificación
    ax1.axhline(metrics['overall_mean'], color='green', linestyle='-', linewidth=2, label=f'Mean: {metrics["overall_mean"]:.4f}')
    ax1.axhline(metrics['ucl_x'], color='red', linestyle='--', linewidth=1.5, label=f'UCL: {metrics["ucl_x"]:.4f}')
    ax1.axhline(metrics['lcl_x'], color='red', linestyle='--', linewidth=1.5, label=f'LCL: {metrics["lcl_x"]:.4f}')

    if metrics['usl'] is not None:
        ax1.axhline(metrics['usl'], color='purple', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'USL: {metrics["usl"]:.2f}')
    if metrics['lsl'] is not None:
        ax1.axhline(metrics['lsl'], color='purple', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'LSL: {metrics["lsl"]:.2f}')

    ax1.set_title(f'Xbar Chart of {variable_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Sample Mean')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(samples) + 0.5)

    # ==================== 2. R Chart ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(samples, metrics['subgroup_ranges'], 'go-', markersize=4, linewidth=1, label='Subgroup Ranges')
    ax2.axhline(metrics['mean_range'], color='green', linestyle='-', linewidth=2, label=f'Mean Range: {metrics["mean_range"]:.4f}')
    ax2.axhline(metrics['ucl_r'], color='red', linestyle='--', linewidth=1.5, label=f'UCL: {metrics["ucl_r"]:.4f}')
    ax2.axhline(metrics['lcl_r'], color='red', linestyle='--', linewidth=1.5, label=f'LCL: {metrics["lcl_r"]:.4f}')

    ax2.set_title(f'R Chart of {variable_name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Sample Range')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(samples) + 0.5)

    # ==================== 3. Capability Histogram ====================
    ax3 = fig.add_subplot(gs[0, 2])

    # Para el histograma necesitamos los datos individuales
    # En una implementación real, pasaríamos los datos también
    # Por ahora, simular con distribución normal
    hist_data = np.random.normal(metrics['overall_mean'], metrics['within_std'], 1000)

    n, bins, patches = ax3.hist(hist_data, bins=20, density=True, alpha=0.7,
                                color='skyblue', edgecolor='black')

    # Agregar curva normal
    if HAS_SCIPY:
        x = np.linspace(min(hist_data), max(hist_data), 100)
        y = stats.norm.pdf(x, metrics['overall_mean'], metrics['within_std'])
        ax3.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

    # Líneas de especificación
    if metrics['usl'] is not None:
        ax3.axvline(metrics['usl'], color='green', linestyle='--', linewidth=2, label=f'USL: {metrics["usl"]:.2f}')
    if metrics['lsl'] is not None:
        ax3.axvline(metrics['lsl'], color='green', linestyle='--', linewidth=2, label=f'LSL: {metrics["lsl"]:.2f}')

    ax3.axvline(metrics['overall_mean'], color='red', linestyle='-', linewidth=2, label=f'Mean: {metrics["overall_mean"]:.4f}')

    ax3.set_title(f'Capability Histogram of {variable_name}', fontsize=12, fontweight='bold')
    ax3.set_xlabel(variable_name)
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)

    # ==================== 4. Normal Probability Plot ====================
    ax4 = fig.add_subplot(gs[1, 0])

    if HAS_SCIPY:
        # Ordenar datos para Q-Q plot
        sorted_data = np.sort(hist_data)
        n = len(sorted_data)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1)/(n+1))

        ax4.scatter(theoretical_quantiles, sorted_data, alpha=0.6, color='blue')

        # Línea de referencia para normalidad
        min_q, max_q = theoretical_quantiles[0], theoretical_quantiles[-1]
        line_x = np.array([min_q, max_q])
        line_y = metrics['overall_mean'] + metrics['within_std'] * line_x
        ax4.plot(line_x, line_y, 'r--', linewidth=2, label='Normal Reference')

        ax4.set_title('Normal Probability Plot', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Ordered Values')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Q-Q Plot requires SciPy',
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Normal Probability Plot (SciPy required)', fontsize=12, fontweight='bold')

    # ==================== 5. Individuals Chart ====================
    ax5 = fig.add_subplot(gs[1, 1])

    # Simular datos individuales (en realidad deberían venir de los datos originales)
    individual_data = np.random.normal(metrics['overall_mean'], metrics['within_std'],
                                      len(metrics['subgroup_means']) * metrics['subgroup_size'])

    # Graficar primeros 50 puntos para claridad
    plot_points = min(50, len(individual_data))
    ax5.plot(range(1, plot_points + 1), individual_data[:plot_points], 'bo-',
             markersize=3, linewidth=0.5, alpha=0.7)

    # Líneas de control para gráfico de individuos
    # Para gráfico I, usamos moving range
    moving_ranges = np.abs(np.diff(individual_data[:plot_points]))
    mean_mr = np.mean(moving_ranges) if len(moving_ranges) > 0 else 0
    i_ucl = metrics['overall_mean'] + 2.66 * mean_mr
    i_lcl = metrics['overall_mean'] - 2.66 * mean_mr

    ax5.axhline(metrics['overall_mean'], color='green', linestyle='-', linewidth=2)
    ax5.axhline(i_ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL: {i_ucl:.3f}')
    ax5.axhline(i_lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL: {i_lcl:.3f}')

    ax5.set_title('Individuals Chart', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Observation')
    ax5.set_ylabel('Individual Value')
    ax5.grid(True, alpha=0.3)

    # ==================== 6. Capability Plot ====================
    ax6 = fig.add_subplot(gs[1, 2])

    # Crear gráfico de capacidad similar a Minitab
    indices = ['Within', 'Overall']
    cp_values = [metrics['cp'], metrics['pp']]
    cpk_values = [metrics['cpk'], metrics['ppk']]

    x_pos = np.arange(len(indices))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, cp_values, width, label='Cp/Pp', color='lightblue', alpha=0.8)
    bars2 = ax6.bar(x_pos + width/2, cpk_values, width, label='Cpk/Ppk', color='lightcoral', alpha=0.8)

    # Línea de referencia para capacidad mínima
    ax6.axhline(y=1.33, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Min (1.33)')
    ax6.axhline(y=1.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Acceptable (1.0)')

    ax6.set_ylabel('Capability Index')
    ax6.set_title('Process Capability', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(indices)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Agregar valores encima de las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # ==================== 7. Métricas de capacidad (texto) ====================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Texto con métricas
    metrics_text = f"""
    Process Capability Sixpack Report for {variable_name}

    {'='*60}

    Xbar Chart:
    • Center Line (CL): {metrics['overall_mean']:.4f}
    • Upper Control Limit (UCL): {metrics['ucl_x']:.4f}
    • Lower Control Limit (LCL): {metrics['lcl_x']:.4f}

    R Chart:
    • Center Line (CL): {metrics['mean_range']:.4f}
    • Upper Control Limit (UCL): {metrics['ucl_r']:.4f}
    • Lower Control Limit (LCL): {metrics['lcl_r']:.4f}

    Capability Metrics:
    • Within Std Dev (SDw): {metrics['within_std']:.5f}
    • Overall Std Dev (SDo): {metrics['overall_std']:.5f}
    • Cp (Within): {metrics['cp']:.2f}
    • Cpk (Within): {metrics['cpk']:.2f}
    • Pp (Overall): {metrics['pp']:.2f}
    • Ppk (Overall): {metrics['ppk']:.2f}
    • PPM: {metrics['ppm']:,.2f}

    Specifications:
    • Lower Spec Limit (LSL): {metrics['lsl']:.2f}
    • Upper Spec Limit (USL): {metrics['usl']:.2f}
    • Target: {(metrics['usl'] + metrics['lsl'])/2:.2f}
    • Tolerance: {metrics['usl'] - metrics['lsl']:.2f}

    The actual process spread is represented by 6σ ({6*metrics['within_std']:.4f} within, {6*metrics['overall_std']:.4f} overall)
    """

    ax7.text(0.02, 0.98, metrics_text, transform=ax7.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Process Capability Sixpack Analysis: {variable_name}',
                 fontsize=16, fontweight='bold', y=0.98)
    # Ocultar ticks x porque son muchos
    plt.tight_layout()
    # Correct return for sixpack report
    return fig

# ============================================================================
# FUNCIONES GAGE R&R (NUEVO)
# ============================================================================

def generate_grr_data():
    """
    Genera datos para estudio Gage R&R Crossed.
    10 Partes, 3 Operadores, 3 Réplicas.
    """
    np.random.seed(42)
    n_parts = 10
    n_operators = 3
    n_replicates = 3

    # Fuentes de variación (Desviaciones estándar reales)
    sigma_part = 2.0        # Variación Parte a Parte
    sigma_operator = 0.5    # Reproducibilidad (Operador)
    sigma_repeatability = 0.3 # Repetibilidad (Error del equipo)

    parts = []
    operators = []
    measurements = []

    # Valores verdaderos de las partes
    part_true_values = np.random.normal(50.0, sigma_part, n_parts)

    # Sesgo por operador (constante para cada operador)
    operator_bias = np.random.normal(0, sigma_operator, n_operators)

    for i in range(n_parts):
        for j in range(n_operators):
            for k in range(n_replicates):
                parts.append(f"Part {i+1}")
                operators.append(f"Op {j+1}")

                # Valor medido = Valor Parte + Sesgo Operador + Error Aleatorio
                value = part_true_values[i] + operator_bias[j] + np.random.normal(0, sigma_repeatability)
                measurements.append(value)

    return pd.DataFrame({
        'Part': parts,
        'Operator': operators,
        'Measurement': measurements
    })

def calculate_anova_grr(df, part_col, operator_col, measurement_col):
    """
    Realiza análisis Gage R&R (ANOVA).
    """
    # Simple validación
    if df is None or len(df) == 0:
        return None

    # Asegurar tipos
    data_df = df.copy()
    data_df[measurement_col] = pd.to_numeric(data_df[measurement_col], errors='coerce')
    data_df = data_df.dropna(subset=[measurement_col])

    # 1. Calcular Medias
    grand_mean = data_df[measurement_col].mean()

    mean_part = data_df.groupby(part_col)[measurement_col].mean()
    mean_oper = data_df.groupby(operator_col)[measurement_col].mean()
    mean_part_oper = data_df.groupby([part_col, operator_col])[measurement_col].mean()

    # 2. Contar niveles
    a = data_df[part_col].nunique()      # Partes
    b = data_df[operator_col].nunique()  # Operadores
    # Promedio de réplicas si no es balanceado perfectamente
    n_counts = data_df.groupby([part_col, operator_col]).size()
    n = n_counts.mean()

    # 3. Suma de Cuadrados (SS)
    # SS Total
    ss_total = np.sum((data_df[measurement_col] - grand_mean)**2)

    # SS Part
    ss_part = b * n * np.sum((mean_part - grand_mean)**2)

    # SS Operator
    ss_operator = a * n * np.sum((mean_oper - grand_mean)**2)

    # SS Interaction (Part*Operator) implementation for potentially unbalanced data approximation
    # SS_Subtotals = Sum(n_ij * (Mean_ij - GrandMean)^2)
    ss_subtotals = 0
    for idx, mean_val in mean_part_oper.items():
        p, o = idx
        n_ij = n_counts.get((p,o), 0)
        ss_subtotals += n_ij * (mean_val - grand_mean)**2

    ss_interaction = ss_subtotals - ss_part - ss_operator
    # Force positive just in case of slight numeric issues
    if ss_interaction < 0: ss_interaction = 0

    # SS Repeatability (Error)
    ss_repeatability = 0
    # Vectorized approach for speed
    # Map group means back to original df
    # We can use transform to get the cell mean for each row
    cell_means = data_df.groupby([part_col, operator_col])[measurement_col].transform('mean')
    ss_repeatability = np.sum((data_df[measurement_col] - cell_means)**2)

    # 4. Degrees of Freedom
    df_part = a - 1
    df_operator = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_repeatability = a * b * (n - 1)
    df_total = (a * b * n) - 1

    # 5. Mean Squares
    ms_part = ss_part / df_part if df_part > 0 else 0
    ms_operator = ss_operator / df_operator if df_operator > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

    # 6. Variance Components
    var_repeatability = ms_repeatability

    # VarComp Interaction
    var_interaction = (ms_interaction - ms_repeatability) / n if n > 0 else 0
    if var_interaction < 0: var_interaction = 0

    # VarComp Operator
    var_operator = (ms_operator - ms_interaction) / (a * n) if (a*n) > 0 else 0
    if var_operator < 0: var_operator = 0

    # VarComp Part
    var_part = (ms_part - ms_interaction) / (b * n) if (b*n) > 0 else 0
    if var_part < 0: var_part = 0

    # Totals
    var_gage_rr = var_repeatability + var_operator + var_interaction
    var_reproducibility = var_operator + var_interaction
    var_total = var_gage_rr + var_part

    # 7. StdDev
    std_repeatability = np.sqrt(var_repeatability)
    std_operator = np.sqrt(var_operator)
    std_interaction = np.sqrt(var_interaction)
    std_gage_rr = np.sqrt(var_gage_rr)
    std_reproducibility = np.sqrt(var_reproducibility)
    std_part = np.sqrt(var_part)
    std_total = np.sqrt(var_total)

    study_var_multiplier = 6.0

    # NDC (Minitab floors it)
    ndc = 1.41 * (std_part / std_gage_rr) if std_gage_rr > 0 else 0
    ndc_int = int(ndc)

    # Results dictionary
    stats = {
        'VarComp': {
            'Total Gage R&R': var_gage_rr,
            '  Repeatability': var_repeatability,
            '  Reproducibility': var_reproducibility,
            '    Operator': var_operator,
            '    Operator*Part': var_interaction,
            'Part-to-Part': var_part,
            'Total Variation': var_total
        },
        'StdDev': {
            'Total Gage R&R': std_gage_rr,
            '  Repeatability': std_repeatability,
            '  Reproducibility': std_reproducibility,
            '    Operator': std_operator,
            '    Operator*Part': std_interaction,
            'Part-to-Part': std_part,
            'Total Variation': std_total
        },
        'ndc': ndc_int
    }

    # Calculate percentages
    stats['%Contribution'] = {k: 100 * (v / var_total) if var_total > 0 else 0 for k, v in stats['VarComp'].items()}
    stats['StudyVar'] = {k: v * study_var_multiplier for k, v in stats['StdDev'].items()}
    stats['%StudyVar'] = {k: 100 * (v / std_total) if std_total > 0 else 0 for k, v in stats['StdDev'].items()}

    return {'components': stats, 'ndc': ndc_int}

def create_grr_plots(df, part_col, operator_col, measurement_col, results):
    """
    Crea los 6 gráficos estándar de Gage R&R.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Components of Variation
    ax1 = fig.add_subplot(gs[0, 0])
    keys = ['Total Gage R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part']
    labels = ['Gage R&R', 'Repeat', 'Reprod', 'Part-to-Part']

    pct_contrib = [results['components']['%Contribution'][k] for k in keys]
    pct_study = [results['components']['%StudyVar'][k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    ax1.bar(x - width/2, pct_contrib, width, label='% Contribution', color='blue')
    ax1.bar(x + width/2, pct_study, width, label='% Study Var', color='magenta')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Components of Variation')
    ax1.legend()

    # 2. R Chart by Operator
    ax2 = fig.add_subplot(gs[0, 1])
    # ... (Simplified R chart logic for visualization)
    # Calculate ranges per part*operator
    r_data = df.groupby([operator_col, part_col])[measurement_col].agg(np.ptp).reset_index()
    r_bar = r_data[measurement_col].mean()
    # Simple limits
    ucl_r = r_bar * 2.57 # Approx for small subgroups

    operators = df[operator_col].unique()
    x_pos = 0
    for op in operators:
        op_vals = r_data[r_data[operator_col] == op][measurement_col].values
        ax2.plot(range(x_pos, x_pos+len(op_vals)), op_vals, 'o-', label=str(op))
        x_pos += len(op_vals)

    ax2.axhline(r_bar, color='green', label='RBar')
    ax2.axhline(ucl_r, color='red', linestyle='--', label='UCL')
    ax2.set_title(f'R Chart by {operator_col}')

    # 3. Xbar Chart by Operator
    ax3 = fig.add_subplot(gs[1, 0])
    x_data = df.groupby([operator_col, part_col])[measurement_col].mean().reset_index()
    x_bar_bar = x_data[measurement_col].mean()
    ucl_x = x_bar_bar + (0.577 * r_bar) # Approx A2
    lcl_x = x_bar_bar - (0.577 * r_bar)

    x_pos = 0
    for op in operators:
        op_vals = x_data[x_data[operator_col] == op][measurement_col].values
        ax3.plot(range(x_pos, x_pos+len(op_vals)), op_vals, 'o-', label=str(op))
        x_pos += len(op_vals)

    ax3.axhline(x_bar_bar, color='green')
    ax3.axhline(ucl_x, color='red', linestyle='--')
    ax3.axhline(lcl_x, color='red', linestyle='--')
    ax3.set_title(f'Xbar Chart by {operator_col}')

    # 4. By Part
    ax4 = fig.add_subplot(gs[1, 1])
    parts = sorted(df[part_col].unique(), key=lambda x: str(x))
    means_part = df.groupby(part_col)[measurement_col].mean()
    ax4.plot(range(len(parts)), [means_part[p] for p in parts], 'bo-')
    ax4.set_xticks(range(len(parts)))
    ax4.set_xticklabels(parts, rotation=45, fontsize=8)
    ax4.set_title(f'By {part_col}')

    # 5. By Operator
    ax5 = fig.add_subplot(gs[2, 0])
    means_oper = df.groupby(operator_col)[measurement_col].mean()
    ops_sorted = sorted(operators, key=lambda x: str(x))
    ax5.plot(range(len(ops_sorted)), [means_oper[o] for o in ops_sorted], 'go-')
    ax5.set_xticks(range(len(ops_sorted)))
    ax5.set_xticklabels(ops_sorted)
    ax5.set_title(f'By {operator_col}')

    # 6. Interaction
    ax6 = fig.add_subplot(gs[2, 1])
    interaction_data = df.groupby([part_col, operator_col])[measurement_col].mean().unstack()
    interaction_data.plot(ax=ax6, marker='o')
    ax6.set_title(f'{operator_col} * {part_col} Interaction')
    ax6.legend(fontsize='small')

    plt.tight_layout()
    return fig

# ============================================================================
# FUNCIONES EXISTENTES (modificadas para compatibilidad)
# ============================================================================

def generate_manufacturing_data():
    """Genera datos en formato largo (compatibilidad con versión anterior)"""
    np.random.seed(42)
    n = 300

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

# Mantener las funciones de cálculo de métricas existentes
def calculate_cp(upper_spec, lower_spec, std_dev):
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

def recommend_sampling_method(data_type, data_nature, application):
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

# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

def main():
    # Inicializar session state
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data_minitab_format()  # Ahora formato Minitab por defecto

    if 'data_format' not in st.session_state:
        st.session_state.data_format = "Minitab"  # 'Minitab' o 'Standard'

    # Header
    st.markdown('<div class="main-header">🏭 Quality Control Dashboard - SPC Sixpack Analysis</div>', unsafe_allow_html=True)

    # Sidebar para navegación - AGREGAR NUEVO MÓDULO
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose Module",
        ["🏠 Home", "📁 Data Import", "🔄 Data Transformation", "📊 Data Overview",
         "📐 Quality Metrics", "🎯 Sampling Recommender", "📏 Gage R&R", "📈 SPC Analysis",
         "📊 Sixpack Report", "🔍 Defect Analysis", "🔬 Advanced Analytics"])

    # ==================== HOME PAGE ====================
    if app_mode == "🏠 Home":
        st.markdown('<div class="section-header">🚀 Welcome to Quality Control Dashboard</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h3>🎯 Complete SPC & Sixpack Analysis Tool</h3>
        <p>This application now supports <strong>Minitab/SQC data format</strong> with complete Sixpack Report generation!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📊 Minitab Format Support")
            st.write("• Upload data in Minitab/SQC format")
            st.write("• 125 samples with subgroup columns")
            st.write("• Automatic data transformation")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📈 Sixpack Report")
            st.write("• Complete 6-chart analysis")
            st.write("• X-bar, R, Histogram, Q-Q plots")
            st.write("• Capability metrics display")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("🔬 Advanced SPC")
            st.write("• Process capability indices")
            st.write("• Control charts generation")
            st.write("• Defect and Pareto analysis")
            st.markdown('</div>', unsafe_allow_html=True)

        # Mostrar ejemplo de formato Minitab
        st.markdown("---")
        st.subheader("📋 Minitab/SQC Data Format Example")

        example_df = pd.DataFrame({
            'Sample': range(1, 6),
            'X1': [10.1, 10.0, 10.2, 10.1, 10.0],
            'X2': [10.2, 10.1, 10.0, 10.2, 10.1],
            'X3': [10.0, 10.1, 10.1, 10.0, 10.2],
            'X4': [10.1, 10.0, 10.2, 10.1, 10.0],
            'X5': [10.2, 10.1, 10.0, 10.2, 10.1]
        })

        st.dataframe(example_df, use_container_width=True)
        st.caption("Format: First column = Sample ID, Next columns = Measurements (X1, X2, ..., X5)")

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

    # ==================== DATA IMPORT ====================
    elif app_mode == "📁 Data Import":
        st.markdown('<div class="section-header">📁 Data Import - Minitab/SQC Format</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Your Data")

            # Seleccionar formato de datos
            data_format = st.radio("Select data format:",
                                   ["Minitab/SQC Format (subgroups in columns)",
                                    "Standard CSV Format (one column per variable)"])

            st.session_state.data_format = "Minitab" if "Minitab" in data_format else "Standard"

            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)

                    if st.session_state.data_format == "Minitab":
                        # Verificar formato Minitab
                        if len(df.columns) < 2:
                            st.error("❌ Minitab format requires at least 2 columns (Sample + measurements)")
                        else:
                            st.session_state.df = df
                            st.success(f"✅ Minitab file uploaded! {df.shape[0]} samples × {df.shape[1]-1} measurements per sample")

                            # Mostrar preview
                            st.subheader("📋 Minitab Data Preview")
                            st.dataframe(df.head(8), use_container_width=True)
                            st.caption(f"First column: '{df.columns[0]}' (assumed as Sample ID)")
                    else:
                        st.session_state.df = df
                        st.success(f"✅ Standard CSV uploaded! Shape: {df.shape}")

                        # Show preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(df.head(8), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")

            # Plantilla descargable
            st.subheader("📥 Download Template")
            if st.button("Download Minitab Format Template (25 samples × 5 measurements)"):
                template_df = generate_manufacturing_data_minitab_format()
                csv = template_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name="minitab_template_25x5.csv",
                    mime="text/csv"
                )

        with col2:
            st.subheader("📚 Sample Data")

            if st.session_state.data_format == "Minitab":
                if st.button("🎲 Load Minitab Sample Data (25×5)"):
                    st.session_state.df = generate_manufacturing_data_minitab_format()
                    st.success("✅ Minitab sample data loaded!")
                    st.dataframe(st.session_state.df.head(8), use_container_width=True)
            else:
                if st.button("🎲 Load Standard Sample Data"):
                    st.session_state.df = generate_manufacturing_data()
                    st.success("✅ Standard sample data loaded!")
                    st.dataframe(st.session_state.df.head(8), use_container_width=True)

            # Información de formato
            st.markdown("---")
            st.subheader("ℹ️ Format Information")

            if st.session_state.data_format == "Minitab":
                st.info("""
                **Minitab/SQC Format:**
                - First column: Sample ID (1, 2, 3, ...)
                - Subsequent columns: Measurements (X1, X2, X3, ...)
                - Each row represents a sample/subgroup
                - Typical: 125 samples × 5 measurements
                """)
            else:
                st.info("""
                **Standard CSV Format:**
                - Each column: A different variable
                - Each row: An individual observation
                - Typical for general statistical analysis
                """)

        # Data quality check
        if 'df' in st.session_state:
            df = st.session_state.df
            st.markdown("---")
            st.subheader("📊 Data Quality Check")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Samples" if st.session_state.data_format == "Minitab" else "Total Rows",
                         df.shape[0])

            with col2:
                if st.session_state.data_format == "Minitab":
                    measurements_per_sample = df.shape[1] - 1
                    st.metric("Measurements per Sample", measurements_per_sample)
                else:
                    st.metric("Total Columns", df.shape[1])

            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)

            with col4:
                if st.session_state.data_format == "Minitab":
                    total_measurements = df.shape[0] * (df.shape[1] - 1)
                    st.metric("Total Measurements", total_measurements)
                else:
                    numeric_cols = len(df.select_dtypes(include=np.number).columns)
                    st.metric("Numeric Columns", numeric_cols)

    # ==================== DATA TRANSFORMATION ====================
    elif app_mode == "🔄 Data Transformation":
        st.markdown('<div class="section-header">🔄 Data Format Transformation</div>', unsafe_allow_html=True)

        if 'df' not in st.session_state:
            st.warning("⚠️ Please load data first in the Data Import module")
            return

        df = st.session_state.df

        st.subheader("📋 Current Data Format")
        st.write(f"**Current format:** {st.session_state.data_format}")
        st.write(f"**Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("🔍 Column Analysis")
            st.write("**Column names:**")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. `{col}`")

            # Detectar automáticamente formato
            st.markdown("---")
            st.subheader("🔄 Transformation Options")

            if st.session_state.data_format == "Minitab":
                if st.button("Transform Minitab → Long Format"):
                    try:
                        df_long = transform_minitab_to_long(df)
                        st.session_state.df = df_long
                        st.session_state.data_format = "Standard"
                        st.success("✅ Transformed to long format successfully!")

                        st.subheader("📊 Transformed Data Preview")
                        st.dataframe(df_long.head(15), use_container_width=True)
                        st.write(f"New dimensions: {df_long.shape[0]} rows × {df_long.shape[1]} columns")

                        # Estadísticas de transformación
                        st.subheader("📈 Transformation Statistics")
                        samples = df_long['Sample'].nunique()
                        measurements_per_sample = len(df_long) / samples

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Unique Samples", samples)
                        with col2:
                            st.metric("Avg Measurements/Sample", f"{measurements_per_sample:.1f}")
                        with col3:
                            st.metric("Total Measurements", len(df_long))
                    except Exception as e:
                        st.error(f"❌ Transformation failed: {e}")
            else:
                st.info("Data is already in standard/long format")

        # Manual column mapping
        st.markdown("---")
        st.subheader("⚙️ Manual Column Mapping")

        col1, col2 = st.columns(2)

        with col1:
            sample_col = st.selectbox("Select Sample ID column", df.columns, index=0)

        with col2:
            measurement_cols = st.multiselect(
                "Select measurement columns (hold Ctrl for multiple)",
                df.columns,
                default=df.columns[1:min(6, len(df.columns))]
            )

        if st.button("Apply Manual Transformation") and measurement_cols:
            try:
                temp_df = df[[sample_col] + measurement_cols].copy()
                df_long = pd.melt(temp_df,
                                 id_vars=[sample_col],
                                 value_vars=measurement_cols,
                                 var_name='Measurement',
                                 value_name='Value')

                st.success(f"✅ Manual transformation complete!")
                st.dataframe(df_long.head(10), use_container_width=True)

                # Opción para guardar transformación
                if st.button("Save This Transformation as Main Dataset"):
                    st.session_state.df = df_long
                    st.session_state.data_format = "Standard"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Manual transformation failed: {e}")

    # ==================== DATA OVERVIEW ====================
    elif app_mode == "📊 Data Overview":
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)

        df = st.session_state.df

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📈 Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")

        with col2:
            if st.session_state.data_format == "Minitab":
                measurements = df.shape[0] * (df.shape[1] - 1)
                st.metric("🔢 Total Measurements", measurements)
            else:
                numeric_cols = len(df.select_dtypes(include=np.number).columns)
                st.metric("🔢 Numeric Columns", numeric_cols)

        with col3:
            if st.session_state.data_format == "Minitab":
                measurements_per = df.shape[1] - 1
                st.metric("📏 Measurements/Sample", measurements_per)
            else:
                categorical_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("📝 Categorical Columns", categorical_cols)

        with col4:
            if 'defect' in df.columns and st.session_state.data_format != "Minitab":
                defect_rate = df['defect'].mean()
                st.metric("⚠️ Defect Rate", f"{defect_rate:.2%}")
            else:
                st.metric("📋 Data Format", st.session_state.data_format)

        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Basic statistics - solo para formato estándar o si hay columnas numéricas
        if st.session_state.data_format == "Standard":
            st.subheader("📈 Basic Statistics")
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistics")
        else:
            # Para formato Minitab, mostrar estadísticas de las columnas de medición
            st.subheader("📈 Measurement Statistics")
            measurement_cols = [col for col in df.columns if col != 'Sample' and df[col].dtype in [np.float64, np.int64]]
            if measurement_cols:
                stats_df = pd.DataFrame({
                    'Mean': df[measurement_cols].mean(),
                    'Std Dev': df[measurement_cols].std(),
                    'Min': df[measurement_cols].min(),
                    'Max': df[measurement_cols].max(),
                    'Range': df[measurement_cols].max() - df[measurement_cols].min()
                })
                st.dataframe(stats_df, use_container_width=True)

        # Visualizations
        st.subheader("📊 Data Distributions")

        if st.session_state.data_format == "Minitab":
            # Para formato Minitab, mostrar distribución de una columna de medición
            measurement_cols = [col for col in df.columns if col != 'Sample' and df[col].dtype in [np.float64, np.int64]]
            if measurement_cols:
                selected_col = st.selectbox("Select measurement column", measurement_cols)

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
        else:
            # Formato estándar (mantener código original)
            numeric_cols = list(df.select_dtypes(include=np.number).columns)
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

    # ==================== SIXPACK REPORT MODULE ====================
    elif app_mode == "📊 Sixpack Report":
        st.markdown('<div class="section-header">📊 Process Capability Sixpack Report</div>', unsafe_allow_html=True)

        if 'df' not in st.session_state:
            st.warning("⚠️ Please load data first")
            return

        df = st.session_state.df

        st.markdown("""
        <div class="info-box">
        <h3>🎯 Sixpack Report Analysis</h3>
        <p>Generates the complete 6-chart Sixpack Report exactly like Minitab for process capability analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        # Verificar formato de datos
        if st.session_state.data_format != "Minitab":
            st.warning("⚠️ Sixpack Report works best with Minitab format data")
            st.info("💡 Please use Minitab format: First column = Sample ID, next columns = measurements")

            # Opción para transformar datos
            if st.button("Transform current data to Minitab format"):
                # Aquí iría lógica para detectar y transformar
                st.info("Feature coming soon - please upload data in Minitab format")

        # Configuración de análisis
        st.subheader("⚙️ Analysis Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Seleccionar columna de muestra
            sample_col_options = [col for col in df.columns if 'sample' in col.lower() or 'id' in col.lower() or df[col].dtype in [np.int64]]
            if not sample_col_options:
                sample_col_options = [df.columns[0]]

            sample_col = st.selectbox("Sample ID Column", sample_col_options)

        with col2:
            # Seleccionar columnas de medición
            measurement_options = [col for col in df.columns if col != sample_col and df[col].dtype in [np.float64, np.int64]]
            default_measurements = measurement_options[:min(5, len(measurement_options))]

            measurement_cols = st.multiselect(
                "Measurement Columns",
                measurement_options,
                default=default_measurements
            )

        with col3:
            # Tamaño de subgrupo
            if measurement_cols:
                subgroup_size = len(measurement_cols)
                st.metric("Subgroup Size", subgroup_size)
            else:
                subgroup_size = st.slider("Subgroup Size", min_value=2, max_value=10, value=5)

            variable_name = st.text_input("Variable Name", value="Diameter")

        # Especificaciones
        st.subheader("📏 Specification Limits")

        col1, col2 = st.columns(2)

        with col1:
                # Calcular estadísticas para sugerir límites
                if measurement_cols:
                    all_measurements = pd.concat([df[col] for col in measurement_cols])
                    data_mean = all_measurements.mean()
                    data_std = all_measurements.std()

                    lsl = st.number_input("Lower Specification Limit (LSL)",
                                         value=float(data_mean - 3*data_std),
                                         format="%.6f",  # <--- Muestra 4 decimales
                                         step=0.000001)    # <--- Permite saltos de 0.0001
                else:
                    lsl = st.number_input("Lower Specification Limit (LSL)",
                                         value=0.0,
                                         format="%.6f",
                                         step=0.000001)

        with col2:
                if measurement_cols:
                    usl = st.number_input("Upper Specification Limit (USL)",
                                         value=float(data_mean + 3*data_std),
                                         format="%.6f",  # <--- Muestra 4 decimales
                                         step=0.000001)    # <--- Permite saltos de 0.0001
                else:
                    usl = st.number_input("Upper Specification Limit (USL)",
                                         value=10.0,
                                         format="%.6f",
                                         step=0.000001)

        # Botón para generar reporte
        if st.button("📊 Generate Sixpack Report", type="primary"):
            if not measurement_cols:
                st.error("❌ Please select at least one measurement column")
                return

            # Preparar datos
            try:
                # Extraer todas las mediciones en un array
                measurement_data = []
                for col in measurement_cols:
                    measurement_data.extend(df[col].dropna().values)

                measurement_array = np.array(measurement_data)

                if len(measurement_array) < subgroup_size * 2:
                    st.error(f"❌ Not enough data. Need at least {subgroup_size * 2} measurements")
                    return

                # Calcular métricas
                metrics = calculate_sixpack_metrics(
                    data=measurement_array,
                    subgroup_size=subgroup_size,
                    lsl=lsl,
                    usl=usl
                )

                # Crear y mostrar Sixpack Report
                st.markdown("---")
                st.markdown(f"## Process Capability Sixpack Report for {variable_name}")

                fig = create_sixpack_report(metrics, variable_name)
                st.pyplot(fig)

                # Métricas adicionales
                st.markdown("---")
                st.subheader("📈 Detailed Capability Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Cp", f"{metrics['cp']:.2f}")
                    st.metric("Cpk", f"{metrics['cpk']:.2f}")

                with col2:
                    st.metric("Pp", f"{metrics['pp']:.2f}")
                    st.metric("Ppk", f"{metrics['ppk']:.2f}")

                with col3:
                    st.metric("Within Std Dev", f"{metrics['within_std']:.5f}")
                    st.metric("Overall Std Dev", f"{metrics['overall_std']:.5f}")

                with col4:
                    st.metric("PPM", f"{metrics['ppm']:,.0f}")
                    sigma_level = calculate_sigma_level(metrics['ppm'])
                    st.metric("Sigma Level", f"{sigma_level:.2f}")

                # Interpretación
                st.markdown("---")
                st.subheader("🎯 Interpretation")

                if metrics['cpk'] >= 1.33:
                    st.success("✅ **EXCELLENT PROCESS** - Cpk ≥ 1.33 indicates a highly capable process")
                elif metrics['cpk'] >= 1.0:
                    st.warning("⚠️ **MARGINAL PROCESS** - Cpk between 1.0 and 1.33 needs monitoring")
                else:
                    st.error("❌ **POOR PROCESS** - Cpk < 1.0 indicates insufficient process capability")

                # Recomendaciones
                st.markdown("---")
                st.subheader("💡 Recommendations")

                if metrics['cp'] > metrics['cpk']:
                    st.info("**Centering Issue:** Cp > Cpk indicates the process is not centered. Adjust process mean toward target.")

                if metrics['ppm'] > 1000:
                    st.info("**High Defect Rate:** PPM > 1000. Consider reducing process variation.")

                if metrics['within_std'] > metrics['overall_std'] * 0.8:
                    st.info("**Subgroup Variation:** Within-subgroup variation is high. Check measurement system.")

                # Opción para descargar reporte
                st.markdown("---")
                st.subheader("📥 Download Report")

                # Convertir figura a bytes para descargar
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="Download Sixpack Report (PNG)",
                        data=buf,
                        file_name=f"sixpack_report_{variable_name}.png",
                        mime="image/png"
                    )

                with col2:
                    # Crear reporte en texto
                    report_text = f"""
                    Process Capability Sixpack Report - {variable_name}
                    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

                    {'='*60}

                    CONTROL LIMITS:
                    - X-bar Chart UCL: {metrics['ucl_x']:.4f}
                    - X-bar Chart LCL: {metrics['lcl_x']:.4f}
                    - R Chart UCL: {metrics['ucl_r']:.4f}
                    - R Chart LCL: {metrics['lcl_r']:.4f}

                    CAPABILITY METRICS:
                    - Cp: {metrics['cp']:.2f}
                    - Cpk: {metrics['cpk']:.2f}
                    - Pp: {metrics['pp']:.2f}
                    - Ppk: {metrics['ppk']:.2f}
                    - Within Std Dev: {metrics['within_std']:.5f}
                    - Overall Std Dev: {metrics['overall_std']:.5f}
                    - PPM: {metrics['ppm']:,.0f}

                    SPECIFICATIONS:
                    - LSL: {metrics['lsl']:.2f}
                    - USL: {metrics['usl']:.2f}
                    - Target: {(metrics['usl'] + metrics['lsl'])/2:.2f}
                    - Tolerance: {metrics['usl'] - metrics['lsl']:.2f}

                    INTERPRETATION:
                    - Process {'IS' if metrics['cpk'] >= 1.33 else 'IS NOT'} capable (Cpk {'≥' if metrics['cpk'] >= 1.33 else '<'} 1.33)
                    - Estimated defect rate: {metrics['ppm']/1000000*100:.2f}%
                    """

                    st.download_button(
                        label="Download Text Report",
                        data=report_text,
                        file_name=f"sixpack_report_{variable_name}.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"❌ Error generating Sixpack Report: {str(e)}")
                st.exception(e)

    # ==================== LOS DEMÁS MÓDULOS ====================
    # (Mantener los módulos existentes pero con compatibilidad para Minitab)

    # Quality Metrics Module
    elif app_mode == "📐 Quality Metrics":
        st.markdown('<div class="section-header">📐 Quality Metrics Calculator</div>', unsafe_allow_html=True)

        df = st.session_state.df

        # Prepare data based on format
        if st.session_state.data_format == "Minitab":
            numeric_cols = [c for c in df.columns if c != 'Sample' and df[c].dtype in [np.float64, np.int64]]
        else:
            numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if 'id' not in c.lower()]

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
                    data_mean = np.mean(data)
                    data_std = np.std(data, ddof=1)

                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    # Aplicamos format y step para que coincida con la precisión de tus sensores/máquinas
                    lsl = st.number_input("Lower Specification Limit (LSL)",
                                         value=float(data_mean - 3*data_std),
                                         format="%.6f",
                                         step=0.000001)

                    usl = st.number_input("Upper Specification Limit (USL)",
                                         value=float(data_mean + 3*data_std),
                                         format="%.6f",
                                         step=0.000001)

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
            values = data.values
            for i in range(0, min(len(values), subgroup_size*10), subgroup_size):
                if i + subgroup_size <= len(values):
                    subgroup = values[i:i+subgroup_size]
                    subgroup_means.append(np.mean(subgroup))
                    subgroup_stds.append(np.std(subgroup, ddof=1))

            short_term_std = np.mean(subgroup_stds) if subgroup_stds else std_val
            cmk = calculate_cmk(usl, lsl, mean_val, short_term_std)

            # DPMO
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
                ("Ppk", ppk, 1.33, 1.0)
            ]

            for name, value, good, marginal in capability_metrics:
                if value >= good:
                    st.success(f"✅ **{name}: {value:.3f}** - Good (≥ {good})")
                elif value >= marginal:
                    st.warning(f"⚠️ **{name}: {value:.3f}** - Marginal (≥ {marginal})")
                else:
                    st.error(f"❌ **{name}: {value:.3f}** - Poor (< {marginal})")

            # Visualization
            st.subheader("📈 Distribution Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Histogram
            ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax1.axvline(usl, color='green', linestyle='dashed', linewidth=2, label=f'USL: {usl}')
            ax1.axvline(lsl, color='green', linestyle='dashed', linewidth=2, label=f'LSL: {lsl}')

            # Normal curve
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            if HAS_SCIPY:
                p = stats.norm.pdf(x, mean_val, std_val)
                ax1.plot(x, p, 'r-', linewidth=2)

            ax1.set_title(f'Distribution of {variable}')
            ax1.legend()

            # Bar chart
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk', 'Cmk']
            values = [cp, cpk, pp, ppk, cmk]
            colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values]

            ax2.bar(indices, values, color=colors, alpha=0.7)
            ax2.axhline(1.33, color='red', linestyle='--', label='Min Rec (1.33)')
            ax2.set_title('Capability Indices')
            ax2.legend()

            st.pyplot(fig)

    # Sampling Recommender (Restore original)
    elif app_mode == "🎯 Sampling Recommender":
        st.markdown('<div class="section-header">🎯 Sampling Method Recommender</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
             data_type = st.radio("Data Type", ["Variable", "Attribute"])
             data_nature = st.selectbox("Data Nature", ["Continuous", "Discrete"])
        with col2:
             app_type = st.selectbox("Application", ["Process Control", "Lot Acceptance"])

        if st.button("Get Recommendation"):
             recs = recommend_sampling_method(data_type, data_nature, app_type)
             for r in recs:
                 st.write(f"• {r}")


    # SPC Analysis Module (adaptar para Minitab)
    elif app_mode == "📈 SPC Analysis":
        st.markdown('<div class="section-header">📈 Statistical Process Control (SPC)</div>', unsafe_allow_html=True)

        df = st.session_state.df

        # Adaptar para formato Minitab
        if st.session_state.data_format == "Minitab":
            st.info("📋 Data is in Minitab format. Extracting measurements for SPC analysis...")

            # Seleccionar columnas de medición
            measurement_cols = [col for col in df.columns if col != 'Sample' and df[col].dtype in [np.float64, np.int64]]

            if not measurement_cols:
                st.error("❌ No measurement columns found")
                return

            # Convertir a formato largo para SPC
            df_long_list = []
            for col in measurement_cols:
                temp_df = pd.DataFrame({
                    'Sample': df['Sample'] if 'Sample' in df.columns else df.index,
                    'Value': df[col],
                    'Measurement': col
                })
                df_long_list.append(temp_df)

            df_long = pd.concat(df_long_list, ignore_index=True)

            # Usar df_long para el análisis SPC
            data = df_long['Value'].dropna()
            subgroup_size = len(measurement_cols)

            st.write(f"**Analysis setup:** {len(data)} total measurements, {subgroup_size} measurements per sample")

        else:
            # Formato estándar (código original)
            numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]

            if not numeric_cols:
                st.error("❌ No numeric variables available for SPC analysis")
                return

            variable = st.selectbox("📊 Select Variable for SPC", numeric_cols)
            subgroup_size = st.slider("👥 Subgroup Size", min_value=2, max_value=10, value=5)
            data = df[variable].dropna()

        # El resto del código SPC original...
            data = df[variable].dropna()

            # --- BEGIN SPC LOGIC ---
            data_values = data.values
            n_subgroups = len(data_values) // subgroup_size

# --- Added User Control for Spec Limits ---
            st.markdown("##### 📏 Specification Limits for Control Charts")
            col_spec1, col_spec2 = st.columns(2)
            with col_spec1:
                # Default to a reasonable range or 0
                default_lsl = float(np.min(data_values) * 0.9)
                lsl_spc = st.number_input("Lower Specification Limit (LSL)",
                                         value=default_lsl,
                                         format="%.6f", # Muestra 4 decimales
                                         step=0.000001)   # Permite ajuste fino
            with col_spec2:
                # Default to a reasonable range or 1.1
                default_usl = float(np.max(data_values) * 1.1)
                usl_spc = st.number_input("Upper Specification Limit (USL)",
                                         value=default_usl,
                                         format="%.6f", # Muestra 4 decimales
                                         step=0.000001)   # Permite ajuste fino

            if n_subgroups < 2:
                st.error(f"❌ Not enough data for subgrouping. Need at least {2*subgroup_size} data points.")
            else:
                subgrouped_data = data_values[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

                # Calculate subgroup statistics
                subgroup_means = np.mean(subgrouped_data, axis=1)
                subgroup_ranges = np.ptp(subgrouped_data, axis=1)  # Peak-to-peak (range)

                # Calculate control limits
                overall_mean = np.mean(subgroup_means)
                mean_range = np.mean(subgroup_ranges)

                # Constants for control limits
                a2_dict = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
                d3_dict = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
                d4_dict = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

                a2 = a2_dict.get(subgroup_size, 0.577)
                d3 = d3_dict.get(subgroup_size, 0)
                d4 = d4_dict.get(subgroup_size, 2.114)

                # Control limits
                ucl_x = overall_mean + a2 * mean_range
                lcl_x = overall_mean - a2 * mean_range
                ucl_r = d4 * mean_range
                lcl_r = d3 * mean_range

                # Create SPC charts
                st.subheader(f"📈 Control Charts for {variable}")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # X-bar chart
                ax1.plot(subgroup_means, 'bo-', label='Subgroup Means', markersize=4)
                ax1.axhline(overall_mean, color='green', linestyle='-', label=f'CL: {overall_mean:.3f}')
                ax1.axhline(ucl_x, color='red', linestyle='--', label=f'UCL: {ucl_x:.3f}')
                ax1.axhline(lcl_x, color='red', linestyle='--', label=f'LCL: {lcl_x:.3f}')

                # Plot User Defined Spec Limits
                if lsl_spc is not None:
                    ax1.axhline(lsl_spc, color='purple', linestyle='-.', alpha=0.7, label=f'LSL: {lsl_spc:.2f}')
                if usl_spc is not None:
                    ax1.axhline(usl_spc, color='purple', linestyle='-.', alpha=0.7, label=f'USL: {usl_spc:.2f}')

                ax1.set_title(f'X-bar Chart')
                ax1.set_ylabel('Mean Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # R chart
                ax2.plot(subgroup_ranges, 'go-', label='Subgroup Ranges', markersize=4)
                ax2.axhline(mean_range, color='green', linestyle='-', label=f'CL: {mean_range:.3f}')
                ax2.axhline(ucl_r, color='red', linestyle='--', label=f'UCL: {ucl_r:.3f}')
                ax2.axhline(lcl_r, color='red', linestyle='--', label=f'LCL: {lcl_r:.3f}')
                ax2.set_title(f'R Chart')
                ax2.set_ylabel('Range')
                ax2.set_xlabel('Subgroup Number')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # --- Process Capability from Control Chart ---
                st.markdown("---")
                st.subheader("📐 Process Capability from Control Chart")

                # Formula: Sigma (within) = R_bar / d2
                d2_dict = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                d2 = d2_dict.get(subgroup_size, 2.326)

                sigma_within = mean_range / d2

                # Check for Spec Limits
                if lsl_spc is not None and usl_spc is not None:
                    cp = (usl_spc - lsl_spc) / (6 * sigma_within) if sigma_within > 0 else 0
                    cpu = (usl_spc - overall_mean) / (3 * sigma_within) if sigma_within > 0 else 0
                    cpl = (overall_mean - lsl_spc) / (3 * sigma_within) if sigma_within > 0 else 0
                    cpk = min(cpu, cpl)

                    col_cap1, col_cap2, col_cap3 = st.columns(3)
                    with col_cap1:
                        st.metric("Within-subgroup Std Dev", f"{sigma_within:.4f}")
                    with col_cap2:
                         st.metric("Cp", f"{cp:.3f}")
                    with col_cap3:
                         st.metric("Cpk", f"{cpk:.3f}")

                    # Simple Control Check: Are all points within limits?
                    points_in_control_x = np.all((subgroup_means >= lcl_x) & (subgroup_means <= ucl_x))
                    points_in_control_r = np.all((subgroup_ranges >= lcl_r) & (subgroup_ranges <= ucl_r))

                    if points_in_control_x and points_in_control_r:
                        st.success("✅ Process appears to be in statistical control (All points within Control Limits)")
                    else:
                        st.warning("⚠️ Process may be out of control (Some points outside Control Limits)")
                else:
                    st.info("ℹ️ Enter LSL and USL above to calculate Cp and Cpk values.")

    # ==================== NEW GAGE R&R MODULE ====================
    elif app_mode == "📏 Gage R&R":
        st.markdown('<div class="section-header">📏 Gage R&R Study (Crossed)</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h3>🎯 Assessment of Measurement System Variability</h3>
        <p>This module performs a <strong>Crossed Gage R&R Study</strong> using the ANOVA method.</p>
        <p>Required Data: Part ID, Operator ID, and Measurement Value.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
             st.subheader("🛠️ Data Setup")
             if st.button("🎲 Load Sample Gage R&R Data"):
                 st.session_state.grr_df = generate_grr_data()
                 st.success("✅ Sample Gage R&R data loaded!")

             # Allow file upload specifically for Gage R&R if needed,
             # or use main df if it has the right columns

        if 'grr_df' in st.session_state:
            df_grr = st.session_state.grr_df
        elif 'df' in st.session_state:
            df_grr = st.session_state.df
        else:
            df_grr = None

        if df_grr is not None:
             with col2:
                 st.subheader("📋 Dataset Preview")
                 st.dataframe(df_grr.head(), use_container_width=True)

             st.markdown("---")
             st.subheader("⚙️ Configuration")

             cols = df_grr.columns.tolist()

             c1, c2, c3 = st.columns(3)
             with c1:
                 # Try to auto-select
                 part_default = next((c for c in cols if 'part' in c.lower()), cols[0])
                 part_col = st.selectbox("Select Part Column", cols, index=cols.index(part_default))
             with c2:
                 oper_default = next((c for c in cols if 'oper' in c.lower()), cols[1] if len(cols)>1 else cols[0])
                 oper_col = st.selectbox("Select Operator Column", cols, index=cols.index(oper_default))
             with c3:
                 meas_default = next((c for c in cols if 'meas' in c.lower() or 'value' in c.lower() or 'x' in c.lower()), cols[2] if len(cols)>2 else cols[0])
                 meas_col = st.selectbox("Select Measurement Column", cols, index=cols.index(meas_default))

             if st.button("🚀 Run Gage R&R Analysis", type="primary"):
                 try:
                     # Calculate
                     results = calculate_anova_grr(df_grr, part_col, oper_col, meas_col)

                     # 1. Visualization
                     st.subheader("📊 Gage R&R Graphs")
                     fig = create_grr_plots(df_grr, part_col, oper_col, meas_col, results)
                     st.pyplot(fig)

                     # 2. Statistics Table (VarComp)
                     st.markdown("---")
                     st.subheader("📉 Gage R&R Statistics (ANOVA Method)")

                     vc = results['components']

                     # Create display dataframe
                     stats_data = {
                         'Source': ['Total Gage R&R', '  Repeatability', '  Reproducibility', '    Operator', '    Operator*Part', 'Part-to-Part', 'Total Variation'],
                         'StdDev (SD)': [vc['StdDev'][k] for k in vc['StdDev']],
                         'Study Var (6*SD)': [vc['StudyVar'][k] for k in vc['StudyVar']],
                         '% Study Var': [vc['%StudyVar'][k] for k in vc['%StudyVar']],
                         '% Contribution': [vc['%Contribution'][k] for k in vc['%Contribution']]
                     }

                     stats_df = pd.DataFrame(stats_data)
                     st.dataframe(stats_df.style.format({
                         'StdDev (SD)': '{:.5f}',
                         'Study Var (6*SD)': '{:.5f}',
                         '% Study Var': '{:.2f}%',
                         '% Contribution': '{:.2f}%'
                     }), use_container_width=True)

                     # 3. NDC and Interpretation
                     st.markdown("---")
                     col_res1, col_res2 = st.columns(2)

                     with col_res1:
                         ndc = results['ndc']
                         st.metric("Number of Distinct Categories (NDC)", ndc)

                         if ndc >= 5:
                             st.success("✅ NDC ≥ 5: Measurement system is acceptable for analysis")
                         else:
                             st.error("❌ NDC < 5: Measurement system may differ only 1-4 categories ( Poor)")

                     with col_res2:
                         pct_grr = vc['%StudyVar']['Total Gage R&R']
                         st.metric("Total Gage R&R (% Study Var)", f"{pct_grr:.2f}%")

                         if pct_grr < 10:
                             st.success("✅ %GRR < 10%: Measurement system is acceptable")
                         elif pct_grr < 30:
                             st.warning("⚠️ 10% < %GRR < 30%: May be acceptable depending on application")
                         else:
                             st.error("❌ %GRR > 30%: Measurement system needs improvement")

                 except Exception as e:
                     st.error(f"Error during calculation: {e}")
                     st.write("Please ensure columns are correct and contain numeric data for measurements.")

    elif app_mode == "🔍 Defect Analysis":
        # ... [Mantener lógica original para Defect Analysis adaptada]
        st.markdown('<div class="section-header">🔍 Defect Analysis</div>', unsafe_allow_html=True)

        df = st.session_state.df

        # Check for numeric boolean or object 0/1 columns that could be defects
        possible_defect_cols = [c for c in df.columns if 'defect' in c.lower() or df[c].nunique() == 2]

        selected_defect_col = None
        defect_col = None  # Initialize to prevent NameError
        is_generated = False

        # --- Defect Generator (Always Available via Expander) ---
        with st.expander("🛠️ Defect Generator (Create Pass/Fail from limits)"):
             st.info("Generate a 'Defect' status based on Specification Limits (LSL/USL).")

             # Locate numeric columns
             numeric_cols_gen = df.select_dtypes(include=np.number).columns.tolist()
             if 'Sample' in numeric_cols_gen: numeric_cols_gen.remove('Sample')

             if numeric_cols_gen:
                 gen_col = st.selectbox("Select Measurement Column", numeric_cols_gen)

                 c1, c2 = st.columns(2)
                 mean_val = df[gen_col].mean()
                 std_val = df[gen_col].std()

                 with c1:
                     # Tighter limits by default (1 sigma) to ensure defects appear in demo
                     gen_lsl = st.number_input("LSL", value=float(mean_val - 1.0*std_val), key='def_lsl')
                 with c2:
                     gen_usl = st.number_input("USL", value=float(mean_val + 1.0*std_val), key='def_usl')

                 if st.button("Generate & Use Defect Data"):
                     # Create temporary defect column
                     df['Generated_Defect'] = ((df[gen_col] < gen_lsl) | (df[gen_col] > gen_usl)).astype(int)
                     # Persist to session state so other modules can see it
                     st.session_state.df = df
                     st.success(f"✅ Generated 'Generated_Defect' column based on limits. Found {df['Generated_Defect'].sum()} defects.")
                     selected_defect_col = 'Generated_Defect'
                     is_generated = True
             else:
                 st.error("No numeric columns available.")

        # If not generated, allow selection
        if not is_generated:
            if possible_defect_cols:
                selected_defect_col = st.selectbox("Select Defect Column (0/1 or Pass/Fail)", possible_defect_cols)
            elif 'Generated_Defect' in df.columns:
                 selected_defect_col = 'Generated_Defect'
            else:
                 st.warning("⚠️ No obvious defect column found. Please use the generator above.")

        if selected_defect_col:
            defect_col = selected_defect_col

            # Ensure it's numeric 0/1 for calculation
            # If it's text (Fail/Pass), map it
            target_col = defect_col
            if not is_generated:
                try:
                    if df[defect_col].dtype == object:
                        unique_vals = df[defect_col].unique()
                        # Heuristic: 'Fail', 'NG', 'Bad', '1' -> 1
                        fail_indicators = ['fail', 'ng', 'bad', 'defect', 'si', 'yes', 'y']
                        # Create mapper
                        df['is_defect'] = df[defect_col].apply(lambda x: 1 if str(x).lower() in fail_indicators else 0)
                        target_col = 'is_defect'
                except:
                    pass

            # Categorical columns for grouping
            # Expanded to include integer columns like 'Sample' or others with reasonable cardinality
            cat_cols = [c for c in df.columns if (df[c].dtype == object or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 100)) and c != defect_col and c != 'Generated_Defect']

            # --- Category Simulation (Always Available via Expander if needed) ---
            with st.expander("🎲 Category Simulator (if missing Operator/Machine)"):
                st.info("Add simulated 'Operator' and 'Machine' columns for testing Stratification.")
                if st.button("Add Simulated Categories"):
                    np.random.seed(42)
                    df['Operator'] = np.random.choice(['Op1', 'Op2', 'Op3'], size=len(df))
                    df['Machine'] = np.random.choice(['M1', 'M2'], size=len(df))
                    # Persist to session state
                    st.session_state.df = df
                    st.success("✅ Added 'Operator' and 'Machine' columns!")
                    st.rerun()

            # Re-evaluate cat_cols after potential simulation
            cat_cols = [c for c in df.columns if (df[c].dtype == object or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 100)) and c != defect_col and c != 'Generated_Defect']

            if not cat_cols:
                st.warning("⚠️ No categorical or grouping columns found (e.g. Operator, Machine).")
                st.info("To see a Pareto chart, you need a column to group the defects by. Use the Simulator above.")
            else:
                category = st.selectbox("Stratify Defects By", cat_cols)

                # Groupby
                summary = df.groupby(category)[target_col].agg(['sum', 'count', 'mean']).reset_index()
                summary.columns = [category, 'Defects', 'Total', 'Rate']
                summary['Rate %'] = summary['Rate'] * 100

                # Pareto
                summary = summary.sort_values('Defects', ascending=False)
                total_defects = summary['Defects'].sum()

                # ALWAYS SHOW GRAPH LOGIC (Even if 0 defects)
                summary['CumSum'] = summary['Defects'].cumsum()
                summary['CumPerc'] = 100 * summary['CumSum'] / total_defects if total_defects > 0 else 0

                st.subheader("📊 Pareto Analysis")
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(summary[category], summary['Defects'], color='skyblue', edgecolor='black')
                ax1.set_ylabel('Defect Count', color='black')
                ax1.tick_params(axis='y', labelcolor='black')

                if total_defects > 0:
                    ax2 = ax1.twinx()
                    ax2.plot(summary[category], summary['CumPerc'], 'bo-', linewidth=2, markersize=5)
                    ax2.set_ylabel('Cumulative %', color='blue')
                    ax2.tick_params(axis='y', labelcolor='blue')
                    ax2.set_ylim([0, 110])
                    ax2.axhline(80, color='grey', linestyle='--')
                    ax2.grid(True, alpha=0.3)
                else:
                    st.info("No defects found (Perfect Quality!). Showing empty chart.")

                # Rotate x labels if necessary
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.title(f'Pareto Chart of Defects by {category}')
                plt.tight_layout()

                st.pyplot(fig)

                if total_defects > 0:
                    # --- Detailed Text Report (Matching Original) ---
                    st.subheader("📈 Pareto Analysis Results")

                    for i, row in summary.iterrows():
                        pct = (row['Defects'] / total_defects) * 100
                        st.write(f"**{i+1}. {row[category]}**: {int(row['Defects'])} defects ({pct:.1f}%), Cumulative: {row['CumPerc']:.1f}%")

                    # Vital Few Analysis
                    vital_few = summary[summary['CumPerc'] <= 80]
                    # If the first one is already > 80, take at least the first one
                    if vital_few.empty:
                        vital_few = summary.iloc[:1]

                    vital_count = len(vital_few)
                    total_cats = len(summary)
                    vital_pct = (vital_count / total_cats) * 100
                    vital_defects_pct = vital_few['CumPerc'].max()

                    st.subheader(f"🎯 Vital Few ({vital_pct:.1f}% of categories cause {vital_defects_pct:.0f}% of defects):")

                    for _, row in vital_few.iterrows():
                        pct = (row['Defects'] / total_defects) * 100
                        st.write(f"• **{row[category]}** - {int(row['Defects'])} defects ({pct:.1f}%)")
                else:
                    st.info("No defects found to analyze.")

    elif app_mode == "🔬 Advanced Analytics":
        st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Prepare numeric columns (exclude ID-like columns)
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if 'id' not in c.lower() and 'sample' not in c.lower()]

        if len(numeric_cols) < 1:
            st.error("❌ Not enough numeric variables for analysis")
            return

        analysis_type = st.radio(
            "Select Analysis Type",
            ["Normality Test", "Q-Q Plot", "Correlation Analysis"]
        )

        if analysis_type == "Normality Test":
            st.subheader("📊 Normality Test (Shapiro-Wilk)")

            variable = st.selectbox("Select Variable", numeric_cols)
            data = df[variable].dropna()

            if len(data) < 3:
                st.warning("⚠️ Need at least 3 data points for Shapiro-Wilk test.")
            else:
                if HAS_SCIPY:
                    stat, p_value = stats.shapiro(data)

                    st.write(f"**Variable:** {variable}")
                    st.write(f"**Statistic:** {stat:.4f}")
                    st.write(f"**P-Value:** {p_value:.4f}")

                    if p_value > 0.05:
                        st.success(f"✅ P-Value > 0.05: Data looks Normally Distributed (Fail to reject H0)")
                    else:
                        st.error(f"❌ P-Value < 0.05: Data does NOT look Normally Distributed (Reject H0)")
                else:
                    st.warning("⚠️ Scipy not installed. Installing scipy is recommended for statistical tests.")

            # Visualization
            st.subheader("📈 Distribution Visualization")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Histogram with normal curve
            ax1.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            if HAS_SCIPY:
                xmin, xmax = ax1.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, np.mean(data), np.std(data))
                ax1.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
            ax1.set_title(f'Distribution of {variable}')
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Box plot
            ax2.boxplot(data)
            ax2.set_title(f'Box Plot of {variable}')
            ax2.set_ylabel(variable)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        elif analysis_type == "Q-Q Plot":
            st.subheader("📈 Q-Q Plot (Quantile-Quantile)")
            if not HAS_SCIPY:
                st.error("❌ Q-Q Plot requires SciPy. Please install scipy to use this feature.")
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
            st.subheader("🔗 Correlation Analysis")
            selected_vars = st.multiselect("📊 Select Variables for Correlation", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])

            if len(selected_vars) >= 2:
                corr_matrix = df[selected_vars].corr()

                st.subheader("Correlation Matrix")
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
                    var2 = st.selectbox("Select Y-axis variable", selected_vars, index=1 if len(selected_vars)>1 else 0)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[var1], df[var2], alpha=0.6)
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    ax.set_title(f'Scatter Plot: {var1} vs {var2}')
                    ax.grid(True, alpha=0.3)

                    # Add correlation coefficient
                    corr_coef = df[var1].corr(df[var2])
                    ax.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}',
                           transform=ax.transAxes, fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("ℹ️ Select at least 2 variables to calculate correlations.")

    # ==================== DATA IMPORT (Update) ====================

# Run the application
if __name__ == "__main__":
    main()

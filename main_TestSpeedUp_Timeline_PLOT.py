import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# 1. GLOBAL STYLING
# ==============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#cccccc"
})

# ==============================================================================
# 2. FILE PATHS
# ==============================================================================

path_case_std  = "./Example_Bentheimer/Permeability.csv"
path_case_ini  = "./Example_Bentheimer/Composed Model/Permeability.csv"
path_case_grad = "./Example_Bentheimer/Gradient_Init/Permeability.csv"

df_std  = pd.read_csv(path_case_std, sep=r'\s+')
df_ini  = pd.read_csv(path_case_ini, sep=r'\s+')
df_grad = pd.read_csv(path_case_grad, sep=r'\s+')

# Strip possible whitespace from column names
df_std.columns  = df_std.columns.str.strip()
df_ini.columns  = df_ini.columns.str.strip()
df_grad.columns = df_grad.columns.str.strip()

# ==============================================================================
# 3. CONVERT DATA TO NUMERIC
# ==============================================================================

for df in [df_std, df_ini, df_grad]:
    df['time'] = pd.to_numeric(df['time'])
    df['absperm(mDa)'] = pd.to_numeric(df['absperm(mDa)'])
    df.sort_values('time', inplace=True)

# ==============================================================================
# 4. EXTRACT DATA
# ==============================================================================

x_std = df_std['time'].to_numpy()
y_std = df_std['absperm(mDa)'].to_numpy()

x_ini = df_ini['time'].to_numpy()
y_ini = df_ini['absperm(mDa)'].to_numpy()

x_grad = df_grad['time'].to_numpy()
y_grad = df_grad['absperm(mDa)'].to_numpy()

# ==============================================================================
# 5. PRINT DATA
# ==============================================================================

print("A - Standard:")
print(df_std[['time', 'absperm(mDa)']].head())
print(df_std[['time', 'absperm(mDa)']].tail())

print("\nB - Initiated:")
print(df_ini[['time', 'absperm(mDa)']].head())
print(df_ini[['time', 'absperm(mDa)']].tail())

print("\nC - Pressure Gradient:")
print(df_grad[['time', 'absperm(mDa)']].head())
print(df_grad[['time', 'absperm(mDa)']].tail())

print("\nTime ordering:")
print("A:", df_std['time'].is_monotonic_increasing)
print("B:", df_ini['time'].is_monotonic_increasing)
print("C:", df_grad['time'].is_monotonic_increasing)

# ==============================================================================
# 6. FIGURE
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# ------------------------------------------------------------------------------
# Standard
# ------------------------------------------------------------------------------

ax.plot(
    x_std, y_std,
    label='Standard (null velocity)',
    color='black',
    linestyle='--',
    linewidth=2.5,
    zorder=3
)

ax.scatter(
    x_std[-1], y_std[-1],
    color='black',
    marker='*',
    s=120,
    zorder=4
)

# ------------------------------------------------------------------------------
# Deep Learning Initialization
# ------------------------------------------------------------------------------

ax.plot(
    x_ini, y_ini,
    label='Initiated (deep learning)',
    color='#16a085',
    linestyle='-',
    linewidth=2.5,
    zorder=2
)

ax.scatter(
    x_ini[-1], y_ini[-1],
    color='#16a085',
    marker='*',
    s=120,
    zorder=4
)

# ------------------------------------------------------------------------------
# Pressure Gradient Initialization
# ------------------------------------------------------------------------------

ax.plot(
    x_grad, y_grad,
    label='Pressure gradient',
    color='#8e44ad',
    linestyle='-.',
    linewidth=2.5,
    zorder=2
)

ax.scatter(
    x_grad[-1], y_grad[-1],
    color='#8e44ad',
    marker='*',
    s=120,
    zorder=4
)

# ==============================================================================
# 7. LEGEND PROXY FOR TOLERANCE
# ==============================================================================

ax.plot(
    [],
    [],
    color='gray',
    marker='*',
    linestyle='None',
    markersize=10,
    label='Tolerance Achieved'
)

# ==============================================================================
# 8. AXIS LABELS
# ==============================================================================

ax.set_xscale('log')

ax.set_xlabel('Timesteps', fontsize=18)
ax.set_ylabel('Absolute Permeability (mDa)', fontsize=18)

# ==============================================================================
# 9. TICKS
# ==============================================================================

ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
ax.xaxis.set_minor_locator(
    ticker.LogLocator(base=10, subs='auto')
)

ax.yaxis.set_major_locator(
    ticker.MaxNLocator(nbins=6)
)

ax.minorticks_on()

# ==============================================================================
# 10. GRID
# ==============================================================================

ax.grid(
    True,
    which='major',
    color='#e0e0e0',
    linestyle='-',
    linewidth=0.8,
    alpha=0.8
)

ax.grid(
    True,
    which='minor',
    axis='y',
    color='#f0f0f0',
    linestyle='--',
    linewidth=0.5,
    alpha=0.8
)

ax.grid(False, which='minor', axis='x')

# ==============================================================================
# 11. LEGEND
# ==============================================================================

ax.legend(
    loc='upper right',
    fontsize=13,
    borderpad=0.6,
    handlelength=2.5
)

# ==============================================================================
# 12. SAVE
# ==============================================================================

plt.tight_layout()

plt.savefig(
    'SingleCase_SpeedUp.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
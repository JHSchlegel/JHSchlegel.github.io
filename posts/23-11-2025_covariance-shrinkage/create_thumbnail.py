import numpy as np
import pandas as pd
import plotnine as pn
from plotnine import *

# Set random seed for reproducibility
np.random.seed(42)

# Define Academic Theme
THEME_ACADEMIC = pn.theme(
    text=pn.element_text(family="monospace"),
    plot_title=pn.element_text(weight="bold", size=12, ha="center"),
    legend_text=pn.element_text(size=10),
    legend_title=pn.element_text(size=10, hjust = 0.5),
    panel_background=pn.element_rect(fill="white"),
    panel_border=pn.element_rect(color="grey", size=0.5),
    axis_ticks=pn.element_line(color="grey"),
    panel_grid_major=pn.element_line(color="grey", size=0.1, alpha=0.3),
    panel_grid_minor=pn.element_line(color="grey", size=0.1, alpha=0.3),
    legend_background=pn.element_rect(fill="white", color=None),
    legend_key=pn.element_rect(fill="white", color=None),
    legend_key_size=18,
    plot_margin=0.01,
    figure_size=(5, 3.5),
    axis_title=element_text(size=12),
    axis_text=element_text(size=10),
    panel_spacing=0.05
)

PALETTE = [
    "#8F44FFFF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF",
    "#80796BFF", "#FFC107FF", "#00C49AFF", "#FF7043FF", "#003366FF",
    "#66BB6AFF", "#BA68C8FF", "#8B0000FF", "#556B2FFF", "#FFD700FF",
    "#40E0D0FF", "#E6E6FAFF", "#800000FF", "#A0522DFF"
]

def nonlinear_shrinkage(X):
    """
    Computes the Nonlinear Shrinkage estimator using the Direct Kernel method.
    """
    T, N = X.shape
    S = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(S)
    
    # Sort eigenvalues
    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # 1. Estimate the Stieltjes transform m(z)
    h = T**(-0.35) 
    lambda_i = evals
    z = lambda_i + 1j * h
    
    diff = evals.reshape(1, -1) - z.reshape(-1, 1) # (N, N)
    m_hat = np.mean(1 / diff, axis=1)
    
    # 2. Apply the nonlinear shrinkage formula
    c = N / T
    denom = np.abs(1 - c - c * lambda_i * m_hat)**2
    d_star = lambda_i / denom
    
    # 3. Reconstruct the matrix
    Sigma_nonlinear = evecs @ np.diag(d_star) @ evecs.T
    
    return Sigma_nonlinear, d_star

# Generate data for visualization
N_viz = 500
T_viz = 1000 # q = 2
X_viz = np.random.normal(0, 1, (T_viz, N_viz))
_, d_star_viz = nonlinear_shrinkage(X_viz)
S_viz = np.cov(X_viz, rowvar=False)
evals_viz = np.linalg.eigvalsh(S_viz)

df_shrinkage = pd.DataFrame({
    'Sample Eigenvalue': evals_viz,
    'Shrunk Eigenvalue': d_star_viz
})

# Theoretical 45-degree line (No Shrinkage)
line_df = pd.DataFrame({'x': [0, max(evals_viz)], 'y': [0, max(evals_viz)]})

plot_func = (
    ggplot(df_shrinkage, aes(x='Sample Eigenvalue', y='Shrunk Eigenvalue'))
    + geom_point(color=PALETTE[0], alpha=0.5, size=1.5)
    + geom_line(aes(x='x', y='y'), data=line_df, linetype='dashed', color='black', size=1)
    + labs(
        title="Nonlinear Shrinkage Function",
        x="Sample Eigenvalue",
        y="Shrunk Eigenvalue"
    )
    + THEME_ACADEMIC
)

# Save the plot
plot_func.save("image.png", dpi=300, width=5, height=3.5)
print("Thumbnail saved as image.png")

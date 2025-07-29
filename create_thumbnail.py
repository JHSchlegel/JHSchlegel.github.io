import numpy as np
import matplotlib.pyplot as plt

# Set style and color palette
plt.style.use('seaborn-v0_8-whitegrid')

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#f0f0f0')

# Time and SDE parameters
T = 1.0
N = 200
dt = T / N
t = np.linspace(0, T, N + 1)
mu = 0.1
sigma = 0.4

# Generate Wiener process
dW = np.random.normal(0, np.sqrt(dt), (10, N))
W = np.cumsum(dW, axis=1)
W = np.hstack([np.zeros((10, 1)), W])

# Solve SDE
X0 = 100
X = X0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Plot SDE paths
for i in range(10):
    ax.plot(t, X[i], lw=2, alpha=0.7)

# Add title and labels
ax.set_title('Stochastic Differential Equation Paths', fontsize=24, fontweight='bold', color='#333333')
ax.set_xlabel('Time', fontsize=16, color='#555555')
ax.set_ylabel('Asset Price', fontsize=16, color='#555555')

# Customize grid and ticks
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12, colors='#555555')

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# Add a subtle background gradient
ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], facecolor='white', alpha=0.5, zorder=-1)

# Save the thumbnail
plt.savefig('posts/03-04-2025_intro_to_sde/img/sde_thumbnail.png', dpi=150, bbox_inches='tight')

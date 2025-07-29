import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean and modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep") # A professional color palette

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

# --- Central SDE Concept: Stochastic Path ---
np.random.seed(42)
T = 1.0
N_steps = 500
dt = T / N_steps
t = np.linspace(0, T, N_steps + 1)

# Simulate multiple stochastic paths (e.g., Brownian motion with drift)
mu_sde = 0.5 # drift
sigma_sde = 0.8 # diffusion
n_paths_sde = 50 # Increased number of paths

for _ in range(n_paths_sde):
    dW = np.random.randn(N_steps) * np.sqrt(dt)
    X = np.zeros(N_steps + 1)
    X[0] = 0.0 # Starting point
    for i in range(N_steps):
        X[i+1] = X[i] + mu_sde * dt + sigma_sde * dW[i]
    ax.plot(t, X, color='#1f77b4', linewidth=0.8, alpha=0.3)

# Add a single prominent stochastic path for clarity
dW_prominent = np.random.randn(N_steps) * np.sqrt(dt)
X_prominent = np.zeros(N_steps + 1)
X_prominent[0] = 0.0
for i in range(N_steps):
    X_prominent[i+1] = X_prominent[i] + mu_sde * dt + sigma_sde * dW_prominent[i]
ax.plot(t, X_prominent, color='#1f77b4', linewidth=2.5, alpha=0.8, label='Stochastic Path')

# Add a smooth trend line (conceptual drift)
ax.plot(t, mu_sde * t, color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.7, label='Deterministic Trend')

# --- Annotations and Text ---
ax.text(0.5, 0.9, 'Stochastic Differential Equations', 
        horizontalalignment='center', verticalalignment='center', 
        transform=ax.transAxes, fontsize=18, fontweight='bold', color='#333333')

ax.text(0.5, 0.8, r'$dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t$', 
        horizontalalignment='center', verticalalignment='center', 
        transform=ax.transAxes, fontsize=16, color='#555555')

# --- Aesthetics and Layout ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.tick_params(axis='both', which='both', length=0) # Remove ticks
ax.set_frame_on(False) # Remove frame
ax.grid(False)

plt.tight_layout()

# Save thumbnail
plt.savefig('/home/janhsc/Documents/projects/JHSchlegel.github.io/posts/03-04-2025_intro_to_sde/img/sde_thumbnail.png', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

plt.close(fig) # Close the figure to free memory
print("New SDE infographic thumbnail generated successfully!")
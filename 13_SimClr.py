import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# 1. Setup and Data Generation
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

generate0 = np.random.randn(300, 2) + np.array([0, 0])
generate1 = np.random.randn(300, 2) + np.array([4, 4])
generate2 = np.random.randn(300, 2) + np.array([0, 4])
X = np.vstack((generate0, generate1, generate2))

# Labels for the linear probe
y = np.array([0]*300 + [1]*300 + [2]*300)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create two views with noise
view1_np = X + 0.3 * np.random.randn(900, 2)
view2_np = X + 0.3 * np.random.randn(900, 2)

view1 = torch.tensor(view1_np, dtype=torch.float32)
view2 = torch.tensor(view2_np, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

# ==========================================
# 2. SimCLR Model Definition
# ==========================================
class SimClr(nn.Module):
    def __init__(self, encoder_input_dim=2, encoder_output_dim=64, projection_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, encoder_output_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        z = F.normalize(z, p=2, dim=-1)
        return h, z

model = SimClr()
tau = 0.01
optimizer_simclr = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 3. Train SimCLR (Unsupervised)
# ==========================================
print("=== Phase 1: Training SimCLR ===")
for epoch in range(1000):
    optimizer_simclr.zero_grad()
    
    _, z1 = model(view1)
    _, z2 = model(view2)
    
    z_all = torch.cat([z1, z2], dim=0)
    N = z1.shape[0]
    
    sim_matrix = (z_all @ z_all.T) / tau
    mask = torch.eye(2 * N, dtype=torch.bool, device=z_all.device)
    sim_matrix_no_self = sim_matrix.masked_fill(mask, -float('inf'))
    
    sim_i = torch.diagonal(sim_matrix, offset=N)
    sim_j = torch.diagonal(sim_matrix, offset=-N)
    pos_sim = torch.cat([sim_i, sim_j], dim=0)
    
    denominator = torch.logsumexp(sim_matrix_no_self, dim=1)
    loss = -torch.mean(pos_sim - denominator)
    
    loss.backward()
    optimizer_simclr.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | SimCLR Loss: {loss.item():.4f}")

# ==========================================
# 4. Linear Probe (Supervised)
# ==========================================
print("\n=== Phase 2: Linear Probe Training ===")

# IMPORTANT: Set model to eval so we don't update SimCLR weights
model.eval()
with torch.no_grad():
    # Get the FINAL trained representations for the original clean data
    h_tensor, _ = model(X_tensor)

# Define Probe
probe = nn.Linear(64, 3)
# FIX: Assign optimizer to a variable
probe_optimizer = optim.Adam(probe.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train the probe
for epoch in range(100): # Increased epochs slightly for stability
    probe_optimizer.zero_grad()
    
    predictions = probe(h_tensor)
    loss = criterion(predictions, y_tensor)
    
    loss.backward()
    probe_optimizer.step()
    
    if epoch % 20 == 0:
        acc = (torch.argmax(predictions, dim=1) == y_tensor).float().mean().item()
        print(f"Probe Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

# Final Evaluation
with torch.no_grad():
    final_preds = probe(h_tensor)
    final_acc = (torch.argmax(final_preds, dim=1) == y_tensor).float().mean().item()

print(f"\n🚀 Final Linear Probe Accuracy: {final_acc * 100:.2f}%")

# ==========================================
# 5. Visualization
# ==========================================
h_np = h_tensor.numpy()
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = y == i
    plt.scatter(h_np[mask, 0], h_np[mask, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='w', s=40)

plt.title(f"Linear Probe Result: {final_acc*100:.1f}% Accuracy\nEncoder Representations", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
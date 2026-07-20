import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from scipy.optimize import root
import datetime
from matplotlib import colors

plt.rcParams.update({'font.size': 12})

# Constants
pKL = 6.3  # pKa of the lipids
pKRNA = 1  # pKa of RNA
vL = 3  # lipid volume [nm^3]
rhoL = 1 / vL  # max lipid charge density [e0/nm^3]
N_mRNA = 1929  # degree of polymarization of mRNA
vRNA = np.pi * N_mRNA * 0.34 * 1.2**2  # RNA volume [nm^3]
rhoRNA = -1 * N_mRNA / vRNA  # max RNA charge density [e0/nm^3]
fw = 0.2  # fraction of volume in LNPs occupied by water

def FRR_to_water_fraction(FRR):

    water_fraction = FRR / (FRR + 1)
    return water_fraction

def water_fraction_to_FRR(water_frac):
    if water_frac >= 1.0:
        return float('inf')
    return water_frac / (1 - water_frac)

def epsilon_from_FRR(FRR):

    water_frac = FRR_to_water_fraction(FRR)
    epsilon_water = 78.5
    epsilon_ethanol = 24.3
    return water_frac * epsilon_water + (1 - water_frac) * epsilon_ethanol

def lb_from_epsilon(epsilon_r):
    return 0.71 * (78.5 / epsilon_r)

def psiDH(rho, R, csalt, lb):
    lD = 1 / np.sqrt(8 * np.pi * lb * 0.60223 * csalt)  # Debye length [nm]
    return (rho * lb * 4 * np.pi * R**2) / (3 * (1 + R * np.sqrt(8 * np.pi * lb * 0.60223 * csalt)))

def rho(psi, XL, XRNA, csalt, ttdpHL, ttdpHRNA, lb):
    term1 = (ttdpHL * np.exp(-psi)) / (1 + ttdpHL * np.exp(-psi)) * rhoL * XL * (1 - fw)
    term2 = (ttdpHRNA * np.exp(psi)) / (1 + ttdpHRNA * np.exp(psi)) * rhoRNA * XRNA * (1 - fw)
    term3 = -2 * fw * csalt * 0.60223 * np.sinh(psi)
    return term1 + term2 + term3

def rhoCR(R, XL, XRNA, csalt, ttdpHL, ttdpHRNA, lb):
    """Find charge regulation density using root finding"""
    def equation(rhox):
        return rho(psiDH(rhox, R, csalt, lb), XL, XRNA, csalt, ttdpHL, ttdpHRNA, lb) - rhox
    sol = root(equation, 0.1)
    return sol.x[0] if sol.success else None

# Generate training data
print("\n[1] Generating training data...")
R_values = np.linspace(1, 30, 20)
XRNAx_values = np.linspace(0, 1, 20)
csalt_values = np.linspace(0.005, 0.15, 10)
FRR_values = np.linspace(1, 9, 9)  # FRR from 1:1 to 9:1 (water:ethanol)

X_train = []
y_train = []

total_combinations = len(R_values) * len(XRNAx_values) * len(csalt_values) * len(FRR_values)

count = 0
failed_count = 0

for FRR in FRR_values:
    epsilon_r = epsilon_from_FRR(FRR)
    lb = lb_from_epsilon(epsilon_r)
    
    for csalt in csalt_values:
        for R in R_values:
            for XRNAx in XRNAx_values:
                count += 1
                if count % 5000 == 0:
                    print(f"  Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)")
                
                ttdpHL = 10**(pKL - 4)
                ttdpHRNA = 10**(4 - pKRNA)
                XL_i = 5 / 6 * (1 - XRNAx)
                XRNA_i = XRNAx
                
                try:
                    rhox = rhoCR(R, XL_i, XRNA_i, csalt, ttdpHL, ttdpHRNA, lb)
                    if rhox is not None:
                        psi = 25 * psiDH(rhox, R, csalt, lb)
                        X_train.append([R, XRNAx, csalt, FRR])
                        y_train.append(psi)
                    else:
                        failed_count += 1
                except:
                    failed_count += 1

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train-validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Initialize MLPRegressor
print("\n[3] Initializing Neural Network...")
model = MLPRegressor(
    hidden_layer_sizes=(20, 20, 20),
    activation='tanh',
    learning_rate_init=0.001,
    max_iter=1,  # We will train manually
    random_state=42,
    alpha=0.001  # L2 regularization
)

print(f"  Activation: tanh")
print(f"  Learning rate: 0.001")

# Training loop with manual early stopping
print("\n[4] Training model...")
train_loss = []
val_loss = []
train_r2 = []
val_r2 = []
iteration_time = []

best_val_loss = float('inf')
patience = 20
patience_counter = 0

for i in range(2000):
    start_time = datetime.datetime.now()
    model.partial_fit(X_train_split, y_train_split)
    end_time = datetime.datetime.now()
    iteration_time.append((end_time - start_time).total_seconds())

    y_train_pred = model.predict(X_train_split)
    y_val_pred = model.predict(X_val_split)
    
    train_mse = mean_squared_error(y_train_split, y_train_pred)
    val_mse = mean_squared_error(y_val_split, y_val_pred)
    
    train_r2_score = r2_score(y_train_split, y_train_pred)
    val_r2_score = r2_score(y_val_split, y_val_pred)

    train_loss.append(train_mse)
    val_loss.append(val_mse)
    train_r2.append(train_r2_score)
    val_r2.append(val_r2_score)

    if i % 50 == 0:
        print(f"  Iter {i:4d} | Train MSE: {train_mse:8.2f} | Val MSE: {val_mse:8.2f} | "
              f"Train R²: {train_r2_score:.4f} | Val R²: {val_r2_score:.4f}")
    
    # Early stopping with patience
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\n[Early Stop] Iteration {i}, validation loss hasn't improved for {patience} iterations.")
        break

print(f"  Final Train MSE: {train_loss[-1]:.2f}")
print(f"  Final Val MSE: {val_loss[-1]:.2f}")
print(f"  Final Train R²: {train_r2[-1]:.4f}")
print(f"  Final Val R²: {val_r2[-1]:.4f}")
print(f"  Total iterations: {len(train_loss)}")
print(f"  Average time per iteration: {np.mean(iteration_time):.4f} seconds")

# Plot training and validation metrics
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(train_loss, label='Training Loss', linewidth=2)
axes[0].plot(val_loss, label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].set_title('Training vs Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# R² plot
axes[1].plot(train_r2, label='Training R²', linewidth=2)
axes[1].plot(val_r2, label='Validation R²', linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Training vs Validation R² Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics_FRR.png', dpi=300, bbox_inches='tight')
plt.show()

# Prediction vs Actual plot
print("\n[5] Generating prediction plots...")
y_train_pred_final = model.predict(X_train_split)
y_val_pred_final = model.predict(X_val_split)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data
axes[0].scatter(y_train_split, y_train_pred_final, alpha=0.3, s=10)
axes[0].plot([y_train_split.min(), y_train_split.max()], 
             [y_train_split.min(), y_train_split.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual ψ [mV]')
axes[0].set_ylabel('Predicted ψ [mV]')
axes[0].set_title(f'Training Set\n(R² = {train_r2[-1]:.4f}, MSE = {train_loss[-1]:.2f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation data
axes[1].scatter(y_val_split, y_val_pred_final, alpha=0.3, s=10, color='orange')
axes[1].plot([y_val_split.min(), y_val_split.max()], 
             [y_val_split.min(), y_val_split.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual ψ [mV]')
axes[1].set_ylabel('Predicted ψ [mV]')
axes[1].set_title(f'Validation Set\n(R² = {val_r2[-1]:.4f}, MSE = {val_loss[-1]:.2f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_accuracy_FRR.png', dpi=300, bbox_inches='tight')
plt.show()

# 3D Surface plots for different FRR values
print("\n[6] Generating 3D surface plots for different FRR values...")
R_test = np.linspace(1, 30, 30)
XRNAx_test = np.linspace(0, 1, 30)
R_grid, XRNAx_grid = np.meshgrid(R_test, XRNAx_test)

csalt_fixed = 0.025
FRR_to_plot = [9, 5, 3, 1]  # 9:1, 5:1, 3:1, 1:1

fig = plt.figure(figsize=(24, 6))

psi_min = -150
psi_max = 150
norm = colors.Normalize(vmin=psi_min, vmax=psi_max)

for idx, FRR in enumerate(FRR_to_plot):
    ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
    
    psi_values_nn = np.zeros(R_grid.shape)
    
    for i in range(R_grid.shape[0]):
        for j in range(R_grid.shape[1]):
            R = R_grid[i, j]
            XRNAx = XRNAx_grid[i, j]
            
            # Predict using neural network
            X_test = np.array([[R, XRNAx, csalt_fixed, FRR]])
            psi_values_nn[i, j] = model.predict(X_test)[0]
    
    # Plot surface
    surf = ax.plot_surface(R_grid, XRNAx_grid, psi_values_nn, 
                          cmap='twilight_r', norm=norm, edgecolor='none', alpha=0.9)
    
    # Labels
    ax.set_ylabel(r'$\phi_{RNA}$', labelpad=10)
    ax.set_xlabel('R [nm]', labelpad=10)
    ax.set_zlabel(r'$\psi$ [mV]', labelpad=10)
    
    # Title with FRR information
    epsilon_r = epsilon_from_FRR(FRR)
    water_frac = FRR_to_water_fraction(FRR)
    ax.set_title(f'FRR = {FRR}:1 (H₂O:EtOH)\n' +
                f'{water_frac*100:.0f}% water, ε = {epsilon_r:.1f}',
                fontsize=14, pad=20)
    
    ax.view_init(elev=20, azim=45)

# Add colorbar
fig.subplots_adjust(right=0.93)
cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
cbar = fig.colorbar(surf, cax=cbar_ax)
cbar.set_label(r'Surface Potential $\psi$ [mV]', rotation=270, labelpad=30, fontsize=14)
plt.show()

# Save the model
model.random_state = None
joblib.dump(model, 'neural_network_model_FRR.pkl', protocol=4)
print("[✔] Model saved as 'neural_network_model_FRR.pkl'")

# Save model info
model_info = {
    'input_features': ['R [nm]', 'φ_RNA', 'c_salt [M]', 'FRR (water:ethanol)'],
    'output': 'ψ [mV]',
    'FRR_range': [1, 9],
    'FRR_interpretation': 'FRR=1 means 1:1 (50% water), FRR=9 means 9:1 (90% water)',
    'architecture': '4 → 20 → 20 → 20 → 1',
    'activation': 'tanh',
    'train_samples': len(X_train_split),
    'val_samples': len(X_val_split),
    'final_train_mse': float(train_loss[-1]),
    'final_val_mse': float(val_loss[-1]),
    'final_train_r2': float(train_r2[-1]),
    'final_val_r2': float(val_r2[-1]),
    'iterations': len(train_loss)
}

joblib.dump(model_info, 'model_info_FRR.pkl')
print(" Model info saved as 'model_info_FRR.pkl'")

# Create FRR conversion table
print("\n" + "="*80)
print("FRR CONVERSION TABLE")
print("="*80)
print(f"{'FRR':<10} {'Ratio':<15} {'Water %':<12} {'Ethanol %':<12} {'ε':<10} {'lb [nm]':<10}")
print("-"*80)
for FRR in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    water_frac = FRR_to_water_fraction(FRR)
    ethanol_frac = 1 - water_frac
    epsilon = epsilon_from_FRR(FRR)
    lb = lb_from_epsilon(epsilon)
    print(f"{FRR:<10} {FRR}:1{'':<11} {water_frac*100:<12.1f} {ethanol_frac*100:<12.1f} {epsilon:<10.2f} {lb:<10.3f}")
print("="*80)

# Test the model with example predictions
print("\n[9] Example predictions:")
print("-" * 90)
print(f"{'R [nm]':<10} {'φ_RNA':<10} {'c_salt [M]':<12} {'FRR':<10} {'Water %':<12} {'Predicted ψ [mV]':<20}")
print("-" * 90)

test_cases = [
    [15, 0.5, 0.025, 9],
    [15, 0.5, 0.025, 5],
    [15, 0.5, 0.025, 3],
    [15, 0.5, 0.025, 1],
    [5, 0.3, 0.05, 9],
    [20, 0.7, 0.1, 5],
]

for case in test_cases:
    pred = model.predict([case])[0]
    water_pct = FRR_to_water_fraction(case[3]) * 100
    print(f"{case[0]:<10.1f} {case[1]:<10.2f} {case[2]:<12.3f} {case[3]}:1{'':<6} {water_pct:<12.0f} {pred:<20.2f}")

print("-" * 90)

# Create a summary report
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Input Features: {', '.join(model_info['input_features'])}")
print(f"Output: {model_info['output']}")
print(f"FRR Range: {model_info['FRR_range'][0]} to {model_info['FRR_range'][1]}")
print(f"Network Architecture: {model_info['architecture']}")
print(f"Total Training Samples: {model_info['train_samples']}")
print(f"Total Validation Samples: {model_info['val_samples']}")
print(f"Training Iterations: {model_info['iterations']}")
print(f"\nFinal Metrics:")
print(f"  Training MSE: {model_info['final_train_mse']:.2f}")
print(f"  Validation MSE: {model_info['final_val_mse']:.2f}")
print(f"  Training R²: {model_info['final_train_r2']:.4f}")
print(f"  Validation R²: {model_info['final_val_r2']:.4f}")
print("="*80)
print("\n All tasks completed successfully!")
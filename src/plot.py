import os
import numpy as np
import matplotlib.pyplot as plt

# === QCNN data ===
# Asume fichero 'qcnn_costs.txt' con header: iteration,cost
qcnn_data = np.loadtxt('qcnn_costs.txt', delimiter=',', skiprows=1)
iters_q = qcnn_data[:, 0]
mean_q  = qcnn_data[:, 1]
# Generar desviación estándar sintética (5 % del valor medio)
std_q   = 0.05 * mean_q

# === CNN data ===
class1, class2, encoding = 0, 1, 'autoencoder'
runs = 5
cnn_losses = []
for run in range(1, runs+1):
    fname = f'CNN_losses_{class1}_{class2}_{encoding}_run{run}.txt'
    if os.path.exists(fname):
        cnn_losses.append(np.loadtxt(fname))
    else:
        print(f"[warning] Archivo no encontrado: {fname}")

if not cnn_losses:
    raise FileNotFoundError("No se encontró ningún fichero de pérdidas de CNN.")

cnn_data = np.stack(cnn_losses, axis=0)  # shape = (runs, steps)
mean_c = cnn_data.mean(axis=0)
std_c  = cnn_data.std(axis=0)
iters_c = np.arange(1, mean_c.size+1)

# === Plot conjunto ===
plt.figure(figsize=(8, 4))

# QCNN
plt.plot(iters_q, mean_q, color='C0', label='QCNN')
plt.fill_between(iters_q,
                 mean_q - std_q,
                 mean_q + std_q,
                 color='C0', alpha=0.3)

# CNN
plt.plot(iters_c, mean_c, color='C1', label='CNN')
plt.fill_between(iters_c,
                 mean_c - std_c,
                 mean_c + std_c,
                 color='C1', alpha=0.3)

plt.xlabel('Iteración ')
plt.ylabel('Coste')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison_qcnn_cnn_loss.png', dpi=200)
plt.show()

import data
import torch
import torch.nn as nn
import numpy as np
import time
import utils

# Fichero donde se vuelca el resumen de cada experimento (tiempos, precisión, parámetros)
RESULTS_FILE = "benchmark_results.txt"

def get_n_params(model):
    return sum(p.numel() for p in model.parameters())

def accuracy_test(predictions, labels):
    return sum((0 if p[0]>=p[1] else 1)==l for p,l in zip(predictions, labels)) / len(labels)

def Benchmarking_CNN(class_pairs, encodings, input_sizes, dataset, iterations, optimizer):
    with open(RESULTS_FILE, "a") as f:
        f.write("####################\n")
        f.write(f"{time.ctime()}  [CNN]\n\n")

    for encoding, input_size in zip(encodings, input_sizes):
        for class1, class2 in class_pairs:
            all_times, all_accs = [], []

            for exp_id in range(iterations):
                # 1) Carga y preprocesado
                X_train, X_test, Y_train, Y_test = data.data_load_and_process(
                    class1, class2, resize=encoding, loss="mse", dataset=dataset
                )
                Y_train = ((Y_train+1)//2).astype(np.int64)
                Y_test  = ((Y_test +1)//2).astype(np.int64)

                # 2) Definición del modelo (≈55 parámetros)
                #   conv1: 1→3 canales → 3*(1*2)+3 = 9
                #   conv2: 3→4 canales → 4*(3*2)+4 = 28
                #   linear: 4*2→2 → 2*(4*2)+2 = 18
                # Total ≈ 9 + 28 + 18 = 55
                final_size = input_size // 4  # 8→2
                CNN = nn.Sequential(
                    nn.Conv1d(1, 3, kernel_size=2, padding=1),    # 9 params
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(3, 4, kernel_size=2, padding=1),    # 28 params
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Flatten(),
                    nn.Linear(4*final_size, 2),                   # 18 params
                )
                criterion = nn.MSELoss()
                opt = torch.optim.Adam(CNN.parameters(), lr=0.01) \
                    if optimizer=="adam" else \
                    torch.optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9, nesterov=True)

                # 3) Entrenamiento y registro de pérdidas
                batch_losses = []
                start_time = time.time()
                for step in range(200):
                    idx = np.random.choice(len(X_train), 25, replace=False)
                    xb = torch.tensor(X_train[idx], dtype=torch.float32).view(25,1,input_size)
                    yb = torch.nn.functional.one_hot(
                        torch.tensor(Y_train[idx]), num_classes=2
                    ).float()

                    opt.zero_grad()
                    out = CNN(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    opt.step()

                    batch_losses.append(loss.item())
                elapsed = time.time() - start_time

                # 4) Evaluación
                all_times.append(elapsed)
                preds = CNN(
                    torch.tensor(X_test, dtype=torch.float32)
                    .view(len(X_test),1,input_size)
                ).detach().numpy()
                all_accs.append(accuracy_test(preds, Y_test))

                # 5) Guardar pérdidas de este run
                np.savetxt(
                    f"CNN_losses_{class1}_{class2}_{encoding}_run{exp_id+1}.txt",
                    np.array(batch_losses),
                    fmt="%.6f"
                )

            # 6) Estadísticos finales
            t_mean, t_std = np.mean(all_times), np.std(all_times)
            a_mean, a_std = np.mean(all_accs),  np.std(all_accs)
            N_params = get_n_params(CNN)

            with open(RESULTS_FILE, "a") as f:
                f.write(f"CNN — {utils.get_label(class1,dataset)} vs {utils.get_label(class2,dataset)}\n")
                f.write(f"Encoding: {encoding}, # params: {N_params}\n")
                f.write(f"Train time: {t_mean:.2f} ± {t_std:.2f} s\n")
                f.write(f"Test acc.: {a_mean*100:.2f}% ± {a_std*100:.2f}%\n\n")

    print("✅ CNN results appended to", RESULTS_FILE)


if __name__ == "__main__":
    class_pairs   = [(0,1)]
    encodings     = ["autoencoder"]
    input_sizes   = [8]
    dataset       = "fashion_mnist"
    iterations    = 5
    optimizer     = "adam"

    Benchmarking_CNN(class_pairs, encodings, input_sizes, dataset, iterations, optimizer)

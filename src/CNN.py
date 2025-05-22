import data
import torch
import torch.nn as nn
import numpy as np
import time
import utils
import matplotlib.pyplot as plt


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def accuracy_test(predictions, labels):
    acc = 0
    for (p, l) in zip(predictions, labels):
        pred = 0 if p[0] >= p[1] else 1
        if pred == l:
            acc += 1
    return acc / len(labels)


def Benchmarking_CNN(class_pairs, Encodings, Encodings_size, dataset, iterations, optimizer, n_feature=2):
    results_file = "results_CNN.txt"

    with open(results_file, "a") as f:
        f.write("#################################\n")
        f.write(f"{time.ctime()}\n\n")

        for encoding, input_size in zip(Encodings, Encodings_size):
            for class1, class2 in class_pairs:
                training_times = []
                test_accuracies = []
                all_losses = []

                for it in range(iterations):
                    X_train, X_test, Y_train, Y_test = data.data_load_and_process(
                        class1, class2, resize=encoding, dataset=dataset
                    )
                    Y_train = ((Y_train + 1) // 2).astype(np.int64)
                    Y_test = ((Y_test + 1) // 2).astype(np.int64)

                    final_layer_size = input_size // 4
                    CNN = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2),
                        nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2),
                        nn.Flatten(),
                        nn.Linear(n_feature * final_layer_size, 2)
                    )

                    criterion = nn.MSELoss()
                    if optimizer == 'adam':
                        opt = torch.optim.Adam(CNN.parameters(), lr=0.01)
                    else:
                        opt = torch.optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9, nesterov=True)

                    batch_size = 25
                    steps = 200
                    losses = []

                    start_time = time.time()
                    for step in range(steps):
                        idx = np.random.choice(len(X_train), batch_size, replace=False)
                        X_batch = torch.tensor(X_train[idx], dtype=torch.float32).view(batch_size, 1, input_size)
                        Y_batch = torch.tensor(Y_train[idx], dtype=torch.long)
                        Y_batch = torch.nn.functional.one_hot(Y_batch, num_classes=2).float()

                        opt.zero_grad()
                        outputs = CNN(X_batch)
                        loss = criterion(outputs, Y_batch)
                        loss.backward()
                        opt.step()

                        losses.append(loss.item())
                    end_time = time.time()

                    training_times.append(end_time - start_time)
                    all_losses.append(losses)

                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(len(X_test), 1, input_size)
                    predictions = CNN(X_test_tensor).detach().numpy()
                    acc = accuracy_test(predictions, Y_test)
                    test_accuracies.append(acc)

                    print(f"Iteration {it+1}: Time Training: {end_time - start_time:.2f}s, Test accuracy: {acc * 100:.2f}%")

                # compute summary stats
                avg_time = np.mean(training_times)
                std_time = np.std(training_times)
                avg_acc = np.mean(test_accuracies)
                std_acc = np.std(test_accuracies)
                N_params = get_n_params(CNN)

                # write results
                f.write(f"Parameters:\n{utils.get_label(class1,dataset)} vs {utils.get_label(class2,dataset)}\n")
                f.write(f"Encoding (Resize): {encoding}\n")
                f.write(f"Number_parameters: {N_params}\n")
                f.write(f"Training Time: {avg_time:.2f} ± {std_time:.2f} s\n")
                f.write(f"Test accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n")

                print(f"\nSummary for this configuration:")
                print(f"Average Training Time: {avg_time:.2f} ± {std_time:.2f} s")
                print(f"Average Test Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
                print(f"Number of Parameters: {N_params}\n")

                # evolution of loss: mean and std
                all_losses_np = np.array(all_losses)  # shape (iterations, steps)
                mean_loss = all_losses_np.mean(axis=0)
                std_loss = all_losses_np.std(axis=0)

                # save to files
                np.savetxt(f"CNN_loss_mean_{class1}_{class2}_{encoding}.txt", mean_loss)
                np.savetxt(f"CNN_loss_std_{class1}_{class2}_{encoding}.txt", std_loss)

                # plot with shaded std band
                plt.figure()
                x = np.arange(len(mean_loss))
                plt.plot(x, mean_loss, label="MSE loss mean")
                plt.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3, label="±1 std")
                plt.xlabel("Training Step")
                plt.ylabel("Loss (MSE)")
                plt.title(f"CNN Loss Evolution ({utils.get_label(class1,dataset)} vs {utils.get_label(class2,dataset)})")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"CNN_loss_curve_{class1}_{class2}_{encoding}.png")
                plt.show()

    print("✅ Results saved to", results_file)


# ==== PARÁMETROS DEL EXPERIMENTO ====
class_pairs = [(0, 1)]
Encodings = ['autoencoder']
Encodings_size = [8]
dataset = 'fashion_mnist'

Benchmarking_CNN(class_pairs, Encodings, Encodings_size, dataset,
                 iterations=5, optimizer='adam', n_feature=3)

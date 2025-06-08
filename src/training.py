# training.py

import pennylane.numpy as pnp
import numpy as np 
import pennylane as qml
import autograd.numpy as anp 

import utils
import circuit

def circuit_training(X_train, Y_train, unitary, embedding, loss):
    num_params = utils.param_num(unitary)
    params = pnp.array(np.random.uniform(0, 2*np.pi, size=num_params), requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=0.01)

    # Prepara archivo para guardar costes
    cost_log = open("qcnn_costs.txt", "w")
    cost_log.write("iteration,cost\n")

    for it in range(200):
        batch_indices = np.random.randint(0, len(X_train), size=25)
        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]
        
        params, cost_new = opt.step_and_cost(
            lambda p: cost(p, X_batch, Y_batch, unitary, embedding, loss),
            params
        )
        
        # Imprimir en pantalla
        print(f"Iteration {it}: Cost = {cost_new:.4f}")
        # Guardar en archivo
        cost_log.write(f"{it},{cost_new:.6f}\n")
    
    cost_log.close()
    return params

def cost(params, X, Y, unitary, embedding, loss):
    predictions = pnp.array([circuit.QCNN(x, params, unitary, embedding, loss) for x in X])

    if loss == "mse":
        return square_loss(Y, predictions)
    else:
        return cross_entropy(Y, predictions)

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    return loss / len(labels)

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * anp.log(p[l]) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -loss

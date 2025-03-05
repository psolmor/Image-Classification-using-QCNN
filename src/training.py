import circuit as cir
import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

def square_loss(labels,predictions):
    loss=0
    for l,p in zip(labels,predictions):
        loss+=(l-p)**2
    loss=loss/len(labels)
    return loss

def cost(params, X, Y, U, U_params):
  
    predictions = [cir.QCNN(x, params, U, U_params) for x in X]
    loss = square_loss(Y, predictions)

    return loss


steps=200
learning_rate=0.01
batch_size=25

def circuit_training(X_train, Y_train, U, U_params, circuit):

    total_params = U_params * 3 + 2 * 3


    params = pnp.array(pnp.random.randn(total_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []

    for it in range(steps):
        batch_index = pnp.random.randint(0, len(X_train), (batch_size))

        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params),params)
        loss_history.append(cost_new)

        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)

    return loss_history, params
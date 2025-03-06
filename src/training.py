import circuit as cir
import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cost(params, X, Y):
  
    predictions = [cir.QCNN(x,params) for x in X]
    loss = square_loss(Y, predictions)

    return loss


steps=200
learning_rate=0.001
batch_size=25

def circuit_training(X_train, Y_train):

    total_params = 12


    params = pnp.array(pnp.random.randn(total_params), requires_grad=True)
    #params = pnp.full(total_params, pnp.pi / 2, requires_grad=True)

    #opt = qml.AdamOptimizer(stepsize=learning_rate)
    opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

    
    def cost_fn(params):
        batch_index = pnp.random.randint(0, len(X_train), batch_size)
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        return cost(params, X_batch, Y_batch)

    
    loss_history = []

    for it in range(steps):

        #batch_index = pnp.random.randint(0, len(X_train), batch_size)
        #X_batch = [X_train[i] for i in batch_index]
        #Y_batch = [Y_train[i] for i in batch_index]
        
        params, cost_new = opt.step_and_cost(cost_fn,params)
        loss_history.append(cost_new)

        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)

    return loss_history, params
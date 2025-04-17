import pennylane.numpy as pnp
import numpy as np 
import pennylane as qml

import utils
import circuit

def circuit_training(X_train, Y_train, unitary, embedding):

    num_params = utils.param_num(unitary)
    params = pnp.array(np.random.uniform(0, 2*np.pi, size=num_params),requires_grad=True)
    
    opt = qml.NesterovMomentumOptimizer(stepsize=0.01)

    for it in range(200):
    
        batch_indices = np.random.randint(0, len(X_train), size=25)
        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]
        
        params, cost_new = opt.step_and_cost(
            lambda p: cost(p, X_batch, Y_batch, unitary, embedding),
            params
        )
        
        if it % 10 == 0:
            print(f"Iteration {it}: Cost = {cost_new:.4f}")
    
    return params

def cost(params, X, Y, unitary, embedding):
    predictions = pnp.array([circuit.QCNN(x, params, unitary, embedding) for x in X])
    return square_loss(Y, predictions)

def square_loss(labels, predictions):
    loss = pnp.mean((labels - predictions) ** 2)
    return loss
import circuit
import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

#Loss Functions

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

#Cost function
def cost(params, X, Y,unitary,embedding):
  
    predictions = [circuit.QCNN(x,params,unitary,embedding) for x in X]
    loss = square_loss(Y, predictions)

    return loss



#Training parameters
steps=200
learning_rate=0.01
batch_size=25

#Function in charge to optimize the circuit
def circuit_training(X_train, Y_train,params_num,unitary,embedding):

    total_params = params_num


    if unitary=="TTN":
        total_params=12
    elif unitary=="CONV":
        total_params=15
       
    params = pnp.array(pnp.random.randn(total_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
       
    def cost_fn(params):
        batch_index = pnp.random.randint(0, len(X_train), batch_size)
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        return cost(params, X_batch, Y_batch,unitary,embedding)


    for it in range(steps):
        params, cost_new = opt.step_and_cost(cost_fn,params)

        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)

    return  params
import numpy as np 

def accuracy_test(predictions, labels):
    acc = 0
    counter=0
    for l, p in zip(labels, predictions):
        if np.abs(l - p) < 1:
            acc = acc + 1
        if counter%10==0:
            print(f"Testing image: {counter}/{len(predictions)}")
    return acc / len(labels)


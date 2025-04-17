def param_num(unitary):
    """Calculate total parameters needed for the circuit"""

    if unitary == "CONV":
        return 57
    elif unitary == "TTN":
        return 45 
    else:
        raise ValueError("Unitary must be 'CONV' or 'TTN'")
    



    
def get_label(label, dataset):
    """
    Returns the description of a label for either MNIST or Fashion MNIST.
    
    Args:
        label (int): Label number (0-9).
        dataset (str): 'mnist' or 'fashion_mnist'.
        
    Returns:
        str: Description of the label.
    """
    if dataset == 'mnist':
        return f"MNIST: Digit {label}"
    elif dataset == 'fashion_mnist':
        fashion_labels = {
            0: "Fashion MNIST: T-shirt/top",
            1: "Fashion MNIST: Trouser",
            2: "Fashion MNIST: Pullover",
            3: "Fashion MNIST: Dress",
            4: "Fashion MNIST: Coat",
            5: "Fashion MNIST: Sandal",
            6: "Fashion MNIST: Shirt",
            7: "Fashion MNIST: Sneaker",
            8: "Fashion MNIST: Bag",
            9: "Fashion MNIST: Ankle boot"
        }
        return fashion_labels.get(label, "Fashion MNIST: Unknown label")
    else:
        return "Dataset not recognized (must be 'mnist' or 'fashion_mnist')"
import numpy as np

def efficiency_fcn(x, x0, A, k):
    """
    Calculate the (1-exponential decay of x).
    :param x: numpy array of the independent variable
    :param A: amplitude or initial value of the dependent variable
    :param k: decay constant
    :return: y, numpy array of the dependent variable
    """
    y =1 - A * np.exp(-k * (x-x0))
    return y

def relu(efficiency_fcn):
    """Applies the rectified linear unit (ReLU) function to the given efficiency function. t"""
    return np.maximum(0, efficiency_fcn)



    
from sklearn.neighbors import KernelDensity

def labels_to_indices(data, labels):
    labels = np.array(labels)
    data = np.array(data)
    label_to_indices = {} # a list of indices is provided for each label, ordering is from the most populated label to the smallest. Note that label_to_indices[i] gives all indices for the label 'i', NOT the i-th element!
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    return label_to_indices

def balance_classes(X, y, threshold=200, band = 0.1):
    """
    Modifies dataset by selecting only threshold=n samples randomly for classes with more than 
    threshold=n samples and KDE sampling new data for classes with less than threshold=n samples.
    
    todo: alternative sampling techniques
    
    Parameters:
        X (numpy.ndarray): input data array
        y (numpy.ndarray): array of labels corresponding to input data
        threshold (int): threshold value for the minimum number of samples per class
    
    Returns:
        tuple: a tuple containing the modified data array X_new and corresponding 
                label array y_new.
    """

    labs = labels_to_indices(X, y)
    new_X, new_y = [], []
    for lab in labs:
        if (len(labs[lab])) > threshold:
            idx = np.random.choice(np.where(y == lab)[0], size=threshold, replace=False)
            new_X.append(X[idx])
            new_y.append(y[idx])
        else:
            bandw_rel = X[labs[lab]].mean() * band
            # Kernel density estimator
            kde = KernelDensity(kernel='gaussian', bandwidth=bandw_rel).fit(X[y == lab])
            # Sample from the KDE to obtain more data
            new_samples = kde.sample(threshold - (len(labs[lab]))) # Convert to int
            new_X.append(X[labs[lab]])
            new_X.append(new_samples)
            new_y.append(y[labs[lab]])
            new_y.append(np.full((new_samples.shape[0],), lab))
    X_new = np.concatenate(new_X, axis=0)
    y_new = np.concatenate(new_y, axis=0)
    return X_new, y_new


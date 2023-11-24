import numpy as np

# Helper function to load the dataset
def load_dataset(path: str) -> (np.ndarray, np.ndarray):
    with np.load(path) as data:
        x, y = data["x"], data["y"]
        
        # Normalize the data
        x -= x.mean(axis=0)
        x /= x.std(axis=0)
        
    return x, y

# Load the dataset using the helper function
X, y = load_dataset("data/lux.npz")
print(X.shape[0], y.shape)

k = 500
s = X.shape[0] // k
indexes = [i * s for i in range(k+1)]
x_folds = [X[indexes[i] : indexes[i + 1]] for i in range(k)]
y_folds = [y[indexes[i] : indexes[i + 1]] for i in range(k)]

x_test = x_folds[:30] + x_folds[31]
#x_test = np.concatenate(x_test[:])
print(x_test)
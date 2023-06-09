import numpy as np
from sklearn.decomposition import IncrementalPCA


def make_tasks(X, y, n_tasks, n_spc):
    """ Arange data into binary classification tasks based on class label """

    TX = np.full((n_tasks, 2 * n_spc, X.shape[1]), np.nan, dtype=np.float32) 
    Ty = np.full((n_tasks, 2 * n_spc), -1, dtype=np.int8)

    for i in range(0, 2 * n_tasks, 2):

        idx = int(i / 2)

        TX[idx] = np.vstack((X[y == i][:n_spc], 
                             X[y == (i + 1)][:n_spc])) 

        Ty[idx] = np.concatenate((y[y == i][:n_spc], 
                                  y[y == (i + 1)][:n_spc]))

    return TX, Ty

    
class ART_IPCA:
    """ Model class in the style of scikit-learn """
    
    def __init__(self, n_components, sim_metric, rho):
        
        # Hyperparamaters
        self.n_components = n_components
        self.sim_metric = sim_metric  # Choose metric from sklearn.metrics.pairwise
        self.rho = rho
        
        # Model parameters
        self.W = None
        self.C = None
        self.N = None
        self.ipca = None
    
    
    def fit(self, TX_train, Ty_train):
        """ Train the model """

        n_tasks = TX_train.shape[0]

        # Num samples, dimensionality
        n = TX_train.shape[0] * TX_train.shape[1]
        d = TX_train.shape[2]

        # Set max number of weights to the number of samples to be safe
        max_nw = n

        # Number of weights learned thus far (index for W, C)
        nw = 0

        # List of weights, mapping from categories to weights, and number of matches per category
        self.W = np.full((max_nw, d), np.nan, dtype=np.float32)
        self.C = np.full(max_nw, -1, dtype=np.int32)
        self.N = np.full(max_nw, -1, dtype=np.int32)

        self.ipca = IncrementalPCA(self.n_components)

        for i in range(n_tasks):

            X_train = TX_train[i]
            y_train = Ty_train[i]

            # Fit on all data seen thus far, transform current task data
            self.ipca.partial_fit(TX_train[i])
            X_train_i = self.ipca.transform(X_train)

            for j in range(X_train.shape[0]):

                x = X_train[j]
                y = y_train[j]

                xr = X_train_i[j]

                # Init first weight with first sample (never before seen class)
                if j == 0:
                    self.W[nw] = x
                    self.C[nw] = y
                    self.N[nw] = 1
                    nw += 1
                    continue

                # Find max similarity across input and weights (rows of W) in reduced space!
                sim = self.sim_metric(xr.reshape(1, -1), self.ipca.transform(self.W[:nw])).ravel()

                # Maximum similarity score, winning category index, and label of winning category
                max_idx = np.argmax(sim)
                max_sim = sim[max_idx]
                max_lab = self.C[max_idx]

                # To update category weight:
                # Input is sufficiently similar to category AND category maps to correct class
                if (max_sim >= self.rho) and (max_lab == y):
                    self.N[max_idx] = self.N[max_idx] + 1
                    self.W[max_idx] = (1 / self.N[max_idx]) * x +  (1 - (1 / self.N[max_idx])) * self.W[max_idx]

                # Otherwise, create new category
                else:
                    self.W[nw] = x
                    self.C[nw] = y
                    self.N[nw] = 1
                    nw += 1

        self.W = self.W[~np.isnan(self.W).any(axis=1)]
        self.C = self.C[self.C != -1]
        self.N = self.N[self.N != -1]

        return None
    
    
    def predict(self, X):
        """ Make predictions """

        # Note: similarity comparison between 2D arrays
        sim = self.sim_metric(self.ipca.transform(X), self.ipca.transform(self.W))
        max_idx = np.argmax(sim, axis=1)
        y_pred = self.C[max_idx]

        return y_pred

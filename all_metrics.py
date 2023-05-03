import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


def make_tasks_train_test(X_train, y_train, X_test, y_test, n_train_spc, n_test_spc, n_tasks):
    """
    """
    
    TX_train = np.full((n_tasks, 2 * n_train_spc, X_train.shape[1]), np.nan, dtype=np.float32) 
    Ty_train = np.full((n_tasks, 2 * n_train_spc), -1, dtype=np.int8)

    TX_test = np.full((n_tasks, 2 * n_test_spc, X_test.shape[1]), np.nan, dtype=np.float32) 
    Ty_test = np.full((n_tasks, 2 * n_test_spc), -1, dtype=np.int8)

    for i in range(0, 2 * n_tasks, 2):

        idx = int(i / 2)

        TX_train[idx] = np.vstack((X_train[y_train == i][:n_train_spc], 
                                    X_train[y_train == (i + 1)][:n_train_spc])) 

        Ty_train[idx] = np.concatenate((y_train[y_train == i][:n_train_spc], 
                                         y_train[y_train == (i + 1)][:n_train_spc]))

        TX_test[idx] = np.vstack((X_test[y_test == i][:n_test_spc], 
                                   X_test[y_test == (i + 1)][:n_test_spc]))

        Ty_test[idx] = np.concatenate((y_test[y_test == i][:n_test_spc], 
                                        y_test[y_test == (i + 1)][:n_test_spc]))

    return TX_train, Ty_train, TX_test, Ty_test


def ART_IPCA_train_test(TX_train, Ty_train, TX_test, Ty_test, 
                        n_components=200, sim_metric=cosine_similarity, rho=0.5):
    """
    """

    n_tasks = TX_train.shape[0]
    
    # Num samples, dimensionality
    n = TX_train.shape[0] * TX_train.shape[1]
    d = TX_train.shape[2]

    # Set max number of weights to the number of samples to be safe
    max_nw = n

    # Number of weights learned thus far (idx for W, C)
    nw = 0

    # List of weights, mapping from categories to weights, and number of matches per category
    W = np.full((max_nw, d), np.nan, dtype=np.float32)
    C = np.full(max_nw, -1, dtype=np.int32)
    N = np.full(max_nw, -1, dtype=np.int32)

    # Accuracy matrix
    sh_acc_mat = np.full((n_tasks, n_tasks), np.nan, dtype=np.float32)
    mh_acc_mat = np.full((n_tasks, n_tasks), np.nan, dtype=np.float32)

    ipca = IncrementalPCA(n_components)

    for i in range(n_tasks):

        X_train = TX_train[i]
        y_train = Ty_train[i]

        # Fit on all data seen thus far, transform current task data
        ipca.partial_fit(TX_train[i])
        X_train_i = ipca.transform(X_train)

        for j in range(X_train.shape[0]):

            x = X_train[j]
            y = y_train[j]

            xr = X_train_i[j]

            # Init first weight with first sample (never before seen class)
            if j == 0:
                W[nw] = x
                C[nw] = y
                N[nw] = 1
                nw += 1
                continue

            # Find max similarity across input and weights (rows of W) in reduced space!
            sim = sim_metric(xr.reshape(1, -1), ipca.transform(W[:nw])).ravel()

            # Maximum similarity score, winning category index, and label of winning category
            max_idx = np.argmax(sim)
            max_sim = sim[max_idx]
            max_lab = C[max_idx]

            # To update category weight:
            # Input is sufficiently similar to category AND category maps to correct class
            if (max_sim >= rho) and (max_lab == y):
                N[max_idx] = N[max_idx] + 1
                W[max_idx] = (1 / N[max_idx]) * x +  (1 - (1 / N[max_idx])) * W[max_idx]

            # Otherwise, create new category
            else:
                W[nw] = x
                C[nw] = y
                N[nw] = 1
                nw += 1

        # When task is complete, evaluate on all previous tasks
        for k in range(i + 1):

            X_test = TX_test[k]
            y_test = Ty_test[k]

            ### Singleheaded ###
            # iPCA learned only on training data
            sim_i = sim_metric(ipca.transform(X_test), ipca.transform(W[:nw]))

            max_idx_i = np.argmax(sim_i, axis=1)
            y_pred_i = C[max_idx_i]

            acc_i = accuracy_score(y_test, y_pred_i)
            sh_acc_mat[i, k] = acc_i

            ### Multiheaded ###
            # Limit to relevant categories (binary classification)
            m_idx = (C == (2 * k)) | (C == (2 * k + 1))
            W_con = W[m_idx]
            C_con = C[m_idx]

            sim_i = sim_metric(ipca.transform(X_test), ipca.transform(W_con))

            max_idx_i = np.argmax(sim_i, axis=1)
            y_pred_i = C_con[max_idx_i]

            acc_i = accuracy_score(y_test, y_pred_i)
            mh_acc_mat[i, k] = acc_i

    W = W[~np.isnan(W).any(axis=1)]
    C = C[C != -1]
    N = N[N != -1]

    print('Overall accuracy \nMultiheaded setting:  %.3f\nSingleheaded setting: %.3f' 
          % (np.mean(mh_acc_mat[-1]), np.mean(sh_acc_mat[-1])))
    
    return mh_acc_mat, sh_acc_mat
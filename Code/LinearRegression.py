
import numpy as np
import sklearn.model_selection

def predict(X, W, Y=None):

    # TODO Your code here
    y = np.dot(X,W)
    tHat = 1 * (y >0.5)
    acc = sklearn.metrics.accuracy_score(tHat.T.ravel(), Y.T.ravel())
    diff = tHat - Y
    return  acc, diff


def train(X_train, t_train, X_val, t_val, n_poch):

    N_train = X_train.shape[0]

    alpha = 0.01      # learning rate
    batch_size = 10000    # batch size

    # initialization
    w = np.zeros([X_train.shape[1], 1]) # 10 feacture

    w_best = None
    acc_best = 0

    for epoch in range(n_poch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            t_batch = t_train[b*batch_size: (b+1)*batch_size].reshape(-1,1)

            acc,diff = predict(X_batch, w, t_batch)
            differences = np.dot(X_batch.T,diff)
            w = w - (alpha / X_batch.shape[0]) * differences

        acc,diff = predict(X_val, w, t_val.reshape(-1,1))

        if acc > acc_best:
            acc_best = acc
            w_best=w
            
    return acc_best,  w_best


def runRegreesion(times):
    X_train = np.loadtxt("xTrain.txt")
    t_train = np.loadtxt("tTrain.txt")
    X_test = np.loadtxt("xTest.txt")
    t_test = np.loadtxt("tTest.txt")

    acc_best, W_best = train(X_train, t_train, X_test, t_test,times)
    valid_accs,diff = predict(X_test, W_best, t_test.reshape(-1,1))

    w_importance = np.absolute(W_best[:,0])
    return acc_best,valid_accs,w_importance

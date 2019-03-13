import numpy as np


def data_load(path):
    fn = np.loadtxt(path, delimiter=",")
    data = fn[:, 0:-1].astype(np.float)
    label = fn[:, -1].astype(np.int16)
    label[label == -1] = 0
    return data, label


def sigmoid(z):
    return 1/(1+np.exp(-z))


def params_init(dim, loc=0, scale=0.1):
    W = np.random.normal(loc, scale, (dim, 1))
    b = 0
    return W, b


def propagate(W, b, X, y):
    N = X.shape[0]
    # forward
    y_pred = sigmoid(np.dot(X, W)+b)
    loss = 0 - (np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))/N

    # backward
    dout = y_pred-y
    dw = (np.dot(X.T, dout))/N
    db = np.sum(dout)/N

    return dw, db, loss


def optimize(X, y, num_iter, lr=0.1, is_print=True):
    loss_history = []
    W, b = params_init(X.shape[1])
    for step in range(num_iter):
        dw, db, loss = propagate(W, b, X, y)
        # update paramters
        W -= lr*dw
        b -= lr*db
        if step % 10 == 0:
            loss_history.append(loss)
        if is_print and step % 100 == 0:
            acc = np.sum(predict(W, b, X) == y)/X.shape[0]
            print('iter: {}\tacc: {}\tloss: {}'.format(step+1, acc, loss))
    return W, b, loss_history


def predict(W, b, X):
    y_pred = sigmoid(np.dot(X, W)+b)
    return np.round(y_pred)


if __name__ == "__main__":
    X, y = data_load(r'project\mechine learning\data\breast-cancer.csv')
    y = np.expand_dims(y, 1)
    X_tr, y_tr = X[0:200], y[0:200]
    X_te, y_te = X[200:], y[200:]
    W, b = params_init(X.shape[1],0,0)
    optimize(X_tr, y_tr, 2000)

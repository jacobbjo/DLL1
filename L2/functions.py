import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def readfile(file):
    file_dict = unpickle(file)
    data = file_dict[b"data"]
    labels = file_dict[b"labels"]

    # Generate one hot representation of the labels:
    one_hots = np.zeros((10, len(labels)))

    for ind, label in enumerate(labels):
        one_hots[label, ind] = 1

    return np.transpose(data/255), one_hots, np.array(labels)


def get_parameters(dim_img, am_labels, am_nodes):
    W = []
    b = []

    am_nodes.insert(0, dim_img)
    am_nodes.append(am_labels)

    for i in range(len(am_nodes)-1):
        # Xavier initialization from lecture notes
        W.append(np.random.normal(0, (1/np.sqrt(dim_img)), (am_nodes[i+1], am_nodes[i])))
        b.append(np.zeros((am_nodes[i+1], 1)))

    return [W, b]


def softmax(s):
    return np.exp(s)/np.sum(np.exp(s), axis=0)


def evaluate_classifier(X, W, b):
    activations = [X]

    for i in range(len(W)):
        s = np.matmul(W[i], activations[-1]) + b[i]
        x = np.maximum(0, s)

        activations.append(x)

    P = softmax(s)
    return [P, activations]


def compute_cost(X, Y, W, b, lamb):
    """the sum of the loss of the network’s predictions for the images in X relative to
    the ground truth labels and the regularization term on W"""

    P = evaluate_classifier(X, W, b)
    YP = Y*P
    YP_log_sum = 0

    for r in range(YP.shape[0]):
        for c in range(YP.shape[1]):
            if YP[r, c] != 0:
                YP_log_sum += np.log(YP[r, c])

    cross = -1/X.shape[1] * YP_log_sum
    J = cross + lamb * np.sum(np.power(W, 2))
    return J


def compute_accuracy(X, y, W, b):
    P = evaluate_classifier(X, W, b)
    preds = np.argmax(P, axis=0) #the predicted class for each picture
    are_equal = np.sum(preds == y)
    return are_equal/preds.shape[0]


def compute_gradients(X, Y, P, H, W, lamb):
    dldb = []
    dldw = []

    for i in range(len(W)):

        # Initialize
        dldb.append(np.zeros((W[i].shape[0], 1)))
        dldw.append(np.zeros(W[i].shape))

    g = -np.transpose(Y - P)

    for i in reversed(range(len(W))):
        gNew = np.zeros((X.shape[1], W[i-1].shape[0]))

        for ind, picture in enumerate(X.T):

            gpart = g[ind, :].reshape(-1, 1)
            hpart = H[i][:, ind].reshape(-1, 1)

            dldb[i] += gpart
            dldw[i] += np.transpose(np.matmul(hpart, np.transpose(gpart)))

            if i > 0:
                ind_fun = np.where(hpart > 0, 1, 0).reshape(-1, 1)

                gTemp = np.matmul(gpart.T, W[i])
                gNew[ind, :] = np.multiply(gTemp.T, ind_fun).T

        g = gNew.copy()

        dldb[i] /= X.shape[1]
        dldw[i] /= X.shape[1]

        dldw[i] += (2 * lamb * W[i])

    return dldb, dldw



def disp_img(image):
    plt.imshow(np.transpose(image.reshape(3, 32, 32), (1, 2, 0)))
    plt.show()


def compute_grads_num_slow(X, Y, W, b, lamb, h):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = compute_cost(X, Y, W, b_try, lamb)

        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, lamb)

        grad_b[i] = ((c2 - c1) / (2 * h))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try[i, j] = W_try[i, j] - h
            c1 = compute_cost(X, Y, W_try, b, lamb)

            W_try = np.copy(W)
            W_try[i, j] = W_try[i, j] + h
            c2 = compute_cost(X, Y, W_try, b, lamb)

            grad_W[i, j] = ((c2 - c1) / (2 * h))

    return grad_b, grad_W

def mini_batch_GD(X, X_val, Y, Y_val, n_batch, eta, n_epochs, W, b, lamb):
    J_tr = []
    J_val = []


    for i in range(n_epochs):
        for j in range(int(X.shape[1]/n_batch)):
            j_start = j * n_batch
            j_end = (j+1) * n_batch
            Xbatch = X[:, j_start: j_end]
            Ybatch = Y[:, j_start: j_end]

            P = evaluate_classifier(Xbatch, W, b)
            djdb, djdw = compute_gradients(Xbatch, Ybatch, P, W, lamb)


            W -= djdw * eta
            b -= djdb * eta


        J_tr.append(compute_cost(X, Y, W, b, lamb))
        J_val.append(compute_cost(X_val, Y_val, W, b, lamb))
        print("TR Cost for epoch ", i, " is ", J_tr[i])
        print("VAL Cost for epoch ", i, " is ", J_val[i])

    Wstar = W
    bstar = b
    epochs = [x + 1 for x in range(n_epochs)]
    plt.plot(epochs, J_tr, label="Training")
    plt.plot(epochs, J_val, label="Validation")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()


    return Wstar, bstar

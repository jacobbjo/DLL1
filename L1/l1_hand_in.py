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


def softmax(s):
    return np.exp(s)/np.sum(np.exp(s), axis=0)


def evaluate_classifier(X, W, b):
    s = np.matmul(W, X) + b
    P = softmax(s)
    return P


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


def compute_gradients(X, Y, P, W, lamb):
    # initializing gradient variables
    dldb = np.zeros((Y.shape[0], 1))
    dldw = np.zeros(W.shape)

    g = -np.transpose(Y - P)

    dldb = np.sum(g, axis=0)

    for ind, picture in enumerate(X.T):
        gpart = g[ind, :].reshape(-1, 1)
        add = np.matmul(gpart, picture.reshape(-1, 1).T)
        dldw += add

    dldb = dldb/X.shape[1]
    dldw = dldw/X.shape[1]


    djdw = dldw + 2 * lamb * W
    djdb = dldb.reshape(-1, 1)

    return djdb, djdw



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


def main():
    am_labels = 10
    X_tr, Y_tr, y_tr = readfile("../Datasets/data_batch_1")
    X_val, Y_val, y_val = readfile("../Datasets/data_batch_2")
    X_test, Y_test, y_test = readfile("../Datasets/test_batch")

    dim_img = len(X_val)


    W = np.random.normal(0, 0.01, (am_labels, dim_img))
    b = np.random.normal(0, 0.01, (am_labels, 1))


    #P = evaluate_classifier(X_tr[:, 0:100], W, b)
    #djdb, djdw = compute_gradients(X_tr[:, 0:100], Y_tr[:, 0:100], P, W, 0.001)
    #djdb2, djdw2 = compute_grads_num_slow(X_tr[:, 0:100], Y_tr[:, 0:100], W, b, 0.001, 0.000001)
    #diff_b =  djdb - djdb2
    #diff_w = djdw - djdw2
    #wsum = np.sum(np.abs(diff_w)) / W.size1
    #print("Arage difference an vs num: ", wsum)


    acc_before = compute_accuracy(X_test, y_test, W, b)
    print("Accuracy with random: ", acc_before)

    lamb = 0
    n_epochs = 40
    n_batch = 100
    eta =.01

    Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb)

    acc = compute_accuracy(X_test, y_test, Wstar, bstar)
    print("Accuracy after training: ", acc)

    #for row in range(Wstar.shape[0]):
    #    W_row = Wstar[row, :]
    #    W_row = ((W_row - np.min(W_row)) / (np.max(W_row) - np.min(W_row)))
    #    disp_img(W_row)


if __name__ == "__main__":
    main()
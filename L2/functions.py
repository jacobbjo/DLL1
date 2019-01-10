import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


def save_mats(mat_list, name, folder):
    for i in range(len(mat_list)):
        file_path = folder + "/" + name + "_" + str(i)
        np.savetxt(file_path, mat_list[i])


def load_mats(name, folder):
    mat_list = []

    i = 0
    while True:
        file_path_str = folder + "/" + name + "_" + str(i)
        file_path = Path(file_path_str)

        if not file_path.is_file():
            print("load_mats:  \"" + file_path_str + "\" is not found, breaking.")
            break

        mat = np.loadtxt(file_path_str)
        if mat.ndim < 2:
            # fixing shape of vectors
            #print("fixing mat shape for: " + name + "_" + str(i))
            mat = mat.reshape(-1, 1)

        mat_list.append(mat)

        i += 1

    return mat_list

def get_parameters(dim_img, am_labels, am_nodes, seed):
    W = []
    b = []

    loop_nodes = am_nodes[:]
    loop_nodes.insert(0, dim_img)
    loop_nodes.append(am_labels)

    for i in range(len(loop_nodes)-1):
        # Xavier initialization from lecture notes
        np.random.seed(seed)
        W.append(np.random.normal(0, (1/np.sqrt(dim_img)), (loop_nodes[i+1], loop_nodes[i])))
        b.append(np.zeros((loop_nodes[i+1], 1)))

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

    P = evaluate_classifier(X, W, b)[0]
    YP = Y*P
    YP_log_sum = 0

    for r in range(YP.shape[0]):
        for c in range(YP.shape[1]):
            if YP[r, c] != 0:
                YP_log_sum += np.log(YP[r, c])

    cross = -1/X.shape[1] * YP_log_sum
    J = cross

    for hl in range(len(W)):
        J += lamb * np.sum(np.power(W[hl], 2))

    return J


def compute_accuracy(X, y, W, b):
    P = evaluate_classifier(X, W, b)[0]
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

    grad_b = []
    grad_W = []

    for i in range(len(W)):
        # Initialize
        grad_b.append(np.zeros(b[i].shape))
        grad_W.append(np.zeros(W[i].shape))

    for hl in range(len(W)): #for every hidden layer

        for i in range(b[hl].shape[0]):
            b_try = b[:]
            b_try[hl][i] = b_try[hl][i] - h
            c1 = compute_cost(X, Y, W, b_try, lamb)

            b_try = b[:]
            b_try[hl][i] = b_try[hl][i] + h
            c2 = compute_cost(X, Y, W, b_try, lamb)

            grad_b[hl][i] = ((c2 - c1) / (2 * h))

        for i in range(W[hl].shape[0]):
            print("W loop, i: ", i)
            for j in range(W[hl].shape[1]):
                W_try = W[:]
                W_try[hl][i, j] = W_try[hl][i, j] - h
                c1 = compute_cost(X, Y, W_try, b, lamb)

                W_try = W[:]
                W_try[hl][i, j] = W_try[hl][i, j] + h
                c2 = compute_cost(X, Y, W_try, b, lamb)

                grad_W[hl][i, j] = ((c2 - c1) / (2 * h))

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

            P, H = evaluate_classifier(Xbatch, W, b)
            djdb, djdw = compute_gradients(Xbatch, Ybatch, P, H, W, lamb)

            for hl in range(len(W)):  # for every hidden layer
                W[hl] -= djdw[hl] * eta
                b[hl] -= djdb[hl] * eta

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

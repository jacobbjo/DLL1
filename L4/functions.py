import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_file(file):
    with open(file, 'r', encoding='utf-8') as fo:
        book_data = fo.read()
        unique_chars = set(book_data)

    print("Found", len(unique_chars), "unique chars in file: ", file)

    char_to_ind = {}
    ind_to_char = {}

    for i, char in enumerate(sorted(unique_chars)):
    #for i, char in enumerate(sorted(unique_chars)[:-2]):
        char_to_ind[char] = i
        ind_to_char[i] = char

    return book_data, char_to_ind, ind_to_char


def get_parameters(k, m, seed=-1):
    sig = 0.01

    if seed > 0:
        np.random.seed(seed)

    b = np.zeros((m, 1))
    c = np.zeros((k, 1))
    U = np.random.rand(m, k)*sig
    W = np.random.rand(m, m)*sig
    V = np.random.rand(k, m)*sig

    return [b, c, U, W, V]


def get_zero_grads(k, m):
    b = np.zeros((m, 1))
    c = np.zeros((k, 1))
    U = np.zeros((m, k))
    W = np.zeros((m, m))
    V = np.zeros((k, m))

    return [b, c, U, W, V]


def softmax(s):
    return np.exp(s)/np.sum(np.exp(s), axis=0)


def get_seq(b, c, U, W, V, h, x, n, ind_to_char):
    Y = np.zeros((x.shape[0], n))

    for t in range(n):
        a = np.matmul(W, h) + np.matmul(U, x) + b
        h = np.tanh(a)
        o = np.matmul(V, h) + c
        p = softmax(o)
        x = one_hot_vec(get_xnext(p), p.shape[0])
        Y[:, t:t+1] = x

    seq = [ind_to_char[ind] for ind in np.argmax(np.array(Y), axis=0)]

    return seq


def get_xnext(p):
    cp = np.cumsum(p, axis=0)[:,0]
    a = np.random.rand(1)
    ixs = np.where((cp - a) > 0)
    ii = ixs[0][0]
    return ii


def one_hot_vec(ind, n):
    vec = np.zeros((n, 1))
    vec[ind, 0] = 1
    return vec


def forward_pass(b, c, U, W, V, X, Y, h):
    m = h.shape[0]
    k = X.shape[0]
    n = X.shape[1]

    l = 0
    A = np.zeros((m, n))
    H = np.zeros((m, n))
    P = np.zeros((k, n))

    for t in range(n):
        x = X[:, t:t+1]
        a = np.matmul(W, h) + np.matmul(U, x) + b
        h = np.tanh(a)
        o = np.matmul(V, h) + c
        p = softmax(o)

        A[:, t:t+1] = a
        H[:, t:t+1] = h
        P[:, t:t+1] = p

        l += np.log(np.matmul(Y[:, t:t+1].T, p))

    l *= -1

    return A, H, P, l


def backward_pass(b, c, U, W, V, X, Y, A, H, P, h):
    m = h.shape[0]
    k = X.shape[0]
    n = X.shape[1]

    dldb = np.zeros(b.shape)
    dldc = np.zeros(c.shape)
    dldU = np.zeros(U.shape)
    dldW = np.zeros(W.shape)
    dldV = np.zeros(V.shape)

    dldo = (-(Y - P))

    dldH = np.zeros((n, m))
    dldH[n-1, :] = np.matmul(dldo[:, n-1:n].T, V)

    dldA = np.zeros((n, m))
    dldA[n-1, :] = np.matmul(dldH[n - 1:n, :], np.diag(1 - np.power(np.tanh(A[:, n - 1]), 2)))


    for t in reversed(range(n-1)):
        dldH[t, :] = np.matmul(dldo[:, t:t+1].T, V) + np.matmul(dldA[t+1:t+2, :], W)
        dldA[t, :] = np.matmul(dldH[t:t+1, :], np.diag(1 - np.power(np.tanh(A[:, t]), 2)))

    dldb += np.sum(dldA, axis=0).reshape(-1, 1)
    dldc += np.sum(dldo, axis=1).reshape(-1, 1)
    dldU += np.matmul(dldA.T, X.T)
    dldV += np.matmul(dldo, H.T)
    dldW += np.matmul(dldA[0:1, :].T, h.T)

    for t in range(1, n):
        dldW += np.matmul(dldA[t:t+1, :].T, H[:, t-1:t].T)

    # clipping the gradients
    for grad in [dldb, dldc, dldU, dldW, dldV]:
        grad[grad > 5] = 5
        grad[grad < -5] = -5


    return dldb, dldc, dldU, dldW, dldV


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


def ComputeGradsNum(X, Y, b, c, U, W, V, h):
    RNN = [b, c, U, W, V]
    num_grads = []

    hprev = np.zeros((W.shape[0], 1))

    for k, f in enumerate(RNN):
        print("Computing numerical gradient for field number: ", k)
        num_grads.append(np.zeros(f.shape))

        for i in range(f.shape[0]):
            #print("   Row", i, "/", f.shape[0], "started")

            for j in range(f.shape[1]):
                f_try = np.copy(f)
                f_try[i, j] -= h
                RNN_try = RNN[:]
                RNN_try[k] = f_try
                l1 = forward_pass(*RNN_try, X, Y, hprev)[-1]

                f_try = np.copy(f)
                f_try[i, j] += h
                RNN_try = RNN[:]
                RNN_try[k] = f_try
                l2 = forward_pass(*RNN_try, X, Y, hprev)[-1]

                num_grads[-1][i, j] = (l2-l1)/(2*h)

    return num_grads


def train_AdaGrad(n_epochs, eta, m, seq_length, book_data, char_to_ind, ind_to_char, RNN):
    eps = 0.000001
    k = len(char_to_ind)
    ada_grads = get_zero_grads(k, m)
    smooth_loss = 0
    x0 = one_hot_vec(30, k)  # H
    iteration = 0

    for ep in range(n_epochs):
        e = 0
        hprev = np.zeros((m, 1))

        while not e > len(book_data)-seq_length-1:
            X_chars = book_data[e: e + seq_length]
            Y_chars = book_data[e + 1: e + seq_length + 1]

            X = np.array([one_hot_vec(char_to_ind[char], k) for char in X_chars])[:, :, 0].T
            Y = np.array([one_hot_vec(char_to_ind[char], k) for char in Y_chars])[:, :, 0].T

            A, H, P, l = forward_pass(*RNN, X, Y, hprev)

            gradients = backward_pass(*RNN, X, Y, A, H, P, hprev)

            hprev = H[:, -1:]

            # AdaGrad
            for g in range(len(gradients)):
                ada_grads[g] += np.power(gradients[g], 2)
                RNN[g] -= (eta * gradients[g]/(np.sqrt(ada_grads[g] + eps)))

            if iteration == 0:
                smooth_loss = l

            smooth_loss = 0.999 * smooth_loss + 0.001 * l

            if iteration%500 == 0 or iteration == 1:
                print("Ep:", ep, "Iteration:", iteration, "smooth loss:", smooth_loss)

                #smooth_loss_list = [smooth_loss_list smooth_loss];
                #iterations_list = [iterations_list iterations];


            if iteration%500 == 0 or iteration == 1:
                x0 = X[:, 1:2]
                print("".join(get_seq(*RNN, hprev, x0, 200, ind_to_char)), "\n")


            iteration += 1
            e += seq_length


    return RNN





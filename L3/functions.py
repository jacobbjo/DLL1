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


def get_parameters(dim_img, am_labels, am_nodes, seed=-1):
    W = []
    b = []

    loop_nodes = am_nodes[:]
    loop_nodes.insert(0, dim_img)
    loop_nodes.append(am_labels)

    if seed > 0:
        np.random.seed(seed)

    for i in range(len(loop_nodes)-1):
        # Xavier initialization from lecture notes
        W.append(np.random.normal(0, (1/np.sqrt(dim_img)), (loop_nodes[i+1], loop_nodes[i])))
        b.append(np.zeros((loop_nodes[i+1], 1)))

    return [W, b]


def get_parameters_he(dim_img, am_labels, am_nodes, seed=-1):
    W = []
    b = []

    loop_nodes = am_nodes[:]
    loop_nodes.insert(0, dim_img)
    loop_nodes.append(am_labels)

    if seed > 0:
        np.random.seed(seed)

    for i in range(len(loop_nodes) - 1):
        W.append(np.random.normal(0, np.sqrt(2 / loop_nodes[i]), (loop_nodes[i+1], loop_nodes[i])))
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


def evaluate_classifier_batch_norm(X, W, b, M=[], V=[]):
    calc_mean_var = False

    if not (len(M) > 0 and len(V) > 0):
        M = []
        V = []
        calc_mean_var = True
        #print("Not using mean and variance input in evaluate_classifier_batch!")

    S = []
    S_N = []
    activations = [X]

    for i in range(len(W)):
        # score
        s = np.matmul(W[i], activations[-1]) + b[i]
        S.append(s)

        if calc_mean_var:
            # mean
            m = np.mean(s, axis=1).reshape(-1, 1)
            M.append(m)

            # variance
            v = np.mean(np.power((s-m), 2), axis=1).reshape(-1, 1)
            V.append(v)

        else:
            # mean
            m = M[i]

            # variance
            v = V[i]

        # normalized score
        s_n = batch_norm(s, m, v)
        S_N.append(s_n)

        # activation
        x = np.maximum(0, s_n)
        activations.append(x)

    P = softmax(S[-1])
    return [P, activations, S, M, V]


def batch_norm(s, m, v):
    return (s-m)/np.sqrt(v + 0.000001)  # small epsilon to not sqrt(0)


def batch_norm_back_pass(djdshat, s, m, v):
    n = s.shape[1]
    vb = v + 0.0000001  # adding small epsilon to avoid zero division
    Vb_12 = np.diag((np.power(vb, -1/2)).reshape(-1))
    Vb_32 = np.diag((np.power(vb, -3/2)).reshape(-1))

    djdvb = np.zeros((1, s.shape[0]))
    djdmy = np.zeros((1, s.shape[0]))

    for i in range(n):
        score_diag = np.diag((s[:, i].reshape(-1, 1) - m).reshape(-1))
        temp_prod = np.matmul(djdshat[i, :].reshape(1, -1), Vb_32)

        djdvb += np.matmul(temp_prod, score_diag)
        djdmy += np.matmul(djdshat[i, :].reshape(1, -1), Vb_12)
    djdvb *= (-1/2)
    djdmy *= (-1)

    djds = np.zeros(djdshat.shape)

    for i in range(n):
        score_diag = np.diag((s[:, i].reshape(-1, 1) - m).reshape(-1))

        part_1 = np.matmul(djdshat[i, :].reshape(1, -1), Vb_12)
        part_2 = (2/n) * np.matmul(djdvb, score_diag)
        part_3 = (1/n) * djdmy

        djds[i, :] = part_1 + part_2 + part_3

    return djds


def compute_cost(X, Y, W, b, lamb):
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


def compute_cost_batch_norm(X, Y, W, b, lamb):
    P = evaluate_classifier_batch_norm(X, W, b)[0]
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


def compute_accuracy_batch_norm(X, y, W, b, M=[], V=[]):
    P = evaluate_classifier_batch_norm(X, W, b, M, V)[0]
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

        if i > 0:
            for ind in range(X.shape[1]):
                gpart = g[ind, :].reshape(-1, 1)
                hpart = H[i][:, ind].reshape(-1, 1)

                dldb[i] += gpart

                dldw[i] += np.matmul(gpart, hpart.T)

                ind_fun = np.where(hpart > 0, 1, 0).reshape(-1, 1)

                gTemp = np.matmul(gpart.T, W[i])
                gNew[ind, :] = np.multiply(gTemp.T, ind_fun).T

        else:
            dldw[i] += np.matmul(g.T, H[i].T)

        g = gNew.copy()

        dldb[i] /= X.shape[1]
        dldw[i] /= X.shape[1]

        dldw[i] += (2 * lamb * W[i])

    return dldb, dldw


def compute_gradients_batch_norm(X, Y, P, H, S, M, V, W, lamb):
    dldb = []
    dldw = []

    for i in range(len(W)):

        # Initialize
        dldb.append(np.zeros((W[i].shape[0], 1)))
        dldw.append(np.zeros(W[i].shape))

    g = -np.transpose(Y - P)

    # Last layer (k)
    dldb[-1] = np.sum(g, axis=0)
    dldb[-1] /= X.shape[1]
    dldb[-1] = dldb[-1].reshape(-1, 1)

    g_new = np.zeros((X.shape[1], W[-2].shape[0]))

    for ind in range(X.shape[1]):
        gpart = g[ind, :].reshape(-1, 1)
        hpart = H[-2][:, ind].reshape(-1, 1)

        # eq 20 from assignment
        dldw[-1] += np.matmul(gpart, hpart.T)

        # eq 21 and 22 from assignment
        g_temp = np.matmul(gpart.T, W[-1])
        ind_fun = np.diag(np.where(hpart[:,0] > 0, 1, 0))
        g_new[ind, :] = np.matmul(g_temp, ind_fun)

    dldw[-1] /= X.shape[1]
    dldw[-1] += (2 * lamb * W[-1])

    g = g_new.copy()

    # Layers = k − 1, . . . , 1
    for i in reversed(range(len(W)-1)):
        g_new = np.zeros((X.shape[1], W[i-1].shape[0]))

        g = batch_norm_back_pass(g, S[i], M[i], V[i])

        dldb[i] = np.sum(g, axis=0)

        if i > 0:
            for ind in range(X.shape[1]):
                gpart = g[ind, :].reshape(-1, 1)
                hpart = H[i][:, ind].reshape(-1, 1)

                dldw[i] += np.matmul(gpart, hpart.T)

                ind_fun = np.diag(np.where(hpart > 0, 1, 0)[:, 0])

                g_temp = np.matmul(gpart.T, W[i])
                g_new[ind, :] = np.matmul(g_temp, ind_fun)

        else:
            dldw[i] += np.matmul(g.T, H[i].T)

        g = g_new.copy()

        dldb[i] /= X.shape[1]
        dldb[i] = dldb[i].reshape(-1, 1)

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
            b_lay = np.copy(b[hl])
            b_lay[i] = b_lay[i] - h
            b_try = b[:]
            b_try[hl] = b_lay
            c1 = compute_cost(X, Y, W, b_try, lamb)

            b_lay = np.copy(b[hl])
            b_lay[i] = b_lay[i] + h
            b_try = b[:]
            b_try[hl] = b_lay
            c2 = compute_cost(X, Y, W, b_try, lamb)

            grad_b[hl][i] = ((c2 - c1) / (2 * h))

        for i in range(W[hl].shape[0]):
            print("W loop, i: ", i)
            for j in range(W[hl].shape[1]):

                W_lay = np.copy(W[hl])
                W_lay[i, j] = W_lay[i, j] - h
                W_try = W[:]
                W_try[hl] = W_lay
                c1 = compute_cost(X, Y, W_try, b, lamb)

                W_lay = np.copy(W[hl])
                W_lay[i, j] = W_lay[i, j] + h
                W_try = W[:]
                W_try[hl] = W_lay
                c2 = compute_cost(X, Y, W_try, b, lamb)

                grad_W[hl][i, j] = ((c2 - c1) / (2 * h))

    return grad_b, grad_W


def mini_batch_GD(X, X_val, Y, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr):

    int_cost = compute_cost(X, Y, W, b, lamb)

    J_tr = []
    J_val = []

    vb = []
    vW = []

    for i in range(len(W)):
        # Initialize
        vb.append(np.zeros((W[i].shape[0], 1)))
        vW.append(np.zeros(W[i].shape))

    for i in range(n_epochs):
        for j in range(int(X.shape[1]/n_batch)):
            j_start = j * n_batch
            j_end = (j+1) * n_batch
            Xbatch = X[:, j_start: j_end]
            Ybatch = Y[:, j_start: j_end]

            P, H = evaluate_classifier(Xbatch, W, b)

            djdb, djdw = compute_gradients(Xbatch, Ybatch, P, H, W, lamb)

            for hl in range(len(W)):  # for every hidden layer
                vb[hl] = (vb[hl] * rho) + (eta * djdb[hl])
                vW[hl] = (vW[hl] * rho) + (eta * djdw[hl])

                b[hl] -= vb[hl]
                W[hl] -= vW[hl]

        J_tr.append(compute_cost(X, Y, W, b, lamb))
        J_val.append(compute_cost(X_val, Y_val, W, b, lamb))

        #print("TR Cost for epoch ", i, " is ", J_tr[i])
        #print("VAL Cost for epoch ", i, " is ", J_val[i])
        print("    Epoch ", i, " TR: ", J_tr[i], " VAL: ", J_val[i])

        if J_tr[-1] > 3* int_cost:
            print("J_tr[-1] > 3* int_cost")
            break

        eta *= dr

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


def mini_batch_GD_batch_norm(X, X_val, Y, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr):
    alpha = 0.99
    int_cost = compute_cost_batch_norm(X, Y, W, b, lamb)

    J_tr = []
    J_val = []

    vb = []
    vW = []

    move_mean = []
    move_vari = []

    for i in range(len(W)):
        # Initialize
        vb.append(np.zeros((W[i].shape[0], 1)))
        vW.append(np.zeros(W[i].shape))

    for i in range(n_epochs):
        for j in range(int(X.shape[1]/n_batch)):
            j_start = j * n_batch
            j_end = (j+1) * n_batch
            Xbatch = X[:, j_start: j_end]
            Ybatch = Y[:, j_start: j_end]

            P, H, S, M, V = evaluate_classifier_batch_norm(Xbatch, W, b)

            if i == 0 and j == 0:
                move_mean = M.copy()
                move_vari = V.copy()
            else:
                for lay in range(len(W)):
                    move_mean[lay] = (alpha*move_mean[lay]) + ((1-alpha)*M[lay])
                    move_vari[lay] = (alpha*move_vari[lay]) + ((1-alpha)*V[lay])

            djdb, djdw = compute_gradients_batch_norm(Xbatch, Ybatch, P, H, S, M, V, W, lamb)

            for hl in range(len(W)):  # for every hidden layer
                vb[hl] = (vb[hl] * rho) + (eta * djdb[hl])
                vW[hl] = (vW[hl] * rho) + (eta * djdw[hl])
                b[hl] -= vb[hl]
                W[hl] -= vW[hl]


        J_tr.append(compute_cost_batch_norm(X, Y, W, b, lamb))
        J_val.append(compute_cost_batch_norm(X_val, Y_val, W, b, lamb))

        #print("TR Cost for epoch ", i, " is ", J_tr[i])
        #print("VAL Cost for epoch ", i, " is ", J_val[i])
        print("    Epoch ", i, " TR: ", J_tr[i], " VAL: ", J_val[i])

        if J_tr[-1] > 3* int_cost:
            print("J_tr[-1] > 3* int_cost")
            break

        eta *= dr

    Wstar = W
    bstar = b

    #epochs = [x + 1 for x in range(n_epochs)]
    #plt.plot(epochs, J_tr, label="Training")
    #plt.plot(epochs, J_val, label="Validation")
    #plt.legend()
    #plt.ylabel("Loss")
    #plt.xlabel("Epochs")
    #plt.show()

    return Wstar, bstar, move_mean, move_vari

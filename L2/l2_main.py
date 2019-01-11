from L2.functions import *
import random

def main():
    am_labels = 10
    X_tr, Y_tr, y_tr = readfile("../Datasets/data_batch_1")
    X_tr1, Y_tr1, y_tr1 = readfile("../Datasets/data_batch_2")
    X_tr2, Y_tr2, y_tr2 = readfile("../Datasets/data_batch_3")
    X_tr3, Y_tr3, y_tr3 = readfile("../Datasets/data_batch_4")
    X_tr4, Y_tr4, y_tr4 = readfile("../Datasets/data_batch_5")

    X_tr = np.concatenate((X_tr, X_tr1), axis=1)
    X_tr = np.concatenate((X_tr, X_tr2), axis=1)
    X_tr = np.concatenate((X_tr, X_tr3), axis=1)
    X_tr = np.concatenate((X_tr, X_tr4[:, :9000]), axis=1)

    Y_tr = np.concatenate((Y_tr, Y_tr1), axis=1)
    Y_tr = np.concatenate((Y_tr, Y_tr2), axis=1)
    Y_tr = np.concatenate((Y_tr, Y_tr3), axis=1)
    Y_tr = np.concatenate((Y_tr, Y_tr4[:, :9000]), axis=1)

    y_tr = np.concatenate((y_tr, y_tr1), axis=0)
    y_tr = np.concatenate((y_tr, y_tr2), axis=0)
    y_tr = np.concatenate((y_tr, y_tr3), axis=0)
    y_tr = np.concatenate((y_tr, y_tr4[:9000]), axis=0)

    X_val = X_tr4[:, 9000:10000]
    Y_val = Y_tr4[:, 9000:10000]
    y_val = y_tr4[9000:10000]

    #X_val, Y_val, y_val = readfile("../Datasets/data_batch_2")

    X_test, Y_test, y_test = readfile("../Datasets/test_batch")

    LOAD = True

    mean_x = np.mean(X_tr, axis=1)

    mean_x = np.reshape(mean_x, (-1, 1))

    X_tr -= mean_x
    X_val -= mean_x
    X_test -= mean_x

    am_nodes = [50]  # number of nodes for the hidden layers

    dim_img = len(X_val)

    W, b = get_parameters(dim_img, am_labels, am_nodes, 1337)

#    P, H = evaluate_classifier(X_tr[:, 0:100], W, b)
#
#    djdb, djdw = compute_gradients(X_tr[:, 0:100], Y_tr[:, 0:100], P, H, W, 0.001)
#
#    if LOAD:
#        djdb2 = load_mats("djdb2", "mats")
#        djdw2 = load_mats("djdw2", "mats")
#    else:
#        djdb2, djdw2 = compute_grads_num_slow(X_tr[:, 0:100], Y_tr[:, 0:100], W, b, 0.001, 0.00001)
#        save_mats(djdb2, "djdb2", "mats")
#        save_mats(djdw2, "djdw2", "mats")
#
#    for lay in range(len(am_nodes) + 1):
#        print("lay: " + str(lay))
#        diff_b = djdb[lay] - djdb2[lay]
#        diff_w = djdw[lay] - djdw2[lay]
#
#
#        bsum = np.sum(np.abs(diff_b)) / b[lay].size
#        wsum = np.sum(np.abs(diff_w)) / W[lay].size
#
#        print("bsum: ", bsum)
#        print("wsum: ", wsum)
#
#
#    acc_before_train = compute_accuracy(X_test, y_test, W, b)
#    print("Accuracy before training: ", acc_before_train)
#
#    n_batch = 100
#    n_epochs = 20
#    rho = 0.9
#    dr = 0.95  # decay rate
#
#    eta_lower = 0.0055
#    eta_upper = 0.008
#
#    lamb_lower = 0.0045
#    lamb_upper = 0.006
#
#    pairing_tries = 100
#
#    results = np.zeros((pairing_tries, 3))
#
#    for t in range(pairing_tries):
#
#        eta = random.uniform(eta_lower, eta_upper)
#        lamb = random.uniform(lamb_lower, lamb_upper)
#
#        Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)
#
#        acc = compute_accuracy(X_test, y_test, Wstar, bstar)
#
#        print("pair: " + str(t) + " acc: " + str(acc), " eta: " + str(eta) + " lamb: " + str(lamb))
#
#        results[t, :] = [acc, eta, lamb]
#
#        # Sort results based on descending accuracy
#        results = results[results[:, 0].argsort()[::-1]]
#
#        np.savetxt("results.txt", results, fmt="%1.5f")
#


    n_batch = 100
    n_epochs = 30
    rho = 0.9
    dr = 0.95  # decay rate
    eta = 0.00560
    lamb = 0.00541

    Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)

    acc = compute_accuracy(X_test, y_test, Wstar, bstar)

    print("Accuracy after training: ", acc)



if __name__ == "__main__":
    main()
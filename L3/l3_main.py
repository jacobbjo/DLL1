from L3.functions import *
import random

def main():
    am_labels = 10
    X_tr, Y_tr, y_tr = readfile("../Datasets/data_batch_1")
#    X_tr1, Y_tr1, y_tr1 = readfile("../Datasets/data_batch_2")
#    X_tr2, Y_tr2, y_tr2 = readfile("../Datasets/data_batch_3")
#    X_tr3, Y_tr3, y_tr3 = readfile("../Datasets/data_batch_4")
#    X_tr4, Y_tr4, y_tr4 = readfile("../Datasets/data_batch_5")
#
#    X_tr = np.concatenate((X_tr, X_tr1), axis=1)
#    X_tr = np.concatenate((X_tr, X_tr2), axis=1)
#    X_tr = np.concatenate((X_tr, X_tr3), axis=1)
#    X_tr = np.concatenate((X_tr, X_tr4[:, :9000]), axis=1)
#
#    Y_tr = np.concatenate((Y_tr, Y_tr1), axis=1)
#    Y_tr = np.concatenate((Y_tr, Y_tr2), axis=1)
#    Y_tr = np.concatenate((Y_tr, Y_tr3), axis=1)
#    Y_tr = np.concatenate((Y_tr, Y_tr4[:, :9000]), axis=1)
#
#    y_tr = np.concatenate((y_tr, y_tr1), axis=0)
#    y_tr = np.concatenate((y_tr, y_tr2), axis=0)
#    y_tr = np.concatenate((y_tr, y_tr3), axis=0)
#    y_tr = np.concatenate((y_tr, y_tr4[:9000]), axis=0)
#
#    X_val = X_tr4[:, 9000:10000]
#    Y_val = Y_tr4[:, 9000:10000]
#    y_val = y_tr4[9000:10000]
#
    X_val, Y_val, y_val = readfile("../Datasets/data_batch_2")

    X_test, Y_test, y_test = readfile("../Datasets/test_batch")

    LOAD = True

    mean_x = np.mean(X_tr, axis=1)
    mean_x = np.reshape(mean_x, (-1, 1))
    X_tr -= mean_x
    X_val -= mean_x
    X_test -= mean_x

    am_nodes = [50]  # number of nodes for the hidden layers

    dim_img = len(X_val)

    W, b = get_parameters_he(dim_img, am_labels, am_nodes, 1337)
#
#    P, H, S, M, V = evaluate_classifier_batch_norm(X_tr[:, 0:100], W, b)
#
#    djdb, djdw = compute_gradients_batch_norm(X_tr[:, 0:100], Y_tr[:, 0:100], P, H, S, M, V, W, 0)
#
#    #djdb, djdw = compute_gradients(X_tr[:, 0:100], Y_tr[:, 0:100], P, H, W, 0.001)
#
#    if LOAD:
#        djdb2 = load_mats("djdb2_50_30", "mats")
#        djdw2 = load_mats("djdw2_50_30", "mats")
#    else:
#        djdb2, djdw2 = compute_grads_num_slow(X_tr[:, 0:100], Y_tr[:, 0:100], W, b, 0, 0.000001)
#        save_mats(djdb2, "djdb2_50_30", "mats")
#        save_mats(djdw2, "djdw2_50_30", "mats")
#
#    for lay in range(len(am_nodes) + 1):
#        print("lay: " + str(lay))
#        diff_b = djdb[lay] - djdb2[lay]
#        diff_w = djdw[lay] - djdw2[lay]
#
#        bsum = np.sum(np.abs(diff_b)) / b[lay].size
#        wsum = np.sum(np.abs(diff_w)) / W[lay].size
#
#        print("bsum: ", bsum)
#        print("wsum: ", wsum)
#
#
#    #acc_before_train = compute_accuracy(X_test, y_test, W, b, M, V)
#    acc_before_train = compute_accuracy_batch_norm(X_test, y_test, W, b)
#
#    print("Accuracy before training: ", acc_before_train)


### FINDING PAIRS

#    n_batch = 100
#    n_epochs = 20
#    rho = 0.9
#    dr = 0.95  # decay rate
#
#    eta_lower = 0.05
#    eta_upper = 0.1
#
#    lamb_lower = 0.002
#    lamb_upper = 0.009
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
#        W, b = get_parameters_he(dim_img, am_labels, am_nodes, 1337)
#
#        Wstar, bstar, move_mean, move_vari = mini_batch_GD_batch_norm(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)
#
#        acc = compute_accuracy_batch_norm(X_val, y_val, Wstar, bstar, move_mean, move_vari)
#
#        print("pair: " + str(t) + " acc: " + str(acc), " eta: " + str(eta) + " lamb: " + str(lamb))
#
#        results[t, :] = [acc, eta, lamb]
#
#        # Sort results based on descending accuracy
#        results = results[results[:, 0].argsort()[::-1]]
#
#        np.savetxt("results.txt", results, fmt="%1.5f")



### FINAL NETWORK STUFF
    n_batch = 100
    n_epochs = 10
    rho = 0.9
    dr = 0.95  # decay rate
    eta = 0.1
    lamb = 0.001

    #n_batch = 100
    #n_epochs = 20
    #rho = 0.9
    #dr = 0.95  # decay rate
    #eta = 0.0575
    #lamb = 0.00118


    #Wstar, bstar, move_mean, move_vari = mini_batch_GD_batch_norm(X_tr[:, 0:100], X_val[:, 0:100], Y_tr[:, 0:100], Y_val[:, 0:100], n_batch, eta, n_epochs, W, b, lamb, rho, dr)
    W#star, bstar, move_mean, move_vari = mini_batch_GD_batch_norm(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)

    #acc = compute_accuracy_batch_norm(X_test, y_test, Wstar, bstar, move_mean, move_vari)

    Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)

    acc = compute_accuracy(X_test, y_test, Wstar, bstar)


    print("Accuracy after training: ", acc)



if __name__ == "__main__":
    main()
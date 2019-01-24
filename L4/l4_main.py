from L4.functions import *

def main():

    book_data, char_to_ind, ind_to_char = read_file("../Datasets/goblet_book.txt")

    k = len(char_to_ind)
    m = 5
    eta = 0.1
    seq_length = 25


    h0 = np.zeros((m, 1))
    x0 = one_hot_vec(12, k)


    b, c, U, W, V = get_parameters(k, m, 1337)

    #a_seq = get_seq(b, c, U, W, V, h0, x0, 50, ind_to_char)
    #print(a_seq)

    X_chars = book_data[0: seq_length]
    Y_chars = book_data[1: seq_length + 1]

    X = np.array([one_hot_vec(char_to_ind[char], k) for char in X_chars])[:, :, 0].T
    Y = np.array([one_hot_vec(char_to_ind[char], k) for char in Y_chars])[:, :, 0].T

    A, H, P, l = forward_pass(b, c, U, W, V, X, Y, h0)

    dldb, dldc, dldU, dldW, dldV = backward_pass(b, c, U, W, V, X, Y, A, H, P, h0)

    print("lol")

#   X_test, Y_test, y_test = readfile("../Datasets/test_batch")

#   LOAD = True

#   mean_x = np.mean(X_tr, axis=1)
#   mean_x = np.reshape(mean_x, (-1, 1))
#   X_tr -= mean_x
#   X_val -= mean_x
#   X_test -= mean_x

#   am_nodes = [50]  # number of nodes for the hidden layers

#   dim_img = len(X_val)

#   W, b = get_parameters_he(dim_img, am_labels, am_nodes, 1337)
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
    #n_batch = 100
    #n_epochs = 10
    #rho = 0.9
    #dr = 0.95  # decay rate
    #eta = 0.1
    #lamb = 0.001

    #n_batch = 100
    #n_epochs = 20
    #rho = 0.9
    #dr = 0.95  # decay rate
    #eta = 0.0575
    #lamb = 0.00118


    #Wstar, bstar, move_mean, move_vari = mini_batch_GD_batch_norm(X_tr[:, 0:100], X_val[:, 0:100], Y_tr[:, 0:100], Y_val[:, 0:100], n_batch, eta, n_epochs, W, b, lamb, rho, dr)
    #Wstar, bstar, move_mean, move_vari = mini_batch_GD_batch_norm(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)

    #acc = compute_accuracy_batch_norm(X_test, y_test, Wstar, bstar, move_mean, move_vari)

    #Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb, rho, dr)

    #acc = compute_accuracy(X_test, y_test, Wstar, bstar)


    #print("Accuracy after training: ", acc)



if __name__ == "__main__":
    main()
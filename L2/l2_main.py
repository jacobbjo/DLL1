from L2.functions import *

def main():
    am_labels = 10
    X_tr, Y_tr, y_tr = readfile("../Datasets/data_batch_1")
    X_val, Y_val, y_val = readfile("../Datasets/data_batch_2")
    X_test, Y_test, y_test = readfile("../Datasets/test_batch")

    mean_x = np.mean(X_tr, axis=1)

    mean_x = np.reshape(mean_x, (-1, 1))

    X_tr -= mean_x
    X_val -= mean_x
    X_test -= mean_x

    am_nodes = [50] # number of nodes for the hidden layers

    dim_img = len(X_val)

    W, b = get_parameters(dim_img, am_labels, am_nodes)


    P = evaluate_classifier(X_tr[:, 0:100], W, b)

    #disp_img(X_test[:, 1])

    #print(P)

    #acc = compute_accuracy(X_tr[:, 0:100], y_tr[:100], W, b)
    #print(acc)

    djdb, djdw = compute_gradients(X_tr[:, 0:100], Y_tr[:, 0:100], P, W, 0.001)

    djdb2, djdw2 = compute_grads_num_slow(X_tr[:, 0:100], Y_tr[:, 0:100], W, b, 0.001, 0.000001)

    diff_b =  djdb - djdb2
    diff_w = djdw - djdw2

    wsum = np.sum(np.abs(diff_w)) / W.size

    #J = compute_cost(X_tr, Y_tr, W, b, 0)


    acc_before = compute_accuracy(X_test, y_test, W, b)
    print("Accuracy with random: ", acc_before)

    n_batch = 100
    eta =.01
    n_epochs = 40
    lamb = 0

    Wstar, bstar = mini_batch_GD(X_tr, X_val, Y_tr, Y_val, n_batch, eta, n_epochs, W, b, lamb)

    acc = compute_accuracy(X_test, y_test, Wstar, bstar)
    print("Accuracy after training: ", acc)

    for row in range(Wstar.shape[0]):
        W_row = Wstar[row, :]
        W_row = ((W_row - np.min(W_row)) / (np.max(W_row) - np.min(W_row)))
        disp_img(W_row)


if __name__ == "__main__":
    main()
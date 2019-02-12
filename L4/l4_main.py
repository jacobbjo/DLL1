from L4.functions import *


def main():

    book_data, char_to_ind, ind_to_char = read_file("../Datasets/goblet_book.txt")

    k = len(char_to_ind)
    m = 5
    eta = 0.1
    seq_length = 25


    h0 = np.zeros((m, 1))
    x0 = one_hot_vec(12, k)

    RNN = get_parameters(k, m, 1337)
#    b, c, U, W, V = get_parameters(k, m, 1337)
#
#    a_seq = get_seq(b, c, U, W, V, h0, x0, 50, ind_to_char)
#    print(a_seq)
#
    X_chars = book_data[0: seq_length]
    Y_chars = book_data[1: seq_length + 1]
#
    X = np.array([one_hot_vec(char_to_ind[char], k) for char in X_chars])[:, :, 0].T
    Y = np.array([one_hot_vec(char_to_ind[char], k) for char in Y_chars])[:, :, 0].T
#
#    A, H, P, l = forward_pass(b, c, U, W, V, X, Y, h0)
#
#    dldb, dldc, dldU, dldW, dldV = backward_pass(b, c, U, W, V, X, Y, A, H, P, h0)
#
#    dldb2, dldc2, dldU2, dldW2, dldV2 = ComputeGradsNum(X, Y, b, c, U, W, V, 0.0001)
#
#    diff_b = np.sum(np.abs(dldb - dldb2)) / b.size
#    diff_c = np.sum(np.abs(dldc - dldc2)) / c.size
#    diff_U = np.sum(np.abs(dldU - dldU2)) / U.size
#    diff_W = np.sum(np.abs(dldW - dldW2)) / W.size
#    diff_V = np.sum(np.abs(dldV - dldV2)) / V.size
#
#    print("diff_b: ", diff_b)
#    print("diff_c: ", diff_c)
#    print("diff_U: ", diff_U)
#    print("diff_W: ", diff_W)
#    print("diff_V: ", diff_V)


    print("lol")



    n_epochs = 20
    train_AdaGrad(n_epochs, eta, m, seq_length, book_data, char_to_ind, ind_to_char, RNN)





def train_AdaGrad(n_epochs, eta, m, seq_length, book_data, char_to_ind, ind_to_char, RNN):
    eps = 0.0000001
    k = len(char_to_ind)
    ada_grads = get_zero_grads(k, m)
    smooth_loss = 0
    x0 = one_hot_vec(30, k)  # H
    iteration = 0

    for ep in range(n_epochs):
        e = 0
        hprev = np.zeros((m, 1))

        while not e > len(book_data)-seq_length:
            X_chars = book_data[e: e + seq_length]
            Y_chars = book_data[e + 1: e + seq_length + 1]

            X = np.array([one_hot_vec(char_to_ind[char], k) for char in X_chars])[:, :, 0].T
            Y = np.array([one_hot_vec(char_to_ind[char], k) for char in Y_chars])[:, :, 0].T

            A, H, P, l = forward_pass(*RNN, X, Y, hprev)

            gradients = backward_pass(*RNN, X, Y, A, H, P, hprev)

            hprev = H[:, -1]

            # AdaGrad
            for g in range(len(gradients)):
                ada_grads[g] += np.power(gradients[g], 2)
                RNN[g] -= (eta * gradients[g]/(np.sqrt(ada_grads[g] + eps)))

            if iteration == 0:
                smooth_loss = l

            smooth_loss = 0.999 * smooth_loss + 0.001 * l

            if iteration%500 == 0 or iteration == 1:
                print("Iteration:", iteration, "smooth loss:", smooth_loss)

                #smooth_loss_list = [smooth_loss_list smooth_loss];
                #iterations_list = [iterations_list iterations];


            if iteration%500 == 0 or iteration == 1:
                x0 = X[:, 1:2]
                print(get_seq(*RNN, hprev, x0, 200, ind_to_char))


            iteration += 1
            e += seq_length




if __name__ == "__main__":
    main()
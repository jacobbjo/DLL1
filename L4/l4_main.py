from L4.functions import *


def main():

    book_data, char_to_ind, ind_to_char = read_file("../Datasets/goblet_book.txt")

    k = len(char_to_ind)
    m = 100
    eta = 0.1
    seq_length = 25

    h0 = np.zeros((m, 1))
    x0 = one_hot_vec(12, k)

    RNN = get_parameters(k, m, 12)

#    b, c, U, W, V = get_parameters(k, m, 1337)
#
#    a_seq = get_seq(b, c, U, W, V, h0, x0, 50, ind_to_char)
#    print(a_seq)
#
#    X_chars = book_data[0: seq_length]
#    Y_chars = book_data[1: seq_length + 1]
#
#    X = np.array([one_hot_vec(char_to_ind[char], k) for char in X_chars])[:, :, 0].T
#    Y = np.array([one_hot_vec(char_to_ind[char], k) for char in Y_chars])[:, :, 0].T
#
#    A, H, P, l = forward_pass(*RNN, X, Y, h0)
#
#    dldb, dldc, dldU, dldW, dldV = backward_pass(*RNN, X, Y, A, H, P, h0)
#
#    dldb2, dldc2, dldU2, dldW2, dldV2 = ComputeGradsNum(X, Y, *RNN, 0.0001)
#
#    diff_b = np.sum(np.abs(dldb - dldb2)) / RNN[0].size
#    diff_c = np.sum(np.abs(dldc - dldc2)) / RNN[1].size
#    diff_U = np.sum(np.abs(dldU - dldU2)) / RNN[2].size
#    diff_W = np.sum(np.abs(dldW - dldW2)) / RNN[3].size
#    diff_V = np.sum(np.abs(dldV - dldV2)) / RNN[4].size
#
#    print("diff_b: ", diff_b)
#    print("diff_c: ", diff_c)
#    print("diff_U: ", diff_U)
#    print("diff_W: ", diff_W)
#    print("diff_V: ", diff_V)

    n_epochs = 3

    NEW_RNN = train_AdaGrad(n_epochs, eta, m, seq_length, book_data, char_to_ind, ind_to_char, RNN)

    print("".join(get_seq(*NEW_RNN, h0, x0, 1000, ind_to_char)))


if __name__ == "__main__":
    main()
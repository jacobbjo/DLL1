from functions import *

def main():
    am_labels = 10
    X_tr, Y_tr, y_tr = readfile("Datasets/data_batch_1")
    X_val, Y_val, y_val = readfile("Datasets/data_batch_2")
    X_test, Y_test, y_test = readfile("Datasets/test_batch")

    dim_img = len(X_val)


    W = np.random.normal(0, 0.01, (am_labels, dim_img))
    b = np.random.normal(0, 0.01, (am_labels, 1))

    #P = evaluate_classifier(X_tr[:, 0:100], W, b)
    #print(P)

    acc = compute_accuracy(X_tr[:, 0:100], y_tr[:100], W, b)
    print(acc)





    disp_img(X_test[:,1])


if __name__ == "__main__":
    main()
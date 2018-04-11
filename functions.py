import matplotlib.pyplot as plt
import numpy as np


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


def softmax(s):
    return np.exp(s)/np.sum(np.exp(s), axis=0)


def evaluate_classifier(X, W, b):
    s = np.matmul(W, X) + b
    P = softmax(s)
    return P


def compute_cost(X, Y, W, b, lamb):
    """the sum of the loss of the network’s predictions for the images in X relative to
    the ground truth labels and the regularization term on W"""

    P = evaluate_classifier(X, W, b)
    cross = -1/X.shape[1] * np.sum(np.log(np.matmul(np.transpose(Y), P)))
    J = cross + lamb * np.sum(np.power(W, 2))
    return J


def compute_accuracy(X, y, W, b):
    P = evaluate_classifier(X, W, b)
    preds = np.argmax(P, axis=0) #the predicted class for each picture
    are_equal = np.sum(preds == y)
    return are_equal/preds.shape[0]


def compute_gradients(X, Y, P, W, lamb):
    # From lecture 3 we can deduct that we need:
    # 𝛿J/𝛿b =
    # 𝛿J/𝛿s * 𝛿s/𝛿b =
    # 𝛿J/𝛿p * 𝛿p/𝛿s * 𝛿s/𝛿b =
    # 𝛿J/𝛿l * 𝛿l/𝛿p * 𝛿p/𝛿s * 𝛿s/𝛿b =
    # 𝛿J/𝛿J * 𝛿J/𝛿l * 𝛿l/𝛿p * 𝛿p/𝛿s * 𝛿s/𝛿b

    # And:
    # 𝛿J/𝛿W =
    # 𝛿J/𝛿z * 𝛿z/𝛿W + 𝛿J/𝛿r * 𝛿r/𝛿w =
    # 𝛿J/𝛿s * 𝛿s/𝛿z * 𝛿z/𝛿W + 𝛿J/𝛿J * 𝛿J/𝛿r * 𝛿r/𝛿W

    # So:

    djdj = 1

    djdl = 1 #*djdj

    dldp = np.sum((Y * (-1/P)), axis=1)




    djdr = lamb


    print("lol")


def disp_img(image):
    plt.imshow(np.transpose(image.reshape(3, 32, 32), (1, 2, 0)))
    plt.show()


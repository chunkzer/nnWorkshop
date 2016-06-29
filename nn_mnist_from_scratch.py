import numpy as np
import os, struct, array


def load_mnist(dataset='training', path='.', digits=np.arange(10)):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def one_hot(i):
    return np.eye(10)[i]

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, index):
    return np.exp(x) / np.sum(np.exp(x))

# derivada de la funcion sigmoid
def derivSigmoid(x):
    return x * (1 - x)


# read the data
images, labels = load_mnist("training", path='data/mnist/')

# parameters
batch_size = 10
np.random.seed(1)
perceptronsHidden = 100
epochs = 2000
learning_rate = 0.02

# initialization

Wxh = 2 * np.random.random((784, perceptronsHidden)) - 1
Whs = 2 * np.random.random((perceptronsHidden, 10)) - 1
Wsy = 2 * np.random.random((10, 10))

# create batches for training
for batch in xrange(len(images) / batch_size):
    X = []
    Y = []
    for i in range(batch * batch_size, (batch + 1) * batch_size):
        # input: data de entrenamiento. dim: batch_size * 784
        X.append(images[i].flatten())
        # expected output: batch_size * 10
        Y.append(one_hot(labels[i][0]))

    # layers:
    l0 = np.array(X)
    l1 = sigmoid(np.dot(l0, Wxh))
    l2 = sigmoid(np.dot(l1, Whs))

    lsoft = softmax(np.dot(l2, Wsy), )

    normalized_l2 = np.array([softmax(layer2) for layer2 in l2])
    # print "Softmax output size: " + `normalized_l2.shape`
    # todo: apply a better error understaing, like cross-entropy
    output_error = Y - normalized_l2

    #Cross Entropy:
    #C = -1/batch_size * (Y * np.log(normalized_l2) + (1 - Y) * np.log(1 - normalized_l2))
    C = (- np.array(Y) * np.log(normalized_l2)).mean()
    # print "Cross Entropy: " + `C`
    # stop = raw_input()
    # gradient descent
    l2_delta =  C * derivSigmoid(normalized_l2)
    hidden_error = l2_delta.dot(Whs.T)
    l1_delta = hidden_error * derivSigmoid(l1)

    # weight adjustment
    Whs += learning_rate * l1.T.dot(l2_delta)
    Wxh += learning_rate * l0.T.dot(l1_delta)

    print "error: " + str(np.mean(np.abs(output_error)))

# add the testing dataset for validation and calculate error.

## Este primer bloque solo carga los datos que necesitamos para el entrenamiento.
import numpy as np
import os, struct, array
import collections


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


# create batches for training
def training(images, labels, Wxh, Why, batch_size, learning_rate):
    # todo iterar correctamente
    for batch in xrange(0):
        X = []
        Y = []
        #creamos los batches
        for i in range(batch * batch_size, (batch + 1) * batch_size):
            # TODO llenar esto
            X.append()
            # todo llenar esto tambien
            Y.append()

        # TODO llenar los layers
        l0 = 0
        l1 = 0
        l2 = 0

        #Lo demas se queda igual, es la misma logica
        cost = np.array(Y) - l2

        # gradient descent
        l2_delta = cost * derivSigmoid(l2)
        hidden_error = l2_delta.dot(Why.T)
        l1_delta = hidden_error * derivSigmoid(l1)

        if batch % 100 == 0:
        print "Error:" + str(np.mean(np.abs(cost)))

        # weight adjustment
        Why += learning_rate * l1.T.dot(l2_delta)
        Wxh += learning_rate * l0.T.dot(l1_delta)

    # add the testing dataset for validation and calculate error.
    print "Finished."
    return Wxh, Why


# Funcion de validacion
def validate(imagesT, labelsT, Wxh, Why, batch_size):
    total7 = totalNon7 = correct7 = falsePositive = 0
    for batch in xrange(len(imagesT) / batch_size):
        X = []
        Y = []
        for i in range(batch * batch_size, (batch + 1) * batch_size):
            X.append(imagesT[i].flatten())
            Y.append([1] if (labelsT[i][0]) == 7 else [0])

        l0 = np.array(X)
        l1 = sigmoid(np.dot(l0, Wxh))
        l2 = sigmoid(np.dot(l1, Why))
        Y = np.array(Y)

        for i in range(len(l2)):
            if l2[i] > 0.5:
                l2[i] = 1
            else:
                l2[i] = 0

        total7 += collections.Counter(Y.flatten())[1]
        totalNon7 += collections.Counter(Y.flatten())[0]
        correct7 += collections.Counter(Y.flatten() + l2.flatten())[2]
        falsePositive += collections.Counter(Y.flatten() - l2.flatten())[-1]

    print "Correct Lucky7: " + `correct7`
    print "Guessed 7, but was wrong: " + `falsePositive`
    print "Total 7s: " + `total7`
    print "Total NonSevens: " + `totalNon7`
    print "Accuracy of our model: " + `100.0 * correct7 / total7`


# Funciones de activacion / utileria
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivSigmoid(x):
    return x * (1 - x)

##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Data
images, labels = load_mnist("training", path='data/mnist/')
imagesT, labelsT = load_mnist("testing", path='data/mnist/')

# Parametros y Modelo
np.random.seed(1)

#todo llenar esto
batch_size = 0
hidden_cells = 0
learning_rate = 0

#todo llenar esto
Wxh = 2 * np.random.random(()) - 1
Why = 2 * np.random.random(()) - 1

# Entrenamineto

validate(imagesT, labelsT, Wxh, Why, batch_size)
Wxh, Why = training(images, labels, Wxh, Why, batch_size, learning_rate)
validate(imagesT, labelsT, Wxh, Why, batch_size)

#####

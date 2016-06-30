import numpy as np

# Nuestras funciones de activación.
def sigmoid()

def sigmoidDeriv()
#Datos de entrada.
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# Nuestra salida objetivo.
y = np.array([[0],[1],[1],[0]])

#Parametros
epochs = # 1 epoch = 1 fwd pass / 1 bwd pass con todos los datos
np.random.seed(1)
#Inicializando los pesos entre las capas.
Wxh =
Why =

#Ciclo de entrenamiento
def training(X, Y, Wxh, Why, epoch)
    for j in xrange(epoch):
        l0 =  # Input layer
        l1 =  # Hidden layer
        l2 =  # Output layer

        Costo =

        if(j % 10000) == 0:
            print "Error:" + str(np.mean(np.abs(Costo)))

        l2_delta =
        l1_error =
        l1_delta =

        Why +=
        Wxh +=

        print "Output after training."
        print l2

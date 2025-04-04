import numpy as np


class MLP:
    def __init__(self, n_entrées, couche_cachés, n_sorties, alpha):
        self.n_entrées = n_entrées  # nombre d'entrées (sans le biais)
        self.couche_cachés = couche_cachés  # Liste du nombre de neurones dans chaque couche cachée
        self.n_sorties = n_sorties  # nombre de neurones de sortie.
        self.alpha = alpha
        # Initialisation des poids et biais
        self.W = []  # Liste des matrices de poids
        self.b = []  # Liste des biais
        self.A = []  #liste pour stoker les activation de chaque cuche

        # Couche d'entrée  à première couche cachée
        self.W.append(np.random.randn(couche_cachés[0], n_entrées))
        self.b.append(np.ones((couche_cachés[0], 1)))

        # (nombre de neurones dans la couche suivante) x (nombre de neurones dans la couche actuelle + 1) (le +1 est pour le biais).
        for i in range(1, len(couche_cachés)):
            self.W.append(np.random.randn(couche_cachés[i], couche_cachés[i - 1]))
            #self.b.append(np.random.randn(couche_cachés[i], 1))
            self.b.append(np.ones((couche_cachés[i], 1)))
        # Dernière couche cachée à la couche de sortie
        """ self.W.append(np.random.randn(1, couche_cachés[-1]))
        self.b.append(np.random.randn(n_sorties, 1))"""
        self.W.append(np.random.randn(n_sorties, couche_cachés[-1]))
        self.b.append(np.ones((n_sorties, 1)))

    # Fonction d'activation sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Dérivée de la sigmoid
    def derivé_sigmoid(self, x):
        return x * (1 - x)

    # Fonction de propagation vers l'avant POUR STOKER LES ACTIVATIONS DES COUCHE
    def forward(self, X):
        """
        Propagation vers l'avant à travers le réseau.
        :X -- Données d'entrée (n_entrées, n_exemples)
        :return -- Sortie du réseau aprés activation.
        """
        A= X.T
        self.A = [A]  # stockage des activationd'entrée
        # propagation a travers les cauches cachees et la cauche de sortie
        for i in range(len(self.couche_cachés) + 1):
            Z = np.dot(self.W[i],A) + self.b[i]
            #z est la valeur avant l'application de la fonction d'activation
            A = self.sigmoid(Z) #la foction d'activation
            self.A.append(A) #sauvgarde de l'activation
        return A


    def backward(self, X, Y):

        Y=Y.reshape(-1,1)
        m = 1# un seule exemple traite a la fois
        #calcul de l'erreur de sortie:On mesure l'écart entre la sortie actuelle et la sortie attendue.
        deltas=[(self.A[-1] - Y) * self.derivé_sigmoid(self.A[-1])]
        #calcule des deltas pour les cauches cachees
        #Propagation de l'erreur en arrière :
        for i in range(len(self.W)-1, 0, -1):
            delta = np.dot(self.W[i].T, deltas[0]) * self.derivé_sigmoid(self.A[i])
            deltas.insert(0, delta)
            # mise a jour des poids et biais
            for i in range(len(self.W)):
                self.W[i] -= self.alpha * np.dot(deltas[i], self.A[i].T)/m
                self.b[i] -= self.alpha * np.sum(deltas[i],axis=1, keepdims=True)/m



    def train(self, X, Y, N_ITER=1000):
        for _ in range(N_ITER):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)  # melanger les exemples
            for i in indices:
                self.forward(X[i].reshape(1, -1))  # Propagation avant pour un seul exemple
                self.backward(X[i].reshape(1, -1), Y[i])  ## Mise à jour après chaque exemple

        # PREDICTION

    def prediction(self, X):
        return self.forward(X)




"""--------------------------------------------------------------------------------------------------------------------------------------"""

#Exemple de l'XOR

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
mlp = MLP(n_entrées=2, couche_cachés=[4],n_sorties=1,alpha=0.1)
mlp.train(X,Y,N_ITER=5000)
predections = mlp.prediction(X)
print("Predection apres entrainement : ", predections.T)
#-------------------4eme QST --------------------------
data_set = np.loadtxt("data.txt")
X = data_set[:,:-1]#extrait les donne sauf le derniere
Y = data_set[:,-1]
print("X : \n ",X,"\n---------------")
print("Y : \n ",Y,)

X = (X- X.mean(axis = 0)/ X.std(axis = 0))
mlp1=MLP(n_entrées=3,couche_cachés=[5], n_sorties=1,alpha=0.01)
mlp1.train(X,Y,N_ITER=50)

predections=mlp1.prediction(X)
print("---------",mlp1.prediction(X))
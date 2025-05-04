import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


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

    def accuracy(self,pred,y_true):

        pred_lbs = (pred > 0.5).astype(int) #convertir les prediction en valeur binaire
        return np.mean(pred_lbs.flatten() == y_true.flatten()) # calculer la precision en comparant les prediction avec les vrais valeur

    def sauvegarder_meill_cordoo(self, X_test, Y_test, predictions, fichier='Fichie_test.txt'):
        predi = (predictions > 0.5).astype(int).flatten()
        Y_test_flat = Y_test.flatten()
        coord = []
        # extraire les cordonnees
        for i in range(len(Y_test)):
            if predi[i] == Y_test_flat[i]:
                x, y, z = X_test[i][:3]

                coord.append([x, y, z])
        # Sauvga si les cordds sont correct
        if coord:
            df = pd.DataFrame(coord, columns=["x", "y", "z"])
            df.to_csv(fichier, index=False, sep=' ', header=False)
            print(f"{len(df)} coordonnees sauvgarder dans le ficier {fichier}")
        else:
            print("Aucune coordonnee correct a sauvgarder.")


    def prdic_et_sauvgarder_coords(self,fichier, fichier_sortie):
        #charger des coordonn
        coords = np.loadtxt(fichier)
        #prediction
        prediction=self.prediction(coords)
        pred_labels = (prediction>0.5).astype(int).flatten()
        reslts = np.hstack((coords, pred_labels.reshape(-1,1)))
        #sauvgarde
        np.savetxt(fichier_sortie, reslts, fmt='%.15f %.15f %.15f %d ',delimiter=' ')
        print(f"{len(reslts)} coordonnees avec prediction sauvgarder dans '{fichier_sortie}'")


    def sauvgarder_poids(self, fichier = 'poids_MLP'):
        donnees =[]
        for i,(w, b) in enumerate(zip(self.W, self.b)):
            for j in range(w.shape[0]):
                ligne = {
                    'cauche':i,
                    'neurone':j,
                    'poids':w[j].tolist(), #liste des poids du neurone j
                    'biais':b[j][0]#Bai de neu j

                }
                donnees.append(ligne)
        df = pd.DataFrame(donnees)
        df.to_csv(fichier, index=False)
        print(f"les poids et les biais sauvgarder dans le fichier {fichier}")


"""    def calculer_r(self,X):
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        z=X[:, 2]
        return np.column_stack((r,z))"""

"""--------------------------------------------------------------------------------------------------------------------------------------"""

#Exemple de l'XOR

"""X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
mlp = MLP(n_entrées=2, couche_cachés=[8],n_sorties=1,alpha=0.1)
mlp.train(X,Y,N_ITER=300)
predections = mlp.prediction(X)
print("Predection apres entrainement : ", predections.T)"""
#-------------------4eme QST --------------------------
data_set = np.loadtxt("data.txt")
X = data_set[:,:-1]#extrait les donne sauf le derniere
Y = data_set[:,-1]
print("X : \n ",X,"\n---------------")
print("Y : \n ",Y,)

ens_entr = int(0.8 * X.shape[0])
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#Melange sleatroirement des indices
X_entr, X_test = X[indices[:ens_entr]], X[indices[ens_entr:]]
Y_entr, Y_test = Y[indices[:ens_entr]], Y[indices[ens_entr:]]

mlp1 = MLP(n_entrées=X_entr.shape[1],couche_cachés=[10], n_sorties=1,alpha=0.01)
mlp1.train(X_entr,Y_entr,N_ITER=300)
predections=mlp1.prediction(X_test)
#fonctin pour cree le fichier test
#mlp1.sauvegarder_meill_cordoo(X_test, Y_test)
mlp1.sauvgarder_poids("poids_biais.csv")

#fonc pour la predict des coord de fichier test
mlp1.prdic_et_sauvgarder_coords('Fichie_test','resultats.txt')

"""print("--Predection: \n ",predections.T)
print("Y:",Y_test)"""

acc = mlp1.accuracy(predections, Y_test)
print(f"Accurcy sur le test : {acc * 100} %")
#----------------------- 5 QST --------------------------------
#obtenir les prediction pour les donnees de test

predictions_test = mlp1.prediction(X_test)
prediction_labs = (predictions_test > 0.5).astype(int).flatten()
#couleur selon la prediction
colors = ['red' if label == 1 else 'blue' for label in prediction_labs]
#tracer les points dans l'espace 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=colors, s=10)
ax.set_title("Reconstruction du vase a partir es predictions MLP")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
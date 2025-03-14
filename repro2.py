# pylint: disable=all
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
        self.b.append(np.random.randn(couche_cachés[0], 1))

        # (nombre de neurones dans la couche suivante) x (nombre de neurones dans la couche actuelle + 1) (le +1 est pour le biais).
        for i in range(1, len(couche_cachés)):
            self.W.append(np.random.randn(couche_cachés[i], couche_cachés[i - 1]))
            self.b.append(np.random.randn(couche_cachés[i], 1))

        # Dernière couche cachée à la couche de sortie
        """ self.W.append(np.random.randn(1, couche_cachés[-1]))
        self.b.append(np.random.randn(n_sorties, 1))"""
        self.W.append(np.random.randn(n_sorties, couche_cachés[-1]))
        self.b.append(np.random.randn(n_sorties, 1))

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


 

import random
import graphviz as g
import pydot
import os
import pandas as pd
import numpy as np
import uuid



class DiGraphe():

    # Matrice de transition
    A: np.array
    # Ensemble des sommets
    E: list
    # Nombre de sommets
    nb_sommets: int
    # Graphe
    graph: g.Digraph
    # Seuil à partir duquel on considère qu'il est inutile de représenter un arc
    proba_mini: float


    def __init__(self, A_df: pd.DataFrame, proba_mini: float = 0.01, proba_faible: float = 0.1):
        self.A = np.array(A_df.transpose())
        self.E = [col for col in A_df.columns]
        self.nb_sommets = len(self.E)
        self.graph = g.Digraph(format="png")
        # Permet de générer des couleurs de façon aléatoire pour chaque sommet
        self.r = lambda: random.randint(0, 200)
        # proba minimale pour prendre l'arc en considération
        self.proba_mini = 0.01
        # proba considérée comme faible (orange)
        self.proba_faible = proba_faible
        # Couleurs par défaut des sommets du graphe
        self.couleurs = [
            (200, 0, 0),
            (0, 0, 200),
            (0, 200, 0),
            (200, 200, 0),
            (int(0.6*200), 0, int(0.6*200)),
            (0, int(0.8*200), int(0.8*200)),
            (200, int(0.4*200), 0),
            (0, 0, 0),
            (200, int(0.6*200), int(0.8*200)),
            (0, int(0.4*200), 0)
        ]

        # Initialisation du graphe
        self.__initialiser()

    def __initialiser(self):
        """
        VA : Permet de définir les attributs du graphe à partir de la matrice de transition
        :return: None
        """
        # Ajout des sommets
        for i in range(len(self.E)):
            self.couleurs.append((self.r(), self.r(), self.r()))
            self.graph.node(self.E[i], self.E[i], color="black", fillcolor='#%02X%02X%02X' % self.couleurs[i], style="filled", fontcolor="white")
        # Ajouter les arcs
        for y in range(self.nb_sommets):
            for i in range(self.nb_sommets):
                # On ne réprésente pas les chemins impossibles ou presque
                if self.A[i][y] > self.proba_mini:
                    print(f"{self.A[i][y]} > {self.proba_mini}")
                    # On met la couleur orange pour les probas faibles 
                    if self.A[i][y] < self.proba_faible:
                        self.graph.edge(self.E[y], self.E[i], label=f"{round(self.A[i][y], 2)}", color="orange")
                    elif self.A[i][y] >= self.proba_faible:
                        self.graph.edge(self.E[y], self.E[i], label=f"{round(self.A[i][y], 2)}", color="red")

    def enregistrer_PNG(self, chemin_dot: str, chemin_png):
        """
        VA : Permet d'enrehistrer le graphe au format PNG à partir d'un dot file
        :param chemin_dot: Chemin où le dot file doit être enregistré
        :param chemin_png: Chemin où le graphe doit être enregistré au format png
        :return: None
        """
        # Sauvegarde du fichier .dot
        self.graph.save(chemin_dot + "graphe.dot")
        # Lecture du dot file qui contient le graphe
        (graphe,) = pydot.graph_from_dot_file(chemin_dot + "graphe.dot")
        #"static/images/graphe.png"
        # Enregistrement au format png du graphe

        graphe.write_png(chemin_png)
        # Suppression du dot file qui est désormais inutile
        os.remove(chemin_dot + "graphe.dot")

    def get_couleurs_01(self):
        """Permet d'obtenir l'équivalent des couleurs 8 bits en pourcentage
        exemple : (255, 0, 127) => (1, 0, 0.5)
        Returns:
            _type_: _description_
        """
        couleurs_01 = []
        for couleur in self.couleurs:
            couleur_01 = (couleur[0]/255, couleur[1]/255, couleur[2]/255)
            couleurs_01.append(couleur_01)
        return couleurs_01



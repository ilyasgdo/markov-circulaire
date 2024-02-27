import pandas as pd
import os
from python_classes.graphe import DiGraphe
from python_classes.plots import Plots
import json
from django.conf import settings
import numpy as np
from numpy.linalg import matrix_power
import csv
# Ajout du dossier bin aux variables d'environnemen

os.environ["PATH"] += ";.."
os.environ["PATH"] += ";.\\bin"


"""
+++++++++++++++++++++++++++++++++++++++++++++++
                   ZONE API
+++++++++++++++++++++++++++++++++++++++++++++++
"""


def get_resume(id_variable: str, unite: str):
    """
    FONCTION DE L'API
    Permet de récupérer le résumé qui correspond aux paramètres

    Args:
        id_variable (str): ID de la variable circulaire (0 à n)
        unite (str): Unité souhaiter pour le résumé

    Returns:
        str: résumé - format json -> dictionnaire 
    """
    with open('out/resumes_circulaires.json', 'r', encoding="utf8") as f:
        data = json.load(f)
    return str(data[id_variable][unite])


def get_moyenne_unite(id_variable: str, unite: str):
    """
    FONCTION DE L'API
    Permet de récupérer la liste des moyennes d'une variable circulaire au format souhaité 

    Args:
        id_variable (str): ID de la variable circulaire (0 à n)
        unite (str): Unité souhaiter pour le résumé

    Returns:
        list[str]: liste des moyennes
    """
    with open('out/resumes_circulaires_moyennes.json', 'r', encoding="utf8") as f:
        data = json.load(f)
    return str(data[id_variable][unite])



"""
+++++++++++++++++++++++++++++++++++++++++++++++
                 ZONE MARKOV
+++++++++++++++++++++++++++++++++++++++++++++++
"""

#fonction pour calculer le temps de convergence
def find_convergence_time(P, stable_matrix, tolerance_factor=0.001, max_iterations=100000):
    current_matrix = np.identity(P.shape[0])

    for iteration in range(1, max_iterations + 1):
        current_matrix =  np.linalg.matrix_power(P, iteration)

        temp = current_matrix[0, :]
        difference = np.abs(temp - stable_matrix)
        max_difference = np.max(difference)


        if max_difference < tolerance_factor:
            return iteration

    # La convergence n'a pas été atteinte dans le nombre maximal d'itérations
    return -1


#fonction pour trouver la loi stable
def loiStable(max_power=100):
    # Répertoire contenant les fichiers CSV
    repertoire_matricecsv = os.path.join(settings.BASE_DIR, 'CsvGraphe')

    sauvegardeLoiStable = []

    # Parcours de tous les fichiers dans le répertoire
    for fichier_csv in os.listdir(repertoire_matricecsv):
        if fichier_csv.endswith(".csv"):
            # Construction du chemin complet du fichier CSV
            chemin_fichier_csv = os.path.join(repertoire_matricecsv, fichier_csv)

            A_df = pd.read_csv(chemin_fichier_csv)

            P = A_df.to_numpy()

            # Initialiser avec une puissance aléatoire
            pi = np.random.rand(P.shape[0])

            # Normaliser pour obtenir une distribution de probabilité
            pi /= np.sum(pi)

            # Nombre maximal d'itérations
            max_iterations = 1500

            # Seuil de convergence
            epsilon = 1e-6

            iteration_count = None  # Initialisez la variable à None ou à une valeur appropriée avant la boucle

            for iteration_count in range(max_iterations):
                # Effectuer une multiplication matricielle pour obtenir la prochaine itération
                next_pi = np.dot(pi, matrix_power(P, max_power))

                # Normaliser la loi stable
                next_pi /= np.sum(next_pi)

                # Vérifier la convergence en comparant avec l'itération précédente
                if np.linalg.norm(next_pi - pi) < epsilon:
                    pi = next_pi
                    break

                pi = next_pi


            rounded_pi = [round(val, 3) for val in pi.real]

            # Afficher la loi stable
            print("Loi Stable:", pi.real)

            # Calculer le temps de convergence
            convergence_time = find_convergence_time(P, pi.real)
            print("Temps de convergence:", convergence_time)

            sauvegardeLoiStable.append((rounded_pi, convergence_time))

    return sauvegardeLoiStable



def sauvegarder_graphe(A_file, format):
    """Permet de créer la chaine de markov et de l'enregistrer au format choisi

    Args:
        A_file (_type_): Chemin vers le fichier markov.csv
        format (_type_): Format auquel on souhaite enregistrer la chaine de markov 

    Returns:
        (list, Graph): (liste des couleurs, graphe (objet))
    """

    # Répertoire contenant les fichiers CSV
    repertoire_matricecsv = os.path.join(settings.BASE_DIR, 'CsvGraphe')

    # Parcours de tous les fichiers dans le répertoire
    for fichier_csv in os.listdir(repertoire_matricecsv):
        if fichier_csv.endswith(".csv"):
            # Construction du chemin complet du fichier CSV
            chemin_fichier_csv = os.path.join(repertoire_matricecsv, fichier_csv)

            # Lecture du fichier CSV et création du DataFrame
            A_df = pd.read_csv(chemin_fichier_csv)

            # Instanciation et sauvegarde du graphe
            graphe = DiGraphe(A_df, proba_mini=0.000001, proba_faible=0.1)

            # Enregistrement de l'image PNG dans le répertoire spécifié
            nom_image = f"{fichier_csv.split('.')[0]}.png"



            chemin_image = os.path.join("./static/graphe", nom_image)
            graphe.enregistrer_PNG("", chemin_image)

    print("Opération terminée.")

    return graphe.get_couleurs_01(), graphe





"""
+++++++++++++++++++++++++++++++++++++++++++++++
                 ZONE PLOTS
+++++++++++++++++++++++++++++++++++++++++++++++
"""


def sauvegarder_plot(couleurs, params):
    """Permet de sauvegarder l'ensemble des plots

    Args:
        couleurs (list): Couleurs du graphe pour garder les mêmes 

    Returns:
        Plots: Objet qui contient toutes les informations sur les plots
    """

    # On vérifie si les fichiers sont présents pour ne pas générer d'erreur
    if "continue.csv" in os.listdir("data"):
        df_continue = pd.read_csv("data/continue.csv")
    else:
        df_continue = None

    if "circulaire.csv" in os.listdir("data"):
        df_circulaire = pd.read_csv("data/circulaire.csv")
    else:
        df_circulaire= None

    if "discrete.csv" in os.listdir("data"):
        df_discrete = pd.read_csv("data/discrete.csv")
    else:
        df_discrete= None

    # Instaciation et sauvegarde des différents plots 
    plots = Plots(couleurs, df_continue, df_circulaire, df_discrete, params)
    plots.enregistrer("png")

    return plots


def noms_plots(n: int, type: str, lois_circulaire=None):
    """Permet de générer les chemins vers les images des plots (utile pour le HTML)

    Args:
        n (int): Nombre de plots
        type (str): Type des plots (circulaire, continue, discret ?)
        lois_circulaire (_type_, optional): Liste des lois circulaires par défaut. Defaults to None.

    Returns:
        list: Liste des chemin vers les images
    """
    noms = []
    for i in range(n):
        # Dans le cas des plots circulaires
        if lois_circulaire is not None:
            for unite in ["angle", "jour", "semaine", "heure"]:
                nom = f"/images/plot_{type}_{lois_circulaire[i]}_{i}_{unite}.png"
                noms.append(nom)
        # Dans le cas "normal"
        else:
            nom = f"/images/plot_{type}_{i}.png"
            noms.append(nom)
    return noms



def hexa_couleurs(couleurs):
    # Définition des couleurs
    nouvelles_couleurs = []
    for couleur in couleurs:
        # On convertit les couleurs en hexadécimale pour le plot
        nouvelles_couleurs.append('#%02x%02x%02x' % (int(couleur[0] * 255), int(couleur[1] * 255), int(couleur[2] * 255)))
    return nouvelles_couleurs


"""
+++++++++++++++++++++++++++++++++++++++++++++++
                 ZONE TABLEAU EN BAS LA 
+++++++++++++++++++++++++++++++++++++++++++++++
"""

def lire_fichier_csv(nom_fichier):
    matrice = []

    with open(nom_fichier, newline='', encoding='utf-8-sig') as csvfile:
        lecteur = csv.reader(csvfile, delimiter=',')
        for ligne in lecteur:
            # Ignorer les lignes avec des commentaires (commençant par //)
            if not ligne[0].startswith('//'):
                # Vérifier que toutes les lignes ont la même longueur
                if len(ligne) > 0 and len(ligne) == len(matrice[-1] if matrice else ligne):
                    # Convertir les chaînes en nombres si possible
                    ligne_numerique = [float(valeur) if valeur.replace('.', '', 1).isdigit() else valeur for valeur in ligne]
                    matrice.append(ligne_numerique)

    return np.array(matrice)


def ranger_series_par_model(nom_fichier):
    matrice = lire_fichier_csv(nom_fichier)


    # Trouver l'indice de la colonne avec la plus grande valeur pour chaque ligne
    indices_max = np.argmax(matrice, axis=1)

    series_par_model = {model: [] for model in set(indices_max)}

    # Remplir le dictionnaire en ajoutant chaque série à son modèle correspondant
    for serie, model in enumerate(indices_max):
        series_par_model[model].append(serie + 1)  # Ajouter 1 pour correspondre aux indices des séries

    return series_par_model

FICHIER_TAU = os.path.join(settings.BASE_DIR, 'CsvTau', 'tau.csv')




#;)
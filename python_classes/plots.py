from re import S
from tracemalloc import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.use('agg')
import math
from pkg_resources import ensure_directory
import scipy.stats as stats
from PIL import Image
import os
import math
from scipy.stats import vonmises    
import seaborn as sns
from python_classes.convertisseur_radians import ConvertisseurRadians
import threading
import json
import statistics as stat


class Plots:

    # Dataframe qui permet de lire et traiter les données upload par l'utilisateur
    df_continue: pd.DataFrame
    df_circulaire: pd.DataFrame
    df_discret: pd.DataFrame
    # Liste qui contient le nom des variables étudiées (CO2, dCO2, ...)
    nom_variables: dict
    # Liste de dataframes qui contiennent les infos de chaque variable
    dataframe_variables: dict
    # Ce dictionnaire contient toutes les informations utiles à la génération des résumés
    resume_variables: dict
    # Contient la liste des couleurs des états
    couleurs: list
    # Liste qui contient les lois par défaut pour les variables circulaires
    defaut_circulaire: list
    #Paramètres de production des plots
    params:dict

    def __init__(self, couleurs, df_continue, df_circulaire, df_discret, params):
        """Constructeur

        Args:
            couleurs (list): Liste des couleurs qui est générée par la classe graphe
            df_continue (_type_): Dataframe qui contient les données sur les variables continues 
            df_circulaire (_type_): idem mais pour circulaire
            df_discret (_type_): idem mais pour discret
        """

        self.df_continue = df_continue
        self.df_circulaire = df_circulaire
        self.df_discret = df_discret
        self.params = params

        self.nom_variables = {
            "continue": [],
            "circulaire": [],
            "discrete": [],
            "discrete_valeurs": []
        }
        self.dataframe_variables = {
            "continue": [],
            "circulaire": [],
            "discrete": [],
            "discrete_valeur": []
        }
        self.resume_variables = {
            "moyennes_circulaires": [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],],
            "moyennes_discretes": [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],],
            "moyennes_continues": [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],],
            "continue": [],
            "circulaire": {},
            "circulaire_moyennes": {},
            "circulaire_defaut": [],
            "discrete": []
        }

        
        self.couleurs = couleurs
        
        self.defaut_circulaire = []

        self.__extraction()

    def __extraction(self):
        """
        VA : Permet d'extraire les données qui contiennent les informations sur les variables
        :return: None
        """
        if self.df_continue is not None:
            self.__extraire_infos_df(self.df_continue, "continue")
        if self.df_circulaire is not None:
            self.__extraire_infos_df(self.df_circulaire, "circulaire")
        if self.df_discret is not None:
            self.__extraire_infos_df(self.df_discret, "discrete")

    def __extraire_infos_df(self, df, nature: str):
        # Quand on lit le dataframe, pandas considère que les premières valeurs sont en fait les noms des colonnes
        # ce qui n'est pas le cas, cette variable permet de palier au problème
        premiere_variable = True
        
        # Les indices permettent de séparer le df
        indice_debut: int = 0
        indice_fin: int = 0
        
        # Contiendra les véritables noms des colonnes pour les différents sous-dataframe
        colonnes = []
        
        for i in range(len(df)):
            
            # Si on rencontre le séparateur, alors on passe à une nouvelle variable
            if df[df.columns[0]][i] == "//":
                
                # Extraction du nom de la variable
                self.nom_variables[nature].append(df[df.columns[1]][i])
                
                # Si on est dans un cas discret alors le traitement est différent, permet également de ne pas perturber
                # le fonctionnement pour les autres types de variables
                if nature == "discrete":
                    if not premiere_variable:
                        # On ne prend pas en compte la ligne qui contient le nom des valeurs
                        indice_debut += 1
                        noms_colonnes = df[indice_debut-1:indice_debut].dropna(axis=1, how='all')
                        # Permet de récupérer le nom des vale
                        for key in noms_colonnes.keys():
                            colonnes.append(noms_colonnes[key][indice_debut-1])
                    else:
                        premiere_variable = False
                        colonnes = df.columns
                    
                # Extraction des moyenens et variances pour la variable
                # dropna() -> Supprime les colonnes où il n'y a que des NaN
                sub_df = df[indice_debut:indice_fin].dropna(axis=1, how='all')
                
                # On renomme les colonnes
                if nature == "discrete":
                    sub_df.columns = colonnes
                
                self.dataframe_variables[nature].append(sub_df)
                indice_debut = indice_fin + 1
                colonnes = []
            indice_fin += 1


    def enregistrer_variable_continue(self, ax, compteur: int):
        """
        VA : Permet de définir un subplot pour visualiser des variables continues
        :param ax: Plot
        :param compteur: indice qui permet de faire le lien avec la variable étudiée
        :return: None
        """

        # Extraction des données qui concernent la variable étudiée
        data = np.array(self.dataframe_variables["continue"][compteur])
        # On parcourt chaque ligne qui se présente sous la forme (moyenne-variance)
        for i in range(len(data)):
            # Extraction de la moyenne et de la variance
            moyenne = float(data[i][0])
            variance = float(data[i][1])
            # la pdf de scipy utilise l'ecart-type et non la variance
            sigma = math.sqrt(variance)
            # Simulation d'un espace continu
            x = np.linspace(moyenne - 3 * sigma, moyenne + 3 * sigma, 1000)
            y = stats.norm.pdf(x, moyenne, sigma)
            # Ajout de la courbe correpond à l'état au plot 
            ax.plot(x, y, c=self.couleurs[i])
            ax.fill_between(x=x, y1=y, color=self.couleurs[i], alpha=.05)
            #legend.append(f"Etat {i + 1}")
            self.resume_variables["moyennes_continues"][compteur].append(moyenne)
        # Paramètrage du plot
        ax.set_title(self.nom_variables["continue"][compteur])

    def cauchy_pdf(self, x, c, mu):
        # Fonction de densité de la loi de cauchy
        fraction1 = 1 / (2 * math.pi)
        numerateur = 1 - c ** 2
        denominateur = (1 + c ** 2) - 2 * c * math.cos(x - mu)
        if numerateur == 0 or denominateur == 0:
            return 0
        fraction2 = numerateur / denominateur
        return fraction1 * fraction2

    def y_ticks_legend(self, y_ticks):
        # Cette fonction est utilisée pour la génération de plot circulaire
        legend = []
        for tick in y_ticks:
            # On ajoute 1 à la valeur du tick (car le cercle blanc est de rayon 1) et
            # on l'arrondi à 2 chiffres après la virgule
            legend.append(str(round(tick + 1, 2)))
        return legend

    def meilleure_loi_etat(self, variance: float, meilleure_loi: list):
        # En fonction de la variance, on voit quelle loi semble adaptée
            if variance <= 1:
                meilleure_loi.append("cauchy")
            else:
                meilleure_loi.append("vonmises")

    def calculer_y_circulaire(self, x, variance, mu, y_ticks, loi_proba):
        # On calcule les valeurs en y à l'aide des fonctions de densité
        if loi_proba == "vonmises":
            y = [vonmises.pdf(nb, variance, loc = mu) + 1 for nb in x]
            y_ticks.append(vonmises.pdf(mu, variance, loc = mu) + 1)
        elif loi_proba == "cauchy":
            y = [self.cauchy_pdf(nb, variance, mu) + 1 for nb in x]
            y_ticks.append(self.cauchy_pdf(mu, variance, mu) + 1)
        return y

    #Méthode permettant de vérifier qu'un des labels n'est pas trop proche des labels natifs (0, pi, pi/2)
    def label_apte(self, radian, x_ticks):
        apte = True
        for tick in x_ticks:
            if (tick > radian and (tick - radian) < 0.1) or (radian > tick and (radian - tick) < 0.1):
                apte = False
        return apte

    def x_ticks_circulaire(self, x_ticks, x_ticks_label, compteur, unite):
        # Permet d'ajouter une légende à la moyenne des courbes
        for mu_circulaire in self.resume_variables["moyennes_circulaires"][compteur]:
            # Si mu est négatif, on doit ajouter un tour entier autour du cercle trigo pour 
            # le rendre positif, sinon on génère un 'bug' d'affichage
            if self.label_apte(mu_circulaire, x_ticks):
                if mu_circulaire < 0:
                    x_ticks.append(mu_circulaire + math.pi * 2)
                    x_ticks_label.append(str(ConvertisseurRadians.radian_to_unite(mu_circulaire + math.pi * 2, unite)))

                else:
                    x_ticks.append(mu_circulaire)
                    x_ticks_label.append(str(ConvertisseurRadians.radian_to_unite(mu_circulaire, unite)))

    def parametrage_plot_circulaire(self, ax, x, y_ticks, compteur, loi_proba, unite: str):
        # Paramètrage du plot
        x_ticks = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        x_ticks_label = []
        for tick in x_ticks :
            x_ticks_label.append(str(ConvertisseurRadians.radian_to_unite(tick, unite)))
        self.x_ticks_circulaire(x_ticks, x_ticks_label, compteur, unite)
        ax.set_title(f"{self.nom_variables['circulaire'][compteur]} ({loi_proba})")
        ax.set_xticks(x_ticks, x_ticks_label)
        y_ticks.append(1)
        #ax.set_yticks(y_ticks, self.y_ticks_legend(y_ticks))
        # Permet de remplir l'aire du cercle de rayon 1 en blanc
        ax.fill_between(x=x, y1=1, color="white", alpha=1)

    #Permet de d'aplatir les données au niveau d'un maximum pour créer un effet de zoom à la génération du Plot.
    def trancon_valeurs(self, y, max):
        for index in range(len(y)):
            if y[index] > max:
                y[index] = max
        return y
    
    #Méthode permettant de jauger quelle maximum serait le meilleur nativement
    #Note : ne sert pas ici - l'utilisateur peut choisir le max - passe dans constructeur -> params
    def ecart_valeurs(self, y):
        valPlus = []
        std = np.std(y)
        for val in y:
            if val > std:
                valPlus.append(val)
        return stat.mean(valPlus)

    def enregistrer_variable_circulaire(self, ax, compteur: int, loi_proba: str, unite: str, enregistre: bool):
        #On vérifie que l'utilisateur ait entré un max - fonctionnalité de zoom.
        if('maxCirculaire' in self.params):
            if float(self.params['maxCirculaire']) > 0:
                max = self.params['maxCirculaire']
        else:
            max = 1.6
        # Liste qui aidera à décider la meilleure loi
        meilleure_loi = []
        # légende en y (fonction_de_densite(mu))
        y_ticks = []
        # Extraction des données qui concernent la variable
        data = np.array(self.dataframe_variables["circulaire"][compteur])
        # Simulation d'un espace continu allant de -pi à pi avec 1000 valeurs entre
        x = np.linspace(-math.pi, math.pi, 1000)
        # On parcourt chaque ligne qui contient deux colonnes (moyenne - variance)
        for i in range(len(data)):
            # Extraction de la variance et de la moyenne
            mu = float(data[i][0])
            variance = float(data[i][1])
            self.meilleure_loi_etat(variance, meilleure_loi)
            y = self.calculer_y_circulaire(x, variance, mu, y_ticks, loi_proba)
            #On va applatir les valeurs au maximum pour réaliser un zoom sur les données (on modifie le max)
            y = self.trancon_valeurs(y, float(max))
            # Ajout de la courbe de l'état au plot de la variable
            ax.plot(x, y, lw=1, c=self.couleurs[i])
            ax.fill_between(x=x, y1=y, color=self.couleurs[i], alpha=.05)
            #On traite plusieurs fois les mêmes valeurs mais avec une mise en forme différente
            #On enregistre donc quand il faut les valeurs de résumé (une fois par modèle)
            if loi_proba == "cauchy" and enregistre:
                self.resume_variables["moyennes_circulaires"][compteur].append(mu)
        self.parametrage_plot_circulaire(ax, x, y_ticks, compteur, loi_proba, unite)
        # Si toutes les variances sont inférieures à 1, alors il y a fort à parier
        # que la loi de cauchy est la plus adapatée
        if loi_proba == "cauchy":
            if len(meilleure_loi) == meilleure_loi.count("cauchy"):
                self.defaut_circulaire.append("cauchy")
            else:
                self.defaut_circulaire.append("vonmises")


    def enregistrer_variable_discrete(self, compteur, df):
        # Pour réaliser le diagramme en baton, il faut restructurer la matrice
        df_dict = df.to_dict()
        # Nouveau dataframe qui correspond à la nouvelle structure
        new_df = pd.DataFrame(columns=["proba", "val", "etat"])
        # Contiendra la probabilité maximale, utile pour les dimensions max du plot (0 à val_max en y)
        val_max = 0 

        for valeur in df_dict.keys(): # valeur 1, 2, ...
            x_ticks = [] # -> Légende des barres
            i = 0
            for etat in df_dict[valeur].keys(): # 0, ..., n état
                i+=1
                # Mise à jour de la probabilité maximale
                if float(df_dict[valeur][etat]) > float(val_max):
                    val_max = float(df_dict[valeur][etat])
                # Ajout d'une ligne au dataframe
                new_df = pd.concat([pd.DataFrame([[float(df_dict[valeur][etat]), valeur, etat]], columns=new_df.columns), new_df], ignore_index=True)
                # Mise à jour des ticks
                x_ticks.append(f"état {i}")

        # Définition des couleurs
        couleurs = []
        for couleur in self.couleurs:
            # On convertit les couleurs en hexadécimale pour le plot
            couleurs.append('#%02x%02x%02x' % (int(couleur[0] * 255), int(couleur[1] * 255), int(couleur[2] * 255)))
        # Définition du plot
        g = sns.catplot(
                x="etat",
                y="proba", 
                col="val",
                data=new_df,
                saturation=.99, # visuel
                kind="bar",
                aspect=.5, # ratio (taile)
                palette=sns.color_palette(couleurs)) # palette de couleur

        (g.set_axis_labels("", "Probabilité")
          .set_xticklabels(x_ticks)
          .set_titles("{col_name} ")
          .set(ylim=(0, val_max))
          .despine(right=True)) # Permet d'afficher l'échelle

        g.savefig(f"static/images/plot_discrete_{compteur}")
        
        return couleurs

    


    def traitement_continue(self, format: str):
        # On enregistre seulement si il y a des variables continues pour éviter les erreurs
        if len(self.dataframe_variables["continue"]) > 0:
            # Boucle autant de fois qu'il y a de variables continues
            for i in range(len(self.dataframe_variables["continue"])):
                # Plot
                fig = plt.figure(figsize=(7, 7))
                # Sous-plot unique
                ax = fig.add_subplot(111)
                # Enregistrement de la variable
                self.enregistrer_variable_continue(ax, i)
                fig.savefig(f"static/images/plot_continue_{i}.{format}")


    def traitement_circulaire(self, format: str):
        # On enregistre seulement si il y a des variables circulaires pour éviter les erreurs
        if len(self.dataframe_variables["circulaire"]) > 0:
            # Boucle autant de fois qu'il y a des variables circulaires
            for i in range(len(self.dataframe_variables["circulaire"])):
                # Pour chaque variable, on enregistre avec la loi de cauchy et de vonmises
                for loi in ["cauchy", "vonmises"]:
                    enregistre = True
                    for unite in ["angle", "jour", "semaine", "heure"]:
                        # Plot
                        fig = plt.figure(figsize=(7, 7))
                        # Sous-plot unit avec projection polaire (ciruclaire)
                        ax = fig.add_subplot(111, projection="polar")
                        #Pour chaque unité dans les deux lois, on va injecter différentes unités (journée, heures...)
                        #On enregistre les données dans le résumé global uniquement à la première itération
                        # L'enregistrement retourne la meilleure loi détectée (loi par défaut)
                        self.enregistrer_variable_circulaire(ax, i, loi, unite, enregistre)
                        fig.savefig(f"static/images/plot_circulaire_{loi}_{i}_{unite}.{format}")
                        plt.close(fig)
                        enregistre = False



    def traitement_discrete(self):
        # Idem que pour les deux autres
        if len(self.dataframe_variables["discrete"]) > 0:
            for i in range(len(self.dataframe_variables["discrete"])):
                couleurs = self.enregistrer_variable_discrete(i, self.dataframe_variables["discrete"][i])

        self.couleurs = couleurs

    def enregistrer(self, format: str):
        """
        VA : Enregistre tous les plots 
        :param format: Format auquel on souhaite enregistrer les images
        :return: None
        """
        threads = [
            threading.Thread(target=self.traitement_continue, args=(format,)), 
            threading.Thread(target=self.traitement_circulaire, args=(format,)),
            threading.Thread(target=self.traitement_discrete) ]
        
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


        self.resume_continue()
        self.resume_circulaire()
        self.resume_discret()

    def resume_circulaire_unite(self, unite, i):
        """Génère le résumé (texte) des variables circulaires en fonction de l'unité

        Args:
            unite (str): (jour, semaine, heure, angle)
            i (int): id de la variable circulaire 
        """

        
        self.resume_variables["circulaire_moyennes"][i][unite] = []

        # Première partie du résumé
        resume = {}
        compteur = 1
        # Pour chaque moyenne des états contenus dans la varibale 
        for mu in self.resume_variables["moyennes_circulaires"][i]:
            # Le convertisseur est programmé pour gérer l'intervalle [0; 2pi]
            # si mu est négatif, on ajoute donc un tour complet pour avoir son équivalent positif
            if mu < 0:
                mu = mu + 2 * math.pi
            # Calcul de la valeur en fonction de l'unité
            if unite == "angle":
                valeur = ConvertisseurRadians.angle(mu)
            elif unite == "jour":
                valeur = ConvertisseurRadians.journee(mu)
            elif unite == "semaine":
                valeur = ConvertisseurRadians.semaine(mu)
            elif unite == "heure":
                valeur = ConvertisseurRadians.heure(mu)

            self.resume_variables["circulaire_moyennes"][i][unite].append(valeur)
            # Ajout de la suite du résumé
            resume[f"état {compteur}"] = valeur
            compteur += 1
        self.resume_variables["circulaire"][i][unite] = resume
        # On ajoute le résumé en angle en tant que résumé par défaut de la variable
        if unite == "angle":
            self.resume_variables["circulaire_defaut"].append(resume)
        

    def resume_circulaire(self):
        """Permet de générer les résumés des variables circulaires
        """

        # Pour chaque variables circulaire, on enregistre les résumés avec chaque unité
        for i in range(len(self.nom_variables["circulaire"])):
            self.resume_variables["circulaire"][i] = {}
            self.resume_variables["circulaire_moyennes"][i] = {}
            self.resume_circulaire_unite("angle", i)
            self.resume_circulaire_unite("jour", i)
            self.resume_circulaire_unite("semaine", i)
            self.resume_circulaire_unite("heure", i)
        
        # Sauvegarde des résumés au format JSON pour l'API
        with open("out/resumes_circulaires.json", "w", encoding="utf8") as outfile:
            json.dump(self.resume_variables["circulaire"], outfile, ensure_ascii=False)
        
        with open("out/resumes_circulaires_moyennes.json", "w", encoding="utf8") as outfile:
            json.dump(self.resume_variables["circulaire_moyennes"], outfile, ensure_ascii=False)


    def resume_discret(self):
        """Permet de générer le résumé des variables discrètes
        """
        for variable in self.dataframe_variables["discrete"]:
            # Le résumé se présente sous la forme d'un tableau
            tableau_signe = []
            # Convertion du dataframe en float
            variable = variable.T.astype(float)
            for column in variable.keys():
                # On calcul la moyenne de chaque variable qui va permettre de décider si un état
                # est élevé, moyen ou faible en fonction de la variable et de sa moyenne
                moyenne_colonne = variable[column].mean()
                # Classement des état
                self.resume_discret_classer_probas(tableau_signe, variable[column], moyenne_colonne)
            # Formatage du tableau de signe 
            tableau_signe = np.flip(np.array(tableau_signe).reshape(variable.shape, order="F").T, axis=1)
            self.resume_variables["discrete"].append(tableau_signe)


    def resume_discret_classer_probas(self, tableau_signe: list, colonne, moyenne_colonne: float):
        """Permet de décider si la probabilité d'un état est faible ou non

        Args:
            tableau_signe (list): Tableau de résumé 
            colonne (_type_): 
            moyenne_colonne (float): Moyenne de toutes les probas de la variable
        """
        
        # Les intervalles sont arbitraires, elles peuvent être facilement modifées en changeant les valeurs
        for proba in colonne: 
            if proba < moyenne_colonne * 0.3:
                tableau_signe.append("--")
            elif moyenne_colonne * 0.3 <= proba < moyenne_colonne * 0.9:
                tableau_signe.append("-")
            elif moyenne_colonne * 0.9 <= proba < moyenne_colonne * 1.1:
                tableau_signe.append("~")
            elif moyenne_colonne * 1.1 <= proba < moyenne_colonne * 1.7:
                tableau_signe.append("+")
            elif moyenne_colonne * 1.7 <= proba:
                tableau_signe.append("++")

    def resume_continue(self):
        """Permet de générer les résumés de variables continues sous la forme d'un tableau
        """
        for i in range(len(self.nom_variables["continue"])):
            # Même méthode que pour le résumé des variables discrètes
            tableau_signe = []
            moyennes : list = self.resume_variables["moyennes_continues"][i]
            moyenne_des_moyennes : float = sum(moyennes) / len(moyennes)
            for moyenne in moyennes:
                if moyenne < moyenne_des_moyennes * 0.3:
                    tableau_signe.append("--")
                elif moyenne_des_moyennes * 0.3 <= moyenne < moyenne_des_moyennes * 0.9:
                    tableau_signe.append("-")
                elif moyenne_des_moyennes * 0.9 <= moyenne < moyenne_des_moyennes * 1.1:
                    tableau_signe.append("~")
                elif moyenne_des_moyennes * 1.1 <= moyenne < moyenne_des_moyennes * 1.7:
                    tableau_signe.append("+")
                elif moyenne_des_moyennes * 1.7 <= moyenne:
                    tableau_signe.append("++")

            self.resume_variables["continue"].append(tableau_signe)

import pandas as pd
import os
import csv
from django.conf import settings

class CirculaireConverter:


    FICHIER_TABLE_CAUCHY = os.path.join(settings.BASE_DIR, 'csvInterval', 'WCD_0.1.csv')
    FICHIER_TABLE_MISES = os.path.join(settings.BASE_DIR, 'csvInterval', 'mises20.csv')
    FICHIER_CIRCULAIRE = os.path.join(settings.BASE_DIR, 'csvInterval', 'circulaire.csv')

    def __init__(self):
        self.matrice_table_cauchy = self.convertir_en_matrice(self.FICHIER_TABLE_CAUCHY)
        self.matrice_table_mises = self.convertir_en_matrice(self.FICHIER_TABLE_MISES)
        # Initialize result storage attributes
        self.resultat_degrees = []
        self.listeMartice = []

    def convertir_en_degres(self, moyenne_rad):
        return moyenne_rad * (180 / 3.14159)

    def trouver_ligne_kappa(self, table, kappa_arrondi):
        for i, ligne in enumerate(table["dispersion"]):
            if ligne == kappa_arrondi:
                return i
        return -1

    def obtenir_ouverture_pourcentage(self, table, pourcentage):
        for i, valeur in enumerate(table.columns[2:]):
            if int(valeur.split('_')[1]) == pourcentage:
                return i + 2
        return -1

    def trouver_intervale(self, moyenne, kappa, pourcentage, table):
        kappa_arrondi = round(kappa, 1)
        ligne_kappa = self.trouver_ligne_kappa(table, kappa_arrondi)
        ouverture = self.obtenir_ouverture_pourcentage(table, pourcentage)

        print(ligne_kappa)
        print(ouverture)

        val = table.iloc[ligne_kappa, ouverture]
        intervalle_inf = int((moyenne - val) % 360)
        intervalle_sup = int((moyenne + val) % 360)
        print(intervalle_inf)
        print(intervalle_sup)
        return [intervalle_inf, intervalle_sup]

    def convertir_en_matrice(self, fichier_csv):
        return pd.read_csv(fichier_csv, delimiter=';', decimal=',')

    def convertir_graphique_en_intervales(self, type_graduation, pourcentage):
        matrices_intervales = []

        if os.path.exists(self.FICHIER_CIRCULAIRE):
            circulaire_matrices = self.lire_csv(self.FICHIER_CIRCULAIRE)

            for matrice in circulaire_matrices:
                matrice_intervales = []
                for ligne in matrice:
                    moyenne_rad, concentration = ligne[0], ligne[1]
                    moyenne_deg = self.convertir_en_degres(moyenne_rad)

                    if type_graduation == "cauchy":
                        table_utilisee = self.matrice_table_cauchy
                    elif type_graduation == "vonmises":
                        table_utilisee = self.matrice_table_mises
                    else:
                        raise ValueError("Type de graduation non pris en charge.")

                    table_utilisee = table_utilisee.apply(pd.to_numeric, errors='coerce')

                    intervalle = self.trouver_intervale(moyenne_deg, concentration, pourcentage, table_utilisee)
                    matrice_intervales.append(intervalle)

                matrices_intervales.append(matrice_intervales)
                self.resultat_degrees.append(matrice_intervales)

        return matrices_intervales

    def lire_csv(self, filename):
        matrices = []
        matrice_actuelle = []

        with open(filename, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if '//' in row[0]:
                    if matrice_actuelle:
                        matrices.append(matrice_actuelle)
                        matrice_actuelle = []
                else:
                    matrice_actuelle.append(list(map(float, row)))

        if matrice_actuelle:
            matrices.append(matrice_actuelle)

        return matrices

    def generer_intervales(self):
        return self.resultat_degrees,



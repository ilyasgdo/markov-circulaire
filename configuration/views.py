from django.shortcuts import render, redirect
from django.conf import settings

from django.http import HttpResponse, HttpRequest
from .utils import *

import pandas as pd
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, HttpRequest, JsonResponse
from .utils import *
from python_classes.intervales import CirculaireConverter
from python_classes.convertisseur_radians import ConvertisseurRadians
import math




def get_resumes(request: HttpRequest, id_variable: str, unite: str):
    """API : Permet de renvoyer les résumés des variables circulaires en fonction des arguments

    Args:
        request (HttpRequest): Requête HTTP
        id_variable (str): ID de la variable circulaire (0 à n)
        unite (str): Unité avec laquelle on souhaite interpréter le résumé (jour, heure, etc)

    Returns:
        _type_: Le résumé (en HTPP)
    """
    return HttpResponse(get_resume(id_variable, unite))


def get_moyennes_unite(request: HttpRequest, id_variable: str, unite: str):
    """API : Permet de renvoyer la liste des moyennes de la variable circulaire choisie (avec l'unité choisie)

    Args:
        request (HttpRequest): Requête HTTP
        id_variable (str): ID de la variable circulaire (0 à n)
        unite (str): Unité avec laquelle on souhaite récupérer les moyennes (jour, heure, angle, etc)

    Returns:
        _type_: Liste des moyennes avec la bonne unité (en HTTP)
    """
    return HttpResponse(get_moyenne_unite(id_variable, unite))


from django.http import JsonResponse

#pour mettre a jour les interalles
def update_interval(request):
    default_pourcentage = 80  # Pourcentage par défaut
    default_loi = "cauchy"
    default_unite = "angle"

    if request.method == 'POST':
        pourcentage = int(request.POST.get('pourcentage', default_pourcentage))
        loi = request.POST.get('loi', default_loi)
        unite = request.POST.get('unite', default_unite)

        converter = CirculaireConverter()
        intervales_degrees = converter.convertir_graphique_en_intervales(loi, pourcentage)
        intervales_radians = [
            [(math.radians(deg[0]), math.radians(deg[1])) for deg in matrice] for matrice in intervales_degrees
        ]

        intervales_temp = []

        for matrice_intervales in intervales_radians:
            texte_matrice = []
            for intervalle in matrice_intervales:
                radian_inf, radian_sup = intervalle
                converted_valueinf = ConvertisseurRadians.radian_to_unite(radian_inf, unite)
                converted_valuesup = ConvertisseurRadians.radian_to_unite(radian_sup, unite)
                texte_matrice.append((converted_valueinf, converted_valuesup))
            intervales_temp.append(texte_matrice)


        response_data = {
            'success': True,
            'message': 'Interval mis à jour avec succès.',
            'intervals': intervales_temp,
            'pourcentage': pourcentage
        }
        return JsonResponse(response_data)
    else:
        response_data = {'success': False, 'message': 'Requête invalide.', 'default_pourcentage': default_pourcentage}
        return JsonResponse(response_data)



def home(request: HttpRequest):
    params = {}
    for nomChamp in request.POST:
        params[nomChamp] = request.POST[nomChamp]

    for nomChamp in request.GET:
        params[nomChamp] = request.GET[nomChamp]

    directory_path = os.path.join(settings.BASE_DIR, 'static','graphe')

    supprimer_images(directory_path)

    FICHIER_TAU = os.path.join(settings.BASE_DIR, 'CsvTau', 'tau.csv')
    resultat_tau = lire_fichier_csv(FICHIER_TAU)
    """
    partie pour generer les tableau
    """
    print("/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§//§/§/§/§/")
    print(resultat_tau)
    print("/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§/§//§/§/§/§/")

    resultat_rangement_tau = ranger_series_par_model(FICHIER_TAU)
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print("/§/§/§/§/§/§/!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(" ")
    print(resultat_rangement_tau)
    print(" ")
    print("/§/§/§/§/§/§/!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    #
    #
    # Si l'utilisateur a introduit le fichier markov dans le dossier data
    if "markov.csv" in os.listdir("data"):
        # Chargement du graphe et récupération des couleurs
        couleurs, graphe = sauvegarder_graphe("data/markov.csv", "csv")
    else:
        # Sinon on définit des couleurs par défaut
        couleurs = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "red", "green", "blue", "cyan",
                    "magenta", "yellow", "black"]

    directory_path = os.path.join(settings.BASE_DIR, 'static','graphe')

    file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Créez une liste de dictionnaires avec le nom et le chemin de chaque fichier
    files = [{'name': f, 'path': os.path.join( 'graphe', f)} for f in file_list]

    # Chargement des différents plots, on respecte le code couleur
    plots = sauvegarder_plot(couleurs, params)

    loiStableSauvegarde = loiStable()


    # partie pour generer des intervales
    type_graduation = "cauchy"
    pourcentage = 80
    converter = CirculaireConverter()
    intervales = converter.convertir_graphique_en_intervales(type_graduation, pourcentage)

    textes_intervalles = []
    for i, matrice_intervales in enumerate(intervales):
        texte_matrice = []
        for j, intervalle in enumerate(matrice_intervales):
            texte_matrice.append(f"Matrice {i + 1} - Intervalle {j + 1}: {intervalle} degrés")
        textes_intervalles.append(texte_matrice)

    for i, matrice_intervales in enumerate(textes_intervalles):
        print(f"Matrice {i + 1} - Intervales:")
        for intervalle in matrice_intervales:
            print(intervalle)
        print("----")

    #pour faire une liste de  5 en 5 de 0  a 100
    values_for_select = list(range(0, 101, 5))





    data = {
        "plots": plots,
        "plots_continues": noms_plots(len(plots.nom_variables["continue"]), "continue"),
        "plots_circulaires": noms_plots(len(plots.nom_variables["circulaire"]), "circulaire",
                                        lois_circulaire=plots.defaut_circulaire),
        "plots_discrets": noms_plots(len(plots.nom_variables["discrete"]), "discrete"),
        "unites": ["angle", "heure", "jour", "semaine"],
        "lois": list(set(plots.defaut_circulaire)),
        "loi_defaut": plots.defaut_circulaire,
        "markov": "markov.csv" in os.listdir("data"),
        "resumes_circulaires": plots.resume_variables["circulaire_defaut"],
        "noms_circulaires": plots.nom_variables["circulaire"],
        "resumes_discrets": plots.resume_variables["discrete"],
        "noms_discrets": plots.nom_variables["discrete"],
        "noms_discrets_valeurs": plots.nom_variables["discrete_valeurs"],
        "resumes_continus": plots.resume_variables["continue"],
        "noms_continus": plots.nom_variables["continue"],
        "couleurs": hexa_couleurs(plots.couleurs) if len(plots.nom_variables["discrete"]) == 0 else plots.couleurs,
        "params": params,
        "files": files,
        "intervales": intervales,
        "intervalles": textes_intervalles,
        'textes_intervalles': textes_intervalles,
        'pourcentage':pourcentage,
        'values_for_select': values_for_select,
        'loiStable': loiStableSauvegarde,
        'files_and_loiStableInfo': zip(files, loiStableSauvegarde),
        "resultat_tau": resultat_tau,
        "resultat_rangement_tau":resultat_rangement_tau,
    }

    # On passe ici en paramètre la position des variables
    if len(plots.nom_variables["circulaire"]) > 0:
        data["indice_debut_circulaire"] = len(plots.nom_variables["continue"]) - 1
        data["indice_fin_circulaire"] = len(plots.nom_variables["continue"]) + len(plots.nom_variables["circulaire"])

    return render(request, 'configuration/resultat.html', data)

def supprimer_images(repertoire):
    # Liste tous les fichiers dans le répertoire
    fichiers = os.listdir(repertoire)

    # Parcours de tous les fichiers dans le répertoire
    for fichier in fichiers:
        chemin_fichier = os.path.join(repertoire, fichier)

        # Vérifie si le fichier est une image (vous pouvez ajuster cette vérification en fonction de vos besoins)
        if fichier.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                # Supprime le fichier
                os.remove(chemin_fichier)
                print(f"Fichier supprimé : {fichier}")
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier {fichier}: {e}")






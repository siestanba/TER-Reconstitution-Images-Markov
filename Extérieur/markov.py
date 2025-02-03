### model_choice = ising or potts / sampling_choice =  gibbs or metropolis or icm

import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
import cv2
from tqdm import tqdm
import pickle
from scipy import stats

PATH_IMG = '/images/'
nom_image0 = 'immonde3.jpg'

# Vérifie si les arguments de la ligne de commande sont présents
def est_argument_vide():
    if len(sys.argv) > 2:
        return sys.argv[1], sys.argv[2]
    else:
        return None, None

# Renvoie les 4 voisins en connexité 4 d'un pixel (i, j) dans une image de dimensions (h, w)
def voisinage_4(i, j, hauteur, largeur):
    return (
        (np.mod(i-1, hauteur), j),
        (i, np.mod(j+1, largeur)),
        (np.mod(i+1, hauteur), j),
        (i, np.mod(j-1, largeur))
    )

# Calcule l'énergie locale d'un état donné dans un champ aléatoire
def cdf_locale_4(i, j, hauteur, largeur, champ_aleatoire, etat_s, poids_aretes, poids_sommets):
    energie_locale = poids_sommets[etat_s] # commence à zéro pour gibbs
    for voisin in voisinage_4(i, j, hauteur, largeur): # on itère dans les voisins (on nous donne les 4 voisins)
        etat_voisin = champ_aleatoire[voisin[0], voisin[1]] # on récupère l'état du voisin
        energie_locale += poids_aretes[etat_s, etat_voisin] # on ajoute l'énergie de la connexion (1 si l'état est différent, 0 sinon) / on compte les voisins différents
    return energie_locale

# Calcule l'attachement des données locales à l'état donné
def attache_donnees_locale(i, j, etat_s, img, moyenne, echelle):
    return np.log(echelle[etat_s]) + (img[i, j] - moyenne[etat_s])**2 / (2 * echelle[etat_s]**2)

# Génère tous les sommets et arêtes pour une image de dimensions (h, w)
def generer_sommets_et_aretes_4(hauteur, largeur):
    for i in range(hauteur):
        for j in range(largeur):
            liste_temporaire = [(i, j)]
            if j + 1 < largeur:
                liste_temporaire += [(i, j+1)]
            if i + 1 < hauteur:
                liste_temporaire += [(i+1, j)]
            yield liste_temporaire 

# Calcule l'énergie globale du champ aléatoire
def energie_globale_4(champ_aleatoire, poids_aretes, poids_sommets):
    energie_totale = 0
    hauteur, largeur = champ_aleatoire.shape
    for element in generer_sommets_et_aretes_4(hauteur, largeur):
        etat_s = champ_aleatoire[element[0]]
        energie_totale += poids_sommets[etat_s]
        for sommet in element[1:]:
            energie_totale += poids_aretes[etat_s, champ_aleatoire[sommet]]
    return energie_totale

# Fonction d'échantillonnage de Gibbs
def echantillonnage_Gibbs(champ_aleatoire, nb_iterations, modele, nom_fichier_png=None):
    """
    Effectue un échantillonnage de Gibbs sur un champ aléatoire de Markov.

    Paramètres :
    ------------
    champ_aleatoire : ndarray
        Champ aléatoire initial (matrice représentant les états des pixels).
    nb_iterations : int
        Nombre d'itérations pour l'échantillonnage de Gibbs.
    modele : dict
        Dictionnaire contenant les paramètres du modèle MRF, 
        comme le nombre d'états, les poids des connexions et les poids des sommets.
    nom_fichier_png : str, optionnel
        Nom du fichier PNG pour sauvegarder les résultats intermédiaires.

    Retour :
    --------
    energie_globale : list
        Liste contenant l'énergie globale (calculée à chaque itération si activée).
    """

    hauteur, largeur = champ_aleatoire.shape  # Dimensions du champ aléatoire
    nb_etats = modele['nb_etats']  # Nombre d'états possibles pour chaque pixel
    poids_aretes = modele['poids_aretes']  # Poids des connexions entre pixels voisins
    poids_sommets = modele['poids_sommets']  # Poids associés aux états individuels des pixels

    # Déterminer la palette de couleurs et le titre du graphe selon le modèle
    if nb_etats == 2:
        palette = 'gist_yarg'  # Palette pour le modèle d’Ising (binaire)
        titre_plot = 'MRF : Modèle d’Ising'
    else:
        palette = 'Blues'  # Palette pour le modèle de Potts (plusieurs états)
        titre_plot = 'MRF : Modèle de Potts'

    # Sauvegarder l'état initial si un fichier PNG est spécifié
    if nom_fichier_png is not None:
        champ_img = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161)
        plt.suptitle(titre_plot)
        plt.title('Initial')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(champ_img, cmap=palette)
        nb_sous_plots = 1

    energie_globale = []  # Liste pour stocker les énergies globales

    # Boucle principale d'échantillonnage
    for k in tqdm(range(nb_iterations)): # tqdm sert à afficher une barre de progression
        # Sélection aléatoire d'un pixel (i_s, j_s) dans le champ
        i_s = np.random.randint(0, hauteur)
        j_s = np.random.randint(0, largeur)

        # Calculer les énergies locales pour chaque état possible
        energies = np.empty(nb_etats)

        # on calcule l'énergie locale pour chaque état possible (delta_U
        for i, etat in enumerate(range(nb_etats)):
            # Calcul de l'énergie locale pour l'état 'etat' au pixel (i_s, j_s)
            energies[i] = cdf_locale_4(i_s, j_s, hauteur, largeur, champ_aleatoire, etat, poids_aretes, poids_sommets)
            print(f"energies[{i}] : {energies[i]}")
        
        print(f"energie finale: {energies}")
        # on convertit les énergies locales en probabilités (distribution de Gibbs)
        probabilites = np.exp(-energies)  # Exponentielle des énergies inversées
        print(f"probabilites : {probabilites}")
        probabilites /= np.sum(probabilites)  # on normalise pour obtenir une probabilité
        print(f"probabilites : {probabilites}")

        # on choisit un nouvel état selon les probabilités calculées
        nouvel_etat = np.random.choice(list(range(nb_etats)), 1, p=probabilites)[0] # on choisit le nouvel état
        champ_aleatoire[i_s, j_s] = nouvel_etat  # on met à jour le pixel

        # Sauvegarder les visualisations intermédiaires si spécifié
        if nom_fichier_png is not None and (
            k == np.floor(0.2 * nb_iterations) or
            k == np.floor(0.4 * nb_iterations) or
            k == np.floor(0.6 * nb_iterations) or
            k == np.floor(0.8 * nb_iterations)
        ):
            champ_img = champ_aleatoire * np.floor(255 / (nb_etats - 1))
            plt.subplot(161 + nb_sous_plots)
            plt.title(str(k))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(champ_img, cmap=palette)
            nb_sous_plots += 1

    # Sauvegarder l'état final
    if nom_fichier_png is not None:
        champ_img = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161 + nb_sous_plots)
        plt.title('Final')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(champ_img, cmap=palette)
        plt.savefig(nom_fichier_png)

    return energie_globale

def echantillonnage_metropolis(champ_aleatoire, nb_iterations, modele, nom_fichier_png=None):
    """
    Fonction d'échantillonnage utilisant l'algorithme de Metropolis.

    Paramètres :
    ------------
    champ_aleatoire : ndarray
        Champ aléatoire initial (matrice représentant les états des pixels).
    nb_iterations : int
        Nombre d'itérations de l'algorithme.
    modele : dict
        Dictionnaire contenant les paramètres du modèle MRF (Markov Random Field), 
        comme le nombre d'états, les poids des connexions et les poids des sommets.
    nom_fichier_png : str, optionnel
        Nom du fichier PNG pour sauvegarder les images intermédiaires de l'évolution du champ.

    Retour :
    --------
    U_global : list
        Liste de l'énergie globale calculée à chaque itération (si activée).
    """
    hauteur, largeur = champ_aleatoire.shape
    nb_etats = modele['nb_etats']
    poids_connexions = modele['poids_connexions']
    poids_sommets = modele['poids_sommets']

    if nb_etats == 2:
        palette = 'gist_yarg'
        titre_principal = 'MRF : Modèle d’Ising'
    else:
        palette = 'Blues'
        titre_principal = 'MRF : Modèle de Potts'

    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161)
        plt.suptitle(titre_principal)
        plt.title('Initial')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        nb_sous_graphes = 1

    energie_globale = []

    for iteration in tqdm(range(nb_iterations)):
        # Sélection aléatoire d'un pixel
        i_pixel = np.random.randint(0, hauteur)
        j_pixel = np.random.randint(0, largeur)

        # État actuel du pixel et son énergie locale
        etat_actuel = champ_aleatoire[i_pixel, j_pixel]
        energie_locale_actuelle = cdf_locale_4(
            i_pixel, j_pixel, hauteur, largeur, champ_aleatoire, 
            etat_actuel, poids_connexions, poids_sommets
        )

        # Proposition d'un nouvel état
        nouvel_etat = np.random.randint(0, nb_etats)
        nouvelle_energie_locale = cdf_locale_4(
            i_pixel, j_pixel, hauteur, largeur, champ_aleatoire, 
            nouvel_etat, poids_connexions, poids_sommets
        )

        # Calcul de la différence d'énergie locale
        delta_energie_locale = nouvelle_energie_locale - energie_locale_actuelle

        # Mise à jour selon la règle de Metropolis
        if delta_energie_locale < 0:
            champ_aleatoire[i_pixel, j_pixel] = nouvel_etat
        else:
            if np.random.rand() < np.exp(-delta_energie_locale):
                champ_aleatoire[i_pixel, j_pixel] = nouvel_etat

        # Sauvegarde des images intermédiaires
        if nom_fichier_png is not None and (
            iteration == np.floor(0.2 * nb_iterations) or
            iteration == np.floor(0.4 * nb_iterations) or
            iteration == np.floor(0.6 * nb_iterations) or
            iteration == np.floor(0.8 * nb_iterations)
        ):
            img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
            plt.subplot(161 + nb_sous_graphes)
            plt.title(str(iteration))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img_champ, cmap=palette)
            nb_sous_graphes += 1

        # Calcul de l'énergie globale (désactivée ici)
        # energie_globale.append(calcul_energie_globale(champ_aleatoire, poids_connexions, poids_sommets))

    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161 + nb_sous_graphes)
        plt.title('Final')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        plt.savefig(nom_fichier_png)

    return energie_globale

def echantillonnage_ICM(champ_aleatoire, nb_iterations, modele, nom_fichier_png=None):
    """
    Fonction d'échantillonnage utilisant l'algorithme ICM (Iterated Conditional Modes).

    Paramètres :
    ------------
    champ_aleatoire : ndarray
        Champ aléatoire initial (matrice représentant les états des pixels).
    nb_iterations : int
        Nombre d'itérations de l'algorithme.
    modele : dict
        Dictionnaire contenant les paramètres du modèle MRF (Markov Random Field),
        comme le nombre d'états, les poids des connexions et les poids des sommets.
    nom_fichier_png : str, optionnel
        Nom du fichier PNG pour sauvegarder les images intermédiaires de l'évolution du champ.

    Retour :
    --------
    U_globale : list
        Liste de l'énergie globale calculée à chaque itération (si activée).
    """
    hauteur, largeur = champ_aleatoire.shape
    nb_etats = modele['nb_etats']
    poids_connexions = modele['poids_connexions']
    poids_sommets = modele['poids_sommets']

    if nb_etats == 2:
        palette = 'gist_yarg'
        titre_principal = 'MRF : Modèle d’Ising'
    else:
        palette = 'Blues'
        titre_principal = 'MRF : Modèle de Potts'

    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161)
        plt.suptitle(titre_principal)
        plt.title('Initial')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        nb_sous_graphes = 1

    energie_globale = []

    for iteration in tqdm(range(nb_iterations)):
        # Sélection aléatoire d'un pixel
        i_pixel = np.random.randint(0, hauteur)
        j_pixel = np.random.randint(0, largeur)

        # Calcul des énergies locales pour chaque état possible
        energies_locales = np.empty(nb_etats)
        for i, etat in enumerate(range(nb_etats)):
            energies_locales[i] = voisinage_4(
                i_pixel, j_pixel, hauteur, largeur, champ_aleatoire, 
                etat, poids_connexions, poids_sommets
            )

        # Sélection de l'état minimisant l'énergie locale
        nouvel_etat = np.random.choice(np.flatnonzero(energies_locales == energies_locales.min()))
        champ_aleatoire[i_pixel, j_pixel] = nouvel_etat

        # Sauvegarde des images intermédiaires
        if nom_fichier_png is not None and (
            iteration == np.floor(0.2 * nb_iterations) or
            iteration == np.floor(0.4 * nb_iterations) or
            iteration == np.floor(0.6 * nb_iterations) or
            iteration == np.floor(0.8 * nb_iterations)
        ):
            img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
            plt.subplot(161 + nb_sous_graphes)
            plt.title(str(iteration))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img_champ, cmap=palette)
            nb_sous_graphes += 1

        # Calcul de l'énergie globale (désactivée ici)
        # energie_globale.append(calcul_energie_globale(champ_aleatoire, poids_connexions, poids_sommets))

    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161 + nb_sous_graphes)
        plt.title('Final')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        plt.savefig(nom_fichier_png)

    return energie_globale

def seg_img_gris_par_mrf(image, champ_aleatoire, mu, ecart_type, nb_iterations, modele, nom_fichier_png=None):
    """
    Segmentation d'une image en niveaux de gris en utilisant un champ aléatoire de Markov (MRF).

    Paramètres :
    ------------
    image : ndarray
        Image d'entrée en niveaux de gris.
    champ_aleatoire : ndarray
        Champ aléatoire initial (matrice représentant les états des pixels).
    mu : tuple
        Moyennes des niveaux de gris pour chaque classe.
    ecart_type : tuple
        Écarts-types des niveaux de gris pour chaque classe.
    nb_iterations : int
        Nombre d'itérations de l'algorithme.
    modele : dict
        Dictionnaire contenant les paramètres du modèle MRF (nombre d'états, poids des connexions, etc.).
    nom_fichier_png : str, optionnel
        Nom du fichier pour sauvegarder les résultats intermédiaires sous forme d'image.

    Retour :
    --------
    erreur_globale : list
        Liste contenant l'erreur globale (calculée à chaque itération si activée).
    """

    hauteur, largeur = champ_aleatoire.shape  # Dimensions du champ aléatoire
    nb_etats = modele['n_etats']  # Nombre d'états possibles
    poids_aretes = modele['poids_aretes']  # Poids des interactions entre pixels voisins
    poids_sommets = modele['poids_sommets']  # Poids associés aux états individuels des pixels

    # Définir la palette et le titre en fonction du modèle
    if nb_etats == 2:
        palette = 'gray'  # Palette de couleurs pour le modèle d'Ising
        titre_principal = 'MRF : Modèle d’Ising'
    else:
        palette = 'Blues'  # Palette de couleurs pour le modèle de Potts
        titre_principal = 'MRF : Modèle de Potts'

    # Sauvegarder l'état initial si un fichier PNG est spécifié
    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161)
        plt.suptitle(titre_principal)
        plt.title('Initial')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        nb_sous_plots = 1

    erreur_globale = []

    # Boucle principale pour l'échantillonnage
    for k in tqdm(range(nb_iterations)):
        # Sélection aléatoire d'un pixel
        i_pixel = np.random.randint(0, hauteur)
        j_pixel = np.random.randint(0, largeur)

        # Calcul des énergies locales pour chaque état possible
        energies_locales = np.empty(nb_etats)
        for i, etat in enumerate(range(nb_etats)):
            # Énergie locale prioritaire
            energies_locales[i] = cdf_locale_4(i_pixel, j_pixel, hauteur, largeur, champ_aleatoire, etat, poids_aretes, poids_sommets)
            # Ajout de l'attachement aux données (énergie liée à l'image)
            energies_locales[i] += attache_donnees_locale(i_pixel, j_pixel, etat, image, mu, ecart_type)

        # Choisir le nouvel état qui minimise l'énergie locale
        nouvel_etat = np.random.choice(np.flatnonzero(energies_locales == energies_locales.min()))
        champ_aleatoire[i_pixel, j_pixel] = int(nouvel_etat)

        # Sauvegarder les états intermédiaires
        if nom_fichier_png is not None and (k == np.floor(0.2 * nb_iterations) or k == np.floor(0.4 * nb_iterations) or k == np.floor(0.6 * nb_iterations) or k == np.floor(0.8 * nb_iterations)):
            img_champ = champ_aleatoire * int(np.floor(255 / (nb_etats - 1)))
            plt.subplot(161 + nb_sous_plots)
            plt.title(str(k))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img_champ, cmap=palette)
            nb_sous_plots += 1

        # Calcul de l'erreur globale (désactivée ici, mais peut être activée)
        # img_empirique = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        # erreur_globale.append(np.linalg.norm(true_img - img_empirique, ord='fro'))

    # Sauvegarde de l'image finale
    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161 + nb_sous_plots)
        plt.title('Final')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        plt.savefig(nom_fichier_png)

    return erreur_globale









# Fonction pour choisir le modèle et l'algorithme
model_choice, sampling_choice = est_argument_vide()

if model_choice == 'ising':
    # Modèle d'Ising
    alpha_00_11 = 0.0  # Poids pour les connexions entre états identiques
    alpha_01_10 = 1.0  # Poids pour les connexions entre états différents
    Ising_model = {
        'nb_etats': 2,
        'poids_aretes': np.array([[alpha_00_11, alpha_01_10], [alpha_01_10, alpha_00_11]]),
        'poids_sommets': np.zeros(2)
    }
    # Initialisation du champ
    hrf, wrf = 50, 50  # Dimensions du champ
    rf = np.uint8(np.floor(Ising_model['nb_etats'] * np.random.rand(hrf, wrf)))  # l'état initial du champ de Markov pour le modèle d'Ising (champ aléatoire)
    rf_img = rf * np.floor(255 / (Ising_model['nb_etats'] - 1))  # Conversion en image (image 0 pour 0 (noir) et 255 pour 1 (blanc)) / on mult par 0 ou par 255

    # Nom du fichier image
    nom_fichier_png = 'ising_gibbs_1.png'

    # Choix de l'algorithme d'échantillonnage
    if sampling_choice == 'gibbs':
        rf = echantillonnage_Gibbs(rf, 17500, Ising_model, nom_fichier_png=nom_fichier_png)
    elif sampling_choice in ['metropolis', 'metro']:
        U_global = echantillonnage_metropolis(rf, 17500, Ising_model, nom_fichier_png=nom_fichier_png)
    elif sampling_choice == 'icm':
        U_global = echantillonnage_ICM(rf, 17500, Ising_model, nom_fichier_png=nom_fichier_png)

    # Affichage du résultat
    plt.imread(nom_fichier_png)
    plt.show()

elif model_choice == 'potts':
    # Modèle de Potts
    alpha_00_11_22 = 0.25
    alpha_01_10 = 0.75
    alpha_02_20 = 2.5
    alpha_12_21 = 0.75
    Potts_model = {}
    Potts_model['nb_etats'] = 3  # Le modèle de Potts contient 3 états
    Potts_model['poids_aretes'] = np.array([ 
        [alpha_00_11_22, alpha_01_10, alpha_02_20],
        [alpha_01_10, alpha_00_11_22, alpha_12_21],
        [alpha_02_20, alpha_12_21, alpha_00_11_22]
    ])
    Potts_model['poids_sommets'] = np.array([1.0, 2.0, 1.0])  # Poids pour chaque état

    # Initialisation du champ
    hrf, wrf = 75, 75  # Dimensions du champ
    rf = np.uint8(np.floor(Potts_model['nb_etats'] * np.random.rand(hrf, wrf)))  # Champ aléatoire
    rf_img = rf * np.floor(255 / (Potts_model['nb_etats'] - 1))  # Conversion en image

    # Nom du fichier image
    nom_fichier_png = 'potts_test.png'

    # Choix de l'algorithme d'échantillonnage
    if sampling_choice == 'gibbs':
        U_global = echantillonnage_Gibbs(rf, 17500, Potts_model, nom_fichier_png=nom_fichier_png)
    elif sampling_choice in ['metropolis', 'metro']:
        U_global = echantillonnage_metropolis(rf, 17500, Potts_model, nom_fichier_png=nom_fichier_png)
    elif sampling_choice == 'icm':
        U_global = echantillonnage_ICM(rf, 17500, Potts_model, nom_fichier_png=nom_fichier_png)

    # Affichage du résultat
    plt.imread(nom_fichier_png)
    plt.show()

# Affichage des images existantes (si elles existent)
if path.exists(PATH_IMG + nom_image0):
    img0 = cv2.imread(PATH_IMG + nom_image0)
    cv2.imshow('Image', img0)
    cv2.waitKey(0)
    print("image importée")


#img0 = cv2.imread('/images/immonde3.jpg')
img0 = cv2.imread('/Users/sebych/Documents/Fac-Local/Code-TER/Codes_extérieurs/images/immonde3.jpg')
cv2.imshow('Image', img0)
cv2.waitKey(0)
print("image importée avec path")

if path.exists(PATH_IMG + 'immonde3.jpg'):
    true_img = cv2.imread(PATH_IMG + 'immonde3.jpg')
    cv2.imshow('Image originale', true_img)
    cv2.waitKey(0)



# Définition du modèle d'Ising
alpha_00_11 = 0.  # Poids des interactions pour des pixels ayant le même état (0-0 ou 1-1)
alpha_01_10 = 2.  # Poids des interactions pour des pixels ayant des états différents (0-1 ou 1-0)

# Dictionnaire pour stocker les paramètres du modèle d'Ising
modele_Ising = {}
modele_Ising['n_etats'] = 2  # Nombre d'états possibles (ici 2, correspondant à 0 et 1)
# Matrice des poids des interactions entre les pixels voisins en fonction de leurs états
modele_Ising['poids_aretes'] = np.array([[alpha_00_11, alpha_01_10], 
                                         [alpha_01_10, alpha_00_11]])
# Poids associés aux états individuels des pixels (nuls ici, donc pas de biais intrinsèque)
modele_Ising['poids_sommets'] = np.zeros(modele_Ising['n_etats'])

# Initialisation d'un champ aléatoire
hauteur_rf = img0.shape[0]  # Hauteur de l'image d'entrée
largeur_rf = img0.shape[1]  # Largeur de l'image d'entrée
# Génération aléatoire d'un champ initial avec des états entre 0 et 1
champ_rf = np.uint8(np.floor(modele_Ising['n_etats'] * np.random.rand(hauteur_rf, largeur_rf)))
# Transformation du champ aléatoire en niveaux de gris pour la visualisation
champ_rf_img = champ_rf * int(np.floor(255 / (modele_Ising['n_etats'] - 1)))

# Débruitage d'image avec le modèle d'Ising
# Paramètres statistiques des deux classes (0 et 1) dans l'image en niveaux de gris
mu = (97.2, 163.9)  # Moyennes des niveaux de gris pour les classes
ecart_type = (22.4, 22.7)  # Écarts-types des niveaux de gris pour les classes

# Nom du fichier PNG pour sauvegarder l'image résultante
nom_fichier_png = 'test_avec_Ising.png'

# Appel à la fonction de segmentation en niveaux de gris par un MRF
# Entrées :
# - Image bruitée : img0[:,:,0]
# - Image originale (vraie image) : true_img[:,:,0]
# - Champ aléatoire initial : champ_rf
# - Moyennes et écarts-types des classes : mu, ecart_type
# - Nombre d'itérations : 1000000
# - Modèle d'Ising défini précédemment : modele_Ising
# - Nom du fichier de sauvegarde : nom_fichier_png
Erreur_globale = seg_img_gris_par_mrf(img0[:, :, 0], true_img[:, :, 0], champ_rf, 
                                     mu, ecart_type, 1000000, modele_Ising, 
                                     nom_fichier_png=nom_fichier_png)

# Lecture et affichage de l'image segmentée débruitée
plt.imread(nom_fichier_png)
plt.show()
### sampling_choice =  gibbs or metropolis

import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

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


# Fonction d'échantillonnage de Gibbs
def echantillonnage_Gibbs(champ_aleatoire, nb_iterations, modele, nom_fichier_png=None):
    hauteur, largeur = champ_aleatoire.shape  # Dimensions du champ aléatoire
    nb_etats = modele['nb_etats']  # Nombre d'états possibles pour chaque pixel
    poids_aretes = modele['poids_aretes']  # Poids des connexions entre pixels voisins
    poids_sommets = modele['poids_sommets']  # Poids associés aux états individuels des pixels

    # Déterminer la palette de couleurs et le titre du graphe selon le modèle
    if nb_etats == 2:
        palette = 'gist_yarg'  # Palette pour le modèle d’Ising (binaire)
        titre_plot = 'MRF : Modèle d’Ising avec échantillonnage de Gibbs'
    else:
        palette = 'Blues'  # Palette pour le modèle de Potts (plusieurs états)
        titre_plot = 'MRF : Modèle de Potts avec échantillonnage de Gibbs'

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
            #print(f"energies[{i}] : {energies[i]}")
        
        #print(f"energie finale: {energies}")
        # on convertit les énergies locales en probabilités (distribution de Gibbs)
        probabilites = np.exp(-energies)  # Exponentielle des énergies inversées
        #print(f"probabilites : {probabilites}")
        probabilites /= np.sum(probabilites)  # on normalise pour obtenir une probabilité
        #print(f"probabilites : {probabilites}")

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
    hauteur, largeur = champ_aleatoire.shape
    nb_etats = modele['nb_etats']
    poids_aretes = modele['poids_aretes']
    poids_sommets = modele['poids_sommets']

    if nb_etats == 2:
        palette = 'gist_yarg'
        titre_principal = 'MRF : Modèle d’Ising avec échantillonnage de Metropolis'
    else:
        palette = 'Blues'
        titre_principal = 'MRF : Modèle de Potts avec échantillonnage de Metropolis'

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
            etat_actuel, poids_aretes, poids_sommets
        )

        # Proposition d'un nouvel état
        nouvel_etat = np.random.randint(0, nb_etats)
        nouvelle_energie_locale = cdf_locale_4(
            i_pixel, j_pixel, hauteur, largeur, champ_aleatoire, 
            nouvel_etat, poids_aretes, poids_sommets
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
        # energie_globale.append(calcul_energie_globale(champ_aleatoire, poids_aretes, poids_sommets))

    if nom_fichier_png is not None:
        img_champ = champ_aleatoire * np.floor(255 / (nb_etats - 1))
        plt.subplot(161 + nb_sous_graphes)
        plt.title('Final')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img_champ, cmap=palette)
        plt.savefig(nom_fichier_png)

    return energie_globale




########################################################################################
## Fonction pour choisir le modèle et l'algorithme ##
########################################################################################
#sampling_choice = est_argument_vide()
#sampling_choice = 'gibbs'
sampling_choice = 'metropolis'


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
nom_fichier_png = f'ising_{sampling_choice}.png'

# Choix de l'algorithme d'échantillonnage
if sampling_choice == 'gibbs':
    rf = echantillonnage_Gibbs(rf, 17500, Ising_model, nom_fichier_png=nom_fichier_png)
elif sampling_choice in ['metropolis', 'metro']:
    U_global = echantillonnage_metropolis(rf, 17500, Ising_model, nom_fichier_png=nom_fichier_png)

# Affichage du résultat
plt.imread(nom_fichier_png)
plt.show()
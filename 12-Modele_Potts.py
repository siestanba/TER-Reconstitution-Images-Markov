import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

def ajouter_bruit_gaussien_discret(champ, sigma, nb_etats):
    bruit = np.random.normal(0, sigma, size=champ.shape)
    champ_bruite = np.round(champ + bruit).astype(int)
    champ_bruite = np.clip(champ_bruite, 0, nb_etats - 1)
    return champ_bruite

def ajouter_bruit_gaussien_remplacement(champ, sigma, nb_etats):
    centre = (nb_etats - 1) / 2
    bruit_gaussien = np.random.normal(centre, sigma, size=champ.shape)
    bruit_gaussien = np.round(bruit_gaussien).astype(int)
    bruit_gaussien = np.clip(bruit_gaussien, 0, nb_etats - 1)

    # Probabilité de remplacement croissante avec sigma
    proba_remplacement = 1 - np.exp(-sigma / 10.0)  # plus sigma est grand, plus la proba approche 1
    masque = np.random.rand(*champ.shape) < proba_remplacement

    return np.where(masque, bruit_gaussien, champ)

def ajouter_bruit(champ, p, nb_etats):
    bruit = np.random.choice(range(nb_etats), size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)

def cdf_locale_4(i, j, H, L, champ, etat, poids_aretes, poids_sommets):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = poids_sommets[etat]
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

def gibbs_classique(champ, nb_iter, modele):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="Gibbs classique (Potts)"):
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        energies = np.array([
            cdf_locale_4(i, j, H, L, champ, s, modele['poids_aretes'], modele['poids_sommets']) 
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
    return champ

def metropolis_classique(champ, nb_iter, modele):
    H, L = champ.shape
    T = 1
    for k in tqdm(range(nb_iter), desc="Metropolis classique (Potts)"):
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        etat_actuel = champ[i, j]
        nouvel_etat = np.random.choice([s for s in range(modele['nb_etats']) if s != etat_actuel])
        energie_actuelle = cdf_locale_4(i, j, H, L, champ, etat_actuel, modele['poids_aretes'], modele['poids_sommets'])
        energie_nouvelle = cdf_locale_4(i, j, H, L, champ, nouvel_etat, modele['poids_aretes'], modele['poids_sommets'])
        delta_E = energie_nouvelle - energie_actuelle
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            champ[i, j] = nouvel_etat
    return champ

def afficher_resultats(img_init, champ_gibbs, img_bruitee, champ_metro, nb_etats, taux, taux_base, taux_gibbs, taux_metro):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)

    imgs = [img_init, champ_gibbs, img_bruitee, champ_metro]
    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Gibbs classique \n(ε={taux_gibbs:.2f})",
        f"Image bruitée \n(ε={taux_base:.2f})", f"Metropolis classique \n(ε={taux_metro:.2f})"
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs = axs.flatten()
    for ax, titre, img in zip(axs, titres, imgs):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def taux_restauration(img_originale, img_restauree):
    pixels_corrects = np.sum(img_originale == img_restauree)
    total_pixels = img_originale.size
    taux = (pixels_corrects / total_pixels) * 100
    return taux


# === Paramètres ===
chemin_image = "images/gris.png"
sigma_bruit = 0.5
p_bruit = 0.5
nb_etats = 3
nb_iter = 100000
beta = 4

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
#img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
img_bruitee = ajouter_bruit_gaussien_discret(img, sigma_bruit, nb_etats)
#img_bruitee = ajouter_bruit_gaussien_remplacement(img, sigma_bruit, nb_etats)
champ_gibbs = img_bruitee.copy()
champ_metro = img_bruitee.copy()

#poids_aretes = {(a, b): -1 if a == b else 1 for a in range(nb_etats) for b in range(nb_etats)}
poids_aretes = {(a, b): beta * (-1 if a == b else 1) for a in range(nb_etats) for b in range(nb_etats)}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

champ_gibbs = gibbs_classique(champ_gibbs, nb_iter, modele)
champ_metro = metropolis_classique(champ_metro, nb_iter, modele)

# Calcul du taux de restauration pour les deux estimateurs

taux = taux_restauration(img, img)
taux_base = taux_restauration(img, img_bruitee)
taux_gibbs = taux_restauration(img, champ_gibbs)
taux_metro = taux_restauration(img, champ_metro)

# Affichage des résultats
afficher_resultats(img, champ_gibbs, img_bruitee, champ_metro, nb_etats, taux, taux_base, taux_gibbs, taux_metro)
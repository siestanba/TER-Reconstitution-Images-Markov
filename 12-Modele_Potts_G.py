import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement de l'image ===
def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

# === Ajout de bruit gaussien ===
def ajouter_bruit_gaussien_discret(champ, sigma, nb_etats):
    bruit = np.random.normal(0, sigma, size=champ.shape)
    champ_bruite = np.round(champ + bruit).astype(int)
    champ_bruite = np.clip(champ_bruite, 0, nb_etats - 1)
    return champ_bruite

# === Calcul d'énergie locale avec observation bruitée ===
def cdf_locale_4_bruit_gaussien(i, j, H, L, champ, etat, poids_aretes, poids_sommets, observation, sigma2):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = ((etat - observation[i, j]) ** 2) / (2 * sigma2)
    energie += poids_sommets[etat]
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

# === Gibbs adapté au bruit gaussien ===
def gibbs_bruit_gaussien(champ, observation, sigma2, nb_iter, modele):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="Gibbs bruit gaussien"):
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        energies = np.array([
            cdf_locale_4_bruit_gaussien(i, j, H, L, champ, s, modele['poids_aretes'], modele['poids_sommets'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
    return champ

# === Metropolis adapté au bruit gaussien ===
def metropolis_bruit_gaussien(champ, observation, sigma2, nb_iter, modele):
    H, L = champ.shape
    T = 1
    for k in tqdm(range(nb_iter), desc="Metropolis bruit gaussien"):
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        etat_actuel = champ[i, j]
        nouvel_etat = np.random.choice([s for s in range(modele['nb_etats']) if s != etat_actuel])
        energie_actuelle = cdf_locale_4_bruit_gaussien(i, j, H, L, champ, etat_actuel, modele['poids_aretes'], modele['poids_sommets'], observation, sigma2)
        energie_nouvelle = cdf_locale_4_bruit_gaussien(i, j, H, L, champ, nouvel_etat, modele['poids_aretes'], modele['poids_sommets'], observation, sigma2)
        delta_E = energie_nouvelle - energie_actuelle
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            champ[i, j] = nouvel_etat
    return champ

# === Affichage des résultats ===
def afficher_resultats(img_init, champ_gibbs, img_bruitee, champ_metro, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    
    taux = taux_restauration(img_init, img_init)
    taux_base = taux_restauration(img_init, img_bruitee)
    taux_gibbs = taux_restauration(img_init, champ_gibbs)
    taux_metro = taux_restauration(img_init, champ_metro)

    imgs = [img_init, champ_gibbs, img_bruitee, champ_metro]
    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Gibbs bruit gaussien \n(ε={taux_gibbs:.2f})",
        f"Image bruitée \n(ε={taux_base:.2f})", f"Metropolis bruit gaussien \n(ε={taux_metro:.2f})"
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    for ax, titre, img in zip(axs, titres, imgs):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# === Calcul du taux de restauration ===
def taux_restauration(img_originale, img_restauree):
    pixels_corrects = np.sum(img_originale == img_restauree)
    total_pixels = img_originale.size
    taux = (pixels_corrects / total_pixels) * 100
    return taux

# === Paramètres ===
chemin_image = "images/test1.png"
sigma_bruit = 0.8
nb_etats = 3
nb_iter = 200000
beta = 1

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit_gaussien_discret(img, sigma_bruit, nb_etats)
champ_gibbs = img_bruitee.copy()
champ_metro = img_bruitee.copy()

poids_aretes = {(a, b): beta * (-1 if a == b else 1) for a in range(nb_etats) for b in range(nb_etats)}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

sigma2 = sigma_bruit ** 2
champ_gibbs = gibbs_bruit_gaussien(champ_gibbs, img_bruitee, sigma2, nb_iter, modele)
champ_metro = metropolis_bruit_gaussien(champ_metro, img_bruitee, sigma2, nb_iter, modele)

# === Affichage final ===
afficher_resultats(img, champ_gibbs, img_bruitee, champ_metro, nb_etats)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

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
    for k in tqdm(range(nb_iter), desc="Gibbs classique"):
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

def afficher_resultats(img_init, img_bruitee, gibbs, metro, nb_etats, taux_gibbs, taux_metro):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)

    imgs = [img_init, gibbs, img_bruitee, metro]
    titres = [
        "Image originale", "Gibbs classique",
        "Image bruitée", "Metropolis classique"
    ]
    taux = [None, taux_gibbs, None, taux_metro]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 = 4 images
    axs = axs.flatten()

    for i, (ax, titre, img, taux_rest) in enumerate(zip(axs, titres, imgs, taux)):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
        
        if taux_rest is not None:
            # Ajouter le taux sous le titre
            ax.text(0.5, -0.2, f'Taux de restauration: {taux_rest:.2f}%', ha='center', va='center', transform=ax.transAxes, fontsize=15)

    plt.tight_layout()
    plt.show()

def taux_restauration(img_originale, img_restauree):
    pixels_corrects = np.sum(img_originale == img_restauree)
    total_pixels = img_originale.size
    taux = (pixels_corrects / total_pixels) * 100
    return taux

# === Paramètres ===
chemin_image = "images/souris.png"
#chemin_image = "images/chat.jpg"
nb_etats = 2
p_bruit = 0.3
nb_iter = 100000
beta = 1

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
champ_gibbs = img_bruitee.copy()
champ_metro = img_bruitee.copy()

#poids_aretes = {(a, b): -1 if a == b else 1 for a in range(nb_etats) for b in range(nb_etats)}
poids_aretes = {
    (a, b): -beta * (1 if a == b else -1)
    for a in range(nb_etats) for b in range(nb_etats)
}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

champ_gibbs = gibbs_classique(champ_gibbs, nb_iter, modele)
champ_metro = metropolis_classique(champ_metro, nb_iter, modele)

# Calcul du taux de restauration pour les deux estimateurs
taux_gibbs = taux_restauration(img, champ_gibbs)
taux_metro = taux_restauration(img, champ_metro)

# Affichage des résultats
afficher_resultats(img, img_bruitee, champ_gibbs, champ_metro, nb_etats, taux_gibbs, taux_metro)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement et quantification de l'image ===
def charger_image_grayscale(path, nb_etats):
    """
    Charge une image, la convertit en niveaux de gris, puis la quantifie sur 'nb_etats' niveaux.
    Cela correspond à x ∈ {0, ..., nb_etats - 1}^S.
    """
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

# === Ajout de bruit : simulateur du modèle d'observation P(Y|X) ===
def ajouter_bruit(champ, p, nb_etats):
    """
    Simule une observation bruitée y ~ P(Y|X), avec un bruit uniforme ajouté avec probabilité p.
    """
    bruit = np.random.choice(range(nb_etats), size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)

# === Calcul de l'énergie locale (U_s) pour un pixel donné ===
def energie_locale(i, j, H, L, champ, etat, poids_aretes, poids_sommets):
    """
    Calcule l'énergie locale si on assigne 'etat' au pixel (i,j), en tenant compte des voisins (cliques d'ordre 2).
    """
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = poids_sommets[etat]  # Potentiel de site (clique d'ordre 1)
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

# === Calcul de l'énergie locale pour MAP (avec attache aux données) ===
def energie_locale_map(i, j, H, L, champ, etat, poids_aretes, observation, sigma2):
    """
    même chose avec attache aux données
    """
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = ((etat - observation[i, j]) ** 2) / (2 * sigma2)
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie


# === Estimateur MAP via recuit simulé (minimisation de U(x|y)) ===
def estimateur_map(champ, observation, nb_iter, modele, sigma2, t_init=1.0):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="MAP - recuit simulé"):
        T = t_init / np.log(2 + k)
        i, j = np.random.randint(H), np.random.randint(L)
        energies = np.array([
            energie_locale_map(i, j, H, L, champ, s, modele['poids_aretes'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies / T)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
    return champ

# === Génération d'échantillons Gibbs pour MPM / TPM ===
def gibbs_mcmc(champ, nb_iter, modele, collect_start=50000, collect_every=1000):
    H, L = champ.shape
    samples = []
    for k in tqdm(range(nb_iter), desc="Gibbs MCMC"):
        i, j = np.random.randint(H), np.random.randint(L)
        energies = np.array([
            energie_locale(i, j, H, L, champ, s, modele['poids_aretes'], modele['poids_sommets'])
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
        if k >= collect_start and (k - collect_start) % collect_every == 0:
            samples.append(champ.copy())
    return samples

# === Estimateur MPM : argmax_s P(X_s | Y) ===
def estimateur_mpm(samples, nb_etats):
    H, L = samples[0].shape
    result = np.zeros((H, L), dtype=int)
    for i in range(H):
        for j in range(L):
            counts = np.bincount([s[i, j] for s in samples], minlength=nb_etats)
            result[i, j] = np.argmax(counts)
    return result

# === Estimateur TPM : E[X_s | Y] (moyenne empirique) ===
def estimateur_tpm(samples):
    return np.round(np.mean(np.array(samples), axis=0)).astype(int)

# === Visualisation ===
def afficher_resultats(img_init, img_bruitee, map_est, mpm_est, tpm_est, nb_etats, taux, taux_base, taux_map, taux_mpm, taux_tpm):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    
    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Image bruitée \n(ε={taux_base:.2f})", 
        f"MAP \n(ε={taux_map:.2f})", f"MPM \n(ε={taux_mpm:.2f})", f"TMP \n(ε={taux_tpm:.2f})"    
    ]
    images = [img_init, img_bruitee, map_est, mpm_est, tpm_est]
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for ax, titre, img in zip(axs, titres, images):
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
chemin_image = "images/test1.png"
nb_etats = 3
p_bruit = 0.3
nb_iter = 100000
beta = 0.5
sigma2 = 2

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
champ_init = img_bruitee.copy()

# Modèle de Potts binaire
#poids_aretes = {(a, b): -1 if a == b else 1 for a in range(nb_etats) for b in range(nb_etats)}
poids_aretes = {
    (a, b): beta * (-1 if a == b else 1)
    for a in range(nb_etats) for b in range(nb_etats)
}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

# MAP
map_est = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele, sigma2)

# MCMC
mcmc_samples = gibbs_mcmc(champ_init.copy(), nb_iter, modele)
mpm_est = estimateur_mpm(mcmc_samples, nb_etats)
tpm_est = estimateur_tpm(mcmc_samples)

# Calcul du taux de restauration pour les deux estimateurs
taux = taux_restauration(img, img)
taux_base = taux_restauration(img, img_bruitee)
taux_map = taux_restauration(img, map_est)
taux_mpm = taux_restauration(img, mpm_est)
taux_tpm = taux_restauration(img, tpm_est)


# Affichage final
afficher_resultats(img, img_bruitee, map_est, mpm_est, tpm_est, nb_etats, taux, taux_base, taux_map, taux_mpm, taux_tpm)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement d'une image personnalisée ===
def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

# === Ajout de bruit aléatoire ===
def ajouter_bruit(champ, p, nb_etats):
    bruit = np.random.choice(range(nb_etats), size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)

# === Énergie locale ===
def cdf_locale_4(i, j, H, L, champ, etat, poids_aretes, poids_sommets):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = poids_sommets[etat]
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

# === Échantillonneur de Gibbs pour MCMC ===
def gibbs_mcmc(champ, nb_iter, modele, collect_start=50000, collect_every=1000):
    H, L = champ.shape
    samples = []
    for k in tqdm(range(nb_iter), desc="Gibbs MCMC"):
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        energies = np.array([
            cdf_locale_4(i, j, H, L, champ, s, modele['poids_aretes'], modele['poids_sommets']) 
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
        if k >= collect_start and (k - collect_start) % collect_every == 0:
            samples.append(champ.copy())
    return samples

# === Estimateurs ===
def estimateur_mpm(samples, nb_etats):
    H, L = samples[0].shape
    result = np.zeros((H, L), dtype=int)
    for i in range(H):
        for j in range(L):
            counts = np.bincount([s[i, j] for s in samples], minlength=nb_etats)
            result[i, j] = np.argmax(counts)
    return result

def estimateur_tpm(samples):
    return np.round(np.mean(np.array(samples), axis=0)).astype(int)

def estimateur_map(champ, nb_iter, modele):
    # Recuit simulé simplifié
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="MAP - Recuit simulé"):
        T = 1 / np.log(2 + k)
        i = np.random.randint(0, H)
        j = np.random.randint(0, L)
        energies = np.array([
            cdf_locale_4(i, j, H, L, champ, s, modele['poids_aretes'], modele['poids_sommets']) 
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-energies / T)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
    return champ

# === Visualisation ===
def afficher_resultats(img_init, img_bruitee, map_est, mpm_est, tpm_est, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    
    titres = ["Image originale", "Image bruitée", "MAP", "MPM", "TPM"]
    images = [img_init, img_bruitee, map_est, mpm_est, tpm_est]
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for ax, titre, img in zip(axs, titres, images):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# === Paramètres ===
chemin_image = "images/test1.jpg"
nb_etats = 3
p_bruit = 0.3
nb_iter = 150000

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
champ_init = img_bruitee.copy()

# Modèle de Potts binaire
poids_aretes = {(a, b): -1 if a == b else 1 for a in range(nb_etats) for b in range(nb_etats)}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

# MAP
map_est = estimateur_map(champ_init.copy(), nb_iter, modele)

# MCMC
mcmc_samples = gibbs_mcmc(champ_init.copy(), nb_iter, modele)
mpm_est = estimateur_mpm(mcmc_samples, nb_etats)
tpm_est = estimateur_tpm(mcmc_samples)

# Affichage final
afficher_resultats(img, img_bruitee, map_est, mpm_est, tpm_est, nb_etats)
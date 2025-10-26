import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement et quantification de l'image ===
def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((200, 200))
    img_np = np.array(img)
    return np.floor(img_np / (256 / nb_etats)).astype(int)

# === Ajout de bruit : simulateur du modèle d'observation P(Y|X) ===
def ajouter_bruit_uniforme(champ, p, nb_etats):
    bruit = np.random.choice(range(nb_etats), size=champ.shape)
    masque = np.random.rand(*champ.shape) < p
    return np.where(masque, bruit, champ)

def ajouter_bruit_gaussien_discret(champ, sigma, nb_etats):
    bruit = np.random.normal(0, sigma, size=champ.shape)
    champ_bruite = np.round(champ + bruit).astype(int)
    champ_bruite = np.clip(champ_bruite, 0, nb_etats - 1)
    return champ_bruite

# === Calcul de l'énergie locale pour MAP (avec attache aux données) ===
def energie_locale_map(i, j, H, L, champ, etat, poids_aretes, observation, sigma2):
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    energie = ((etat - observation[i, j]) ** 2) / (2 * sigma2)
    for dx, dy in voisins:
        ni, nj = i + dx, j + dy
        if 0 <= ni < H and 0 <= nj < L:
            energie += poids_aretes[(etat, champ[ni, nj])]
    return energie

# === Énergie globale ===
def calcul_energie_globale(champ, observation, modele, sigma2):
    H, L = champ.shape
    energie = 0
    for i in range(H):
        for j in range(L):
            etat = champ[i, j]
            energie += ((etat - observation[i, j]) ** 2) / (2 * sigma2)
            for dx, dy in [(-1, 0), (0, -1)]:  # éviter double comptage
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < L:
                    energie += modele['poids_aretes'][(etat, champ[ni, nj])]
    return energie

# === Estimateur MAP via descente locale ===
def estimateur_map(champ, observation, nb_iter, modele, sigma2):
    H, L = champ.shape
    energies = []
    for k in tqdm(range(nb_iter), desc="MAP"):
        i, j = np.random.randint(H), np.random.randint(L)
        local_energies = np.array([
            energie_locale_map(i, j, H, L, champ, s, modele['poids_aretes'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        champ[i, j] = np.argmin(local_energies)
        if k % 1000 == 0:
            energies.append(calcul_energie_globale(champ, observation, modele, sigma2))
    return champ, energies

# === Estimateur MAP avec recuit simulé ===
def estimateur_map_recuit(champ, observation, nb_iter, modele, sigma2, t_init=1.0):
    H, L = champ.shape
    energies = []
    for k in tqdm(range(nb_iter), desc="MAP recuit simulé"):
        T = t_init / np.log(2 + k)
        i, j = np.random.randint(H), np.random.randint(L)
        local_energies = np.array([
            energie_locale_map(i, j, H, L, champ, s, modele['poids_aretes'], observation, sigma2)
            for s in range(modele['nb_etats'])
        ])
        proba = np.exp(-local_energies / T)
        proba /= np.sum(proba)
        champ[i, j] = np.random.choice(range(modele['nb_etats']), p=proba)
        if k % 1000 == 0:
            energies.append(calcul_energie_globale(champ, observation, modele, sigma2))
    return champ, energies

# === Visualisation comparée MAP vs MAP recuit simulé ===
def afficher_map_vs_recuit(img_init, img_bruitee, map_est, map_recuit_est, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)
    
    taux = taux_restauration(img_init, img_init)
    taux_base = taux_restauration(img_init, img_bruitee)
    taux_map = taux_restauration(img_init, map_est)
    taux_mapr = taux_restauration(img_init, map_recuit)


    images = [img_init, img_bruitee, map_est, map_recuit_est]
    #titres = ["Image originale", "Image bruitée", "MAP (descente locale)", "MAP (recuit simulé)"]

    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Image bruitée \n(ε={taux_base:.2f})", 
        f"MAP (descente locale) \n(ε={taux_map:.2f})", f"MAP (recuit simulé) \n(ε={taux_mapr:.2f})"   
    ]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, titre, img in zip(axs, titres, images):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# === Tracer l'évolution de l'énergie ===
def tracer_energie(energies_map, energies_recuit):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(energies_map)) * 1000, energies_map, label="MAP (descente locale)")
    plt.plot(np.arange(len(energies_recuit)) * 1000, energies_recuit, label="MAP (recuit simulé)")
    plt.xlabel("Itérations")
    plt.ylabel("Énergie globale")
    plt.title("Évolution de l'énergie au cours des itérations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def taux_restauration(img_originale, img_restauree):
    pixels_corrects = np.sum(img_originale == img_restauree)
    total_pixels = img_originale.size
    taux = (pixels_corrects / total_pixels) * 100
    return taux


# === Paramètres ===
chemin_image = "images/test1.png"
sigma_bruit = 0.5
p_bruit = 0.3
nb_etats = 16
nb_iter = 200000
beta = 2
sigma2 = 10
t_init = 1

# === Chargement et préparation des données ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit_uniforme(img, p_bruit, nb_etats)
#img_bruitee = ajouter_bruit_gaussien_discret(img, sigma_bruit, nb_etats)
champ_init = img_bruitee.copy()

# === Définition du modèle Potts ===
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

# === Estimation MAP classique et recuit simulé ===
map_est, energie_map = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele, sigma2)
map_recuit, energie_recuit = estimateur_map_recuit(champ_init.copy(), img_bruitee, nb_iter, modele, sigma2, t_init)

# === Affichage des résultats ===
afficher_map_vs_recuit(img, img_bruitee, map_est, map_recuit, nb_etats)
tracer_energie(energie_map, energie_recuit)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# === Chargement et quantification de l'image ===
def charger_image_grayscale(path, nb_etats):
    img = Image.open(path).convert("L").resize((100, 100))
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

# === Estimateur MAP avec recuit simulé ===
def estimateur_map(champ, observation, nb_iter, modele, sigma2, t_init=1.0):
    H, L = champ.shape
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
    return champ

# === Visualisation comparée MAP vs MAP recuit simulé ===
def afficher_map_vs_recuit(img_init, img_bruitee, map_est, nb_etats, taux, taux_base, taux_map, taux_mapr):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)

    images = [img_init, img_bruitee, map_est]
    #titres = ["Image originale", "Image bruitée", "MAP (descente locale)", "MAP (recuit simulé)"]

    titres = [
        f"Image originale \n(ε={taux:.2f})", f"Image bruitée \n(ε={taux_base:.2f})", 
        f"MAP (descente locale) \n(ε={taux_map:.2f})"   
    ]

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
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

# === Fonction pour accumuler les restaurations ===
def generer_images_restaures(img_bruitee, nb_images, nb_iter, modele, sigma2):
    H, L = img_bruitee.shape
    toutes_images = np.zeros((nb_images, H, L), dtype=int)
    for idx in tqdm(range(nb_images), desc="Génération des images restaurées"):
        champ_init = img_bruitee.copy()
        champ_rest = estimateur_map(champ_init, img_bruitee, nb_iter, modele, sigma2)
        toutes_images[idx] = champ_rest
    return toutes_images



# === Fonction pour calculer l'argmax pixel par pixel ===
def argmax_pixel_par_pixel(samples, nb_etats):
    H, L = samples[0].shape
    result = np.zeros((H, L), dtype=int)
    for i in range(H):
        for j in range(L):
            counts = np.bincount([s[i, j] for s in samples], minlength=nb_etats)
            result[i, j] = np.argmax(counts)
    return result

# === Affichage final comparatif ===
def afficher_comparaison(img_ref, img_100, img_500, img_1000, original, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)

    taux_ref = taux_restauration(original, img_ref)
    taux_100 = taux_restauration(original, img_100)
    taux_500 = taux_restauration(original, img_500)
    taux_1000 = taux_restauration(original, img_1000)


    titres = [
        f"MAP (base)\n(\u03b5={taux_ref:.2f})",
        f"Argmax sur 10 images\n(ε={taux_100:.2f})",
        f"Argmax sur 50 images\n(ε={taux_500:.2f})",
        f"Argmax sur 100 images\n(ε={taux_1000:.2f})"
    ]
    images = [img_ref, img_100, img_500, img_1000]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, titre, img in zip(axs, titres, images):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=20)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# === Paramètres ===
chemin_image = "images/test1.png"
sigma_bruit = 0.5
p_bruit = 0.3
nb_etats = 3
nb_iter = 60000
beta = 0.8
sigma2 = 5
t_init = 1

# === Chargement et préparation des données ===
img = charger_image_grayscale(chemin_image, nb_etats)
#img_bruitee = ajouter_bruit_uniforme(img, p_bruit, nb_etats)
img_bruitee = ajouter_bruit_gaussien_discret(img, sigma_bruit, nb_etats)
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
map_est = estimateur_map(champ_init.copy(), img_bruitee, nb_iter, modele, sigma2)

# === Génération et traitement ===
toutes_images = generer_images_restaures(img_bruitee, 100, nb_iter, modele=modele, sigma2=sigma2)

img_100 = argmax_pixel_par_pixel(toutes_images[:10], nb_etats)
img_500 = argmax_pixel_par_pixel(toutes_images[:50], nb_etats)
img_1000 = argmax_pixel_par_pixel(toutes_images, nb_etats)

# === Affichage ===
afficher_comparaison(map_est, img_100, img_500, img_1000, img, nb_etats)
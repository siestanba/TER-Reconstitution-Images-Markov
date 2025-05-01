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

def gibbs_recuit(champ, nb_iter, modele):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="Gibbs recuit simulé (Potts)"):
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

def metropolis_recuit(champ, nb_iter, modele):
    H, L = champ.shape
    for k in tqdm(range(nb_iter), desc="Metropolis recuit simulé (Potts)"):
        T = 1 / np.log(2 + k)
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

def afficher_resultats(img_init, img_bruitee, gibbs, gibbs_rec, metro, metro_rec, nb_etats):
    def to_image(img):
        return (img * (255 / (nb_etats - 1))).astype(np.uint8)

    imgs = [img_init, img_bruitee, gibbs, gibbs_rec, metro, metro_rec]
    titres = [
        "Image originale", "Gibbs classique", 
        "Gibbs recuit simulé", "Image bruitée",
        "Metropolis classique", "Metropolis recuit simulé"
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    for ax, titre, img in zip(axs, titres, imgs):
        ax.imshow(to_image(img), cmap="gray")
        ax.set_title(titre, fontsize=25)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# === Paramètres ===
chemin_image = "images/carre.png"
nb_etats = 4
p_bruit = 0.3
nb_iter = 100000

# === Exécution ===
img = charger_image_grayscale(chemin_image, nb_etats)
img_bruitee = ajouter_bruit(img, p_bruit, nb_etats)
champ_gibbs = img_bruitee.copy()
champ_gibbs_rec = img_bruitee.copy()
champ_metro = img_bruitee.copy()
champ_metro_rec = img_bruitee.copy()

poids_aretes = {(a, b): -1 if a == b else 1 for a in range(nb_etats) for b in range(nb_etats)}
poids_sommets = [0] * nb_etats
modele = {
    'nb_etats': nb_etats,
    'poids_aretes': poids_aretes,
    'poids_sommets': poids_sommets
}

champ_gibbs = gibbs_classique(champ_gibbs, nb_iter, modele)
champ_gibbs_rec = gibbs_recuit(champ_gibbs_rec, nb_iter, modele)
champ_metro = metropolis_classique(champ_metro, nb_iter, modele)
champ_metro_rec = metropolis_recuit(champ_metro_rec, nb_iter, modele)

afficher_resultats(img, champ_gibbs, champ_gibbs_rec, img_bruitee, champ_metro, champ_metro_rec, nb_etats)
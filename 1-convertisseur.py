from PIL import Image
import numpy as np

def convert(path_entree, nb_etats, taille=(100, 100)):
    """
    Charge une image en niveaux de gris, la redimensionne à 'taille',
    la quantifie sur 'nb_etats' niveaux de gris, et l'enregistre en 8 bits.

    - path_entree : chemin de l’image d’origine
    - path_sortie : chemin où sauvegarder l’image traitée
    - nb_etats : nombre de niveaux de gris souhaités (par exemple 2, 3, 4...)
    - taille : taille finale en pixels (par défaut : 100x100)
    """
    # 1. Chargement de l'image en niveaux de gris
    img = Image.open(path_entree).convert("L").resize(taille)

    # 2. Conversion en tableau NumPy
    img_np = np.array(img)

    # 3. Quantification : projection sur nb_etats niveaux entiers
    img_quant = np.floor(img_np / (256 / nb_etats)).astype(int)

    # 4. Reprojection sur [0, 255] pour enregistrement (8 bits)
    img_255 = (img_quant * (255 / (nb_etats - 1))).astype(np.uint8)

    # 5. Sauvegarde
    img_finale = Image.fromarray(img_255, mode="L")
    img_finale.save('nb.png')

    return img_255  # Pour usage ultérieur si besoin

#=== Paramètres ===#
image = "images/nb.png"
nb_etats = 2

convert(image, nb_etats)
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype,
#       -toc
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   notebookname: indexation & slicing
# ---

# %% [markdown]
# <div class="licence">
# <span>Licence CC BY-NC-ND</span>
# <span>UE12</span>
# <span><img src="media/ensmp-25-alpha.png" /></span>
# </div>

# %%
from IPython.display import HTML
HTML('<link rel="stylesheet" href="slides-notebook.css" />')

# %% [markdown]
# # python-numérique - suite du TP simple avec des images
#
# merci à Wikipedia et à stackoverflow
#
# **le but de ce TP n'est pas d'apprendre le traitement d'image  
# on se sert d'images pour égayer des exercices avec `numpy`  
# (et parce que quand on se trompe ça se voit)**

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# **notions intervenant dans ce TP**
#
# sur les tableaux `numpy.ndarray`
# * `reshape()`, tests, masques booléens, *ufunc*, agrégation, opérations linéaires sur les `numpy.ndarray`
# * les autres notions utilisées sont rappelées (très succinctement)
#
# pour les lecture, écriture, affichage d'images
# * utilisez `plt.imread`, `plt.imshow`
# * utilisez `plt.show()` entre deux `plt.imshow()` dans la même cellule
#
# **note**
#
# * nous utilisons les fonctions de base sur les images de `pyplot` par souci de simplicité
# * nous ne signifions pas là du tout que ce sont les meilleures  
# par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
# alors que la fonction `save` de `PIL` le permet
# * vous êtes libres d'utiliser une autre librairie comme `opencv`  
#   si vous la connaissez assez pour vous débrouiller, les images ne sont qu'un prétexte
#
# **n'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# ## lecture des codes RGB de couleurs d'un fichier

# %% [markdown]
# 1. Le fichier `rgb-codes.txt` contient une table de couleurs:
# ```
# AliceBlue 240 248 255
# AntiqueWhite 250 235 215
# Aqua 0 255 255
# .../...
# YellowGreen 154 205 50
# ```
# Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
# Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.
# <br>
#
# 1. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
# `'Red'`, `'Lime'`, `'Blue'`
# <br>
#
# 1. Faites une fonction `patchwork` qui  
#    * prend une liste de couleurs et la structure donnant le code des couleurs RGB
#    * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
#    * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
#    si besoin de compléter l'image mettez du noir rayé de blanc)
# <br>
# <br>   
# 1. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.
# <br>
#
# 1. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
# même chose pour des jaunes  
# <br>
#
# 1. Appliquez la fonction à toutes les couleurs du fichier  
# et sauver ce patchwork dans le fichier 'patchwork.jpg' avec `plt.imsave`
# <br>
#
# 1. Relisez et Affichez votre fichier  
#    attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

# %%
# votre code

# %% [markdown]
# ## somme des valeurs RGB d'une image

# %% [markdown]
# 0. Lisez l'image `les-mines.jpg`
#
# 1. Créez un nouveau tableau `numpy.ndarray` en sommant avec l'opérateur `+` les valeurs RGB des pixels de votre image  
#
# 2. Affichez l'image (pas terrible), son maximum et son type
#
# 3. Créez un nouveau tableau `numpy.ndarray` en sommant avec la fonction d'agrégation `np.sum` les valeurs RGB des pixels de votre image
#
# 4. Affichez l'image
#
# 5. Pourquoi cette différence ? Utilisez le help `np.sum?`
#
# 6. Passez l'image en niveaux de gris de type en entiers non-signés 8 bits  
# (de la manière que vous préférez)
#
# 7. Affichez l'image en niveaux de gris avec une table des couleurs comme Purples
#
# 8. Remplacez dans l'image en niveaux de gris,   
# les valeurs >= à 127 par 255 et celles inférieures par 0  
# Affichez l'image avec une carte des couleurs des niveaux de gris  
# vous pouvez utilisez la fonction `numpy.where`
#
# 9. avec la fonction `numpy.unique`  
# regardez les valeurs différentes que vous avez dans votre image en noir et blanc

# %%
# votre code

# %% [markdown]
# ## Image en sépia

# %% [markdown]
# Pour passer en sépia les valeurs R, G et B d'un pixel (dont encodées sur un entier non-signé 8 bits)  
#
# 1. convertir ces valeurs par cette transformation  
# $x_R = 0.393\, R + 0.769\, G + 0.189\, B$  
# $x_G = 0.349\, R + 0.686\, G + 0.168\, B$  
# $x_B = 0.272\, R + 0.534\, G + 0.131\, B$  
# avec des $x_R$, $x_G$ et $x_B$ de type flottants  
# 1. convertir les valeurs $x_R$, $x_G$ et $x_B$ en `int`  
# pas `uint8` pour ne pas avoir d'overflow (genre de 256 devenant 0)  
# 1. puis seuiller les valeurs qui sont plus grandes que `255` à `255` 
#
# **Exercice** en utilisant la fonction `numpy.dot` et autres fonctions `numpy` de base sans aucune boucle  `for`
# 1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
# (vous pouvez généralisez votre fonction à toute transformation linéaire de RGB)  
# 1. Passez votre patchwork de couleurs en sépia  
# Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso
# 2. Passez l'image `les-mines.jpg` en sépia
# 4. en option - Faites en sorte que votre fonction sepia puisse prendre un RGB-A  
#    (la transformation en sépia doit simplement conserver la même transparence)

# %%
# votre code

# %% [markdown]
# ## exemple de qualité de compression

# %% [markdown]
# 1. importez la librairie `Image`de `PIL` (pillow) 
# (vous devez peut être installer PIL (pillow) dans votre environnement)
#
# 2. lisez le fichier 'les-mines.jpg' avec `Image.open` et avec `plt.imread`  
#
# 3. vérifier que les valeurs contenues dans deux objets sont proches
#
# 4. sauver (toujours avec de nouveaux noms de fichiers)  
# l'image lue par `imread` avec `plt.imsave` et celle lue par `Image.open` avec `save`  
# (`save` s'applique à l'objet créé par `Image.open`)
#
# 5. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
# Que constatez-vous ?
#
# 6. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence  

# %%
# votre code


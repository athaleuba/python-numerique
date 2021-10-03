# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
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
#   nbhosting:
#     title: masques et conditions
# ---

# %%
from IPython.display import HTML
HTML('<link rel="stylesheet" href="slides-notebook.css" />')

# %% [markdown]
# # Python-numérique - conditions

# %%
import pandas as pd
import numpy as np

# %% [markdown] tags=["framed_cell"]
# ## conditions sur les éléments d'une colonne
#
# <br>
#
#
# dans les analyses de tables de données  
# il est fréquent de **sélectionner des données par des conditions**
#
# <br>
#
# en `pandas`, comme en `numpy`, les fonctions sont **vectorisées**  
# par souci de rapidité du code
#     
# <br>
#
# les conditions s'appliquent, à la fois
# * à tout un tableau
# * ou à toute une colonne
# * ou à toute une ligne
# * ou à tout un sous-tableau, sous-colonne, sous-ligne 
#
# <br>
#
#  
# il ne faut jamais itérer avec un `for-python` sur les valeurs d'une table    
# (les itérations se font dans le code des fonctions `numpy` et `pandas`)
#     
# <br>    
#     
# Combien de passagers avaient moins de 12 ans ?
#
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
#     
# df['Age'] < 12 # l'opérateur < est vectorisé 
#
# -> PassengerId
# 1      False # le passager d'id 1 a plus de 12 ans
# 2      False # le passager d'id 2 a plus de 12 ans
#        ...
# 890    False # le passager d'id 1 a plus de 12 ans
# 891    False # le passager d'id 1 a plus de 12 ans
# Name: Age, Length: 891, dtype: bool
# ```
#
#     
# <br>
#     
#   
# cette expression retourne un **tableau de booléens** (un masque, un filtre)    
# dans une `pandas.Series` dont le type est naturellement `bool`    
# avec, pour chaque valeur de la colonne, sa réponse au test
#  
# <br>
#     
# Comment calculer le nombre d'enfants ?
#
# <br>
#
# nous pouvons sommer les `True` avec `pandas.Series.sum`
#     
# ```python
# children = df['Age'] < 12
# children.sum()
# -> 68
# ```
#
# <br>
#     
# en `pandas` comme en `numpy` pour combiner les conditions  
# * on utilise `&` (et) `|` (ou) et `~` (non)   
# ou les `numpy.logical_and`, `numpy.logical_or`, `numpy.logical_not`
# * et **pas** `and`, `or` et `not` (opérateurs `Python` non vectorisés)
# * on parenthèse les expressions
#  
# ```python
# girls = (df['Age'] < 12) & (df['Sex'] == 'female') 
# girls.sum()
# -> 32
# ```
#     
# <br>
#     
# ou utiliser la méthodes `pandas.Series.value_counts`  
# qui donne le nombre de valeurs différentes dans une colonne  
#
#    
# ```python
# children = df['Age'] < 12
# children.value_counts()
# -> False    823
#    True      68
#    Name: Age, dtype: int64
# ```
#     
# la méthode vous indique la colonne `Age` et son type `int64`
#   
#     
# <br>
#     
# ainsi parmi les passagers dont on connait l'age  
# `68` passagers,  ont moins de `12` ans
#     
# <br>
#     
# certaines colonnes ont des valeurs manquantes

# %% scrolled=false
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
children = df['Age'] < 12

girls = (df['Age'] < 12) & (df['Sex'] == 'female') 
girls.sum()


print(children.dtype)
print(children.sum())
children.value_counts()

# %%
# le code

# %% [markdown]
# ## valeurs manquantes

# %% [markdown] tags=["framed_cell"]
# ### contexte général
#
# <br>
#
# on peut calculer le nombre de valeurs manquantes des data-frames et des séries    
# dans l'exemple du Titanic, ce sont les valeurs qui ne sont pas renseignées dans le `csv`  
#     
#
# <br>
#
# NA signifie Non-Available et NaN Not-a-Number
#     
# <br>
#     
# sur les `pandas.DataFrame` et les `pandas.Series`  
# la méthode `isna` rend (resp.) une data-frame ou une série de de `True` et de `False` où
# * `True` signifie que la valeur est manquante
# * `False` que la valeur ne l'est pas
#    
# <br>
#     
# il existe son contraire qui est `notna`  
# il existe aussi la méthode `notnull`  
# **préférez** utiliser `isna`
#    
# <br>
#
# on pourra ensuite utiliser ces tableaux de booléens  
# * pour leur appliquer des fonctions  
# * comme des masques pour sélectionner des sous-tableaux

# %% [markdown]
# ***

# %% [markdown] tags=["framed_cell"]
# ### valeurs manquantes sur les colonnes
# <br>
#     
# la méthode `pandas.Series.isna`  
# retourne une `pandas.Series` de `bool` où
# * `True` signifie que la valeur est manquante
# * `False` que la valeur ne l'est pas
#    
# <br>
#
#     
# regardons les valeurs manquantes d'une colonne
#     
# ```python
# df['Age'].isna()
# -> PassengerId
#     1      False
#     2      False
#            ...  
#     889     True
#     890    False
#     891    False
#     Name: Age, Length: 891, dtype: bool
# ```
# <br>
#     
# l'age du passager d'`Id` 889 est manquant  
# on peut le voir par `,,` dans le fichier en format `csv`
#     
# ```
# 889,0,3,"Johnston, Miss. Catherine Helen ""Carrie""",female,,1,2,W./C. 6607,23.45,,S
# ```
#
# <br>
#
# Combien d'ages sont-ils manquants ?
#
# ```python
# df['Age'].isna().sum()
# np.sum(df['Age'].isna()) # aussi
# -> 177
# ```

# %% scrolled=true
# le code
df['Age'].isna()

# %%
# le code
df['Age'].isna().sum()
np.sum(df['Age'].isna())

# %%
# le code
df['Age'].notna().sum()

# %% [markdown] tags=["framed_cell"]
# ### valeurs manquantes sur une data-frame
#
# <br>
#     
# la méthode `pandas.DataFrame.isna`  
# retourne une data-frame de `True` et de `False` où
# * `True` signifie que la valeur est manquante
# * `False` que la valeur ne l'est pas
#    
# <br>
#
#    
# regardons les valeurs manquantes d'une data-frame
#     
# ```python
# df.isna()
# ->           Survived  Pclass  Name    Sex     Age     SibSp   Parch   Ticket   Fare   Cabin   Embarked
# PassengerId                                                                                        
#           1  False     False   False   False   False   False   False   False   False   True    False
#           2  False     False   False   False   False   False   False   False   False   False   False
# ...
#         890  False     False   False   False   False   False   False   False   False   False   False
#         891  False     False   False   False   False   False   False   False   False   True    False
# ```
#
# <br>
#
# vous remarquez une dataframe de valeurs  
# `True` (valeur absente)  
# `False` (valeur présente)
#     
# <br>
#     
# c'est un tableau en dimension 2 avec une forme `(nb lignes, nb colonnes)`
#     
# <br>
#     
# comme en `numpy` je peux appliquer une fonction en précisant l'`axis`  
# `0` on applique la fonction dans l'axe des lignes (le défaut)    
# `1` on applique la fonction dans l'axe des colonnes  
# l'objet retourné est une série contenant le résultat de la fonction
#
# <br>
#     
# exemple avec la somme (`sum`) des valeurs manquantes sur l'axe des lignes `axis=0`  
# (qui `sum` les lignes entre elles)
#     
# ```python
# df.isna().sum()
# df.isna().sum(axis=0)
#     
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64    
# ```
# <br>
#     
# nous remarquons des valeurs manquantes dans les colonnes `Cabin`, `Age` et `Embarked`
#     
# <br>
#     
# exemple de la somme des valeurs manquantes sur l'axe des colonnes
#
# ```python
# df.isna().sum(axis=1):
# PassengerId
# 1      1
# 2      0
# 3      1
# 4      0
# 5      1
#       ..
# 887    1
# 888    0
# 889    2
# 890    0
# 891    1
# Length: 891, dtype: int64
# ```
# <br>
#     
# le passager d'id `889` a deux valeurs manquantes

# %%
# le code
df.isna()

# %%
# le code
df.isna().sum(axis=0)
df.isna().sum()

# %% scrolled=false
# le code
df.isna().sum(axis=1)

# %% [markdown] tags=["framed_cell"]
# ### utilisation des fonctions `numpy`
#
# <br>
#     
#     
# les méthodes `numpy` s'appliquent sur des `pandas.DataFrame` et des `pandas.Series`
#     
# <br>
#     
# on précise l'`axis`  
# `0` pour l'axe des lignes et **aussi le défaut**  
# `1` pour l'axe des colonnes  
#
# <br>
#
# différence avec `numpy`
# * par défaut d'`axis` la fonction ne donne **pas** le résultat global  
# * elle donne le résultat sur l'axe des lignes
#
# <br>
#     
# si on désire le résultat global  
# il faut passer par le sous-tableau `numpy`  
# et là la fonction `numpy` donnera le résultat global
#
# <br>
#     
# la méthode `pandas.DataFrame.to_numpy` retourne le tableau `numpy.ndarray` de la DataFrame `pandas` 
#
# ```python
# df.isna().to_numpy()
# -> array([[False, False, False, ..., False,  True, False],
#           [False, False, False, ..., False, False, False],
#           ...,
#           [False, False, False, ..., False,  True, False],
#           [False, False, False, ..., False,  True, False]])
# ```
#     
# <br>
#     
# on somme
#     
# ```python
# np.sum(df.isna().to_numpy())
# df.isna().to_numpy().sum()
# -> 866
# ```    
# <br>
#
# il y a `866` valeurs manquantes dans toute la data-frame

# %%
# le code
df.isna().to_numpy()

# %%
# le code
np.sum(df.isna().to_numpy())
df.isna().to_numpy().sum()

# %% [markdown]
# ## **exercice** valeurs uniques

# %% [markdown]
# **exercice**
#
# 1. Lisez la data-frame du Titanic `df`
# <br>
#
# 1. Utilisez la méthode `pd.Series.unique` (1) pour comptez le nombre de valeurs uniques  
# des colonnes `'Survived'`, `'Pclass'`, `'Sex'` et `'Embarked'`  
# vous pouvez utiliser un for-python pour parcourir la liste `cols` des colonnes choisies
# <br>
#
# 1. Utilisez l'expression `df[cols]` pour sélectionner la sous-dataframe réduite à ces 4 colonnes
# <br>
#
# 1. Utilisez l'attribut `dtypes` des `pandas.DataFrame` pour afficher le type de ces 4 colonnes
# <br>
#
# 1. Que constatez-vous ?  
# Quel type serait plus approprié pour ces colonnes ?
#
#
# (1) servez-vous du help `pd.Series.unique?`

# %% [markdown]
# ## **exercice** conditions
#
# 1. Lisez la data-frame des passagers du titanic
# 1. Calculez les valeurs manquantes: totales, des colonnes et des lignes
# 1. Calculez le nombre de classes du bateau
# 1. Calculez le taux d'hommes et de femmes
# 1. Calculez le taux de personnes entre 20 et 40 ans
# 1. Calculez le taux de survie des passagers
# 1. Calculez le taux de survie des hommes et des femmes par classes  
# on reverra ces décomptes d'une autre manière


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
#     title: "acc\xE8s et slicing"
# ---

# %%
from IPython.display import HTML
HTML('<link rel="stylesheet" href="slides-notebook.css" />')

# %% [markdown]
# # python-numérique - accès aux sous-tableaux

# %%
import pandas as pd
import numpy as np # pandas reposant sur numpy on a souvent besoin des deux librairies

# %% [markdown] tags=["framed_cell"]
# ## introduction
# <br>
#
# manipuler des parties de nos données  
# est une opération fréquente en traitement des données
#     
# <br>
#     
# d'où l'importance de savoir localiser dans nos tables `pandas` des sous-parties   
# (élément, ligne, colonne, sous-séries, sous dataframes)  
# afin de leur appliquer une fonction
#     
# <br>
#     
# `pandas` a mis ses efforts sur la gestion d'une indexation des lignes et des colonnes
#     
# <br>
#
# ils ont priviligié le repérage des éléments d'une dataframe **par des index**  
# (les **noms** de colonnes et les **labels** de lignes)  
# et **pas** par des **indices** comme en Python et en `numpy`
#
# <br>
#     
# Pourquoi ?
#     
# * parce que vous utilisez `pandas`  
# si vous avez besoin de voir vos données  
# sous la forme d'une table avec des labels pour indexer les lignes et les colonnes
#     
#     
# * si vous n'avez pas besoin d'index particuliers  
# si vos données se manipulent facilement à base d'indices  
# autant rester avec des tableaux 2D `numpy`  
# avec leurs indices de ligne et de colonne
#     
# <br>
#     
# mais `pandas` va aussi vous permettre d'accéder à vos sous-tableaux par indices

# %% [markdown]
# ***

# %% [markdown] tags=["framed_cell"]
# ## copier une dataframe ou une série
#
# <br>
#     
# pour dupliquer une dataframe ou une série (ligne ou colonne)   
# toujours la méthode classique `copy` des objets `Python`
#     
#  
# <br>
#     
# vous allez utiliser la méthode `pandas.DataFrame.copy` ou `pandas.Series.copy`
#   
#     
# <br>
#     
# construisons une dataframe
#     
# ```python
# df_aux = pd.read_csv('titanic.csv', index_col='PassengerId')
# ```
#     
# <br>
#     
# copions la
#     
# ```python
# df = df_aux.copy()
# ```
#     
# <br>
#     
# supprimons la
#     
# ```python
# del df_aux
# ```   
# <br>
#     
# la copie existe toujours
#     
# ```python
# df.head(2)
# ->   Survived Pclass ...
# PassengerId	...
# ```
#     
# <br>
#     
# `df` est une nouvelle dataframe  
# avec les mêmes valeurs que l'originale `df_aux`  
# mais totalement indépendante

# %%
# le code
df_aux = pd.read_csv('titanic.csv', index_col='PassengerId')
df = df_aux.copy()
del df_aux
df.head(2)

# %% [markdown] tags=["framed_cell"]
# ## créer une nouvelle colonne
#
# <br>
#     
# on crée une nouvelle colonne  
# en la rajoutant dans le dictionnaire des colonnes
#     
# <br>
#     
# souvent on crée une nouvelle colonne  
# en faisant un calcul sur des colonnes existantes
#     
# <br>
#     
# les opérations sur les colonnes peuvent utiliser la forme `df[nom_de_colonne]`
#     
# <br>
#     
# dans la dataframe du titanic  
# créons une colonne des décédés (donc 1 - les survivants)
#     
# ```python
# df['Deceased'] = 1 - df['Survived']
# ```
#
# <br>
#     
# nous avons rajouté la clé `'Deceased'` comme index des colonnes  
# `pandas` voit sa dataframe comme un dictionnaire des colonnes  
# (mais avec des index non uniques)

# %%
# le code
df['Deceased'] = 1 - df['Survived']
df.head(3)

# %%
df.head(3)

# %% [markdown]
# ## localiser en `pandas`

# %% [markdown] tags=["framed_cell"]
# ### localisation `python`, `numpy`
#
# <br>
#
# pour accéder ou modifier des sous-parties de dataframe, nous sommes tentés:
#
# * d'utiliser les syntaxes classiques d'accès aux éléments d'un tableau par leur indice  
# comme vous le feriez en Python
#
# ```python
# L = [10, 20, 30, 40, 60]
# L[0] = "Hello !"
# print(L) # ['Hello !', 20, 30, 40, 60]
# L[1:3] = [200, 300, 500]
# L
# -> L[1:3] = [200, 300, 500]
# ```
#
# <br>
#
# * ou d'utiliser l'accès à un tableau par une paires d'**indices**  
# comme vous le feriez en `numpy`
#
#     créons une matrice `numpy` (4, 4)  
#     et modifions une sous-matrice
#
# ```python
# mat = np.arange(12).reshape((4, 3))
# mat[0:2, 0:2] = 999
# mat
# -> [[999, 999,   2],
#     [999, 999,   5],
#     [  6,   7,   8],
#     [  9,  10,  11]])
# ```
#
# <br>
#     
# vous vous souvenez que la borne supérieure est toujours **exclue**

# %%
# le code
L = [10, 20, 30, 40, 60]
L[0] = "Hello !"
print(L)
L[1:3] = [200, 300, 500]
L

# %%
# le code
mat = np.arange(12).reshape((4, 3))
mat[0:2, 0:2] = 999
mat

# %% [markdown] tags=["framed_cell"]
# ### localisation `pandas`
#
# <br>
#
# `pandas` va offrir des techniques assez similaires pour l'accès et le slicing
#     
# <br>
#     
#     
# on va accéder à des sous-dataframe  
# en étendant l'opération d'indexation `[i]` à des slices `[start:stop:step]`  
# comme en `python` et `numpy`
#     
# <br>
#     
# deux méthodes apparaissent: `pandas.DataFrame.loc` et `pandas.DataFrame.iloc`  
# pour localizer des sous-dataframes resp. par index et par **i**ndice
#     
# <br>
#     
# ainsi qu'une **grande différence**  
# **dans le cas des index** les slices de dataframes **contiennent les bornes**
#
#     
# <br>
#     
# nous allons étudier tous ces cas

# %% [markdown]
# ***

# %% [markdown] tags=["framed_cell"]
# ### localiser des éléments
#
# <br>
#     
# on peut localiser des éléments par indice ou par index
#
# <br>
#     
# la méthode `pandas.DataFrame.iloc` permet d'accéder aux éléments par `indice`
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df.iloc[0, 2] # première ligne, troisième colonne
# -> 'Braund, Mr. Owen Harris'
# ```
#
# <br>
#
# pareil avec les index  sachant que
# * ceux des lignes vont de `1` à `891`  
# * ceux des colonnes vont de `'Survived'` à `'Embarked'`
#
# <br> 
#     
# la méthode `pandas.DataFrame.loc` permet d'accéder aux éléments par `index`
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df.loc[1, 'Name'] # passage d'id 1 et colonne 'Name'
# -> 'Braund, Mr. Owen Harris'
# ```
#     
# <br>
#     
#       
# remarque
# * on est dans le cas d'accès `numpy` à un tableau en 2 dimension    
# * la première valeur concerne les lignes
# * le second indice est la colonne
#     
# <br>
#
# le type de l'objet obtenu dépend du type des colonnes

# %%
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
df.iloc[0, 2]

# %%
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
df.loc[1, 'Name']

# %% [markdown] tags=["framed_cell"]
# ### localiser des sous-tableaux
#
# <br>
#     
# <br>
#     
# avec le *slicing*, par indice et index, on peut obtenir des sous-tableaux    
# <br>
#     
# on peut `slicer` sur les indices  
# `pandas.DataFrame.iloc[start:stop:step, start:stop:step]`
#     
# <br>
#
# on prend les lignes d'indice `0` à `10` (non-compris) et les colonnes d'indice `0` à `5` (non-compris)
# ```python
# df.iloc[0:10, 0:5].shape
# -> (10, 5)
# ```
#     
# <br>
#     
# on peut aussi slicer sur les index  
# `pandas.DataFrame.loc[start:stop:step, start:stop:step]`  
# **MAIS ATTENTION** pour les **index** `stop` est compris    
#
#     
# <br>
#  
# on prend les lignes d'index `1` à `11` **compris**  
# et les colonnes d'index `'Survived'` à `'SibSp'` **compris**
# ```python
# df.loc[1:11, 'Survived':'SibSp'].shape
# -> (11, 6)
# ```
#  
# <br>
#     
# on obtient des objets de type `pandas.DataFrame`

# %%
# le code
df.iloc[0:10, 0:5].shape

# %%
# le code
df.loc[1:11, 'Survived':'SibSp'].shape

# %% [markdown] tags=["framed_cell"]
# ### localiser des lignes et des colonnes
#
# <br>
#
# ***ou sous-lignes et sous-colonnes***
#     
# <br>
#     
# avec le *slicing*, par indice et index, on peut obtenir des lignes et des colonnes  
# ou des sous-lignes et des sous-colonnes
#     
# <br>
#   
# on peut slicer, par indice, **pour obtenir une ligne**
#     
# ```python
# df.iloc[0, :] # première ligne (toutes les colonnes)
# df.iloc[0, :].shape
# -> (11,)
# ```
#   
# notez qu'on peut alors omettre les colonnes puisqu'on les prend toutes
#     
#   
# ```python
# df.iloc[0] # première ligne (toutes les colonnes)
# df.iloc[0].shape
# -> (11,)
# ```   
#     
# <br>
#
# on peut slicer, par indice,  **pour obtenir une colonne**
#     
# ```python
# df.iloc[:, 0] # première colonne (toutes les lignes)
# df.iloc[:, 0].shape
# -> (891,)
# ```    
#
# <br>
#     
# on obtient des objets de type `pandas.Series`
#     
# <br>
#   
# on peut slicer, par index, pour obtenir une ligne
#     
# ```python
# df.loc[1, :] # première ligne (toutes les colonnes)
# df.loc[1, :].shape
# -> (11,)
# ```
#     
# <br>
#
# on peut slicer, par index,  pour obtenir une colonne
#     
# ```python
# df.loc[:, 'Survived'] # première colonne (toutes les lignes)
# df.loc[:, 'Survived'].shape
# -> (891,)
# ```    
#
# </div>

# %%
# le code
df.iloc[0, :].shape
df.iloc[0].shape

# %%
# le code
df.iloc[:, 0].shape

# %%
# le code
df.loc[1, :].shape
df.loc[1].shape

# %%
# le code
df.loc[:, 'Survived'].shape

# %% [markdown]
# **exercice**
#
# 1. lisez le titanic et mettez les `passengerId` comme index des lignes
# <br>
#
# 1. localisez l'élément d'index `40` de deux manières   
# Quel est le type de l'élément ?  
# <br>
#
# 1. localisez le nom du passager d'index `40` ? 
# <br>
#
# 1. faites de même pour l'élément d'indice `40`
# <br>
#
# 1. localisez les 3 derniers éléments de la ligne d'index `40`
# <br>
#
# 1. localisez les 4 derniers éléments de la colonne d'index `Cabin`

# %% [markdown] tags=["framed_cell"]
# ### autres manières d'accéder à des éléments
#
# <br>
#
# on accède à une colonne par sa clé (son nom)  
# on obtient une `pandas.Series`
#     
# ```python
# df['Age']
# ```
#     
# <br>
#
# à partir d'une colonne, on peut accéder aux éléments  
# (attention aux index)
#     
#
# <br>
#     
# exemple avec un index de lignes explicite
#     
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# # on a indiqué l'index de lignes
# df['Age'][1] # 1 est un index ici le passager d'id 1
# -> 22.0
# ```
#     
# <br>
#     
# exemple avec un index de lignes implicite
#     
# ```python
# df = pd.read_csv('titanic.csv') # on n'a pas d'index
# # par défaut, les index de lignes sont leurs indices
# df['Age'][0] # 0 est un indice ici la première ligne
# -> 22.0
# ```

# %%
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
# on a des index de lignes
df['Age'][1] # 1 est un index ici le passager d'id 1

# %%
# le code
df = pd.read_csv('titanic.csv') # on n'a pas d'index
# les index de lignes sont leurs indices
df['Age'][0] # 0 est un indice ici la première ligne
             # passager d'id 1 aussi

# %%
# le code
df.iloc[-1]

# %% [markdown]
# ## autres mécanismes d'indexation

# %% [markdown] tags=["framed_cell"]
# ### accès à une liste explicite de lignes ou colonnes
#
# <br>
#
# supposons que la sous-partie, qui nous intéresse, d'une dataframe  
# **ne s'exprime pas** sous la forme d'une slice  
#
# <br>
#
# nous possédons par contre  
# la **liste des lignes et des colonnes** à conserver dans la  sous-dataframe
#
# <br>
#
# pour le faire en `pandas`
#
# * on utilise les méthodes `pandas.DataFrame.loc` ou `pandas.DataFrame.iloc`  
# suivant que nous ayons des index ou des indices
#
#
# * auxquelle nous passons, non plus des slices, mais des listes  
# avec les éléments dans l'ordre que vous voulez pour la sortie
#
# <br>
#
# par exemple, gardons:
# * les lignes d'index 450, 3, 67, 800 et 678
# * et les colonnes `Age`, `Pclass` et `Survived`
#
# <br>
#
# ce sont des index, donc nous utilisons `loc`  
# nous référons ainsi une partie de la data-frame
#
# ```python
# df.loc[[450, 3, 67, 800, 678], ['Age', 'Pclass', 'Survived']]
# ->           Age   Pclass Survived
# PassengerId
#         450  52.0  1      1
#         3    26.0  3      1
#         67   29.0  2      1 
#         800  30.0  3      0
#         678  18.0  3      1
# ```
#
# <br>
#     
# pour obtenir la même chose avec les indices
#     
# ```python
# df.iloc[[449, 2, 66, 799, 677], [4, 1, 0]]
# ```

# %%
# le code
print(df.loc[[450, 3, 67, 800, 678], ['Age', 'Pclass', 'Survived']])
df.iloc[[449, 2, 66, 799, 677], [4, 1, 0]]

# %% [markdown] tags=["framed_cell"]
# ### rappel sur les conditions
#
# <br>
#
# nous avons vu comment appliquer des conditions  
# à une colonne ou à une data-frame  
# et comment utiliser ce tableau de booléens pour des décomptes
#     
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df_survived = (df['Survived'] == 1)
# df_survived.sum()/len(df)
# ->  0.3838383838383838
# ```
# <br>
#
# on a vu comment combiner ces conditions  
# vous ne pouvez **pas** utiliser `and`, `or` et `not` python (pas vectorisés)  
# et devez utiliser `&`, `|` et `~`  
# ou `np.logical_and`, `np.logical_or` et `np.logical_not`
#     
# taux de survie des passagers femmes de première classe
#     
# ```python
#
# ( ((df['Sex'] == 'female') & (df['Survived'] == 1) & (df['Pclass'] == 1)).sum()
#   /((df['Sex'] == 'female') & (df['Pclass'] == 1)).sum()   )
#     
# ```

# %%
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
df_survived = (df['Survived'] == 1)
print(   df_survived.shape   )
( ((df['Sex'] == 'female') & (df['Survived'] == 1) & (df['Pclass'] == 1)).sum()
  /((df['Sex'] == 'female') & (df['Pclass'] == 1)).sum()   )

# %% [markdown] tags=["framed_cell"]
# ### sélection par masque booléen
#
# <br>
#
# nous avons vu, ci-dessus la manière de combiner des conditions  
# pour repérer des éléments de la data-frame
#     
# <br>
#     
# pour accéder à des sous-parties d'une dataframe  
# on va **indexer** une dataframe par un **masque de booléens** sur la colonne des `index`  
# i.e. on va isoler avec `loc` (pas `iloc`) les lignes de la dtaframe où la valeur du booléen est vraie
#
# <br>
#     
# faisons le *masque* des passagers de 3-ième classe  
#
# ```python
# # le code
# mask = df['Pclass'] >= 3
# type(mask)   # pandas.core.series.Series
# mask.dtype   # dtype('bool')
# mask.shape   # (891,)
# mask.head()
# -> PassengerId
#              1  True
#              2  False
#              3  True
#              4  False
#              5  True
# Name: Pclass, dtype: bool
# ```
#     
# <br>
#
# vous obtenez une `pandas.Series` de `bool`  
# de la taille du nombre de lignes de votre dataframe  
# indiquant le résultat de la condition pour tous les passagers  
# le passager d'`Id` `1` est un passager de 3-ième classe
#     
# <br>
#         
# pour extraire la sous-dataframe des voyageurs en 3-ième classe  
# on **indexe** notre dataframe, par cet objet de type `Series` de booléens
#     
# seules sont conservées, les lignes, dont les booléens sont vrais
#     
# <br>
#     
# ```python
# df.loc[ mask ]
# ->            Survived   Pclass   Name                      Sex     Age    ...
# PassengerId                                 
#           1   0          3        Braund, Mr. Owen Harris   male    22.0   ...          
#           3   1          3        Heikkinen, Miss. Laina    female  26.0   ...
#           5   0          3        Allen, Mr. William Henry  male    35.0   ...
# ...
# ```
#     
# <br>
#     
# `df.loc[mask]`  
# dans les crochets on n'a plus ni une slice, ni une liste  
# mais une colonne, une Series, de booléens  
# appelée un masque
#     
# <br>
#     
# pour un code concis et lisible  
# il est recommandé d'écrire directement la version abrégée
#
# ```python
# df.loc[df['Pclass'] >= 3]
# ```

# %%
# le code
mask = df['Pclass'] >= 3
type(mask)   # pandas.core.series.Series
mask.dtype   # dtype('bool')
mask.shape
mask.head() # un masque de booléens sur la colonne des index donc la colonne PassengerId

# %%
# le code
df.loc[mask].head(3) # on n'affiche que les 3 premiers

# %%
# le code 
df.loc[df['Pclass'] >= 3].head(3) # on n'affiche que les 3 premiers

# %% [markdown]
# ### **exercice** combinaison d'expressions booléennes

# %% [markdown]
# **exercice**
#
# 1. en une seule ligne sélectionner la sous-dataframe des passagers  
# qui ne sont pas en première classe  
# et dont l'age est supérieur ou égal à 70 ans
# <br>
#
# 1. Combien trouvez-vous de passagers ?  
# <br>
#
# 1. Accédez à la valeur `Survived` du premier de ces passagers
# <br>
#
# 1. Faites la même expression que la question 1  
# en utilisant les fonctions `numpy.logical_and`, `numpy.logical_not`
# <br>

# %% [markdown] tags=["framed_cell"]
# ## résumé des méthodes d'indexation
#
# <br>
#     
# trois méthodes d'indexation utilisables avec `pandas.DataFrame.loc` (pas `iloc`)    
#
# * les slices, comme on manipule des `index` et **non** des `indices`  
#     **les bornes sont inclusives**
# * les liste explicite  
# pour choisir exactement et dans le bon ordre les index qui nous intéressent
# * les masques  
# une colonne obtenue en appliquant une expression booléenne à la dataframe de départ
#
# <br>
#     
# on peut mélanger les trois méthodes d'indexation
#     
# <br>
#     
# une liste pour les lignes et une slice pour les colonnes
# ```python
# df.loc[
#     # dans la dimension des lignes: une liste
#     [450, 3, 67, 800, 678], 
#     # dans la dimension des colonnes: une slice
#     'Sex':'Cabin':2]
# ->
#               Sex     SibSp       Ticket  Cabin
# PassengerId            
#         450   male    0           113786  C104
#           3   female  0 STON/O2. 3101282  NaN
#          67   female  0       C.A. 29395  F33
#         800   female  1           345773  NaN
#         678   female  0           4138    NaN
# ```
#     
# <br>
#
# un masque booléen pour les liste et une liste pour les colonnes  
# les colonnes `Sex` et `Survived` des passagers de plus de 71 ans 
# ```python
# df.loc[df['Age'] >= 71, ['Sex', 'Survived']]
# ->          Sex  Survived
# PassengerId
#          97 male 0
#         494 male 0
#         631 male 1
#         852 male 0
# ```

# %% tags=["level_advanced"]
# le code
df.loc[
    # dans la dimension des lignes: une liste
    [450, 3, 67, 800, 678], 
    # dans la dimension des colonnes: une slice
    'Sex':'Cabin':2]

# %%
# le code
df.loc[df['Age'] >= 71, ['Sex', 'Survived']]

# %% [markdown]
# ## règles des modifications

# %% [markdown] tags=["framed_cell"]
# ### sélections de parties de dataframe
#
# <br>
#
#
# une opération sur une dataframe `pandas` renvoie une **sous-partie** de la dataframe
#
# <br>
#
# **le problème**
# * savoir si cette sous-partie **réfère** la dataframe initiale ou est une **copie** de la data-frame initiale 
# * ...ça dépend du contexte
#
# <br>
#
# vous devez vous en soucier ?
# * **dès que** vous essayez de modifier des sous-parties de dataframe
# * tant que vous ne faites que lire, tout va bien
#
# <br>
#
# en effet
# * si c'est une **copie**  
#  votre modification ne sera **pas prise en compte** sur la dataframe d'origine
# * si c'est une **référence partagée**  
# vos modifications dans la sélection, seront bien **répercutées** dans les données d'origine
#
# <br>
#
# **donc**  
# savoir si une opération retourne une copie ou une référence devient important  
# et dépend toujours du contexte
#
# <br>
#
# **à retenir**
#
# * en utilisant les méthodes `pandas.DataFrame.loc[line, column]` et `pandas.DataFrame.iloc[line, column]`  
# on ne **crée pas de copie** mais des **références partagées**  
#
#
#
# * dès que vous utiliser un **chaînage d'indexation** pour modifier   
# que ce soit `df[l][c]` ou `df.loc[l][c]` ou `df.iloc[l][c]`   
#  **vous ne pouvez pas compter sur le résultat**  
# ça fonctionne par hasard
#
# <br>
#     
# (pour les avancés) ce *problème* s'appelle le *chained indexing*  
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

# %% [markdown] tags=["level_intermediate"]
# ***

# %% [markdown] tags=["level_intermediate", "framed_cell"]
# ### modification d'une copie
#
# <br> 
# **par chainage d'indexations**
# <br>
#     
# prenons une dataframe et accèdons à une colonne  
# en utilisant la syntaxe classique d'accès à une colonne comme à une clé d'un dictionnaire
#     
# <br>
#
# la colonne des survivants `'Survived'` 
#     
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df['Survived']
# ```
# <br>
#     
# on obtient une colonne de type `pandas.Series`  
# accédons à l'élément d'index `1` de la colonne  
#  
# ```python
# df = pd.read_csv('titanic.csv', index_col='PassengerId')
# df['Survived'][1]
# -> 0
# ```
#   
# <br>
#
# Pouvons-nous utiliser cette manière d'accéder pour modifier l'élément ?  
# et ressusciter le passager d'index 1 en changeant son état de survie
#     
# <br>
#     
# essayons, on obtient un message d'erreur:
#     
# ```python
# df['Survived'][1] = 1
# ```
# ```
# A value is trying to be set on a copy of a slice from a DataFrame
#
# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   df['Survived'][1] = 1
#     
#     
# ```
#
# <br>
#     
# **non**
# * `df['Survived'][1]` est clairement une indexation par chaînage, on voit les `[][]`
# * ce n'est pas une référence
# * toutes les indexations par chaînage sont des copies
# * elle ne doivent pas être utilisées pour des modifications
#
# <br>
#     
# si ça fonctionne c'est *par hasard*, vous **devez utiliser** `loc` ou `iloc` !
#     
# <br>
#     
#     
# ```python
# df.loc[1, 'Survived'] = 1
# ```

# %% scrolled=true
# le code
df = pd.read_csv('titanic.csv', index_col='PassengerId')
df['Survived']
df['Survived'][1]
df['Survived'][1] = 1
# possible que df['Survived'][1] soit passé à 1, par hasard
# mais votre code est faux

df.loc[1, 'Survived'] = 1

# %% [markdown] tags=["framed_cell"]
# ### récapitulatif sur les modifications
#
# <br>
#
# vous voulez modifier une partie de votre `pandas.DataFrame`
#     
# <br>
#     
# lors d'accès à cette sous-dataframe
# * `pandas` peut retourner une copie de la sous data-frame
# * sauf si vous utilisez `loc` et `iloc` (correctement i.e. sans chaînage)  
# il retourne alors une partie de la dataframe existante
#     
# <br>
#     
# Qu'est-ce-qu'un chaînage ?  
#    
# l'expression `df['Age'][889]` comporte un chaînage d'index que vous remarquez par les `[][]`  
# * on accède à la colonne d'index `Age` de la DataFrame `df`
# * cet accès retourne la série (`pandas.Series`) représentant la colonne `df['Age']`
# * on accède à l'index `889` de cette série
#
# <br>
#     
# donc `pandas` ne fera correctement la modification souhaitée de votre `pandas.DataFrame`  
# que si vous utilisez `loc` ou `iloc` pour accéder à cette partie
#     
# <br>
#     
# sinon il vous dira *A value is trying to be set on a copy of a slice from a DataFrame*  
# vous pouvez même avoir l'impression qu'il a ait l'affectation  
# mais vous ne pouvez et ne devez pas compter dessus
#     
#     
# <br>
#
# OUI
# ```python
# df.loc[889, 'Age'] = 27.5
# ```
#
#
# NON
# ```python
# df.loc[889]['Age'] = 28.5  
# df['Age'][889] = 28.5```
# ```
#
# <br>
#  
# donc, pour modifier (écrire dans) une cellule, **il ne faut PAS faire**  
# ~~`df.loc[889]['Age'] = 28.5`~~  
# ~~`df['Age'][889] = 28.5`~~
#
# et si ça fonctionne, c'est par accident
#
# <br>
#     
# La **bonne méthode**, prenez-en l'habitude, consiste à utiliser cet idiome :
#
# * `df.loc[889, 'Age'] = 10`

# %%
# le code
print(df['Age'][889])

# le code
df.loc[889, 'Age'] = 27.5

# le code
df['Age'][889] = 27.5

# %% [markdown] tags=["level_intermediate", "framed_cell"]
# ### récapitulatif indexation et modification
#
# <br>
#     
# deux possibilité lors d'extractions de sous-partie d'une dataframe  
# (obtenue par découpage de la dataframe d'origine)
# * c'est une copie **implicite** de la dataframe: vous ne devez pas la modifier
#     
#     ```python
#     df1 = df[ ['Survived', 'Pclass', 'Sex'] ] # df1 est une copie implicite ...
#     df1.loc[1, 'Survived'] = 1 # loc fait sur une copie donc le warning suivant apparaît
#     -> SettingWithCopyWarning:
#           A value is trying to be set on a copy of a slice from a DataFrame.
#     ```
#     (le warning apparaît une seule fois, mais il continue à être vrai ...)
# <br>
#     
# * c'est une référence sur la dataframe: vous pouvez la modifier  
# mais donc vous modifiez la dataframe d'origine 
#     ```python
#     df1 = df.loc[ :, ['Survived', 'Pclass', 'Sex'] ]
#     df1.loc[1, 'Survived'] = 1
#     ```
#      
# <br>
#
# vous ne voulez pas modifier la dataframe d'origine ?  
# Faites une copie **explicite** de la sous-dataframe
#     
# ```python
# df2 = df[ ['Survived', 'Pclass', 'Sex'] ].copy() # copie explicite
# df2.loc[1, 'Survived']     # 1
# df2.loc[1, 'Survived'] = 0 # on le passe à 0
# df2.loc[1, 'Survived']     # 0 maintenant
# df.loc[1, 'Survived']      # toujours 1 dans la dataframe d'origine df
# ```
#
# <br>
#      
#
# si l'idée est de ne modifier qu'une copie d'une dataframe  
# utilisez `copy` pour maîtriser ce que vous faites  
# et coder ainsi explicitement et proprement

# %% tags=["level_intermediate"]
# le code
df1 = df[ ['Survived', 'Pclass', 'Sex'] ]
df1.loc[1, 'Survived'] = 1

# %%
# le code
df1 = df.loc[ :, ['Survived', 'Pclass', 'Sex'] ]
df1.loc[1, 'Survived'] = 1

# %% tags=["level_intermediate"]
# le code
df2 = df[ ['Survived', 'Pclass', 'Sex'] ].copy()
print(df2.loc[1, 'Survived'])
df2.loc[1, 'Survived'] = 0
print(df2.loc[1, 'Survived'])
print(df.loc[1, 'Survived'])


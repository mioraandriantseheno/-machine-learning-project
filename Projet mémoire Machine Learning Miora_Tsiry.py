#!/usr/bin/env python
# coding: utf-8




# Introduction ##
# 
# Mon choix d'étude porte sur une open data à disposition sur le site data.gouv.fr :  les accidents corporels de la circulation routière pour les années 2015 à 2019. 
# 
# Sont répertoriées sur ce site 4 catégories de données sur les accidents : les caractéristiques, les usagers, les lieux et les véhicules impliqués.
# Après analyse des 4 jeux de données, la variable qui m'a paru la plus importante à expliquer est la variable "grav" de la dataset "usagers" ; cette dernière nous indique par 4 attributs l'indice de gravité de l'accident : indemne, blessé léger, blessé hospitalisé et tué. 
# Placé dans un contexte actuariel, la prédiction de la gravité d'un accident s'avère fort utile dans la mesure où cela peut impliquer de lourdes conséquences en terme de provisionnement et d'indemnisation vis-à-vis des victimes. Pour l'étude, et dans un souci de simplicité, je vais regrouper les modalités de la variable en 2 classes : la classe des indemnes et blessés légers et la classe des tués et des blessés hospitalisés.  
# Ainsi, ma nouvelle variable à prédire sera une variable binaire.  
# 
# Pour cela, j'entreprendrai différentes méthodes de machine learning afin d'obtenir le meilleur modèle prédictif (régression logistique GLM, arbre CART, random forest, xgboost...).
# 
# 

# ## 2. Présentation des Données et contexte actuariel ## 

# ### *2.1 Description Open Data et Insee Data*

# ### *2.2 Nettoyage des données et mise en forme*

# #### *2.2.1 Importation des données principales*

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 
from sklearn.pipeline import Pipeline
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[7]:


veh15 = pd.read_csv('vehicules_2015.csv',sep=',')
veh16 = pd.read_csv('vehicules_2016.csv',sep=',')
veh17 = pd.read_csv('vehicules-2017.csv',sep=',')
veh18 = pd.read_csv('vehicules-2018.csv',sep=',')
veh19 = pd.read_csv('vehicules-2019.csv',sep=';')
#veh20 = pd.read_csv('vehicules-2020.csv',sep=';')


lieux15 = pd.read_csv('lieux_2015.csv',sep=',',low_memory=False)
lieux16 = pd.read_csv('lieux_2016.csv',sep=',',low_memory=False)
lieux17 = pd.read_csv('lieux-2017.csv',sep=',',low_memory=False)
lieux18 = pd.read_csv('lieux-2018.csv',sep=',',low_memory=False)
lieux19 = pd.read_csv('lieux-2019.csv',sep=';')
#lieux20 = pd.read_csv('lieux-2020.csv',sep=';')


carac15 = pd.read_csv("caracteristiques_2015.csv",encoding = "ISO-8859-1")
carac16 = pd.read_csv("caracteristiques_2016.csv",encoding = "ISO-8859-1")
carac17 = pd.read_csv("caracteristiques-2017.csv",encoding = "ISO-8859-1")
carac18 = pd.read_csv("caracteristiques-2018.csv",encoding = "ISO-8859-1")
carac19 = pd.read_csv('caracteristiques-2019.csv',sep=';')
#carac20 = pd.read_csv('caracteristiques-2020.csv',sep=';')


usag15 = pd.read_csv('usagers_2015.csv',sep=',')
usag16 = pd.read_csv('usagers_2016.csv',sep=',')
usag17 = pd.read_csv('usagers-2017.csv',sep=',')
usag18 = pd.read_csv('usagers-2018.csv',sep=',')
usag19 = pd.read_csv('usagers-2019.csv',sep=';')
#usag20 = pd.read_csv('usagers-2020.csv',sep=';')


# _**Import des données INSEE supplémentaires**_

# In[8]:


# Population
pop15_met =pd.read_excel('base-ic-evol-struct-pop-2015.xls',header=5)
pop15_dom =pd.read_excel('base-ic-evol-struct-pop-2015-com.xls', header=5)

pop16_met =pd.read_excel('base-ic-evol-struct-pop-2016.xls',sheet_name=0, header=5)
pop16_dom =pd.read_excel('base-ic-evol-struct-pop-2016-com.xls', header=5)

pop17_met =pd.read_csv('base-ic-evol-struct-pop-2017.CSV',sep=";")
pop17_dom =pd.read_csv('base-ic-evol-struct-pop-2017-com.CSV',sep=";")

pop18_met =pd.read_csv('base-ic-evol-struct-pop-2018.CSV',sep=";")
pop18_dom =pd.read_csv('base-ic-evol-struct-pop-2018-com.CSV',sep=";")

pop19_met = pd.read_csv('base-ic-evol-struct-pop-2019.CSV',sep=";")
pop19_dom =pd.read_csv('base-ic-evol-struct-pop-2019-com.CSV',sep=";")

# Statistiques départements - communes
stats = pd.read_excel('base_cc_comparateur.xlsx',header=5)


# Revenus des ménages
rev15 = pd.read_excel('base-cc-filosofi-2015.xls', sheet_name = 3, header=5)
rev16 = pd.read_excel('base-cc-filosofi-2016.xls', sheet_name = 3, header=5)
rev17 = pd.read_csv('cc_filosofi_2017_DEP.CSV', sep=";")
rev18 = pd.read_csv('cc_filosofi_2018_DEP-geo2021.CSV', sep=";")
rev19 = pd.read_csv('cc_filosofi_2019_DEP.csv',sep=";")


# Activités des résidents
activ15_met =pd.read_excel('base-ic-activite-residents-2015.xls', header=5)
activ15_dom =pd.read_excel('base-ic-activite-residents-2015-com.xls',header=5)

activ16_met =pd.read_excel('base-ic-activite-residents-2016.xls',header=5)
activ16_dom =pd.read_excel('base-ic-activite-residents-2016-com.xls',header=5)

activ17_met =pd.read_csv('base-ic-activite-residents-2017.CSV',sep=";")
activ17_dom =pd.read_csv('base-ic-activite-residents-2017-com.CSV',sep=";")

activ18_met =pd.read_csv('base-ic-activite-residents-2018.CSV',sep=";")
activ18_dom =pd.read_csv('base-ic-activite-residents-2018-com.CSV',sep=";")

activ19_met = pd.read_csv('base-ic-activite-residents-2019.CSV',sep=";")
activ19_dom =pd.read_csv('base-ic-activite-residents-2019-com.CSV',sep=";")


# Dans les tables Population, nous n'allons retenir que les 15 colonnes suivantes : 
# 
#     - COM : commune  
#     - P15_POP : population en 2015 (resp. en 2016 à 2019)  
#     - P15_POPH : population masculine en 2015 (resp. en 2016 à 2019)  
#     - P15_POPF : population féminine en 2015 (resp. en 2016 à 2019)  
#     - C15_POP15_CS1 : population >15ans dans la CSP Agriculteurs exploitants en 2015 (resp. 2016 à 2019)  
#     - C15_POP15_CS2 : population >15 ans dans la CSP Artisans, Comm., Chefs entr. en 2015 (resp. 2016 à 2019)  
#     - C15_POP15_CS3 : population >15 ans dans la CSP Cadres, Prof. intel. sup. en 2015 (resp. 2016 à 2019)  
#     - C15_POP15_CS4 : population >15 ans dans la CSP Prof. intermédiaires en 2015 (resp. 2016 à 2019)  
#     - C15_POP15_CS5 : population >15 ans dans la CSP Employés en 2015 (resp. 2016 à 2019)  
#     - C15_POP15_CS6 : population >15 ans dans la CSP Ouvriers en 2015 (resp. 2016 à 2019)   
#     - C15_POP15_CS7 : population >15 ans dans la CSP Retraités en 2015 (resp. 2016 à 2019)   
#     - C15_POP15_CS8 : population >15 ans dans la CSP Autres en 2015 (resp. 2016 à 2019)  
#     - P15_POP_FR : population de nationalité française (resp. en 2016 à 2019)  
#     - P15_POP_ETR : population de nationalité étrangère (resp. en 2016 à 2019)  
#     - P15_POP_IMM : population immigrée 
#     
# Dans la table stats, nous ne retenons que 2 colonnes :  
# 
#     - DEP : département  
#     - SUPERF : superficie
#     
# Dans les tables Revenu, nous allons retenir les 11 colonnes suivantes :   
# 
#     - CODGEO : département  
#     - LIBGEO : nom du département  
#     - TP6015 : taux de pauvreté en 2015 (%) (resp. 2016 à 2019 )  
#     - MED15 : médiane du niveau de vie en 2015 (resp. 2016 à 2019)  
#     
# Dans les tables Activité, nous allons retenir les 11 colonnes suivantes :  
# 
#     - COM : commune  
#     - P15_ACT1564 : actifs 15-64 ans en 2015 (resp. 2016 à 2019)  
#     - P15_CHOM1564 : chômeurs 15-64 ans en 2015 (resp. 2016 à 2019)  
#     - P15_INACT1564 : Inactifs 15-64 ans en 2015 (resp. 2016 à 2019)  
#     - P15_RETR1564 : retraité ou pré-retraités 15-64 ans en 2015 (resp. 2016 à 2019)  
#     - C15_ACTOCC15P_PAS : Actifs occupés >15 ans n'utilisant pas de transport pour aller au travail en 2015 (resp. 2016 à 2019)  
#     - C15_ACTOCC15P_MAR : Actifs occupés >15 ans marchant à pied pour aller au travail en 2015 (resp. 2016 à 2019)    
#     - C15_ACTOCC15P_DROU (C15_ACTOCC15P_2ROUESMOT) : Actifs occupés >15 ans utilisant un deux-roues motoriéé pour aller au travail en 2015 (resp. 2016 à 2019)  
#     - C15_ACTOCC15P_VOIT : Actifs occupés >15 ans utilisant la voiture pour aller travailler en 2015 (resp. 2016 à 2019)  
#     - C15_ACTOCC15P_TCOM : Actifs occupés >15 ans utilisant les transports en commun pour aller travailler en 2015 (resp. 2016 à 2019)  

# *Pré-traitement des données INSEE*

# Population

# In[9]:


#Sélection des variables intéressantes
pop15_met=pop15_met[['COM','P15_POP','P15_POPH','P15_POPF','C15_POP15P_CS1','C15_POP15P_CS2','C15_POP15P_CS3',
                    'C15_POP15P_CS4','C15_POP15P_CS5','C15_POP15P_CS6','C15_POP15P_CS7','C15_POP15P_CS8','P15_POP_FR',
                    'P15_POP_ETR','P15_POP_IMM']]
pop15_dom=pop15_dom[['COM','P15_POP','P15_POPH','P15_POPF','C15_POP15P_CS1','C15_POP15P_CS2','C15_POP15P_CS3',
                    'C15_POP15P_CS4','C15_POP15P_CS5','C15_POP15P_CS6','C15_POP15P_CS7','C15_POP15P_CS8','P15_POP_FR',
                    'P15_POP_ETR','P15_POP_IMM']]

pop16_met=pop16_met[['COM','P16_POP','P16_POPH','P16_POPF','C16_POP15P_CS1','C16_POP15P_CS2','C16_POP15P_CS3',
                    'C16_POP15P_CS4','C16_POP15P_CS5','C16_POP15P_CS6','C16_POP15P_CS7','C16_POP15P_CS8','P16_POP_FR',
                    'P16_POP_ETR','P16_POP_IMM']]
pop16_dom=pop16_dom[['COM','P16_POP','P16_POPH','P16_POPF','C16_POP15P_CS1','C16_POP15P_CS2','C16_POP15P_CS3',
                    'C16_POP15P_CS4','C16_POP15P_CS5','C16_POP15P_CS6','C16_POP15P_CS7','C16_POP15P_CS8','P16_POP_FR',
                    'P16_POP_ETR','P16_POP_IMM']]

pop17_met=pop17_met[['COM','P17_POP','P17_POPH','P17_POPF','C17_POP15P_CS1','C17_POP15P_CS2','C17_POP15P_CS3',
                    'C17_POP15P_CS4','C17_POP15P_CS5','C17_POP15P_CS6','C17_POP15P_CS7','C17_POP15P_CS8','P17_POP_FR',
                    'P17_POP_ETR','P17_POP_IMM']]
pop17_dom=pop17_dom[['COM','P17_POP','P17_POPH','P17_POPF','C17_POP15P_CS1','C17_POP15P_CS2','C17_POP15P_CS3',
                    'C17_POP15P_CS4','C17_POP15P_CS5','C17_POP15P_CS6','C17_POP15P_CS7','C17_POP15P_CS8','P17_POP_FR',
                    'P17_POP_ETR','P17_POP_IMM']]

pop18_met=pop18_met[['COM','P18_POP','P18_POPH','P18_POPF','C18_POP15P_CS1','C18_POP15P_CS2','C18_POP15P_CS3',
                    'C18_POP15P_CS4','C18_POP15P_CS5','C18_POP15P_CS6','C18_POP15P_CS7','C18_POP15P_CS8','P18_POP_FR',
                    'P18_POP_ETR','P18_POP_IMM']]
pop18_dom=pop18_dom[['COM','P18_POP','P18_POPH','P18_POPF','C18_POP15P_CS1','C18_POP15P_CS2','C18_POP15P_CS3',
                    'C18_POP15P_CS4','C18_POP15P_CS5','C18_POP15P_CS6','C18_POP15P_CS7','C18_POP15P_CS8','P18_POP_FR',
                    'P18_POP_ETR','P18_POP_IMM']]

pop19_met=pop19_met[['COM','P19_POP','P19_POPH','P19_POPF','C19_POP15P_CS1','C19_POP15P_CS2','C19_POP15P_CS3',
                    'C19_POP15P_CS4','C19_POP15P_CS5','C19_POP15P_CS6','C19_POP15P_CS7','C19_POP15P_CS8','P19_POP_FR',
                    'P19_POP_ETR','P19_POP_IMM']]
pop19_dom=pop19_dom[['COM','P19_POP','P19_POPH','P19_POPF','C19_POP15P_CS1','C19_POP15P_CS2','C19_POP15P_CS3',
                    'C19_POP15P_CS4','C19_POP15P_CS5','C19_POP15P_CS6','C19_POP15P_CS7','C19_POP15P_CS8','P19_POP_FR',
                    'P19_POP_ETR','P19_POP_IMM']]

stats=stats[['DEP','SUPERF']]

rev15=rev15[['CODGEO','LIBGEO','TP6015','MED15']]
rev16=rev16[['CODGEO','LIBGEO','TP6016','MED16']]
rev17=rev17[['CODGEO','TP6017','MED17']]
rev18=rev18[['CODGEO','TP6018','MED18']]
rev19=rev19[['CODGEO','TP6019','MED19']]

activ15_met=activ15_met[['COM','P15_ACT1564','P15_CHOM1564','P15_INACT1564','P15_RETR1564','C15_ACTOCC15P_PAS',
                         'C15_ACTOCC15P_MAR','C15_ACTOCC15P_DROU','C15_ACTOCC15P_VOIT','C15_ACTOCC15P_TCOM']]
activ15_dom=activ15_dom[['COM','P15_ACT1564','P15_CHOM1564','P15_INACT1564','P15_RETR1564','C15_ACTOCC15P_PAS',
                         'C15_ACTOCC15P_MAR','C15_ACTOCC15P_DROU','C15_ACTOCC15P_VOIT','C15_ACTOCC15P_TCOM']]
activ16_met=activ16_met[['COM','P16_ACT1564','P16_CHOM1564','P16_INACT1564','P16_RETR1564','C16_ACTOCC15P_PAS',
                         'C16_ACTOCC15P_MAR','C16_ACTOCC15P_DROU','C16_ACTOCC15P_VOIT','C16_ACTOCC15P_TCOM']]
activ16_dom=activ16_dom[['COM','P16_ACT1564','P16_CHOM1564','P16_INACT1564','P16_RETR1564','C16_ACTOCC15P_PAS',
                         'C16_ACTOCC15P_MAR','C16_ACTOCC15P_DROU','C16_ACTOCC15P_VOIT','C16_ACTOCC15P_TCOM']]
activ17_met=activ17_met[['COM','P17_ACT1564','P17_CHOM1564','P17_INACT1564','P17_RETR1564','C17_ACTOCC15P_PAS',
                         'C17_ACTOCC15P_MAR','C17_ACTOCC15P_2ROUESMOT','C17_ACTOCC15P_VOIT',
                         'C17_ACTOCC15P_TCOM']]
activ17_dom=activ17_dom[['COM','P17_ACT1564','P17_CHOM1564','P17_INACT1564','P17_RETR1564','C17_ACTOCC15P_PAS',
                         'C17_ACTOCC15P_MAR','C17_ACTOCC15P_2ROUESMOT','C17_ACTOCC15P_VOIT',
                         'C17_ACTOCC15P_TCOM']]
activ17_dom.rename(columns={'C17_ACTOCC15P_2ROUESMOT':'C17_ACTOCC15P_DROU'},inplace=True)
activ17_met.rename(columns={'C17_ACTOCC15P_2ROUESMOT':'C17_ACTOCC15P_DROU'},inplace=True)

activ18_met=activ18_met[['COM','P18_ACT1564','P18_CHOM1564','P18_INACT1564','P18_RETR1564','C18_ACTOCC15P_PAS',
                         'C18_ACTOCC15P_MAR','C18_ACTOCC15P_2ROUESMOT','C18_ACTOCC15P_VOIT',
                         'C18_ACTOCC15P_TCOM']]
activ18_dom=activ18_dom[['COM','P18_ACT1564','P18_CHOM1564','P18_INACT1564','P18_RETR1564','C18_ACTOCC15P_PAS',
                         'C18_ACTOCC15P_MAR','C18_ACTOCC15P_2ROUESMOT','C18_ACTOCC15P_VOIT',
                         'C18_ACTOCC15P_TCOM']]
activ18_dom.rename(columns={'C18_ACTOCC15P_2ROUESMOT':'C18_ACTOCC15P_DROU'},inplace=True)
activ18_met.rename(columns={'C18_ACTOCC15P_2ROUESMOT':'C18_ACTOCC15P_DROU'},inplace=True)

activ19_met=activ19_met[['COM','P19_ACT1564','P19_CHOM1564','P19_INACT1564','P19_RETR1564','C19_ACTOCC15P_PAS',
                         'C19_ACTOCC15P_MAR','C19_ACTOCC15P_2ROUESMOT','C19_ACTOCC15P_VOIT',
                         'C19_ACTOCC15P_TCOM']]
activ19_dom=activ19_dom[['COM','P19_ACT1564','P19_CHOM1564','P19_INACT1564','P19_RETR1564','C19_ACTOCC15P_PAS',
                         'C19_ACTOCC15P_MAR','C19_ACTOCC15P_2ROUESMOT','C19_ACTOCC15P_VOIT',
                         'C19_ACTOCC15P_TCOM']]
activ19_dom.rename(columns={'C19_ACTOCC15P_2ROUESMOT':'C19_ACTOCC15P_DROU'},inplace=True)
activ19_met.rename(columns={'C19_ACTOCC15P_2ROUESMOT':'C19_ACTOCC15P_DROU'},inplace=True)


#Retraitement de com
pop15_dom= pop15_dom.assign(COM2 = pop15_dom.COM.astype(str))
pop15_dom.drop(columns=['COM'],inplace=True)
pop15_dom.rename(columns = {'COM2':'COM'}, inplace = True)

pop16_dom= pop16_dom.assign(COM2 = pop16_dom.COM.astype(str))
pop16_dom.drop(columns=['COM'],inplace=True)
pop16_dom.rename(columns = {'COM2':'COM'}, inplace = True)

pop17_dom= pop17_dom.assign(COM2 = pop17_dom.COM.astype(str))
pop17_dom.drop(columns=['COM'],inplace=True)
pop17_dom.rename(columns = {'COM2':'COM'}, inplace = True)
pop17_met= pop17_met.assign(COM2 = pop17_met.COM.astype(str))
pop17_met.drop(columns=['COM'],inplace=True)
pop17_met.rename(columns = {'COM2':'COM'}, inplace = True)

pop18_dom= pop18_dom.assign(COM2 = pop18_dom.COM.astype(str))
pop18_dom.drop(columns=['COM'],inplace=True)
pop18_dom.rename(columns = {'COM2':'COM'}, inplace = True)
pop18_met= pop18_met.assign(COM2 = pop18_met.COM.astype(str))
pop18_met.drop(columns=['COM'],inplace=True)
pop18_met.rename(columns = {'COM2':'COM'}, inplace = True)

pop19_dom= pop19_dom.assign(COM2 = pop19_dom.COM.astype(str))
pop19_dom.drop(columns=['COM'],inplace=True)
pop19_dom.rename(columns = {'COM2':'COM'}, inplace = True)
pop19_met= pop19_met.assign(COM2 = pop19_met.COM.astype(str))
pop19_met.drop(columns=['COM'],inplace=True)
pop19_met.rename(columns = {'COM2':'COM'}, inplace = True)


activ15_dom= activ15_dom.assign(COM2 = activ15_dom.COM.astype(str))
activ15_dom.drop(columns=['COM'],inplace=True)
activ15_dom.rename(columns = {'COM2':'COM'}, inplace = True)

activ16_dom= activ16_dom.assign(COM2 = activ16_dom.COM.astype(str))
activ16_dom.drop(columns=['COM'],inplace=True)
activ16_dom.rename(columns = {'COM2':'COM'}, inplace = True)

activ17_dom= activ17_dom.assign(COM2 = activ17_dom.COM.astype(str))
activ17_dom.drop(columns=['COM'],inplace=True)
activ17_dom.rename(columns = {'COM2':'COM'}, inplace = True)
activ17_met= activ17_met.assign(COM2 = activ17_met.COM.astype(str))
activ17_met.drop(columns=['COM'],inplace=True)
activ17_met.rename(columns = {'COM2':'COM'}, inplace = True)

activ18_dom= activ18_dom.assign(COM2 = activ18_dom.COM.astype(str))
activ18_dom.drop(columns=['COM'],inplace=True)
activ18_dom.rename(columns = {'COM2':'COM'}, inplace = True)
activ18_met= activ18_met.assign(COM2 = activ18_met.COM.astype(str))
activ18_met.drop(columns=['COM'],inplace=True)
activ18_met.rename(columns = {'COM2':'COM'}, inplace = True)

activ19_dom= activ19_dom.assign(COM2 = activ19_dom.COM.astype(str))
activ19_dom.drop(columns=['COM'],inplace=True)
activ19_dom.rename(columns = {'COM2':'COM'}, inplace = True)
activ19_met= activ19_met.assign(COM2 = activ19_met.COM.astype(str))
activ19_met.drop(columns=['COM'],inplace=True)
activ19_met.rename(columns = {'COM2':'COM'}, inplace = True)

# Concaténation tables met et dom
pop15 = pd.concat([pop15_met,pop15_dom],ignore_index=True) 
pop16 = pd.concat([pop16_met,pop16_dom],ignore_index=True)
pop17 = pd.concat([pop17_met,pop17_dom],ignore_index=True) 
pop18 = pd.concat([pop18_met,pop18_dom],ignore_index=True) 
pop19 = pd.concat([pop19_met,pop19_dom],ignore_index=True)

activ15 = pd.concat([activ15_met,activ15_dom],ignore_index=True)
activ16 = pd.concat([activ16_met,activ16_dom],ignore_index=True)
activ17 = pd.concat([activ17_met,activ17_dom],ignore_index=True)
activ18 = pd.concat([activ18_met,activ18_dom],ignore_index=True)
activ19 = pd.concat([activ19_met,activ19_dom],ignore_index=True)



# In[10]:


# Création dep dans pop et activ
pop15COM = pop15['COM']
pop15DEP = []
for i in range(len(pop15COM)):
    pop15DEP.append(pop15COM[i][0:2])
    if pop15DEP[i] in (['97','98']):
        pop15DEP[i]=pop15COM[i][0:3]
pop15 = pop15.assign(dep=pop15DEP)    

activ15COM = activ15['COM']
activ15DEP = []
for i in range(len(activ15COM)):
    activ15DEP.append(activ15COM[i][0:2])
    if activ15DEP[i] in (['97','98']):
        activ15DEP[i]=activ15COM[i][0:3]
activ15 = activ15.assign(dep=activ15DEP)    


pop16COM = pop16['COM']
pop16DEP = []
for i in range(len(pop16COM)):
    pop16DEP.append(pop16COM[i][0:2])
    if pop16DEP[i] in (['97','98']):
        pop16DEP[i]=pop16COM[i][0:3]
pop16 = pop16.assign(dep=pop16DEP)
activ16COM = activ16['COM']
activ16DEP = []
for i in range(len(activ16COM)):
    activ16DEP.append(activ16COM[i][0:2])
    if activ16DEP[i] in (['97','98']):
        activ16DEP[i]=activ16COM[i][0:3]
activ16 = activ16.assign(dep=activ16DEP)    

pop17COM = pop17['COM']
pop17DEP = []
for i in range(len(pop17COM)):
    if len(pop17COM[i])==4:
        pop17DEP.append(pop17COM[i][0:1])
    else:    
        pop17DEP.append(pop17COM[i][0:2])
        if pop17DEP[i] in (['97','98']):
            pop17DEP[i]=pop17COM[i][0:3]
pop17 = pop17.assign(dep=pop17DEP)
activ17COM = activ17['COM']
activ17DEP = []
for i in range(len(activ17COM)):
    if len(activ17COM[i])==4:
        activ17DEP.append(activ17COM[i][0:1])
    else:    
        activ17DEP.append(activ17COM[i][0:2])
        if activ17DEP[i] in (['97','98']):
            activ17DEP[i]=activ17COM[i][0:3]
activ17 = activ17.assign(dep=pop17DEP)

pop18COM = pop18['COM']
pop18DEP = []
for i in range(len(pop18COM)):
    if len(pop18COM[i])==4:
        pop18DEP.append(pop18COM[i][0:1])
    else:    
        pop18DEP.append(pop18COM[i][0:2])
        if pop18DEP[i] in (['97','98']):
            pop18DEP[i]=pop18COM[i][0:3]
pop18 = pop18.assign(dep=pop18DEP)
activ18COM = activ18['COM']
activ18DEP = []
for i in range(len(activ18COM)):
    if len(activ18COM[i])==4:
        activ18DEP.append(activ18COM[i][0:1])
    else:    
        activ18DEP.append(activ18COM[i][0:2])
        if activ18DEP[i] in (['97','98']):
            activ18DEP[i]=activ18COM[i][0:3]
activ18 = activ18.assign(dep=activ18DEP)

pop19COM = pop19['COM']
pop19DEP = []
for i in range(len(pop19COM)):
    if len(pop19COM[i])==4:
        pop19DEP.append(pop19COM[i][0:1])
    else:    
        pop19DEP.append(pop19COM[i][0:2])
        if pop19DEP[i] in (['97','98']):
            pop19DEP[i]=pop19COM[i][0:3]
pop19 = pop19.assign(dep=pop19DEP)
activ19COM = activ19['COM']
activ19DEP = []
for i in range(len(activ19COM)):
    if len(activ19COM[i])==4:
        activ19DEP.append(activ19COM[i][0:1])
    else:    
        activ19DEP.append(activ19COM[i][0:2])
        if activ19DEP[i] in (['97','98']):
            activ19DEP[i]=activ19COM[i][0:3]
activ19 = activ19.assign(dep=activ19DEP)


# Suppression de COM
pop15.drop(columns=['COM'],inplace=True)
pop16.drop(columns=['COM'],inplace=True)
pop17.drop(columns=['COM'],inplace=True)
pop18.drop(columns=['COM'],inplace=True)
pop19.drop(columns=['COM'],inplace=True)

activ15.drop(columns=['COM'],inplace=True)
activ16.drop(columns=['COM'],inplace=True)
activ17.drop(columns=['COM'],inplace=True)
activ18.drop(columns=['COM'],inplace=True)
activ19.drop(columns=['COM'],inplace=True)


# In[11]:


# Garder les libellés départements et codes de départements
departements = rev16[['CODGEO','LIBGEO']]


# In[12]:


# Renommer les colonnes
pop15.rename(columns = {'P15_POP':'POP','P15_POPH':'POPH','P15_POPF':'POPF','C15_POP15P_CS1':'POP15P_CS1',
                        'C15_POP15P_CS2':'POP15P_CS2','C15_POP15P_CS3':'POP15P_CS3','C15_POP15P_CS4':'POP15P_CS4',
                        'C15_POP15P_CS5':'POP15P_CS5','C15_POP15P_CS6':'POP15P_CS6','C15_POP15P_CS7':'POP15P_CS7',
                        'C15_POP15P_CS8':'POP15P_CS8','P15_POP_FR':'POP_FR','P15_POP_ETR':'POP_ETR',
                        'P15_POP_IMM':'POP_IMM'}, inplace = True)
pop16.rename(columns = {'P16_POP':'POP','P16_POPH':'POPH','P16_POPF':'POPF','C16_POP15P_CS1':'POP15P_CS1',
                        'C16_POP15P_CS2':'POP15P_CS2','C16_POP15P_CS3':'POP15P_CS3','C16_POP15P_CS4':'POP15P_CS4',
                        'C16_POP15P_CS5':'POP15P_CS5','C16_POP15P_CS6':'POP15P_CS6','C16_POP15P_CS7':'POP15P_CS7',
                        'C16_POP15P_CS8':'POP15P_CS8','P16_POP_FR':'POP_FR','P16_POP_ETR':'POP_ETR',
                        'P16_POP_IMM':'POP_IMM'}, inplace = True)
pop17.rename(columns = {'P17_POP':'POP','P17_POPH':'POPH','P17_POPF':'POPF','C17_POP15P_CS1':'POP15P_CS1',
                        'C17_POP15P_CS2':'POP15P_CS2','C17_POP15P_CS3':'POP15P_CS3','C17_POP15P_CS4':'POP15P_CS4',
                        'C17_POP15P_CS5':'POP15P_CS5','C17_POP15P_CS6':'POP15P_CS6','C17_POP15P_CS7':'POP15P_CS7',
                        'C17_POP15P_CS8':'POP15P_CS8','P17_POP_FR':'POP_FR','P17_POP_ETR':'POP_ETR',
                        'P17_POP_IMM':'POP_IMM'}, inplace = True)
pop18.rename(columns = {'P18_POP':'POP','P18_POPH':'POPH','P18_POPF':'POPF','C18_POP15P_CS1':'POP15P_CS1',
                        'C18_POP15P_CS2':'POP15P_CS2','C18_POP15P_CS3':'POP15P_CS3','C18_POP15P_CS4':'POP15P_CS4',
                        'C18_POP15P_CS5':'POP15P_CS5','C18_POP15P_CS6':'POP15P_CS6','C18_POP15P_CS7':'POP15P_CS7',
                        'C18_POP15P_CS8':'POP15P_CS8','P18_POP_FR':'POP_FR','P18_POP_ETR':'POP_ETR',
                        'P18_POP_IMM':'POP_IMM'}, inplace = True)
pop19.rename(columns = {'P19_POP':'POP','P19_POPH':'POPH','P19_POPF':'POPF','C19_POP15P_CS1':'POP15P_CS1',
                        'C19_POP15P_CS2':'POP15P_CS2','C19_POP15P_CS3':'POP15P_CS3','C19_POP15P_CS4':'POP15P_CS4',
                        'C19_POP15P_CS5':'POP15P_CS5','C19_POP15P_CS6':'POP15P_CS6','C19_POP15P_CS7':'POP15P_CS7',
                        'C19_POP15P_CS8':'POP15P_CS8','P19_POP_FR':'POP_FR','P19_POP_ETR':'POP_ETR',
                        'P19_POP_IMM':'POP_IMM'}, inplace = True)


rev15.rename(columns = {'CODGEO':'dep','TP6015':'TAUX_P','MED15':'MED_VIE'}, inplace = True)
rev15.drop(columns=['LIBGEO'],inplace=True)
rev16.rename(columns = {'CODGEO':'dep','TP6016':'TAUX_P','MED16':'MED_VIE'}, inplace = True)
rev16.drop(columns=['LIBGEO'],inplace=True)
rev17.rename(columns = {'CODGEO':'dep','TP6017':'TAUX_P','MED17':'MED_VIE'}, inplace = True)
rev18.rename(columns = {'CODGEO':'dep','TP6018':'TAUX_P','MED18':'MED_VIE'}, inplace = True)
rev19.rename(columns = {'CODGEO':'dep','TP6019':'TAUX_P','MED19':'MED_VIE'}, inplace = True)


activ15.rename(columns = {'P15_ACT1564':'ACT_1564','P15_CHOM1564':'CHOM_1564','P15_INACT1564':'INACT_1564',
                          'P15_RETR1564':'RETR_1564','C15_ACTOCC15P_PAS':'ACTOCC15P_PAS',
                          'C15_ACTOCC15P_MAR':'ACTOCC15P_MAR','C15_ACTOCC15P_DROU':'ACTOCC15P_DROU',
                        'C15_ACTOCC15P_VOIT':'ACTOCC15P_VOIT','C15_ACTOCC15P_TCOM':'ACTOCC15P_TCOM'}, inplace = True)
activ16.rename(columns = {'P16_ACT1564':'ACT_1564','P16_CHOM1564':'CHOM_1564','P16_INACT1564':'INACT_1564',
                          'P16_RETR1564':'RETR_1564','C16_ACTOCC15P_PAS':'ACTOCC15P_PAS',
                          'C16_ACTOCC15P_MAR':'ACTOCC15P_MAR','C16_ACTOCC15P_DROU':'ACTOCC15P_DROU',
                          'C16_ACTOCC15P_VOIT':'ACTOCC15P_VOIT','C16_ACTOCC15P_TCOM':'ACTOCC15P_TCOM'}, inplace = True)
activ17.rename(columns = {'P17_ACT1564':'ACT_1564','P17_CHOM1564':'CHOM_1564','P17_INACT1564':'INACT_1564',
                          'P17_RETR1564':'RETR_1564','C17_ACTOCC15P_PAS':'ACTOCC15P_PAS',
                          'C17_ACTOCC15P_MAR':'ACTOCC15P_MAR',
                          'C17_ACTOCC15P_DROU':'ACTOCC15P_DROU','C17_ACTOCC15P_VOIT':'ACTOCC15P_VOIT',
                          'C17_ACTOCC15P_TCOM':'ACTOCC15P_TCOM'}, inplace = True)
activ18.rename(columns = {'P18_ACT1564':'ACT_1564','P18_CHOM1564':'CHOM_1564','P18_INACT1564':'INACT_1564',
                          'P18_RETR1564':'RETR_1564','C18_ACTOCC15P_PAS':'ACTOCC15P_PAS',
                          'C18_ACTOCC15P_MAR':'ACTOCC15P_MAR',
                          'C18_ACTOCC15P_DROU':'ACTOCC15P_DROU','C18_ACTOCC15P_VOIT':'ACTOCC15P_VOIT',
                          'C18_ACTOCC15P_TCOM':'ACTOCC15P_TCOM'}, inplace = True)
activ19.rename(columns = {'P19_ACT1564':'ACT_1564','P19_CHOM1564':'CHOM_1564','P19_INACT1564':'INACT_1564',
                          'P19_RETR1564':'RETR_1564','C19_ACTOCC15P_PAS':'ACTOCC15P_PAS',
                          'C19_ACTOCC15P_MAR':'ACTOCC15P_MAR',
                          'C19_ACTOCC15P_DROU':'ACTOCC15P_DROU','C19_ACTOCC15P_VOIT':'ACTOCC15P_VOIT',
                          'C19_ACTOCC15P_TCOM':'ACTOCC15P_TCOM'}, inplace = True)



# In[13]:


# Aggrégation des tables pop et activ 
pop15 = pop15.groupby(['dep'], as_index=False).sum()
pop16 = pop16.groupby(['dep'], as_index=False).sum()
pop17 = pop17.groupby(['dep'], as_index=False).sum()
pop18 = pop18.groupby(['dep'], as_index=False).sum()
pop19 = pop19.groupby(['dep'], as_index=False).sum()


activ15 = activ15.groupby(['dep'], as_index=False).sum()
activ16 = activ16.groupby(['dep'], as_index=False).sum()
activ17 = activ17.groupby(['dep'], as_index=False).sum()
activ18 = activ18.groupby(['dep'], as_index=False).sum()
activ19 = activ19.groupby(['dep'], as_index=False).sum()


# In[14]:


# Ajout annee
pop15=pop15.assign(annee=2015)
pop16=pop16.assign(annee=2016)
pop17=pop17.assign(annee=2017)
pop18=pop18.assign(annee=2018)
pop19=pop19.assign(annee=2019)

rev15=rev15.assign(annee=2015)
rev16=rev16.assign(annee=2016)
rev17=rev17.assign(annee=2017)
rev18=rev18.assign(annee=2018)
rev19=rev19.assign(annee=2019)

activ15=activ15.assign(annee=2015)
activ16=activ16.assign(annee=2016)
activ17=activ17.assign(annee=2017)
activ18=activ18.assign(annee=2018)
activ19=activ19.assign(annee=2019)

# Concaténation des tables 
pop=pd.concat([pop15,pop16,pop17,pop18,pop19],ignore_index=True)
rev=pd.concat([rev15,rev16,rev17,rev18,rev19],ignore_index=True)
activ=pd.concat([activ15,activ16,activ17,activ18,activ19],ignore_index=True)

# Correction
pop["dep"]=pop["dep"].replace("1","01")
pop["dep"]=pop["dep"].replace("2","02")
pop["dep"]=pop["dep"].replace("3","03")
pop["dep"]=pop["dep"].replace("4","04")
pop["dep"]=pop["dep"].replace("5","05")
pop["dep"]=pop["dep"].replace("6","06")
pop["dep"]=pop["dep"].replace("7","07")
pop["dep"]=pop["dep"].replace("8","08")
pop["dep"]=pop["dep"].replace("9","09")

activ["dep"]=activ["dep"].replace("1","01")
activ["dep"]=activ["dep"].replace("2","02")
activ["dep"]=activ["dep"].replace("3","03")
activ["dep"]=activ["dep"].replace("4","04")
activ["dep"]=activ["dep"].replace("5","05")
activ["dep"]=activ["dep"].replace("6","06")
activ["dep"]=activ["dep"].replace("7","07")
activ["dep"]=activ["dep"].replace("8","08")
activ["dep"]=activ["dep"].replace("9","09")

rev["TAUX_P"]=rev["TAUX_P"].replace(',','.', regex=True).astype(float)


# In[15]:


# Concaténation des 3 tables insee
pop_activ=pd.merge(pop,activ,on=['dep','annee'])
pop_activ_rev=pd.merge(pop_activ,rev,how='left',on=['dep','annee'])
pop_activ_rev=pop_activ_rev[['dep','annee','TAUX_P','MED_VIE','POP','POPH','POPF','POP_FR','POP_ETR','POP_IMM',
                             'POP15P_CS1','POP15P_CS2','POP15P_CS3','POP15P_CS4','POP15P_CS5','POP15P_CS6',
                             'POP15P_CS7','POP15P_CS8','ACT_1564','CHOM_1564','INACT_1564','RETR_1564','ACTOCC15P_PAS',
                             'ACTOCC15P_MAR','ACTOCC15P_DROU','ACTOCC15P_VOIT','ACTOCC15P_TCOM']]


# In[16]:


# Dans la table stats : aggréger les lignes pour sommer les superf 

superficie = stats.groupby(['DEP'], as_index=False).sum()
superficie.rename(columns = {'DEP':'dep'}, inplace = True)


# In[17]:


superficie.head()


# In[18]:


# Fusionner les données superficies avec la base_insee et calcul de densité de population

base_insee=pd.merge(pop_activ_rev,superficie,how='left',on=['dep'])
base_insee=base_insee.assign(DENSITE=base_insee['POP']/base_insee['SUPERF'])


# In[19]:


base_insee.head()


# Etant donné que les données insee recueillies ne sont pas disponibles pour l'année 2020, nous n'allons pas retenir les tables de 2020 quant aux bases d'accidents corporels.
# L'étude portera donc sur les 5 années : 2015 à 2019 incluses.  

# ### *2.2 Description des variables dans chaque table*

# _**Base Vehicules :**_
# 
# Num_Acc : numéro d'identifiant de l'accident  
# id_vehicule : identifiant unique du véhicule repris pour chacun des usagers occupant le véhicule (code numérique)  
# Num_Veh : identifiant unique du véhicule repris pour chacun des usagers occupant le véhicule (code alphanumérique)  
# senc : sens de circulation 
# 
#     -1 -> Non renseigné
#      0 -> Inconnu  
#      1 -> PK ou PR ou numéro d'adresse postale croissant  
#      2 -> PK ou PR ou numéro d’adresse postale décroissant  
#      3 -> Absence de repère  
#  
# catv : catégorie du véhicule  
#     
#     00 -> indéterminable
#     01 -> bicyclette 
#     02 -> cyclomoteur <50cm3
#     03 -> voiturette
#     04 -> scooter immatriculé
#     05 -> motocyclette
#     06 -> side-car
#     07 -> VL seul
#     08 -> VL+caravane
#     09 -> VL+remorque
#     10 -> VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque
#     11 -> VU seul 1,5T <= PTAC <= 3,5T + caravane
#     12 -> VU seul 1,5T <= PTAC <= 3,5T + remorque
#     13 -> PL seul 3,5T < PTAC <= 7,5T  
#     14 -> PL seul > 7,5T
#     15 -> PL > 3,5T + remorque 
#     16 -> tracteur routier seul
#     17 -> tracteur routier + semi-remorque 
#     18 -> transports en commun 
#     19 -> tramway 
#     20 -> engin spécial 
#     21 -> tracteur agricole
#     30 -> scooter <50cm3
#     31 -> motocyclette > 50cm3 <= 125cm3
#     32 -> scooter > 50cm3 <= 125cm3
#     33 -> motocyclette > 125cm3
#     34 -> scooter > 125cm3
#     35 -> quad léger <=50cm3
#     36 -> quad lourd > 50cm3
#     37 -> autobus 
#     38 -> autocar
#     39 -> train 
#     40 -> tramway 
#     41 -> 3RM <= 50cm3
#     42 -> 3RM > 50cm3 <=125cm3
#     43 -> 3RM > 125cm3
#     50 -> EDP à moteur
#     60 -> EDP sans moteur
#     80 -> VAE
#     99 -> Autre véhicule
#     
# obs : obstacle fixé heurté  
# 
#     -1-> Non renseigné
#     0 -> Sans objet
#     1 -> Véhicule en stationnement
#     2 -> Arbre
#     3 -> Glissière métallique
#     4 -> Glissière en béton 
#     5 -> Autre glissière
#     6 -> Batiment, mur, pile de pont
#     7 -> Support de signalisation verticale ou poste d'appel d'urgence
#     8 -> Poteau
#     9 -> Mobilier urbain
#     10 -> Parapet
#     11 -> Ilot, refuge, borne haute
#     12 -> Bordure de trottoir
#     13 -> Fossé, talus, paroi rocheuse 
#     14 -> Autre obstacle fixe sur chaussée 
#     15 -> Autre obstacle fixe sur trottoir ou accotement
#     16 -> Sortie de chaussée sans obstacle 
#     17 -> Buse - tête d’aqueduc 
#     
# obsm : obstacle mobile heurté
# 
#     -1-> Non renseigné
#     0 -> Sans objet 
#     1 -> Piéton
#     2 -> Véhicule
#     4 -> Véhicule sur rails
#     5 -> Animal domestique
#     6 -> Animal sauvage 
#     9 -> Autre
#     
# choc : point de choc initial 
# 
#     -1-> Non renseigné
#     0 -> Aucun
#     1 -> Avant
#     2 -> Avant droit
#     3 -> Avant gauche
#     4 -> Arrière
#     5 -> Arrière droit
#     6 -> Arrière gauche
#     7 -> Côté droit
#     8 -> Côté gauche
#     9 -> Chocs multiples (tonneaux)
#     
# manv : manoeuvre principale avant l'accident
# 
#     -1-> Non renseigné
#     0 -> Inconnue
#     1 -> Sans changement de direction
#     2 -> Même sens, même file
#     3 -> Entre 2 files
#     4 -> Marche arrière
#     5 -> A contre sens
#     6 -> En franchissant le terre plein central 
#     7 -> Dans le couloir bus, dans le même sens 
#     8 -> Dans le couloir bus, en sens Inverse 
#     9 -> En s’insérant 
#     10-> En faisant demi-tour sur la chaussée 
# 
# Changeant de file
# 
#     11-> A gauche 
#     12-> A droite
#     
# Déporté
# 
#     13-> A gauche 
#     14-> A droite
# 
# Tournant
# 
#     15-> A Gauche 
#     16-> A droite 
# 
# Dépassant
# 
#     17-> A Gauche 
#     18-> A droite 
#     
# Divers
# 
#     19-> Traversant la chaussée
#     20-> Manoeuvre de stationnement 
#     21-> Manoeuvre d'évitement
#     22-> Ouverture de porte 
#     23-> Arrêté (hors stationnement)
#     24-> En stationnement (avec occupant)
#     25-> Circulant sur trottoir 
#     26-> Autres manoeuvres
#     
#     
# motor : type de motorisation du véhicule 
# 
#     -1-> Non renseigné
#     0 -> inconnue
#     1 -> Hydrocarbures 
#     2 -> Hybride électrique
#     3 -> Electrique
#     4 -> Hydrogène
#     5 -> Humaine
#     6 -> Autre
#     
# occutc : nombre d’occupants dans le transport en commun 
# 
# 
# 
# 
# _**Base Lieux :**_  
# 
# Num_Acc : numéro d'identifiant de l'accident  
# catr : catégorie de route   
# 
#     1 -> autoroute  
#     2 -> route nationale  
#     3 -> route départementale  
#     4 -> voie communale  
#     5 -> hors réseau public  
#     6 -> parc de stationnement ouvert à la circulation publique  
#     9 -> autre 
#     
# voie : numéro de la route  
# V1 : indice numérique du numéro de route   
# V2 : lettre indice alphanumérique de la route
# 
# circ : régime de circulation
# 
#     -1 -> non renseigné
#     1 -> a sens unique  
#     2 -> bidirectionnelle  
#     3 -> a chaussées séparées  
#     4 -> avec voies d’affectation variable 
#     
# nbv : nombre total de voies de circulation  
# vosp : signale l'existence d'une voie réservée, indépendamment du fait que l'accident ait lieu ou non sur la voie 
# 
#     -1 -> non renseigné
#     0 -> sans objet
#     1 -> piste cyclable  
#     2 -> bande cyclable  
#     3 -> voie réservée 
#     
# prof : profil en long décrit la déclivité de la route à l’endroit de l'accident 
# 
#     -1 -> non renseigné
#     1 -> plat 
#     2 -> pente 
#     3 -> sommet de côte 
#     4 -> bas de côte 
# pr : numéro du PR "point de référence" de rattachement (numéro de la borne amont)   
# pr1 : distance en mètres du PR "point de référence" (par rapport à la borne amont)   
# plan : tracé en plan 
# 
#     -1 -> non renseigné
#     1 -> partie rectiligne  
#     2 -> en courbe à gauche  
#     3 -> en courbe à droite  
#     4 -> en "S" 
#     
# lartpc : largeur du terre plein central (TPC) s'il existe   
# larrout : largeur de la chaussée affectée à la circulation des véhicules ne sont pas compris les bandes d'arrêt  d’urgence, les TPC et les places de stationnement 
# 
# surf : état de la surface 
# 
#     -1 -> non renseigné
#     1 -> normale 
#     2 -> mouillée 
#     3 -> flaques 
#     4 -> inondée 
#     5 -> enneigée 
#     6 -> boue 
#     7 -> verglacée 
#     8 -> corps gras - huile 
#     9 -> autre 
#     
# infra : aménagement - infrastructure  
# 
#     -1 -> non renseigné
#     0 -> Aucun  
#     1 -> souterrain- tunnel 
#     2 -> pont - autopont 
#     3 -> bretelle d'échangeur ou de raccordement 
#     4 -> voie ferrée 
#     5 -> carrefour aménagé 
#     6 -> zone piétonne 
#     7 -> zone de péage
#     8 -> Chantier  
#     9 -> Autres  
#     
#     
# situ : situation de l'accident 
# 
#     -1 -> non renseigné  
#     0 -> Aucun  
#     1 -> sur chaussée 
#     2 -> sur bande d'arrêt d’urgence  
#     3 -> sur accotement 
#     4 -> sur trottoir 
#     5 -> sur piste cyclable  
#     6 -> sur autre voie spéciale  
#     8 -> autres  
#     
# env1 : point école, proximité d'une école  
# vma : vitesse maximale autorisée sur le lieu et au moment de l'accident   
# 
# 
# 
#     
# _**Base Caractéristiques :**_ 
# 
# Num_Acc : Numéro d'identifiant de l'accident   
# jour : jour de l'accident   
# mois : mois de l'accident   
# an : année de l'accident   
# hrmn : heure et minutes de l'accident   
# lum : lumière, conditions d'éclairage dans lesquelles l'accident s'est produit   
# 
#     1 -> plein jour 
#     2 -> crépuscule ou aube 
#     3 -> nuit sans éclairage public 
#     4 -> nuit avec éclairage public non allumé 
#     5 -> nuit avec éclairage public allumé 
#     
# dep : département code Insee   
# com : commune code Insee   
# agg : localisation 
# 
#     1 -> hors agglomération 
#     2 -> en agglomération 
#     
# int : intersection 
# 
#     1 -> hors intersection 
#     2 -> intersection en X 
#     3 -> intersection en T 
#     4 -> intersection en Y 
#     5 -> intersection à plus de 4 branches 
#     6 -> giratoire 
#     7 -> place 
#     8 -> passage à niveau 
#     9 -> autre intersection 
#     
#  atm : conditions atmosphériques 
#  
#     -1 -> non renseigné
#     1 -> normale 
#     2 -> pluie légère 
#     3 -> pluie forte 
#     4 -> neige - grêle 
#     5 -> brouillard - fumée 
#     6 -> vent fort - tempête 
#     7 -> temps éblouissant  
#     8 -> temps couvert  
#     9 -> autre 
#     
# col : type de collision 
# 
#     -1 -> non renseigné
#     1 -> deux véhicules - frontale 
#     2 -> deux véhicules - par l'arrière 
#     3 -> deux véhicules  - par le côté 
#     4 -> trois véhicules et plus - en chaîne 
#     5 -> trois véhicules et plus - collisions multiples 
#     6 -> autre collision 
#     7 -> sans collision  
#     
# adr : adresse postale   
# gps : codage gps  
# 
#     M -> Métropole 
#     A -> Antilles (Martinique ou Guadeloupe) 
#     G -> Guyane  
#     R -> Réunion 
#     Y -> Mayotte  
#     
# lat : latitude   
# long : longitude   
# 
# 
#         
#         
# _**Base Usagers :**_
# 
# Num_Acc : numéro d'identifiant de l'accident   
# Num_Veh : identificant du véhicule repris pour chacun des usagers occupant ce véhicule  
# place : permet de situer la place occupée dans le véhicule par l'usager au moment de l’accident  
# catu : catégorie d'usager 
# 
#     1 -> conducteur 
#     2 -> passager 
#     3 -> piéton 
#     
# grav : gravité de l’accident 
# 
#     1 -> indemne 
#     2 -> tué 
#     3 -> blessé hospitalisé  
#     4 -> blessé léger 
#     
# sexe : sexe de l'usager 
# 
#     1 -> masculin 
#     2 -> féminin 
#     
# an_nais : année de naissance de l'usager   
# 
# trajet : motif du déplacement au moment de l’accident  
# 
#     -1 -> non renseigné  
#     0 -> non renseigné  
#     1 -> domicile - travail  
#     2 -> domicile - école 
#     3 -> courses - achats 
#     4 -> utilisation professionnelle  
#     5 -> promenade - loisirs 
#     9 -> autre 
#     
# secu : sur 2 caractères (jusqu'en 2018), le premier concerne l'existence d'un équipement de sécurité 
# 
#     1 -> ceinture 
#     2 -> casque 
#     3 -> dispositif enfant 
#     4 -> équipement réfléchissant  
#     9 -> autre 
#     
#    le second concerne l’utilisation de l'Équipement de sécurité
#     
#     1 -> oui 
#     2 -> non 
#     3 -> non déterminable  
# 
# secu1 : le renseignement du caractère indique la présence et l'utilisation de l'équipement de sécurité  
#     
#     -1 -> non renseigné  
#     0 -> aucun équipement  
#     1 -> ceinture  
#     2 -> casque  
#     3 -> dispositif enfant  
#     4 -> gilet réfléchissant  
#     5 -> airbag (2RM/3RM)  
#     6 -> gants (2RM/3RM)  
#     7 -> gants + airbag (2RM/3RM)  
#     8 -> non déterminable  
#     9 -> autre 
#  
# secu2 : le renseignement du caractère indique la présence et l'utilisation de l'équipement de sécurité  
#     
#     -1 -> non renseigné  
#     0 -> aucun équipement  
#     1 -> ceinture  
#     2 -> casque  
#     3 -> dispositif enfant  
#     4 -> gilet réfléchissant  
#     5 -> airbag (2RM/3RM)  
#     6 -> gants (2RM/3RM)  
#     7 -> gants + airbag (2RM/3RM)  
#     8 -> non déterminable  
#     9 -> autre 
#     
# secu3 : le renseignement du caractère indique la présence et l'utilisation de l'équipement de sécurité  
#     
#     -1 -> non renseigné  
#     0 -> aucun équipement  
#     1 -> ceinture  
#     2 -> casque  
#     3 -> dispositif enfant  
#     4 -> gilet réfléchissant  
#     5 -> airbag (2RM/3RM)  
#     6 -> gants (2RM/3RM)  
#     7 -> gants + airbag (2RM/3RM)  
#     8 -> non déterminable  
#     9 -> autre  
#     
#     
# locp : localisation du piéton   
#    Sur chaussée 
#     
#     1 -> A >50m du passage piéton 
#     2 -> A <50m du passage piéton 
#     
#    Sur passage piéton 
#     
#     3 -> Sans signalisation lumineuse  
#     4 -> Avec signalisation lumineuse
#     
#    Divers 
#    
#     5 -> sur trottoir  
#     6 -> sur accotement  
#     7 -> sur refuge ou BAU 
#     8 -> sur contre allée 
#     9 -> inconnue  
#     
# actp : action du piéton 
#     se déplaçant 
#     
#     -1 -> non renseigné  
#     0 -> non renseigné ou sans objet 
#     1 -> sens véhicule heurtant 
#     2 -> sens inverse du véhicule 
#     
#    divers 
#    
#     3 -> traversant 
#     4 -> masqué 
#     5 -> jouant-courant 
#     6 -> avec animal 
#     9 -> autre   
#     
# etatp : permet de situer si le piéton accidenté était seul ou non   
# 
#     -1 -> non renseigné  
#     1 -> seul  
#     2 -> accompagné  
#     3 -> en groupe  
# 

# ### *2.3 Homogénéisation des tables : nettoyage, reformatage et recodage des données par table*

# Les tables avant 2019 et depuis 2019 peuvent ne pas contenir les mêmes informations et/ou en avoir en supplément. C'est pourquoi, un retraitement/nettoyage préalable de toutes les tables est à effectuer en amont afin d'homogénéiser celles-ci.  

# #### _2.3.1 **Table Usagers**_

# In[20]:


usag15.head()


# In[21]:


usag16.head()


# In[22]:


usag17.head()


# In[23]:


usag18.head()


# In[24]:


usag19.head()


# *Homogénéiser les tables usagers*  
# 
# Etant donné qu'il y a eu une évolution des tables à partir de 2019, il y a plusieurs variables des tables 2019 qui n'existaient pas dans celle de 2018 et auparavant : par exemple la variable secu, secu1, secu2, secu3. Pour simplifier, dans la table 2019, je retraite et garde uniquement la variable secu1 en la transformant au même format que secu dans les autres tables. Etant donné la définition des codes, nous simplifions le retraitement de secu1 en réaffectant 0 à 92 (pour signifier autre équipement non utilisé), et 5-6-7-8  à 91 (pour signifier autre équipement utilisé).    
# La variable an_nais n'est pas au même format dans les tables : je la retraite.  
# Les variables place et trajet ne sont pas au même format dans les tables : je les retraite.  

# In[25]:


#variable secu, id_vehicule, secu1, secu2, secu3#
usag19.drop(columns=["id_vehicule","secu2","secu3"], inplace=True)
usag19.rename(columns = {'secu1': 'secu'},inplace=True)
usag19['secu'] = usag19['secu'].replace(0,92)
usag19['secu'] = usag19['secu'].replace(5,93)
usag19['secu'] = usag19['secu'].replace(6,93)
usag19['secu'] = usag19['secu'].replace(7,93)
usag19['secu'] = usag19['secu'].replace(8,93)
usag19['secu'] = usag19['secu'].replace(1,11)
usag19['secu'] = usag19['secu'].replace(2,21)
usag19['secu'] = usag19['secu'].replace(3,31)
usag19['secu'] = usag19['secu'].replace(4,41)
usag15['secu'] = usag15['secu'].fillna(-1).astype(int)
usag16['secu'] = usag16['secu'].fillna(-1).astype(int)
usag17['secu'] = usag17['secu'].fillna(-1).astype(int)
usag18['secu'] = usag18['secu'].fillna(-1).astype(int)

#Variable an_nais#
usag15['an_nais'] = usag15['an_nais'].fillna(0).astype(int)
usag16['an_nais'] = usag16['an_nais'].fillna(0).astype(int)
usag17['an_nais'] = usag17['an_nais'].fillna(0).astype(int)
usag18['an_nais'] = usag18['an_nais'].fillna(0).astype(int)

#Variable trajet#
usag15['trajet'] = usag15['trajet'].fillna(-1).astype(int)
usag16['trajet'] = usag16['trajet'].fillna(-1).astype(int)
usag17['trajet'] = usag17['trajet'].fillna(-1).astype(int)
usag18['trajet'] = usag18['trajet'].fillna(-1).astype(int)

#Variable place#
usag15['place'] = usag15['place'].fillna(0).astype(int)
usag16['place'] = usag16['place'].fillna(0).astype(int)
usag17['place'] = usag17['place'].fillna(0).astype(int)
usag18['place'] = usag18['place'].fillna(0).astype(int)

#Variables locp, etatp, actp
usag15['locp'] = usag15['locp'].fillna(-1).astype(int)
usag15['etatp'] = usag15['etatp'].fillna(-1).astype(int)
usag15['actp'] = usag15['actp'].fillna(-1).astype(int)
usag16['locp'] = usag16['locp'].fillna(-1).astype(int)
usag16['etatp'] = usag16['etatp'].fillna(-1).astype(int)
usag16['actp'] = usag16['actp'].fillna(-1).astype(int)
usag17['locp'] = usag17['locp'].fillna(-1).astype(int)
usag17['etatp'] = usag17['etatp'].fillna(-1).astype(int)
usag17['actp'] = usag17['actp'].fillna(-1).astype(int)
usag18['locp'] = usag18['locp'].fillna(-1).astype(int)
usag18['etatp'] = usag18['etatp'].fillna(-1).astype(int)
usag18['actp'] = usag18['actp'].fillna(-1).astype(int)
usag19['actp'] = pd.to_numeric(usag19['actp'],errors='coerce')
usag19['actp'] = usag19['actp'].fillna(-1).astype(int)


# In[26]:


#Réordonner les colonnes#
usag15.reindex(columns=['Num_Acc','num_veh','sexe','an_nais','place','secu','catu','grav','trajet','locp','actp',
                        'etatp'])


# In[27]:


#Réordonner les colonnes#
usag16.reindex(columns=['Num_Acc','num_veh','sexe','an_nais','place','secu','catu','grav','trajet','locp','actp',
                        'etatp'])


# In[28]:


#Réordonner les colonnes#
usag17.reindex(columns=['Num_Acc','num_veh','sexe','an_nais','place','secu','catu','grav','trajet','locp','actp',
                        'etatp'])


# In[29]:


#Réordonner les colonnes#
usag18.reindex(columns=['Num_Acc','num_veh','sexe','an_nais','place','secu','catu','grav','trajet','locp','actp',
                        'etatp'])


# In[30]:


usag19.reindex(columns=['Num_Acc','num_veh','sexe','an_nais','place','secu','catu','grav','trajet','locp','actp',
                        'etatp'])


# In[31]:


#Regroupement des classes de grav en 2 classes 0 et 1 

usag15['grav'] = usag15['grav'].replace(1,0)
usag15['grav'] = usag15['grav'].replace(4,0)
usag15['grav'] = usag15['grav'].replace(2,1)
usag15['grav'] = usag15['grav'].replace(3,1)

usag16['grav'] = usag16['grav'].replace(1,0)
usag16['grav'] = usag16['grav'].replace(4,0)
usag16['grav'] = usag16['grav'].replace(2,1)
usag16['grav'] = usag16['grav'].replace(3,1)

usag17['grav'] = usag17['grav'].replace(1,0)
usag17['grav'] = usag17['grav'].replace(4,0)
usag17['grav'] = usag17['grav'].replace(2,1)
usag17['grav'] = usag17['grav'].replace(3,1)

usag18['grav'] = usag18['grav'].replace(1,0)
usag18['grav'] = usag18['grav'].replace(4,0)
usag18['grav'] = usag18['grav'].replace(2,1)
usag18['grav'] = usag18['grav'].replace(3,1)

usag19['grav'] = usag19['grav'].replace(1,0)
usag19['grav'] = usag19['grav'].replace(4,0)
usag19['grav'] = usag19['grav'].replace(2,1)
usag19['grav'] = usag19['grav'].replace(3,1)


#Ajout de l'année
usag15['an']=2015
usag16['an']=2016
usag17['an']=2017
usag18['an']=2018
usag19['an']=2019


# Dans la table usagers, il y a tous les usagers impliqués dans un accident (1 ou plusieurs victimes). 
#A la fusion, je conserverai toutes les lignes par accident, ainsi j'aurai toutes les victimes usagers impliquées.
# In[32]:



# In[33]:


#Concaténation des tables Usag#
usag = pd.concat([usag15,usag16,usag17,usag18,usag19],ignore_index=True)


# In[34]:


usag.head()


# *Modifier les tables usagers : rendre les libellés des modalités plus lisibles*

# In[35]:


#Création variable age
usag['age']=usag['an']-usag['an_nais']
indexrows = usag[usag['age']>2000].index #il y a des valeurs avec an_nais=0, je remplace
usag.loc[indexrows,['age']]=round(usag['age'].mean())


#Création variable classe d'age
usag['classe age']=pd.cut(x=usag['age'], bins=[0,10,20,30,40,50,60,70,80,90,100,110,120],include_lowest=True)



#Recodage variable catu
usag["catu"]=usag["catu"].replace(1,"Conducteur")
usag["catu"]=usag["catu"].replace(2,"Passager")
usag["catu"]=usag["catu"].replace(3,"Piéton")
usag["catu"]=usag["catu"].replace(4,"Piéton en mouvement sur roller/trottinette")

#Recodage variable Sexe
usag["sexe"]=usag["sexe"].replace(1,"Homme")
usag["sexe"]=usag["sexe"].replace(2,"Femme")

#Recodage variable trajet
usag["trajet"]=usag["trajet"].replace(-1,"Non renseigné")
usag["trajet"]=usag["trajet"].replace(0,"Non renseigné")
usag["trajet"]=usag["trajet"].replace(1,"Domicile Travail")
usag["trajet"]=usag["trajet"].replace(2,"Domicile Ecole")
usag["trajet"]=usag["trajet"].replace(3,"Courses Achats")
usag["trajet"]=usag["trajet"].replace(4,"Utilisation professionnelle")
usag["trajet"]=usag["trajet"].replace(5,"Promenade loisirs")
usag["trajet"]=usag["trajet"].replace(9,"Autre")

#Suppression de an_nais (inutile)
usag.drop(columns=['an_nais'],inplace=True)
                                      


# #### _1.3.2 **Table Lieux**_

# In[36]:


lieux15.head()


# In[37]:


lieux16.head()


# In[38]:


lieux17.head()


# In[39]:


lieux18.head()


# In[40]:


lieux19.head()


# *Homogénéiser les tables lieux*  
#  
#  Les tables ne contiennent pas forcément les mêmes colonnes : je choisis de supprimer les variables env1 et vma.  
# 
#  D'autre part, les variables conservées ne sont pas au même format, je les retraite.  

# In[41]:


#Variables env1, vma, voie
lieux15.drop(columns=["env1"],inplace=True)
lieux16.drop(columns=["env1"],inplace=True)
lieux17.drop(columns=["env1"],inplace=True)
lieux18.drop(columns=["env1"],inplace=True)
lieux19.drop(columns=["vma"],inplace=True)


# In[42]:


#Variables pr, pr1
lieux19['pr'] = pd.to_numeric(lieux19['pr'], errors='coerce')
lieux19['pr1'] = pd.to_numeric(lieux19['pr1'], errors='coerce')


# In[43]:


#Concaténation des tables Lieux#
lieux = pd.concat([lieux15,lieux16,lieux17,lieux18,lieux19],ignore_index=True)


# In[44]:


lieux.head()


# In[45]:


#Recodage de variables prof, plan, circ, vosp, catr, surf, infra, situ

lieux["prof"]=lieux["prof"].replace(-1,"Non renseigné") 
lieux["prof"]=lieux["prof"].replace(1,"Plat")
lieux["prof"]=lieux["prof"].replace(2,"Pente")
lieux["prof"]=lieux["prof"].replace(3,"Sommet de côte")
lieux["prof"]=lieux["prof"].replace(4,"Bas de côte")

lieux["plan"]=lieux["plan"].replace(-1,"Non renseigné") 
lieux["plan"]=lieux["plan"].replace(1,"Partie rectiligne")
lieux["plan"]=lieux["plan"].replace(2,"En courbe à gauche")
lieux["plan"]=lieux["plan"].replace(3,"En courbe à droite")
lieux["plan"]=lieux["plan"].replace(4,"En S")


lieux["circ"]=lieux["circ"].replace(-1,"Non renseigné") 
lieux["circ"]=lieux["circ"].replace(1,"A sens unique ")
lieux["circ"]=lieux["circ"].replace(2,"Bidirectionnelle")
lieux["circ"]=lieux["circ"].replace(3,"A chaussées séparées")
lieux["circ"]=lieux["circ"].replace(4,"Avec voies d'affectation variables")

lieux["vosp"]=lieux["vosp"].replace(-1,"Non renseigné") 
lieux["vosp"]=lieux["vosp"].replace(0,"Sans objet")
lieux["vosp"]=lieux["vosp"].replace(1,"Piste cyclable")
lieux["vosp"]=lieux["vosp"].replace(2,"Bande cyclable")
lieux["vosp"]=lieux["vosp"].replace(3,"Voie réservée")

lieux["catr"]=lieux["catr"].replace(-1,"Non renseigné")
lieux["catr"]=lieux["catr"].replace(1,"Autoroute")
lieux["catr"]=lieux["catr"].replace(2,"Route nationale")
lieux["catr"]=lieux["catr"].replace(3,"Route Départementale")
lieux["catr"]=lieux["catr"].replace(4,"Voie communale")
lieux["catr"]=lieux["catr"].replace(5,"Hors réseau public")
lieux["catr"]=lieux["catr"].replace(6,"Parc de stationnement ouvert à la circulation publique")
lieux["catr"]=lieux["catr"].replace(7,"Route de métropole urbaine")
lieux["catr"]=lieux["catr"].replace(9,"Autre")

lieux["surf"]=lieux["surf"].replace(-1,"Non renseigné") 
lieux["surf"]=lieux["surf"].replace(1,"Normale")
lieux["surf"]=lieux["surf"].replace(2,"Mouillée")
lieux["surf"]=lieux["surf"].replace(3,"Flaques")
lieux["surf"]=lieux["surf"].replace(4,"Inondée")
lieux["surf"]=lieux["surf"].replace(5,"Enneigée")
lieux["surf"]=lieux["surf"].replace(6,"Boue")
lieux["surf"]=lieux["surf"].replace(7,"Verglacée")
lieux["surf"]=lieux["surf"].replace(8,"Corps Gras-Huile")
lieux["surf"]=lieux["surf"].replace(9,"Autre")

lieux["infra"]=lieux["infra"].replace(-1,"Non renseigné")
lieux["infra"]=lieux["infra"].replace(0,"Aucun")
lieux["infra"]=lieux["infra"].replace(1,"Souterrain-Tunnel")
lieux["infra"]=lieux["infra"].replace(2,"Pont-Autopont")
lieux["infra"]=lieux["infra"].replace(3,"Bretelle d'échangeur ou de raccordement")
lieux["infra"]=lieux["infra"].replace(4,"Voie ferrée")
lieux["infra"]=lieux["infra"].replace(5,"Carrefour aménagé")
lieux["infra"]=lieux["infra"].replace(6,"Zone piétonne")
lieux["infra"]=lieux["infra"].replace(7,"Zone de péage")
lieux["infra"]=lieux["infra"].replace(8,"Chantier")
lieux["infra"]=lieux["infra"].replace(9,"Autre")

lieux["situ"]=lieux["situ"].replace(-1,"Non renseigné") 
lieux["situ"]=lieux["situ"].replace(0,"Aucun")
lieux["situ"]=lieux["situ"].replace(1,"Sur chaussée")
lieux["situ"]=lieux["situ"].replace(2,"Sur bande d'arrêt d'urgence")
lieux["situ"]=lieux["situ"].replace(3,"Sur accotement")
lieux["situ"]=lieux["situ"].replace(4,"Sur trottoir")
lieux["situ"]=lieux["situ"].replace(5,"Sur piste cyclable")
lieux["situ"]=lieux["situ"].replace(6,"Sur autre voie spéciale")
lieux["situ"]=lieux["situ"].replace(8,"Autre")


# In[46]:


#Réordonner les colonnes#
lieux.reindex(columns = ['Num_Acc', 'catr', 'v1','v2','voie','circ','nbv','vosp','surf','infra','situ','pr',
                           'pr1','prof','plan','lartpc','larrout'])


# #### _1.3.3 **Table Caractéristiques**_

# In[47]:


carac15.head()


# In[48]:


carac16.head()


# In[49]:


carac17.head()


# In[50]:


carac18.head()


# In[51]:


carac19.head()


# *Homogénéiser les tables carac*  
# 
# La variable com n'est pas renseignée correctement dans la table 2018 : je choisis donc de supprimer cette variable dans toutes les tables.  
# Aussi, la variable adr (adresse) donne l'information précise sur le lieu de l'accident mais je ne peux conserver cette variable nominale (type object) qui sera difficilement exploitable : je me contente donc de garder la variable dep pour l'information de l'adresse, d'autant plus qu'elle nous sera précieuse pour la suite de l'étude avec les données insee supplémentaires.   
# Les variables lat et long ne sont pas exprimées dans la même unité de mesure sur toutes les tables : il faut les retraiter.   
# Les variables dep, an, hrmn, atm, col n'ont pas le même format également dans les tables : je les retraite.   
# La variable gps n'existe que dans la table 2018 : je choisis de la supprimer (comme j'ai déjà conservé dep).  

# In[52]:


#Variable com et adr#
carac15.drop(columns=["com","adr"], inplace=True)
carac16.drop(columns=["com","adr"], inplace=True)
carac17.drop(columns=["com","adr"], inplace=True)
carac18.drop(columns=["com","adr"], inplace=True)
carac19.drop(columns=["com","adr"], inplace=True)

#Variable lat et long#
carac15.lat = carac15.lat/100000
carac15.long = carac15.long/100000
carac16.lat = carac16.lat/100000
carac16.long = carac16.long/100000
carac17.lat = carac17.lat/100000
carac17.long = carac17.long/100000
carac18.lat = carac18.lat/100000
carac18.long = carac18.long/100000

#2019
lat_long = ['lat','long']
df19_lat_long = carac19[lat_long]
lat19=list(df19_lat_long.lat)
long19=list(df19_lat_long.long)
lat19_2 = []
long19_2 = []
for i in range(len(df19_lat_long.lat)):
    nb = df19_lat_long.lat[i]
    nbavtvirg = float(nb[0:nb.find(',')])
    nbaprvirg = float(nb[nb.find(',')+1:])/10000000
    if nbavtvirg <0 :
        nbcvti = nbavtvirg-nbaprvirg
    else :
        nbcvti = nbavtvirg+nbaprvirg
    lat19_2.append(nbcvti)
    
for i in range(len(df19_lat_long.long)):
    nb = df19_lat_long.long[i]
    nbavtvirg = float(nb[0:nb.find(',')])
    nbaprvirg = float(nb[nb.find(',')+1:])/10000000
    if nbavtvirg <0 :
        nbcvti = nbavtvirg-nbaprvirg
    else :
        nbcvti = nbavtvirg+nbaprvirg
    long19_2.append(nbcvti)
    
carac19.drop(columns=['lat','long'],inplace=True)
carac19 = carac19.assign(lat=lat19_2,long=long19_2)

#Variable dep#
carac15 = carac15.assign(dep2=carac15.dep%10)
carac15.dep2.astype(float)
carac15 = carac15.assign(dep3=(carac15.dep-carac15.dep2)/10)
carac15[['dep3']] = carac15[['dep3']].astype(int)
carac15.drop(columns=["dep","dep2"], inplace=True)
carac15.rename(columns = {'dep3': 'dep'},inplace=True)
carac15['dep'] = carac15['dep'].astype(str)

carac16 = carac16.assign(dep2=carac16.dep%10)
carac16.dep2.astype(float)
carac16 = carac16.assign(dep3=(carac16.dep-carac16.dep2)/10)
carac16[['dep3']] = carac16[['dep3']].astype(int)
carac16.drop(columns=["dep","dep2"], inplace=True)
carac16.rename(columns = {'dep3': 'dep'},inplace=True)
carac16['dep'] = carac16['dep'].astype(str)

carac17 = carac17.assign(dep2=carac17.dep%10)
carac17.dep2.astype(float)
carac17 = carac17.assign(dep3=(carac17.dep-carac17.dep2)/10)
carac17[['dep3']] = carac17[['dep3']].astype(int)
carac17.drop(columns=["dep","dep2"], inplace=True)
carac17.rename(columns = {'dep3': 'dep'},inplace=True)
carac17['dep'] = carac17['dep'].astype(str)

carac18 = carac18.assign(dep2=carac18.dep%10)
carac18.dep2.astype(float)
carac18 = carac18.assign(dep3=(carac18.dep-carac18.dep2)/10)
carac18[['dep3']] = carac18[['dep3']].astype(int)
carac18.drop(columns=["dep","dep2"], inplace=True)
carac18.rename(columns = {'dep3': 'dep'},inplace=True)
carac18['dep'] = carac18['dep'].astype(str)


#Variable an#
carac15=carac15.assign(annee=carac15.an+2000)
carac15.drop(columns=["an"], inplace=True)
carac15['annee']=carac15['annee'].astype(int)

carac16=carac16.assign(annee=carac16.an+2000)
carac16.drop(columns=["an"], inplace=True)
carac16['annee']=carac16['annee'].astype(int)

carac17=carac17.assign(annee=carac17.an+2000)
carac17.drop(columns=["an"], inplace=True)
carac17['annee']=carac17['annee'].astype(int)

carac18=carac18.assign(annee=carac18.an+2000)
carac18.drop(columns=["an"], inplace=True)
carac18['annee']=carac18['annee'].astype(int)

carac19.rename(columns = {'an': 'annee'},inplace=True)

#Variable hrmn#
carac19[['hrmn']] = carac19[['hrmn']].astype(str)
carac19 = carac19.assign(heure=carac19['hrmn'].str[:2],
                                             minute=carac19['hrmn'].str[3:5])
carac19['heure']=carac19['heure'].astype(int)
carac19['minute']=carac19['minute'].astype(int)



carac15_hrmnss100 = carac15[carac15.hrmn<100]
carac15_hrmnss1000 = carac15[(carac15.hrmn<1000) & (carac15.hrmn>=100)]
carac15_hrmnov1000 = carac15[carac15.hrmn>=1000]

carac15_hrmnss100[['hrmn']] = carac15_hrmnss100[['hrmn']].astype(str)
carac15_hrmnss1000[['hrmn']] = carac15_hrmnss1000[['hrmn']].astype(str)
carac15_hrmnov1000[['hrmn']] = carac15_hrmnov1000[['hrmn']].astype(str)

carac15_hrmnss100 = carac15_hrmnss100.assign(heure='00',
                                             minute=carac15_hrmnss100['hrmn'].str[0:2])

carac15_hrmnss1000 = carac15_hrmnss1000.assign(heure=carac15_hrmnss1000['hrmn'].str[:1],
                                             minute=carac15_hrmnss1000['hrmn'].str[1:3])

carac15_hrmnov1000 =carac15_hrmnov1000.assign(heure=carac15_hrmnov1000['hrmn'].str[:2],
                                             minute=carac15_hrmnov1000['hrmn'].str[2:4])


carac16_hrmnss100 = carac16[carac16.hrmn<100]
carac16_hrmnss1000 = carac16[(carac16.hrmn<1000) & (carac16.hrmn>=100)]
carac16_hrmnov1000 = carac16[carac16.hrmn>=1000]

carac16_hrmnss100[['hrmn']] = carac16_hrmnss100[['hrmn']].astype(str)
carac16_hrmnss1000[['hrmn']] = carac16_hrmnss1000[['hrmn']].astype(str)
carac16_hrmnov1000[['hrmn']] = carac16_hrmnov1000[['hrmn']].astype(str)

carac16_hrmnss100 = carac16_hrmnss100.assign(heure='00',
                                             minute=carac16_hrmnss100['hrmn'].str[0:2])

carac16_hrmnss1000 = carac16_hrmnss1000.assign(heure=carac16_hrmnss1000['hrmn'].str[:1],
                                             minute=carac16_hrmnss1000['hrmn'].str[1:3])

carac16_hrmnov1000 =carac16_hrmnov1000.assign(heure=carac16_hrmnov1000['hrmn'].str[:2],
                                             minute=carac16_hrmnov1000['hrmn'].str[2:4])



carac17_hrmnss100 = carac17[carac17.hrmn<100]
carac17_hrmnss1000 = carac17[(carac17.hrmn<1000) & (carac17.hrmn>=100)]
carac17_hrmnov1000 = carac17[carac17.hrmn>=1000]

carac17_hrmnss100[['hrmn']] = carac17_hrmnss100[['hrmn']].astype(str)
carac17_hrmnss1000[['hrmn']] = carac17_hrmnss1000[['hrmn']].astype(str)
carac17_hrmnov1000[['hrmn']] = carac17_hrmnov1000[['hrmn']].astype(str)

carac17_hrmnss100 = carac17_hrmnss100.assign(heure='00',
                                             minute=carac17_hrmnss100['hrmn'].str[0:2])

carac17_hrmnss1000 = carac17_hrmnss1000.assign(heure=carac17_hrmnss1000['hrmn'].str[:1],
                                             minute=carac17_hrmnss1000['hrmn'].str[1:3])

carac17_hrmnov1000 = carac17_hrmnov1000.assign(heure=carac17_hrmnov1000['hrmn'].str[:2],
                                             minute=carac17_hrmnov1000['hrmn'].str[2:4])


carac18_hrmnss100 = carac18[carac18.hrmn<100]
carac18_hrmnss1000 = carac18[(carac18.hrmn<1000) & (carac18.hrmn>=100)]
carac18_hrmnov1000 = carac18[carac18.hrmn>=1000]

carac18_hrmnss100[['hrmn']] = carac18_hrmnss100[['hrmn']].astype(str)
carac18_hrmnss1000[['hrmn']] = carac18_hrmnss1000[['hrmn']].astype(str)
carac18_hrmnov1000[['hrmn']] = carac18_hrmnov1000[['hrmn']].astype(str)

carac18_hrmnss100 = carac18_hrmnss100.assign(heure='00',
                                             minute=carac18_hrmnss100['hrmn'].str[0:2])

carac18_hrmnss1000 = carac18_hrmnss1000.assign(heure=carac18_hrmnss1000['hrmn'].str[:1],
                                             minute=carac18_hrmnss1000['hrmn'].str[1:3])

carac18_hrmnov1000 = carac18_hrmnov1000.assign(heure=carac18_hrmnov1000['hrmn'].str[:2],
                                             minute=carac18_hrmnov1000['hrmn'].str[2:4])


# Réunion des 3 dataframes

carac15_b = pd.concat([carac15_hrmnss100,carac15_hrmnss1000,carac15_hrmnov1000])
carac16_b = pd.concat([carac16_hrmnss100,carac16_hrmnss1000,carac16_hrmnov1000])
carac17_b = pd.concat([carac17_hrmnss100,carac17_hrmnss1000,carac17_hrmnov1000])
carac18_b = pd.concat([carac18_hrmnss100,carac18_hrmnss1000,carac18_hrmnov1000])
carac15_b.drop(columns=["hrmn"], inplace=True)
carac16_b.drop(columns=["hrmn"], inplace=True)
carac17_b.drop(columns=["hrmn"], inplace=True)
carac18_b.drop(columns=["hrmn"], inplace=True)
carac19.drop(columns=["hrmn"], inplace=True)

carac15_b['heure']=carac15_b['heure'].astype(int)
carac15_b['minute']=carac15_b['minute'].astype(int)
carac16_b['heure']=carac16_b['heure'].astype(int)
carac16_b['minute']=carac16_b['minute'].astype(int)
carac17_b['heure']=carac17_b['heure'].astype(int)
carac17_b['minute']=carac17_b['minute'].astype(int)
carac18_b['heure']=carac18_b['heure'].astype(int)
carac18_b['minute']=carac18_b['minute'].astype(int)


#Variable atm et col#
carac15_b['atm']=carac15_b['atm'].fillna(0).astype(int)
carac15_b['col']=carac15_b['col'].fillna(0).astype(int)

carac16_b['atm']=carac16_b['atm'].fillna(0).astype(int)
carac16_b['col']=carac16_b['col'].fillna(0).astype(int)

carac17_b['atm']=carac17_b['atm'].fillna(0).astype(int)
carac17_b['col']=carac17_b['col'].fillna(0).astype(int)

carac18_b['atm']=carac18_b['atm'].fillna(0).astype(int)
carac18_b['col']=carac18_b['col'].fillna(0).astype(int)


#Variable gps#
carac15_b.drop(columns=['gps'], inplace=True)
carac16_b.drop(columns=['gps'], inplace=True)
carac17_b.drop(columns=['gps'], inplace=True)
carac18_b.drop(columns=['gps'], inplace=True)


# In[53]:


#Réordonner les colonnes#
carac15_b.reindex(columns = ['Num_Acc', 'jour', 'mois','annee','heure','minute','dep','lum','agg','int','atm',
                           'col','lat','long'])


# In[54]:


#Réordonner les colonnes#
carac16_b.reindex(columns = ['Num_Acc', 'jour', 'mois','annee','heure','minute','dep','lum','agg','int','atm',
                           'col','lat','long'])


# In[55]:


#Réordonner les colonnes#
carac17_b.reindex(columns = ['Num_Acc', 'jour', 'mois','annee','heure','minute','dep','lum','agg','int','atm',
                           'col','lat','long'])


# In[56]:


#Réordonner les colonnes#
carac18_b.reindex(columns = ['Num_Acc', 'jour', 'mois','annee','heure','minute','dep','lum','agg','int','atm',
                           'col','lat','long'])


# In[57]:


carac19.reindex(columns = ['Num_Acc', 'jour', 'mois','annee','heure','minute','dep','lum','agg','int','atm',
                           'col','lat','long'])


# In[58]:


#Concaténation des tables Carac#
carac = pd.concat([carac15_b,carac16_b,carac17_b,carac18_b,carac19],ignore_index=True)


# In[59]:


#Recodage des libellés de variables lum, agg, int, atm, col,dep 
carac["lum"]=carac["lum"].replace(1,"Plein jour")
carac["lum"]=carac["lum"].replace(2,"Crépuscule ou aube")
carac["lum"]=carac["lum"].replace(3,"Nuit sans éclairage public")
carac["lum"]=carac["lum"].replace(4,"Nuit avec éclairage public non allumé")
carac["lum"]=carac["lum"].replace(5,"Nuit avec éclairage public allumé")

carac["agg"]=carac["agg"].replace(1,"Hors agglomération")
carac["agg"]=carac["agg"].replace(2,"En agglomération")

carac["int"]=carac["int"].replace(1,"Hors intersection")
carac["int"]=carac["int"].replace(2,"Intersection en X")
carac["int"]=carac["int"].replace(3,"Intersection en T")
carac["int"]=carac["int"].replace(4,"Intersection en Y")
carac["int"]=carac["int"].replace(5,"Intersection à plus de 4 branches")
carac["int"]=carac["int"].replace(6,"Giratoire")
carac["int"]=carac["int"].replace(7,"Place")
carac["int"]=carac["int"].replace(8,"Passage à niveau")
carac["int"]=carac["int"].replace(9,"Autre intersection")
carac["int"]=carac["int"].replace(0,"Non renseigné")

carac["atm"]=carac["atm"].replace(-1,"Non renseigné")
carac["atm"]=carac["atm"].replace(0,"Non renseigné")
carac["atm"]=carac["atm"].replace(1,"Normale")
carac["atm"]=carac["atm"].replace(2,"Pluie légère")
carac["atm"]=carac["atm"].replace(3,"Pluie forte")
carac["atm"]=carac["atm"].replace(4,"Neige-grêle")
carac["atm"]=carac["atm"].replace(5,"Brouillard-fumée")
carac["atm"]=carac["atm"].replace(6,"Vent fort-tempête")
carac["atm"]=carac["atm"].replace(7,"Temps éblouissant")
carac["atm"]=carac["atm"].replace(8,"Temps couvert")
carac["atm"]=carac["atm"].replace(9,"Autre")

carac["col"]=carac["col"].replace(-1,"Non renseigné")
carac["col"]=carac["col"].replace(0,"Non renseigné")
carac["col"]=carac["col"].replace(1,"Deux véhicules - frontale")
carac["col"]=carac["col"].replace(2,"Deux véhicules - par l'arrière")
carac["col"]=carac["col"].replace(3,"Deux véhicules - par le côté")
carac["col"]=carac["col"].replace(4,"Trois véhicules et plus - en chaîne")
carac["col"]=carac["col"].replace(5,"Trois véhicules et plus - collisions multiples")
carac["col"]=carac["col"].replace(6,"Autre collision")
carac["col"]=carac["col"].replace(7,"Sans collision")

carac["dep"]=carac["dep"].replace("1","01")
carac["dep"]=carac["dep"].replace("2","02")
carac["dep"]=carac["dep"].replace("3","03")
carac["dep"]=carac["dep"].replace("4","04")
carac["dep"]=carac["dep"].replace("5","05")
carac["dep"]=carac["dep"].replace("6","06")
carac["dep"]=carac["dep"].replace("7","07")
carac["dep"]=carac["dep"].replace("8","08")
carac["dep"]=carac["dep"].replace("9","09")


# In[60]:


#Création de variables 
date=[]
for i in tqdm(range(0,len(carac))):
    date.append(pd.datetime(carac.iloc[i,11],carac.iloc[i,1],carac.iloc[i,2]))
carac["Date"]=date

week=[]
wday=[]
for i in tqdm(range(0,len(carac))):
    date=carac.iloc[i,14]
    wday.append(int(date.weekday()))

    if int(date.weekday()) in [5,6]:
        week.append("weekend")
    else:
        week.append("hors weekend")
        
carac["Week_end"]=week
carac["weekday"]=wday

carac["weekday"]=carac["weekday"].replace(0,"lundi")
carac["weekday"]=carac["weekday"].replace(1,"mardi")
carac["weekday"]=carac["weekday"].replace(2,"mercredi")
carac["weekday"]=carac["weekday"].replace(3,"jeudi")
carac["weekday"]=carac["weekday"].replace(4,"vendredi")
carac["weekday"]=carac["weekday"].replace(5,"samedi")
carac["weekday"]=carac["weekday"].replace(6,"dimanche")


# In[61]:


carac.head()


# #### _1.3.4 **Table Véhicules**_

# In[62]:


veh15.head()


# In[63]:


veh16.head()


# In[64]:


veh17.head()


# In[65]:


veh18.head()


# A partir de 2019, l'enregistrement des bases inclue id_vehicule et motor. Ces variables seront écartées de l'étude pour faciliter.  

# In[66]:


veh19.head()


# In[67]:


#Variables à convertir#
veh15[['obs','obsm','choc','manv']] = veh15[['obs','obsm','choc','manv']].fillna(-1).astype(int)
veh16[['senc','obs','obsm','choc','manv']] = veh16[['senc','obs','obsm','choc','manv']].fillna(-1).astype(int)
veh17[['senc','obs','obsm','choc','manv']] = veh17[['senc','obs','obsm','choc','manv']].fillna(-1).astype(int)
veh18[['senc','obs','obsm','choc','manv']] = veh18[['senc','obs','obsm','choc','manv']].fillna(-1).astype(int)
veh19[['occutc']] = veh19[['occutc']].fillna(0).astype(int)

veh19.drop(columns=["motor","id_vehicule"], inplace=True)    


# Rappelons nous plus tard que les valeurs manquantes des 5 variables senc, obs, obsm, choc et manv ont été imputées à -1 et celles de occutc à 0.  

# In[68]:


# Concaténation des 5 veh
veh = pd.concat([veh15,veh16,veh17,veh18,veh19],ignore_index=True)


# In[69]:


veh.shape


# Les informations enregistrées dans cette table concernent tous les véhicules impliqués dans chaque accident (1, 2 ou plusieurs véhicules). 
# Des regroupements de modalités doivent être opérés.
# 
# Par ailleurs, je ne conserverai pas les variables senc et num_veh.  

# In[70]:


# Regroupement pour catv

catv_voit = [7,8,9,10,11,12]
catv_pl = [13,14,15]
catv_eng = [16,17,20,21]
catv_tc = [18,19,37,38,39,40]
catv_drou = [1,2,3,4,5,6,30,31,32,33,34,35,36,41,42,43,80]
catv_autr = [99,0]
catv_edp = [50,60]

veh["catv"]=veh["catv"].replace(catv_voit,"CATV_VOIT")
veh["catv"]=veh["catv"].replace(catv_pl,"CATV_PL")
veh["catv"]=veh["catv"].replace(catv_eng,"CATV_ENG")
veh["catv"]=veh["catv"].replace(catv_tc,"CATV_TC")
veh["catv"]=veh["catv"].replace(catv_drou,"CATV_DROU")
veh["catv"]=veh["catv"].replace(catv_autr,"CATV_AUTR")
veh["catv"]=veh["catv"].replace(catv_edp,"CATV_EDP")



# In[71]:


veh.shape


# In[72]:


# Regroupement pour obs

obs_ni = [-1,0]
stat_veh = [1]
arbre = [2]
glissiere = [3,4,5]
bati = [6,9,10]
poteau = [7,8,11]
autre_obst = [12,13,14,15,16,17]

veh["obs"]=veh["obs"].replace(obs_ni,"OBS_NI")
veh["obs"]=veh["obs"].replace(stat_veh,"OBS_VEH_STAT")
veh["obs"]=veh["obs"].replace(arbre,"OBS_ARB")
veh["obs"]=veh["obs"].replace(glissiere,"OBS_GLIS")
veh["obs"]=veh["obs"].replace(bati,"OBS_BAT")
veh["obs"]=veh["obs"].replace(poteau,"OBS_POT")
veh["obs"]=veh["obs"].replace(autre_obst,"OBS_AUTR")



# In[73]:


veh.head()


# In[74]:


# Pas de regroupement pour obsm (on garde les 10 modalités)


veh["obsm"]=veh["obsm"].replace(0,"OBSM_NI")
veh["obsm"]=veh["obsm"].replace(1,"OBSM_AUCUN")
veh["obsm"]=veh["obsm"].replace(2,"OBSM_PIETON")
veh["obsm"]=veh["obsm"].replace(3,"OBSM_VEH")
veh["obsm"]=veh["obsm"].replace(4,"OBSM_VEHR")
veh["obsm"]=veh["obsm"].replace(5,"OBSM_ANIDOM")
veh["obsm"]=veh["obsm"].replace(6,"OBSM_ANISAUV")
veh["obsm"]=veh["obsm"].replace(7,"OBSM_AUTR")



# In[75]:


veh.shape


# In[76]:


# Regroupement pour choc

choc_ni = [-1,0]
choc_avant = [1,2,3]
choc_arriere = [4,5,6]
choc_cote = [7,8]
choc_multi = [9]

veh["choc"]=veh["choc"].replace(choc_ni,"CHOC_NI")
veh["choc"]=veh["choc"].replace(choc_avant,"CHOC_AVANT")
veh["choc"]=veh["choc"].replace(choc_arriere,"CHOC_ARRIERE")
veh["choc"]=veh["choc"].replace(choc_cote,"CHOC_COTE")
veh["choc"]=veh["choc"].replace(choc_multi,"CHOC_MULTI")



# In[77]:


veh.shape


# In[78]:


# Regroupement pour manv

manv_ni = [-1,0]
manv_schg = [1]
manv_msmf = [2]
manv_e2f = [3]
manv_marr = [4]
manv_contr = [5]
manv_tpc = [6]
manv_clrbus = [7,8]
manv_ins = [9]
manv_demit = [10]
manv_chg = [11,12]
manv_depo = [13,14]
manv_tour = [15,16]
manv_depa = [17,18]
manv_div = [19,20,21,22,23,24,25,26]

veh["manv"]=veh["manv"].replace(manv_ni,"MANV_NI")
veh["manv"]=veh["manv"].replace(manv_schg,"MANV_SCHG")
veh["manv"]=veh["manv"].replace(manv_msmf,"MANV_MSMF")
veh["manv"]=veh["manv"].replace(manv_e2f,"MANV_E2F")
veh["manv"]=veh["manv"].replace(manv_marr,"MANV_MARR")
veh["manv"]=veh["manv"].replace(manv_contr,"MANV_CONTR")
veh["manv"]=veh["manv"].replace(manv_tpc,"MANV_TPC")
veh["manv"]=veh["manv"].replace(manv_clrbus,"MANV_CLRBUS")
veh["manv"]=veh["manv"].replace(manv_ins,"MANV_INS")
veh["manv"]=veh["manv"].replace(manv_demit,"MANV_DEMIT")
veh["manv"]=veh["manv"].replace(manv_chg,"MANV_CHG")
veh["manv"]=veh["manv"].replace(manv_depo,"MANV_DEPO")
veh["manv"]=veh["manv"].replace(manv_tour,"MANV_TOUR")
veh["manv"]=veh["manv"].replace(manv_depa,"MANV_DEPA")
veh["manv"]=veh["manv"].replace(manv_div,"MANV_DIV")



# In[79]:


veh.head()


# In[80]:

# Supprimer les colonnes senc
veh.drop(columns=['senc'],inplace=True)


# In[81]:


veh.head()


# ### *1.4 Fusion en une seule base*

# In[82]:


print(usag.shape)
print(carac.shape)
print(lieux.shape)
print(veh.shape)


# In[83]:


#Merge des 3 tables usag, lieux, carac#
df_carac_lie=pd.merge(carac,lieux,on=['Num_Acc'])
df_carac_lie_usag=pd.merge(df_carac_lie,usag,on=['Num_Acc'], how='right')
df_acc=pd.merge(df_carac_lie_usag,veh,on=['Num_Acc','num_veh'],how='left')


# In[84]:


df_acc.drop(columns=['an','num_veh'],inplace=True)
df_acc.drop_duplicates(inplace=True)

# In[85]:


#Merge avec données insee
df_base_finale=pd.merge(df_acc,base_insee,how='left',on=['dep','annee'])


# In[129]:


pd.set_option('display.max_columns',None)
df_base_finale.head()


# *Regroupement par discrétisation par quantile (décile) pour la variable dep (en fonction du taux de fréquence de grave)*

# In[86]:


df_dep=df_base_finale[['dep','grav']]
df_freqgrav = df_dep.groupby(['dep']).sum()
df_freqgrav = df_freqgrav.assign(Nb = pd.DataFrame(df_dep['dep'].value_counts()))
df_freqgrav= df_freqgrav.assign(Freq = df_freqgrav['grav']/df_freqgrav['Nb'])
df_freqgrav.drop(columns=['grav','Nb'],inplace=True)
df_freqgrav.reset_index(drop = True, inplace = True)




# In[87]:


from sklearn.preprocessing import KBinsDiscretizer

bins=10
Discretizor=KBinsDiscretizer(n_bins=bins,encode="ordinal",strategy="quantile")
Discretizor.fit_transform(df_freqgrav)
df_disc_qt=pd.DataFrame(Discretizor.fit_transform(df_freqgrav),columns=df_freqgrav.columns,dtype="int")

df_disc_qt.columns=['Gpe_dep']
df_disc_qt

df_dep_disc = pd.concat([df_dep.groupby(['dep'],as_index=False).sum(),df_disc_qt],axis=1,ignore_index=False)
df_dep_disc.drop(columns=['grav'],inplace=True)
df_base_finale=pd.merge(df_base_finale,df_dep_disc,how='left',on=['dep'])




# In[88]:


df_base_finale.shape
df_base_finale.head(1)

# ## Partie 2 : Analyse exploratoire des données ##

# ### *2.1 Analyse de la forme* ##

# In[131]:


pd.set_option('display.max_columns', None)



# Après les premiers prétraitements, la dataset finale contient 662 036 lignes et 78 colonnes.   

# #### *2.1.1 Description de la typologie des variables*

# Les **variables qualitatives** sont :  
# 
# _**grav**_ (la variable cible recodifiée en 0 ou 1)   
# lum  
# agg   
# int   
# atm   
# col  
# dep   
# week_end (créée)  
# weekday (créée)  
# catr  
# voie      
# v2  
# circ    
# vosp     
# prof  
# plan   
# surf   
# infra   
# situ  
# place   
# catu   
# sexe   
# trajet  
# secu  
# locp   
# actp  
# etatp    
# classe age (créée)  
# catv  
# obs  
# obsm  
# choc  
# manv  
# Gpe_dep (créée)  
# 
# 
# Les **variables quantitatives** sont :  
# 
# Num_Acc  
# mois  
# jour  
# lat  
# long  
# annee 
# heure (créée)   
# minute (créée)   
# v1    
# nbv  
# pr  
# pr1   
# lartpc    
# larrout  
# age (créée)  
# occutc   
# TAUX_P  
# MED_VIE  
# POP  
# POPH  
# POPF  
# POP_FR  
# POP_ETR  
# POP_IMM  
# POP15P_CS1  
# POP15P_CS2  
# POP15P_CS3  
# POP15P_CS4  
# POP15P_CS5  
# POP15P_CS6  
# POP15P_CS7  
# POP15P_CS8  
# ACT_1564  
# CHOM_1564  
# INACT_1564  
# RETR_1564  
# ACTOCC15P_PAS  
# ACTOCC15P_MAR  
# ACTOCC15P_DROU  
# ACTOCC15P_VOIT  
# ACTOCC15P_TCOM  
# SUPERF  
# DENSITE  
# 
# Une **variable au format date** :   
# Date (créée)  

# #### _**2.1.2 Data visualisation**_

# In[234]:


#import matplotlib.pyplot as plt
#sb.set_style('whitegrid')
#import plotly.express as px
#import geopandas as gpd
#import json
#import geojson
#import folium


# In[235]:


# Accidents graves et pas graves
#dep_accident_grav = df_base_finale[df_base_finale['grav']==1].dep.value_counts().to_frame().reset_index()
#dep_accident_grav['indice_de_gravite']=np.log10(dep_accident_grav['dep'])
#print(dep_accident_grav.head())

#dep_accident_pgrav = df_base_finale[df_base_finale['grav']==0].dep.value_counts().to_frame().reset_index()
#dep_accident_pgrav['indice_de_gravite']=np.log10(dep_accident_pgrav['dep'])


# In[236]:


#with open('departements.geojson',encoding='UTF-8') as dep:
#    departement = geojson.load(dep)


# In[237]:


#for feature in departement['features']:
#    feature['id']= feature['properties']['code']


# In[238]:


#fig1 = px.choropleth_mapbox(dep_accident_grav, locations = 'index',
#                            geojson= departement,
#                            color='indice_de_gravite',
#                            color_continuous_scale=["green","orange","red"],
#                            range_color=[2,3.5],
#                            hover_name='index',
#                            hover_data=['dep'],
#                            title="Répartition des accidents graves en France",
#                            mapbox_style="open-street-map",
#                            center= {'lat':46, 'lon':2},
#                            zoom =4, 
#                            opacity= 0.8)



#fig2 = px.choropleth_mapbox(dep_accident_pgrav, locations = 'index',
#                            geojson= departement,
#                            color='indice_de_gravite',
#                            color_continuous_scale=["green","orange","red"],
#                            range_color=[2,3.5],
#                            hover_name='index',
#                            hover_data=['dep'],
#                            title="Répartition des accidents pas graves en France",
#                            mapbox_style="open-street-map",
#                            center= {'lat':46, 'lon':2},
#                            zoom =4, 
#                            opacity= 0.8)

#fig1.show()
#fig2.show()


# In[239]:


#df_sample= df_base_finale.sample(int(len(df_base_finale)*0.01))
#df_sample['nombre']= np.random.randint(1, 6, df_sample.shape[0])


# In[240]:


#fig3 = px.density_mapbox(df_sample, lat='lat',lon='long', z='nombre', radius=10,
 #                       center=dict(lat=46, lon=2), zoom=4,
#                        mapbox_style="stamen-watercolor")
#fig3.show()


# #### *2.1.3 Missing data visualisation*

# Sur le graphe ci-dessous, les valeurs manquantes représentées sont celles indiquant les NaN. Or, il y a parmi les données manquantes celles qui sont également codifiées en "0", "-1", "Non renseigné" et celles réaffectées dans les variables OBS_NI, OBSM_NI, CHOC_NI, MANV_NI (lors de l'étape de formatage plus haut).  
# Ici, le heatmap représenté ne prend en compte que les valeurs NaN restées telles quelles dans notre dataset. Plus loin, j'analyserai les valeurs manquantes et toutes les autres recodifiées dans sa globalité.    

# In[798]:


plt.figure(figsize=(30,40))
sb.heatmap(df_base_finale.isna(),cbar=False)


# Le graphe indique la présence de valeurs manquantes (les tracés en blanc) pour plusieurs variables : lat, long, voie, v1, v2, pr, pr1, prof, plan, lartpc, larrout, circ, nbv, vosp, surf, infra, situ, classe age et de TAUX_P jusqu'à DENSITE. 
# Ce qui est remarquable c'est qu'il apparaît souvent comme des lignes au niveau des valeurs manquantes : ce sont en fait les mêmes observations qui sont concernées par plusieurs variables, prouvant potentiellement un lien entre ces variables.  
# Observons le pourcentage de valeurs manquantes marquées en NaN.   

# In[799]:


pd.set_option('display.max_rows', None)
print(df_base_finale.isna().sum()/df_base_finale.shape[0])


# Certaines variables ont manifestement beaucoup de valeurs NaN : v1(83%), v2(95%), pr(43%), pr1(43%), lartpc(37%), larrout(36%) avec au moins 36% de valeurs manquantes. Je ne conserverai pas ces colonnes. Je supprimerai également voie qui contient une information texte difficile à exploiter.     
# 
# Sur les variables quantitatives TAUX_P à DENSITE, il reste également environ 3,7% de données manquantes, comme vu sur le heatmap, qui sont potentiellement liées entre elles.  
# 
# Aussi, comme évoqué dans le descriptif, il est important de noter que la plupart des variables peuvent contenir des cellules vides, des 0, des points, et aussi des -1. Ces modalités sont au même titre que les valeurs manquantes marquées en NaN. Je les analyserai plus loin.      
#       

# ### *2.2 Analyse du fond* ##

# #### *2.2.1 Visualisation initiale : élimination des colonnes avec trop de NaN*

# Comme vu au paragraphe précédent, les variables présentant beaucoup de données manquantes seront supprimées (voie, v1, v2, pr, pr1, lartpc, larrout).   

# In[89]:


df_base_finale.drop(columns=["voie","v1","v2","pr","pr1","lartpc","larrout"],inplace=True)


# In[88]:


pd.set_option('display.max_rows', None)
df_base_finale.isna().sum()


# Examinons les autres groupes de variables avec modalités à -1 "Non renseigné", ou vides, ou "." ou 0 "sans objet" dont certaines ont été reformatées en remplissant aussi les NaN en -1 ou 0.  

# In[550]:


print(df_base_finale.lat.unique()) # les 0  => manquantes
print(df_base_finale.long.unique()) # les 0  => manquantes

print(df_base_finale.locp.unique()) # les 0 et -1 => manquantes
print(df_base_finale.actp.unique()) # les 0, 0.0, -1 => manquantes
print(df_base_finale.etatp.unique()) # les 0, -1 => manquantes

print(df_base_finale.secu.unique()) # les -1 => manquantes

print(df_base_finale.infra.unique()) # les Non renseigné, Aucun, NaN => manquantes
print(df_base_finale.situ.unique()) # les NaN, Aucun, Non renseigné => manquantes

print(df_base_finale.circ.unique()) # les NaN, 0.0, Non renseigné => manquantes
print(df_base_finale.nbv.unique())  # les NaN, 0, -1 => manquantes
print(df_base_finale.vosp.unique()) # les Sans objet, NaN, Non renseigné => manquantes
print(df_base_finale.surf.unique()) # les NaN, 0.0, Non renseigné => manquantes

print(df_base_finale.prof.unique()) # les NaN, 0.0, Non renseigné => manquantes
print(df_base_finale.plan.unique()) # les 0.0, NaN, Non renseigné => manquantes

print(df_base_finale.atm.unique())  # les Non renseigné => manquantes
print(df_base_finale.col.unique())  # les Non renseigné => manquantes
print(df_base_finale.int.unique())  # les Non renseigné => manquantes

print(df_base_finale.trajet.unique()) # les Non renseigné => manquantes

print(df_base_finale.place.unique()) # les 0 => manquantes

print(df_base_finale['classe age'].unique()) # les NaN => manquantes


# Comptabilisons à présent les données manquantes avec toutes les données non renseignées (en pourcentage). 

# In[ ]:


#Variables lat, long 
print((df_base_finale["lat"].isin([0]).sum()+df_base_finale["lat"].isna().sum())/df_base_finale.shape[0])
print((df_base_finale["long"].isin([0]).sum()+df_base_finale["long"].isna().sum())/df_base_finale.shape[0])

#Variables locp, actp, etatp 
print((df_base_finale["locp"].isin ([-1,0]).sum()+df_base_finale["locp"].isna().sum())/df_base_finale.shape[0]) 
print((df_base_finale["actp"].isin (['0',-1,0.0]).sum()+df_base_finale["actp"].isna().sum())/df_base_finale.shape[0]) 
print((df_base_finale["etatp"].isin ([-1,0]).sum()+df_base_finale["etatp"].isna().sum())/df_base_finale.shape[0])

#Variables secu
print((df_base_finale["secu"].isin ([-1]).sum()+df_base_finale["secu"].isna().sum())/df_base_finale.shape[0]) 

#Variables infra, situ 
print((df_base_finale["infra"].isin(['Aucun','Non renseigné']).sum()+df_base_finale["infra"].isna().sum())/df_base_finale.shape[0])
print((df_base_finale["situ"].isin(['Aucun','Non renseigné']).sum()+df_base_finale["situ"].isna().sum())/df_base_finale.shape[0])

#Variables circ, nbv, vosp, surf 
print(((df_base_finale["circ"].isin([0,'Non renseigné'])).sum()+df_base_finale["circ"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["nbv"].isin([-1,0])).sum()+df_base_finale["nbv"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["vosp"].isin(['Sans objet','Non renseigné'])).sum()+df_base_finale["vosp"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["surf"].isin([0,'Non renseigné'])).sum()+df_base_finale["surf"].isna().sum())/df_base_finale.shape[0])

#Variables prof, plan
print(((df_base_finale["prof"].isin([0,'Non renseigné'])).sum()+df_base_finale["prof"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["plan"].isin([0,'Non renseigné'])).sum()+df_base_finale["plan"].isna().sum())/df_base_finale.shape[0])

#Variables atm, col, int 
print(((df_base_finale["atm"].isin(['Non renseigné'])).sum()+df_base_finale["atm"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["col"].isin(['Non renseigné'])).sum()+df_base_finale["col"].isna().sum())/df_base_finale.shape[0])
print(((df_base_finale["int"].isin(['Non renseigné'])).sum()+df_base_finale["int"].isna().sum())/df_base_finale.shape[0])

#Variable trajet 
print(((df_base_finale["trajet"].isin(['Non renseigné'])).sum()+df_base_finale["trajet"].isna().sum())/df_base_finale.shape[0])

#Variable place (que les Null)
print(((df_base_finale["place"].isin([0])).sum())/df_base_finale.shape[0])


# Après avoir examiné ce groupe de variables, et avoir rassemblé les modalités qui correspondent à aucune information, j'ai finalement beaucoup de données manquantes encore, dont le pourcentage est très important (>80%). C'est notamment le cas pour : locp, etatp, actp, infra, vosp. C'est pourquoi, je supprimerai ces variables.  

# In[90]:


#Variables vosp, infra, locp, etatp, actp
df_base_finale.drop(columns=['vosp','infra','locp','etatp','actp'],inplace=True)


# In[91]:


df_base_finale.shape


# La dataset finale après suppression de ces variables comporte alors 66 colonnes.   

# #### *2.2.2 Visualisation de la variable grav*

# In[135]:


df_base_finale['grav'].value_counts(normalize=True)


# Nos classes sont assez déséquilibrées.  Pour rappel, la variable initiale était décrite comme suit : 
# 
#     1 -> indemne 
#     2 -> tué 
#     3 -> blessé hospitalisé  
#     4 -> blessé léger  
# 
# Cette variable grav a été transformée en ayant regroupé 1 "Indemne" et 4 "blessé léger" en 0 et 2 "tué" et 3 "blessé hospitalisé" en 1.   
# Ainsi faite, notre dataset finale comporte : 22% de classe 1 positive et 78% de classe 0 négative.

# #### *2.2.3 Relations univariées*

# In[92]:


print(df_base_finale.shape[1])


# In[246]:
df_base_finale.columns

df_base_finale.head(1)


# In[555]:


print(df_base_finale.select_dtypes(include='float').columns) 
print(df_base_finale.select_dtypes(include='int').columns)
print(df_base_finale.select_dtypes(include='object').columns)
print(df_base_finale.select_dtypes(include='category').columns)


# In[247]:


#Split des variables quanti-quali
var_quant = df_base_finale[['Num_Acc','mois','jour','annee','heure','minute','nbv','age','lat','long',
                            'occutc','TAUX_P', 'MED_VIE', 'POP', 'POPH', 'POPF',
       'POP_FR', 'POP_ETR', 'POP_IMM', 'POP15P_CS1', 'POP15P_CS2',
       'POP15P_CS3', 'POP15P_CS4', 'POP15P_CS5', 'POP15P_CS6', 'POP15P_CS7',
       'POP15P_CS8', 'ACT_1564', 'CHOM_1564', 'INACT_1564', 'RETR_1564',
       'ACTOCC15P_PAS', 'ACTOCC15P_MAR', 'ACTOCC15P_DROU', 'ACTOCC15P_VOIT',
       'ACTOCC15P_TCOM', 'SUPERF', 'DENSITE']]
var_quali = df_base_finale[['lum','agg','atm','int','col','Week_end','weekday','catr','circ','prof','plan',
                            'surf','situ','secu','place','catu','sexe','trajet','catv', 'obs', 'obsm', 'choc',
                            'manv','classe age','Gpe_dep']]
print(var_quant.shape[1])
print(var_quali.shape[1])


# Il y a en tout : 38 variables quantitatives, 25 qualitatives, la variable cible qualitative (grav), la variable dep et une variable Date.  

# _**Variables quantitatives**_

# In[806]:


var_quant.describe()


# In[558]:


i=0
for col in var_quant:
    plt.figure()
    sb.displot(var_quant[col],kde=True)
    i=i+1


# In[559]:


# Distribution de age
col_palette = ["#67E568","#257F27","#08420D","#FFF000","#FFB62B","#E56124","#E53E30","#7F2353","#F911FF","#9F8CA6","#257F27"]

plt.figure()
sb.displot(df_base_finale['age'],color=col_palette[7],kde=True)


# D'après l'illustration, la distribution est asymétrique et semble présenter des valeurs aberrantes (outliers). Nous le confirmons ci-dessous avec la boite à moustaches.   

# In[373]:


plt.figure()
sb.boxplot(data=df_base_finale['age'],color=col_palette[9])


# Les victimes âgées d'environ 120 ans, sont-elles des informations exactes ou erronnées ? 

# In[568]:


index_120a = df_base_finale[df_base_finale['age'].isin([118,119,120])]

index_120a['sexe'].value_counts()


# Il y a 186 personnes victimes d'accidents enregistrées dans les bases, qui sont âgées de plus de 118 ans et sont pour la plupart des conducteurs masculins. Ne sachant pas plus sur ces observations, je ne vais pas les supprimer.   

# *Observations des variables quantitatives*
# 
#     - Caractéristiques temporelles :  
#     
#  Sur les mois de l'année, il apparaît que ce sont les mois de juin, juillet, septembre et octobre qui enregistrent les pics d'accidents. A contrario, les mois de février et avril sont les mois avec le moins d'accidents en France sur ces années.    
#  
#  Concernant les jours, ce sont généralement les 6, 7, 12, 13 et 16 du mois où les accidents ont lieu.
#  
#  Sur les 4 années observées, l'année 2020 enregistre le moins d'accidents (effet covid).
#  
#  Quant au moment de la journée, le créneau horaire enregistrant le plus d'accidents est 17h-18h, c-à-d aux heures de pointe des usagers de la route (à la fin de la journée).  La distribution est asymétrique.  
#  
#  Le graphe illustrant la distribution des minutes nous indique que les accidents ont souvent lieu en heure pleine (00 min) ou à la demi-heure (30 minutes). Sur cet histogramme, on voit se dessiner une courbe avec une sorte de saisonnalité. A 0 ou 30 minutes nous avons des pics, puis une forte baisse, puis une hausse entre 10 et 15 minutes, et à nouveau une baisse aux 20-25 minutes de chaque heure.   
#  
#     - Caractéristiques environnementales : 
#     
#  A noter également que les routes ayant 2 voies semblent être les plus accidentogènes. La distribution est asymétrique.  
#  
#      - Caractéristiques des individus :
#      
#  Après avoir remplacé les 3 lignes ayant pour année de naissance 0 (probablement des erreurs de saisies), le graphe illustrant l'âge présente une asymétrie à droite. Les individus impliqués dans le plus d'accidents sont les jeunes âgés de 20-30 ans environ. Ceux ayant le moins d'accident sont plus âgés (>60 ans).  
#  Aussi, on peut noter des valeurs aberrantes (sur le boxplot des âges) avec des individus ayant aux alentours de 120 ans. Est-ce-que ce sont des observations erronnées ? Nous n'en savons pas plus.  
#  
#     - Caractéristiques autour des véhicules :  
#     
# La plupart des variables de comptage sur les véhicules, les obstacles, les chocs et les manoeuvres présente une densité avec une forte concentration sur quelques valeurs (faibles) seulement : ceci signifie qu'il y a le plus souvent peu de véhicules impliqués lors d'un accident, mettant en cause peu d'obstacles fixes ou mobiles, en faisant peu de manoeuvres et causant peu de chocs. C'est ce contexte d'accidentologie qui est le plus reproduit. 
#     
#     
#     
#     - Caractéristiques démographiques et sociales :
# L'immense majorité des accidents ont lieu dans des départements ayant un taux de pauvreté dans la fourchette [10%;20%]. Quant au niveau de vie, ce sont les départements de revenus médians [20 000;23 000] qui sont les plus accidentogènes.  
# Il est à noter que là où l'on enregistre le plus d'accidents (pic) nous nous trouvons en lieu où la population avoisine les 2,2M, une population à faible représentativité en terme d'étrangers et d'immigrés.  
# Les graphes sur la catégorie socio-professionnelle apporte également beaucoup d'informations : les endroits peu peuplés en CS1 (agriculteurs) et CS3 (cadres) sont très accidentogènes. Aussi, les courbes de représentations pour les CS4, CS5 et CS8 connaissent à peu près une tendance similaire avec un pic aux points aux environs de 200 000-300 000.  
#     
# Rappelons les CSP ici :   
#     - CS1 : Agriculteurs exploitants  
#     - CS2 : Artisans, Commerçants, Chefs d'entreprise  
#     - CS3 : Cadres et Professions intellectuelles supérieures  
#     - CS4 : Professions intermédiaires  
#     - CS5 : Employés  
#     - CS6 : Ouvriers  
#     - CS7 : Retraités  
#     - CS8 : Autres sans activité professionnelle  
#     
# Les 3 prochaines courbes sur les actifs, chomêurs et inactifs sont assez identiques en notant cette particularité : il semble se dessiner un creux à chaque courbe montrant qu'il y a un intervalle où il n'y a quasiment aucun accident. Celle des retraités ne montre pas la même tendance, mais présente toujours ce creux. 
# Nous pouvons soulever exactement la même remarque sur les courbes concernant les habitudes de déplacement des actifs occupés.  
# 
# Parmi ceux qui ont l'habitude de prendre un 2 roues motorisé pour aller se rendre au travail, ce sont dans les départements où il y en a le moins que les accidents arrivent le plus.   
# Enfin, parmi les actifs qui ont l'habitude de prendre les transports en commun pour aller travailler, ce sont aux endroits les moins peuplés et les plus peuplés (aux extrêmes) qu'on enregistre le plus d'accidents, assez surprenant.  
# Pour terminer, quant à la densité, l'accident est plus fréquent là où celle-ci est plus faible. 
# 

# _**Variables qualitatives**_

# In[95]:


var_quali.head()


# In[248]:


col_palette = sb.color_palette("dark", 20)

i=0
for col in var_quali:
    plt.figure()
    var_quali[col].value_counts().plot.bar()
    i+=1


# *Observations des variables qualitatives*  
# 
#     - Les caractéristiques environnementales, météorologiques, temporelles :
#  Les accidents ont lieu plus fréquemment en plein jour, en conditions météorologiques normales, hors week-end et plus précisément les journées du vendredi. 
#  
#     - Les caractéristiques routières : 
# La plupart des accidents répertoriés ont eu lieu dans ces circonstances : en agglomération, sur des voies communales ou routes départementales, sur un régime de circulation bidirectionnelle, en profil plat, sur une partie rectiligne, en situation de surface de route normale (sans pluie), sur une chaussée, sur des trajets promenade-loisirs, et essentiellement dans le département 75 (paris).  
# 
#     - Les caractéristiques des victimes impliquées :
# Les personnes impliquées dans les accidents les plus fréquents sont ceux situés sur la place 1, dans la grande majorité donc les conducteurs, de sexe féminin (il est important de rappeler ici que nous avons conservé les victimes dont l'état est le plus grave, il y a donc beaucoup plus de femmes), avec l'équipement de sécurité utilisé, dans la tranche d'âge 20-30 ans. 
#     


# 
# #### *2.2.4 Relations bivariées et tests statistiques (target-features, features-features)*
# 

# In[570]:


positive_df_base_finale = df_base_finale[df_base_finale['grav']==1]
negative_df_base_finale = df_base_finale[df_base_finale['grav']==0]


# _**Relations variables quantitatives-target**_

# In[571]:


i=0
for col in var_quant:
    plt.figure()
    sb.distplot(positive_df_base_finale[col], label='positive',kde=True)
    sb.distplot(negative_df_base_finale[col],label='negative',kde=True)
    plt.legend()
    i+=1


# *Observations des croisements target-variables quantitatives*  
# 
#     - Caractéristiques temporelles :
# Les accidents graves ont lieu souvent aux mois de juin-juillet-août (période estivale).  
# Quant aux moins graves, ils sont plus souvent associés  aux mois de septembre-octobre-novembre.  
# 
# 2015, 2016 et 2017 sont des années marquées par plus d'accidents graves que les autres années où la tendance est inversée. 
# 
# Dans les heures creuses de la journée (9h-14h) surviennent les accidents moins graves puis c'est en fin de journée (16h-18h) que les plus graves arrivent le plus et bien entendu aux heures nocturnes(23h-6h) où la visibilité est moins bonne et/ou les conditions physiques du conducteur ne sont pas optimales.  
#     
#     - Caractéristiques routières :
# Les routes à 2 voies sont les endroits où on enregistre le plus d'accidents graves et moins graves. Les zones aux latitudes (à peu près 0) connaissent des accidents plus graves.  
# 
#     - Caractéristiques des usagers victimes :
# Concernant l'âge, la classe la plus touchée est toujours celle des 20-30 ans dans tous les cas. Cependant, on peut remarquer que dès 45-50 ans, les 2 courbes s'inversent et fait apparaître plus d'accidents graves que de moins graves. A partir de cette tranche d'âge, il y a donc un gros facteur de risque sur l'apparition d'accidents corporels graves.  
# 
#      - Caractéristiques démographiques et sociales :
#      
# La gravité de l'accident augmente avec le nombre d'occupants dans le transport en commun impliqué. De façon identique, si plusieurs véhicules et/ou piétons sont impliqués, la gravité s'accroît si le nombre de véhicules et/ou piétons augmente (2 roues, voitures,...).
# Chez les revenus médians faibles à modérés, la gravité des accidents s'avère plus importante.
# De plus, nous constatons que là où la population est le plus faible, il y a plus d'accidents graves, c'est notamment les endroits à faible population étrangère et immigrée.  
# Plus tôt, nous avons vu que les départements touchés par beaucoup d'accidents sont ceux où il y a peu de populations de CS1 ; mais il s'agissait de tous les accidents confondus. Nuançons ce point car, ici, force est de constater que ce sont plus souvent des accidents moins graves qui arrivent en ces lieux précités, et les accidents plus graves se produisent aux endroits à plus forte concentration en population de CS1. Sur les autres courbes, nous remarquons que les accidents les plus graves ont lieu dans les zones où la population CS2, CS3, CS4, CS5, CS6, CS7, CS8 sont faibles. 
# Aussi, sur les courbes des actifs, chomeurs, inactifs et retraités, cette tendance se dessine également : les accidents les plus graves arrivent aux endroits où il y a le moins de population.  
# Sur les habitudes de déplacement, ce sont aussi aux endroits les moins fréquentés par les usagers qu'arrivent les accidents les plus graves (excepté pour les déplacements en voiture). 
# 
# Nous allons vérifier plus loin les corrélations avec le V de Cramer.   
# 
# 


# In[99]:


i=0
for col in var_quant:
    plt.subplots(figsize=(8,5))
    sb.boxplot(x="grav", y=df_base_finale[col], data=df_base_finale)
    i+=1


# *Autres observations sur les boxplot :* 
# 
# Comme constaté plus haut, sur les premières variables, il est difficile d'établir de lien avec la variable grav car les boxplot sont presque tous identiques dans le groupe grav=0 et grav=1 pour chaque variable. Il est assez difficile de dégager des particularités.  
# 
# Pour les autres variables quantitatives (obtenues sur les bases de l'INSEE), nous avons en revanche plus d'informations intéressantes. Par exemple sur les revenus médians (MED_VIE), les 2 boxplot montrent des différences : la moyenne est sensiblement plus faible dans le groupe grav=1 que dans l'autre ce qui signifie que les accidents ont une gravité plus importante chez les populations dont la moyenne des revenus médians est située en dessous de 21000 alors que celle chez les populations ayant un accident moins grave se situe aux environs de 22000. De plus, pour les accidents graves, il y a des valeurs aberrantes au delà des moustaches.
# 
# Les boxplot suivants sont encore plus révélateurs d'informations : les accidents graves se retrouvent dans les endroits où la population moyenne est en-dessous de 1 million, alors que les accidents moins graves arrivent généralement aux endroits plus peuplés en moyenne (1,3millions) ; de même pour les boxplots concernant la population. 
# 
# Comme vu déjà plus haut, la population moyenne en CS1 est plus élevée pour les accidents graves que les moins graves. Il semble y avoir un lien entre la gravité de l'accident et la population de cette catégorie. Dans les autres CSP, la tendance est inversée : c'est la population moyenne chez les accidents moins graves qui est plus élevée. Cette tendance est également observée dans les variables suivantes. 
# En ce qui concerne la densité de popuplation, les boites sont très différentes avec des points aberrants sur le groupe grav=1, des moyennes à peu près au même niveau.  
# 
# 
# Pour conforter tous nos constats, nous allons plus loin réaliser des tests d'hypothèses afin de vérifier les relations évoquées.  
# 
# 

# _**Relations variables qualitatives-target**_

# In[807]:

i=0
list_quali = var_quali.columns.tolist()
for col in var_quali:
    
    plt.figure()
    g = sb.catplot(x = list_quali[i], hue = "grav", data = df_base_finale, kind = "count")
    g.set_xticklabels(rotation=90)
    plt.show()
    i+=1


# In[573]:


#Variable dep et target
dep_grav1=df_base_finale[df_base_finale['grav']==1]
dep_grav0=df_base_finale[df_base_finale['grav']==0]

fig = plt.figure(figsize=(10,15))

ax1 = fig.add_subplot(121)
dep1 = dep_grav1.dep.value_counts().to_frame()
ax1.pie(dep1.dep,autopct='%1.1f%%',labels=dep1.index)
ax1.set_title('Accidents graves')
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
figu = plt.gcf()
figu.gca().add_artist(centre_circle)


ax2 = fig.add_subplot(122)
dep2 = dep_grav0.dep.value_counts().to_frame()
ax2.pie(dep2.dep,autopct='%1.1f%%',labels=dep2.index)
ax2.set_title('Accidents pas graves')
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
figu = plt.gcf()
figu.gca().add_artist(centre_circle)

plt.show()


# *Observations des relations target-variables qualitatives*  
# 
#     - Les caractéristiques environnementales, météorologiques, temporelles :
#  Les conditions les plus propices à l'apparition d'accidents plus graves que moins graves sont : 
#  la nuit sans éclairage public, hors agglomération, avec une légère accentuation par temps éblouissant et/ou en temps de brouillard/fumée. En complément, il apparaît que le top 5 des départements les plus gravement accidentogènes est : 13, 69, 93, 97, 59 tandis que le top 5 des départements où ont lieu les accidents les moins graves est : 75, 13, 93, 94, 92 (forte représentation des départements d'ile-de-france).  
#  
#  
#     - Les caractéristiques routières : 
# Les routes départementales sont les endroits les plus sujets aux accidents graves que moins graves. Aussi, les chemins en courbe sont les plus favorables aux accidents graves.  
# 
# 
#     - Les caractéristiques des victimes et véhicules impliqués :
#  Les hommes sont les victimes les plus représentées des accidents graves.  
# 
# 


# _**Relation target-date**_ 

# In[575]:


import calplot

df_base_finale['Date'] = pd.to_datetime(df_base_finale['Date'])
df_base_finale.set_index('Date', inplace = True)


# In[576]:


calplot.calplot(data = df_base_finale['grav'],how = 'sum', cmap = ('Reds'), colorbar=True,figsize = (16, 8),
                suptitle = "Gravité des accidents")


# *Observations sur le calendrier des jours d'accidents graves*
# 
# De manière globale, chaque année est caractérisée par ce que nous avons déjà constaté plus haut : les jours les plus à risques sont les fins de semaine (vendredis, samedis et dimanches), les périodes de grandes vacances scolaires ((juin-juillet) ou de veille de vacances, et parfois de lendemain de jours fériés ou jour de fête.  
# Plus précisément, en comparant les années les unes par rapport aux autres, l'année 2017 est quand même nettement plus marquée par ces accidents graves, surtout les mois de juin, juillet. Les jours particulièrement dangereux sont les fins de semaine : vendredi à dimanche.
# 
# En 2016, les jours avec beaucoup d'accidents graves sont :  
#     - vendredis 9 et 30 septembre  
#     - week-ends au mois de juin-juillet
# 
# En 2017, les jours où se sont produits le plus d'accidents graves sont :  
#     - dimanche 9 avril, week-end des vacances scolaires (>130 accidents)  
#     - vendredi 13 octobre (>120 accidents)  
#     - les week-ends au mois de juin-juillet  
#     - le lendemain de l'ascension le 26 mai  
# 
# En 2018, les jours où se sont produits beaucoup d'accidents graves sont :   
#     - vendredi 6 avril, début de vacances scolaires  
#     - vendredi 5 octobre  
#     - week-ends de mai, juin, juillet    
#     
# En 2019, ce sont les mêmes tendances.
#    

#  _**Mesures de liaisons entre les variables et la target (tests statistiques)**_

# * V de Cramer pour les variables qualitatives discrètes et grav  
# 
# On peut appliquer les test de chi-2 sous réserve de respecter les conditions de son application. En effet, pour pouvoir être valide, ce test requiert que chaque cellule des tableaux croisés constitués des modalités des variables testées contienne au minimum un effectif de 5. Cela signifie que pour les mesures entre nos variables et grav, il faut qu'il y ait au minimum 5 accidents pour chaque modalité de la variable testée.  
# Vérifions les tableaux croisés pour chaque variable.        
# 
#     
#     
#     
# 
# * Test de Student pour les variables quantitatives avec grav  
# 
# Dans ce test statistique, nous allons tester l'hypothèse nulle à savoir que les moyennes sur chaque groupe (grav=0 et grav=1) sont égales. 

# In[249]:


for col in var_quali:
    table = pd.crosstab(var_quali[col],df_base_finale['grav'])
    print(pd.DataFrame(table))


# Pour quelques variables nous avons peu d'effectifs (<5), il sera difficile d'appliquer le V de Cramer dans ces cas (dep,int).  

# In[251]:


var_qual = var_quali.drop(columns=[#'dep',
                                   'int'])


# In[254]:


#Dépendance entre grav et variables qualitatives (V de Cramer)
from scipy.stats import chi2_contingency

def cramers_V(var1,var2) :
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) 
    stat = chi2_contingency(crosstab)[0] 
    obs = np.sum(crosstab) 
    mini = min(crosstab.shape)-1 

col = []
grav = df_base_finale['grav']
for var1 in var_quali:    
    cramers =cramers_V(var_quali[var1], grav)
    col.append(round(cramers,2)) 
#    col.append(cramers)  
df_V_Cramer = pd.DataFrame(col)
df_V_Cramer.index=var_quali.columns
df_V_Cramer.sort_values(by = 0,ascending=False)


# Pour toutes les variables, les valeurs sont plus proches de 0 que de 1, il y a donc peu de liaisons avec la variable grav. Celles qui ont le coefficient le plus proche de 1 sont : secu, Gpe_dep, col, obs et catr.  

# In[810]:


#Dépendance entre les variables quantitatives et grav
from scipy import stats
for col in var_quant.columns:
    pos = df_base_finale[df_base_finale['grav']==1][col].dropna()
    neg = df_base_finale[df_base_finale['grav']==0][col].dropna()
    neg2=neg.sample(pos.shape[0])
    print("Pour",col, "le résultat est : ",stats.ttest_ind(pos,neg2))


# Pour les variables mois et jour le test aboutit à un non-rejet de l'hypothèse nulle car la p-value >5%,donc les moyennes sont significativement égales, et on ne peut pas considérer qu'il y ait un lien entre la variable grave et celles-ci.
# 
# En revanche, pour toutes les autres variables, les p-value <5% signifient que l'hypothèse nulle est rejetée et qu'il y a donc bien un lien entre ces variables et grav. Ces variables là seront intéressantes pour la suite.  
# 


# #### *2.2.5 Analyse statistique multivariée*

# *Relation variables quantitatives/variables quantitatives*

# In[580]:


sb.pairplot(df_acc)
plt.show()
fig0 = plt.figure(figsize=(40,30))
ax1=plt.subplot()
sb.heatmap(df_acc.corr('pearson'), annot=True)


# In[129]:


sb.pairplot(base_insee)


# In[581]:


fig4 = plt.figure(figsize=(40,30))
ax1=plt.subplot()
sb.heatmap(base_insee.corr('pearson'), annot=True)


# In[218]:


var_select=(base_insee.corr('pearson')>= -0.5) & (base_insee.corr('pearson')<= 0.5)
var_select


# In[811]:


var_select=base_insee[['annee','TAUX_P','MED_VIE','POP','POP15P_CS1','SUPERF','DENSITE']]
fig6= plt.figure(figsize=(7,4))
ax1=plt.subplot()
sb.heatmap(var_select.corr('pearson'),annot=True)
           


# In[582]:


fig5 = plt.figure(figsize=(40,30))
ax1=plt.subplot()
sb.heatmap(var_quant.corr('spearman'), annot=True)


# *Relation variables qualitatives/variables qualitatives*

# In[255]:


DataMatrix = pd.get_dummies(var_quali)

plt.figure(figsize=(25,20)) 
sb.heatmap(DataMatrix.corr('pearson'), cmap='coolwarm', center=0)


# In[256]:


rows= []
for var1 in var_quali:
  col = []
  for var2 in var_quali :
    cramers =cramers_V(var_quali[var1], var_quali[var2]) 
    col.append(round(cramers,2))   
  rows.append(col)
  
cramers_results = np.array(rows)
df_var_quali = pd.DataFrame(cramers_results, columns = var_quali.columns, index =var_quali.columns)
df_var_quali


# In[259]:


mask = np.zeros_like(df_var_quali, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

with sb.axes_style("white"):
    ax = sb.heatmap(df_var_quali, mask=mask,vmin=0., vmax=1, square=True, annot=True)
plt.show()


# *Conclusion*
# 
#    - Quantitative/quantitative :   
#   Dans le groupe des variables récoltées via nos données principales, il y a certaines corrélations très fortes (>0,8) : occutc-catv_tc (0,8), obs_ni-obsm_veh (0,85) et moyennement fortes (>0,5) comme catv_voit-obs_ni (0,5), 
#   catv_voit-obsm_veh (0,54), catv_drou-catv_voit (-0,54), obsm_aucun-obsm_veh (-0,56), obsm_veh-obsm_pieton(-0,51).  
# En revanche, dans le groupe des variables recueillies de l'INSEE, nous observons de très fortes corrélations parmi les variables relatives à la population exceptée POP15P_CS1 (qui d'ailleurs nous l'avons vu était bien corrélée à grav). D'autre part, DENSITE et TAUX_P ont également peu de corrélations avec les autres variables.    
#   
#   
#   
#    - Qualitative/qualitative :   
#   Dans le groupe de ces variables, il y a très peu de corrélations :  seules les variables catr-agg (0,42) et catu-place (0,39) apparaissent avec les plus fortes valeurs de corrélations mais on pourrait quand même les conserver.
# 
# 
# 
# 

# **Conclusion de l'EDA**
# 
# Les variables corrélées à grav et intéressantes à conserver (non corrélées aux autres) sont : 
# 
#     - quantitatives : 
# 
# annee   
# heure   
# minute    
# nbv 
# age  
# lat   
# long    
# CATV_EDP  
# CATV_ENG   
# CATV_PL   
# CATV_TC  
# CATV_VOIT  
# OBS_ARB   
# OBS_AUTR    
# OBS_BAT   
# OBS_POT    
# OBS_VEH_STAT     
# OBSM_ANIDOM    
# OBSM_AUTR 
# OBSM_VEH  
# CHOC_ARRIERE      
# CHOC_AVANT   
# CHOC_COTE    
# CHOC_MULTI  
# CHOC_NI  
# MANV_CHG  
# MANV_CLRBUS  
# MANV_CONTR  
# MANV_DEMIT  
# MANV_DEPA  
# MANV_DEPO  
# MANV_DIV  
# MANV_E2F  
# MANV_INS  
# MANV_MARR  
# MANV_MSMF  
# MANV_NI  
# MANV_SCHG  
# MANV_TOUR  
# MANV_TPC  
# TAUX_P  
# MED_VIE  
# POP  
# POP15P_CS1   
# SUPERF  
# DENSITE
#     
#     - qualitatives : 
# Toutes (sauf place et dep) car nous conservons Gpe_dep pour la notion de département.  
#    
#    

# In[93]:


# Base retenue

df_base_finale.drop(columns=['Num_Acc','mois','occutc','jour',
                             'place',
                             'dep',
                             'POPH', 'POPF','POP_FR', 'POP_ETR', 'POP_IMM','POP15P_CS2',
                             'POP15P_CS3', 'POP15P_CS4', 'POP15P_CS5', 'POP15P_CS6', 'POP15P_CS7',
                             'POP15P_CS8', 'ACT_1564', 'CHOM_1564', 'INACT_1564', 'RETR_1564',
                             'ACTOCC15P_PAS', 'ACTOCC15P_MAR', 'ACTOCC15P_DROU', 'ACTOCC15P_VOIT',
                             'ACTOCC15P_TCOM'],inplace=True)


# #### *2.2.6 Identification des outliers*

# 

# ## Partie 3 : Pre-processing

# ### *3.1 Split Train-Test* ##

# In[94]:


df_base_finale.shape


# In[95]:


df_base_finale_y = df_base_finale['grav']
df_base_finale_X = df_base_finale.drop(columns=['grav'])
X_train, X_test, y_train, y_test = train_test_split(df_base_finale_X, df_base_finale_y, test_size=0.3, 
                                                    random_state=0,
                                                    stratify = df_base_finale_y)


# In[96]:


print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])
X_train.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)


# ### *3.2 Traitement des données manquantes : dropna(), imputation*

# In[97]:


from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Imputation des variables 
var_quali1_na = ['circ','prof','plan','surf','situ']
var_quali2_na = ['nbv']
var_quanti_na = ['lat','long','TAUX_P','MED_VIE','POP','POP15P_CS1','SUPERF','DENSITE']
var_quali3_na = ['circ','prof','plan','surf']
var_quali4_na = ['nbv']

# Imputer 1 : pour les var quali 3 et 4 où il reste des 0 et -1 en tant que données manquantes (vus plus haut)
def impute_retraite_3(df):
    for col in var_quali3_na:
        df[col] = df[col].replace(0,"Non renseigné")
    return df

def impute_retraite_4(df):
    for col in var_quali4_na:
        df[col] = df[col].replace(-1,0)
    return df

# Imputer 2
imputation_var_quali1 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='Non renseigné')
imputation_var_quali2 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
imputation_var_quanti = SimpleImputer()

# Pipeline avec Imputer 2
pipeline_preproc_impute = make_column_transformer((imputation_var_quali1,var_quali1_na),
                                                  (imputation_var_quali2,var_quali2_na),
                                                  (imputation_var_quanti,var_quanti_na))


# Compilation des Imputer 
impute_retraite_3(X_train)
impute_retraite_4(X_train)
impute_retraite_3(X_test)
impute_retraite_4(X_test)

df_impute_na = pd.DataFrame(pipeline_preproc_impute.fit_transform(X_train)).reset_index(drop='index')
column_names = ['circ','prof','plan','surf','situ','nbv','lat','long','TAUX_P','MED_VIE',
                'POP','POP15P_CS1','SUPERF','DENSITE']
df_impute_na.columns=column_names
X_train.drop(columns=['circ','prof','plan','surf','situ','nbv','lat','long','TAUX_P','MED_VIE',
                    'POP','POP15P_CS1','SUPERF','DENSITE'],inplace=True)
X_train = pd.concat([X_train,df_impute_na],axis=1)


df_impute_na_test = pd.DataFrame(pipeline_preproc_impute.fit_transform(X_test)).reset_index(drop='index')
column_names = ['circ','prof','plan','surf','situ','nbv','lat','long','TAUX_P','MED_VIE',
                'POP','POP15P_CS1','SUPERF','DENSITE']
df_impute_na_test.columns=column_names
X_test.drop(columns=['circ','prof','plan','surf','situ','nbv','lat','long','TAUX_P','MED_VIE',
                    'POP','POP15P_CS1','SUPERF','DENSITE'],inplace=True)
X_test = pd.concat([X_test,df_impute_na_test],axis=1)

# ### *3.3 Standardisation & Encodage*
# 

# In[98]:


from sklearn.preprocessing import StandardScaler


var_quali = ['lum', 'agg', 'int', 'atm', 'col','Week_end', 'weekday', 'catr',
             'circ', 'prof', 'plan', 'surf', 'situ', 'catu','sexe', 'trajet', 'secu','classe age',
             'catv', 'obs', 'obsm', 'choc', 'manv','Gpe_dep']

var_quanti = ['annee', 'heure', 'minute','lat','long', 'nbv', 'age',
              'TAUX_P','MED_VIE','POP','POP15P_CS1','SUPERF','DENSITE']

#encodage_var_quali = OneHotEncoder()
standardisation_var_quanti = make_pipeline(StandardScaler())

standard_var_quanti = pd.DataFrame(standardisation_var_quanti.fit_transform(X_train[var_quanti]))
standard_var_quanti.columns=X_train[var_quanti].columns

column_names_for_onehot = X_train[var_quali].columns[0:]
encodage_var_quali = pd.get_dummies(X_train[var_quali], columns=column_names_for_onehot)




standard_var_quanti_test = pd.DataFrame(standardisation_var_quanti.fit_transform(X_test[var_quanti]))
standard_var_quanti_test.columns=X_test[var_quanti].columns

column_names_for_onehot = X_test[var_quali].columns[0:]
encodage_var_quali_test = pd.get_dummies(X_test[var_quali], columns=column_names_for_onehot)



# Base pour modélisation
X_train_model = pd.concat([standard_var_quanti,encodage_var_quali],axis=1)
X_test_model = pd.concat([standard_var_quanti_test,encodage_var_quali_test],axis=1)


# In[99]:


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_train_model.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train_model.columns.values]
X_test_model.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test_model.columns.values]

X_train_model.rename(columns={"int":"intersect","classe age":"classe_age"},inplace=True)
X_test_model.rename(columns={"int":"intersect","classe age":"classe_age"},inplace=True)


# ### *3.4 Premier modèle simple (GLM)*

# Une première idée de modélisation serait de faire une approche classique GLM sur la fréquence de gravité par model point c'est-à-dire par agrégation de données.   
# En effet, je vais d'abord construire des Model Points en regroupant les accidents selon des caractéristiques communes. 
# Je vais les regrouper selon les 5 variables catégorielles les plus corrélées à la variable cible, examinées plus haut : secu, Gpe_dep, col, obs et catr
# A l'issue de cette étape, je modéliserai la fréquence de gravité des accidents dans les clusters identifiés au moyen d'une régression classique GLM.  
#  
# 

# *Regroupement simple de données choisies les plus corrélées à grav*

# In[100]:



df_for_reg = df_base_finale.copy()
colonne = ['secu', 'Gpe_dep', 'col', 'obs', 'catr']
for col in df_for_reg[colonne]:
    df_for_reg[col]= df_for_reg[col].astype(str)

# In[370]:


espace = ["_" for i in range(len(df_for_reg))]

Agg_var =     [secu + espace +
               Gpe_dep + espace +
               col + espace +  
               obs + espace +
               catr 

               for secu, espace,
                   Gpe_dep, espace,
                   col, espace, 
                   obs, espace,
                   catr
               in zip(df_for_reg['secu'], espace, 
                      df_for_reg['Gpe_dep'], espace,       
                      df_for_reg['col'], espace, 
                      df_for_reg['obs'], espace,
                      df_for_reg['catr']
                     )]
df_Agg_Var = pd.DataFrame(Agg_var)
df_Agg_Var.columns=['Agg_Var']

# Rajout de la nouvelle variable Agg_Var dans X_train

df_for_reg.reset_index(drop = True, inplace = True)

df_for_reg_1 = pd.concat([df_for_reg,df_Agg_Var],ignore_index=False,axis=1)

# In[371]:


# Avec cette nouvelle variable,  je vais aggréger les observations ayant les mêmes caractéristiques.  

# In[ ]:
df_agg_var_grav = df_for_reg_1[['Agg_Var','grav']]
df_agg_var_grav.head()
df_freqgrav_aggvar = df_agg_var_grav.groupby(['Agg_Var']).sum()
df_freqgrav_aggvar = df_freqgrav_aggvar.assign(Nb = pd.DataFrame(df_agg_var_grav['Agg_Var'].value_counts()))
df_freqgrav_aggvar= df_freqgrav_aggvar.assign(Freq = df_freqgrav_aggvar['grav']/df_freqgrav_aggvar['Nb'])
df_freqgrav_aggvar.drop(columns=['grav','Nb'],inplace=True)
df_freqgrav_aggvar.reset_index(drop = True, inplace = True)
df_freqgrav_aggvar.shape[0]


df_agg_var_freq = pd.concat([df_agg_var_grav.groupby(['Agg_Var'],as_index=False).sum(),df_freqgrav_aggvar],axis=1,ignore_index=False)
df_agg_var_freq.drop(columns=['grav'],inplace=True)
df_for_reg_1=pd.merge(df_for_reg_1,df_agg_var_freq,how='left',on=['Agg_Var'])


df_for_reg_1.rename(columns={'Freq':'Freq_grav'},inplace=True)

df_for_reg_1.head()


# In[95]:

df_for_reg_y = df_for_reg_1['Freq_grav']
df_for_reg_X = df_for_reg_1.drop(columns=['Freq_grav'])
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(df_for_reg_X, df_for_reg_y, test_size=0.3, 
                                                    random_state=0)
                                                  

# X_train_model reste la base des variables explicatives, mais y_train_reg et y_test_reg deviennent nos variables à expliquer

# In[96]:


print(X_train_model.shape[0])
print(X_test_model.shape[0])
print(y_train_reg.shape[0])
print(y_test_reg.shape[0])
X_train_model.reset_index(drop = True, inplace = True)
X_test_model.reset_index(drop = True, inplace = True)
y_train_reg.reset_index(drop = True, inplace = True)
y_test_reg.reset_index(drop = True, inplace = True)




# In[ ]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve

X = np.array(X_train_model)
y=np.array(y_train_reg)
model_LinR = LinearRegression()
reg =model_LinR.fit(X, y)

def adj_r2(test_data, r2_score):
    records_num = test_data.shape[0]
    feature_num = test_data.shape[1]
    adj_r2_score = 1 - ((records_num - 1) / (records_num - feature_num - 1) * (1 - r2_score))
    return adj_r2_score

model_LinR_predictions = model_LinR.predict(X_test_model)
model_LinR_rmse = mean_squared_error(y_test_reg, model_LinR_predictions, squared=False)
model_LinR_r2 = r2_score(y_test_reg, model_LinR_predictions)
model_LinR_adj_r2 = adj_r2(X_test_model, model_LinR_r2)
print(f'RMSE: {model_LinR_rmse} \n R2: {model_LinR_r2} \n Adj_R2: {model_LinR_adj_r2}')


# In[ ]:
LinR_pipe = make_pipeline(LinearRegression())
# lr_pipe.get_params().keys()
LinR_param = {
    'linearregression__fit_intercept': [True, False]
}

LinR_grid = GridSearchCV(LinR_pipe, 
                         LinR_param,
                         scoring="neg_root_mean_squared_error",
                         n_jobs=-1,
                         cv = 5)
LinR_grid.fit(X_train_model, y_train_reg)
print(f'Best Params: {LinR_grid.best_params_} \nBest score: {-(LinR_grid.best_score_)}')



N,train_score, test_score = learning_curve(LinR_grid.best_estimator_, X_train_model, y_train_reg, cv=3, scoring='neg_root_mean_squared_error',
                                               train_sizes=np.linspace(0.1,1,5))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()



# In[ ]:


# *ACP*

# Avant le passage à la modélisation , je réalise une analyse ACP puor visualiser en 2D ou 3D les observations.   

# In[101]:


from sklearn.decomposition import PCA

np.random.seed(123)

pca=PCA(n_components=3)
df_reduced=pca.fit_transform(X_train_model)


# In[102]:


PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# In[146]:


pca.explained_variance_ratio_.sum()


# In[147]:


eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(3)], 
        "Variance expliquée" : pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    }
)
eig



# *Cercle des corrélations des variables*

# In[395]:


# Corrélations entre variables et axes
n_components=3
sqrt_vp = np.sqrt(pca.explained_variance_)
varfac = np.zeros((X_train_model.shape[1],X_train_model.shape[1]))
for i in range(n_components): 
    varfac[:,i] = pca.components_[i,:] * sqrt_vp[i]

Contr1 = pd.DataFrame({'Variable':X_train_model.columns,'Composante_1':varfac[:,0]})
Contr2 = pd.DataFrame({'Variable':X_train_model.columns,'Composante_2':varfac[:,1]})
Contr3 = pd.DataFrame({'Variable':X_train_model.columns,'Composante_3':varfac[:,2]})
Contr1_trie = Contr1[(Contr1['Composante_1']<=-0.1) | (Contr1['Composante_1']>=0.1) ]
Contr2_trie = Contr2[(Contr2['Composante_2']<=-0.1) | (Contr2['Composante_2']>=0.1) ]
Contr3_trie = Contr3[(Contr3['Composante_3']<=-0.1) | (Contr3['Composante_3']>=0.1) ]
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
ax.barh(Contr1_trie[:10].Variable,Contr1_trie[:10].Composante_1)
ax.set_xticks([-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ax.set_xticklabels([-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ax.set_title('Contribution des variables Composante 1')


fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.barh(Contr2_trie[:10].Variable,Contr2_trie[:10].Composante_2)
ax.set_xticks([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
ax.set_xticklabels([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
ax.set_title('Contribution des variables Composante 2')


fig3 = plt.figure()
ax = fig3.add_axes([0,0,1,1])
ax.barh(Contr3_trie[:10].Variable,Contr3_trie[:10].Composante_3)
ax.set_xticks([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ax.set_xticklabels([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ax.set_title('Contribution des variables Composante 3')


plt.show()
#plt.savefig(os.path.join('contribution des variables PCA.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial 


# La visualisation des contributions à la formation des axes montre que :
# 
#     -SUPERF et POP15P_CS1 (et agg_Hors agglomeration dans une moindre mesure) contribuent fortement positivement à la construction de l'axe 1 ainsi que POP, MED_VIE et DENSITE mais de façon négative.  
#     
#     -TAUX_P contribue très fortement (et positivement) à la formation de l'axe 2, et on peut dire que MED_VIE et  lat contribuent également mais de façon négative.  
#     
#     -nbv est celle qui contribue le plus à l'axe 3 (avec annee, lat et TAUX_P dans une moindre mesure).  
#     

# In[321]:


# Cercle des corrélations

c1 = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
c2 = pca.components_[1] * np.sqrt(pca.explained_variance_[1])

fig1 = plt.figure(figsize=(5,5))
ax = fig1.add_subplot(1, 1, 1)

for i, j, nom in zip(c1, c2, X_train_model.columns):
    plt.text(i, j, nom, fontsize=10)
    plt.arrow(0, 0, i, j, color='black')
# Ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_artist(cercle)

plt.axis((-1,1,-1,1))
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.show()

c1 = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
c3 = pca.components_[2] * np.sqrt(pca.explained_variance_[2])

fig2 = plt.figure(figsize=(5,5))
ax = fig2.add_subplot(1, 1, 1)

for i, j, nom in zip(c1, c3, X_train_model.columns):
    plt.text(i, j, nom, fontsize=10)
    plt.arrow(0, 0, i, j, color='black')
# Ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_artist(cercle)

plt.axis((-1,1,-1,1))
plt.xlabel('Dim 1')
plt.ylabel('Dim 3')
plt.show()


c2 = pca.components_[1] * np.sqrt(pca.explained_variance_[1])
c3 = pca.components_[2] * np.sqrt(pca.explained_variance_[2])

fig2 = plt.figure(figsize=(5,5))
ax = fig2.add_subplot(1, 1, 1)

for i, j, nom in zip(c2, c3, X_train_model.columns):
    plt.text(i, j, nom, fontsize=10)
    plt.arrow(0, 0, i, j, color='black')
# Ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_artist(cercle)

plt.axis((-1,1,-1,1))
plt.xlabel('Dim 2')
plt.ylabel('Dim 3')
plt.show()



# Nous confirmons les constats précédemment évoqués :   
#    - sur l'**axe 1**, les accidents ayant lieu dans les endroits à superficies élevées où loge la population de la catégorie CS1 (agriculteurs exploitants) s'opposent aux accidents aux endroits à forte densité de population. Avec la première analyse ci-dessus, nous pouvons résumer sur cet axe les accidents ayant lieu hors agglomération (larges superficies avec des agriculteurs) par opposition à ceux en agglomération (population dense, revenus élevés).    
#     
#    - l'**axe 2** oppose quant à lui les accidents ayant lieu autour des zones dont le taux de pauvreté est très élevé avec les accidents survenant aux zones où le revenu médian de vie est important : cet axe-ci retrace les accidents aux lieux de niveau de vie différents.  
#     
#    - l'**axe 3**, enfin, les accidents ayant lieu aux zones où le nombre de voies est élevé.
#     



# In[103]:
# *Nuage des individus*

# Coordonnées des individus (accidents)
X_pca = pd.DataFrame({
    "PC1" : df_reduced[:,0], 
    "PC2" : df_reduced[:,1],
    "PC3" : df_reduced[:,2]
})


# In[276]:


X_pca.plot(x='PC1',y='PC2',kind='scatter',figsize=(8,6),color='grey')
X_pca.plot(x='PC1',y='PC3',kind='scatter',figsize=(8,6),color='grey')
X_pca.plot(x='PC2',y='PC3',kind='scatter',figsize=(8,6),color='grey')


# Sur le premier plan factoriel, les points se rassemblent à peu près en 5 clusters nets et avec un cluster plus dominant que les autres au milieu.   
# Sur le deuxième plan, il apparaît plutôt 4 clusters, le cluster au milieu rassemble toujours la majorité des individus.   
# Enfin, le troisième plan dessine 3 clusters nets à droite et au milieu un gros cluster.



# In[104]:


data_acp = X_pca[['PC1','PC2','PC3']]
data_acp.head()


# In[202]:


eigpp=pca.components_.T


# In[205]:


eigpp.shape


# In[105]:


#Ajout de PC1, PC2 et PC3
X_train_model["PC1"]=data_acp['PC1']
X_train_model["PC2"]=data_acp['PC2']
X_train_model["PC3"]=data_acp['PC3']

#Application projection sur X_test
X_reduced_test=pca.transform(X_test_model)
X_pca_test = pd.DataFrame({
    "PC1" : X_reduced_test[:,0], 
    "PC2" : X_reduced_test[:,1],
    "PC3" : X_reduced_test[:,2]
})
data_acp_test = X_pca_test[['PC1','PC2','PC3']]

X_test_model["PC1"]=data_acp_test['PC1']
X_test_model["PC2"]=data_acp_test['PC2']
X_test_model["PC3"]=data_acp_test['PC3']




# In[151]:


# In[ ]:
# ##### Régression régularisée de ELASTICNET pour réduire la dimension de X_train
# ELASTICNET
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)
elasticnet = ElasticNetCV(cv=skf, random_state=0,max_iter=5000).fit(X_train_model, y_train)


print('Selected Features:', list(X_train_model.columns[np.where(elasticnet.coef_!=0)[0]]))

X_train_reduced = X_train_model[X_train_model.columns[np.where(elasticnet.coef_!=0)[0]]]
X_test_reduced = X_test_model[X_test_model.columns[np.where(elasticnet.coef_!=0)[0]]]

X_train_reduced.shape

# ##### Comparaison de différents modèles et évaluation de performances
# 
# Différents modèles sont évalués : régression logistique, arbre de décision, random forest et XGBoost.
# Tous sont entraînés avec des paramètres par défaut initialement. Puis, je procèderai à l'hyperparamétrage pour chaque modèle.   
# Pour évaluer leurs performances, je vais me baser sur 4 métriques : recall, precision, f1-score, roc auc.  

# In[106]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, validation_curve, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score, ConfusionMatrixDisplay, roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import PolynomialFeatures


# In[346]:


pd.set_option('display.max_columns',None)
print(X_train_reduced.columns)
# In[107]:


def draw_confusion_matrix(classifier):
    fig, ax = plt.subplots(figsize=(8, 5))
    ConfusionMatrixDisplay.from_estimator(classifier, X_test_reduced, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
    ax.set_title("Confusion Matrix", fontsize = 15)
    plt.show()
    

# In[108]:


def evaluation(model):
    model.fit(X_train_reduced,y_train)
    y_pred = model.predict(X_test_reduced)
    draw_confusion_matrix(model)
    print(classification_report(y_test,y_pred))
    # balanced_accuracy
    model_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f'balanced_accuracy: {model_balanced_accuracy}')
    
    # ROC_AUC score
    model_roc_macro = roc_auc_score(y_test, model.predict_proba(X_test_reduced)[:,1])
    print(f"roc_macro: {model_roc_macro}")    



# In[155]:


# Logistic Regression
model_LR = LogisticRegression(C=10, fit_intercept=False, solver='liblinear')
evaluation(model_LR)
y_pred = model_LR.predict(X_test_reduced)
model_LR_recall = recall_score(y_test, y_pred)
model_LR_precision = precision_score(y_test, y_pred)
model_LR_f1_score = f1_score(y_test, y_pred)
model_LR_roc_macro = roc_auc_score(y_test, model_LR.predict_proba(X_test_reduced)[:,1])


print(f'recall: {model_LR_recall}')
print(f'precision: {model_LR_precision}')
print(f'f1_score: {model_LR_f1_score}')
print(f"roc_macro: {model_LR_roc_macro}") 

# In[156]:


N,train_score, test_score = learning_curve(model_LR, X_train_reduced, y_train, cv=3, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,5))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()


# * Decision Tree

# In[157]:


model_Tree = DecisionTreeClassifier(random_state=0)
evaluation(model_Tree)
y_pred = model_Tree.predict(X_test_reduced)
model_Tree_recall = recall_score(y_test, y_pred)
model_Tree_precision = precision_score(y_test, y_pred)
model_Tree_f1_score = f1_score(y_test, y_pred)
model_Tree_roc_macro = roc_auc_score(y_test, model_Tree.predict_proba(X_test_reduced)[:,1])

print(f'recall: {model_Tree_recall}')
print(f'precision: {model_Tree_precision}')
print(f'f1_score: {model_Tree_f1_score}')
print(f"roc_macro: {model_Tree_roc_macro}") 

# In[158]:


N,train_score, test_score = learning_curve(model_Tree, X_train_reduced, y_train, cv=3, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,5))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()



# In[159]:


# Random Forest
model_RF = RandomForestClassifier(random_state=0)
evaluation(model_RF)
y_pred = model_RF.predict(X_test_reduced)
model_RF_recall = recall_score(y_test, y_pred)
model_RF_precision = precision_score(y_test, y_pred)
model_RF_f1_score = f1_score(y_test, y_pred)
model_RF_roc_macro = roc_auc_score(y_test, model_RF.predict_proba(X_test_reduced)[:,1])


print(f'recall: {model_RF_recall}')
print(f'precision: {model_RF_precision}')
print(f'f1_score: {model_RF_f1_score}')
print(f"roc_macro: {model_RF_roc_macro}") 

N,train_score, test_score = learning_curve(model_RF, X_train_reduced, y_train, cv=3, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,5))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()


# * XGBOOST

# In[161]:


# XGB Model
model_XGB = XGBClassifier(learning_rate=0.3, max_depth=10, n_estimators=100, eval_metric='logloss', random_state=0)
evaluation(model_XGB)
y_pred = model_XGB.predict(X_test_reduced)
model_XGB_recall = recall_score(y_test, y_pred)
model_XGB_precision = precision_score(y_test, y_pred)
model_XGB_f1_score = f1_score(y_test, y_pred)
model_XGB_roc_macro = roc_auc_score(y_test, model_XGB.predict_proba(X_test_reduced)[:,1])

print(f'recall: {model_XGB_recall}')
print(f'precision: {model_XGB_precision}')
print(f'f1_score: {model_XGB_f1_score}')
print(f"roc_macro: {model_XGB_roc_macro}") 


N,train_score, test_score = learning_curve(model_XGB, X_train_reduced, y_train, cv=4, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()




# Récap des modèles best


modele_LR = pd.DataFrame(data=[model_LR_recall,
                             model_LR_precision, 
                             model_LR_f1_score,
                             model_LR_roc_macro], 
             columns=['Logistic Regression'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


modele_Tree = pd.DataFrame(data=[model_Tree_recall,
                               model_Tree_precision, 
                               model_Tree_f1_score,
                               model_Tree_roc_macro], 
             columns=['Decision Tree'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


modele_forest = pd.DataFrame(data=[model_RF_recall,
                                 model_RF_precision, 
                                 model_RF_f1_score,
                                 model_RF_roc_macro], 
             columns=['Random Forest'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])
 


modele_XGB = pd.DataFrame(data=[model_XGB_recall,
                              model_XGB_precision, 
                              model_XGB_f1_score,
                              model_XGB_roc_macro], 
             columns=['XGBOOST'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])




df_models = round(pd.concat([modele_LR,
                             modele_Tree,
                             modele_forest,
                             modele_XGB
                             ], axis=1),6)


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

colour_palette = ["lightgray","lightgray","#0f4c81"]
colormap = mcolors.LinearSegmentedColormap.from_list("", colour_palette)

background_color = "#fbfbfb"

figrecap_modeles = plt.figure(figsize=(10,8))
gs = figrecap_modeles.add_gridspec(4, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = figrecap_modeles.add_subplot(gs[0, :])

sb.heatmap(df_models.T, cmap=colormap,annot=True,fmt=".1%",vmin=0,vmax=0.95, linewidths=2.5,cbar=False,ax=ax0,annot_kws={"fontsize":12})
figrecap_modeles.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-2.15,'Model Comparison',fontsize=18,fontweight='bold',fontfamily='serif')
#ax0.text(0,-0.9,'Random Forest performs the best for overall Accuracy,\nbut is this enough? Is Recall more important in this case?',fontsize=14,fontfamily='serif')
ax0.tick_params(axis=u'both', which=u'both',length=0)


plt.show()



# **Optimisation**  
# *Hyperparameter tuning*

# In[114]:

# *Logistic Regression
LR_pipe = make_pipeline(LogisticRegression(solver='liblinear'))
LR_param = {
    'logisticregression__penalty' : ['l1','l2'],
    'logisticregression__C': [0.1, 1.0, 10],
    'logisticregression__fit_intercept': [True, False]
    
}

LR_grid = GridSearchCV(LR_pipe, 
                      LR_param,
                      scoring="f1",
                      n_jobs=-1,
                      cv = 4)

LR_grid.fit(X_train_reduced, y_train)
print(f'Best Params: {LR_grid.best_params_} \nBest score: {-(LR_grid.best_score_)}')

best_LR = (LR_grid.best_estimator_)
y_pred = best_LR.predict(X_test_reduced)


best_LR_recall = recall_score(y_test, y_pred)
best_LR_precision = precision_score(y_test, y_pred)
best_LR_f1_score = f1_score(y_test, y_pred)
best_LR_roc_macro = roc_auc_score(y_test, best_LR.predict_proba(X_test_reduced)[:,1])

print(f'recall: {best_LR_recall}')
print(f'precision: {best_LR_precision}')
print(f'f1_score: {best_LR_f1_score}')
print(f"roc_macro: {best_LR_roc_macro}") 


draw_confusion_matrix(best_LR)

N, train_score, test_score = learning_curve(best_LR, X_train_reduced, y_train,cv=4,
                                            scoring='f1',train_sizes=np.linspace(0.1,1,10))

plt.figure(figsize=(8,8))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()

# In[122]:
# Decision Tree

Tree_pipe = make_pipeline(DecisionTreeClassifier(random_state=0))
Tree_param = {
    'decisiontreeclassifier__max_depth': [5, 10, 20],
    'decisiontreeclassifier__min_samples_leaf': [2, 5, 10],
    'decisiontreeclassifier__min_samples_split': [2, 4, 6, 8]
}

Tree_grid = GridSearchCV(Tree_pipe, 
                         Tree_param,
                         scoring="f1",
                         n_jobs=-1,
                         cv = 4)
Tree_grid.fit(X_train_reduced, y_train)

best_Tree = (Tree_grid.best_estimator_)
print(Tree_grid.best_params_)
y_pred = best_Tree.predict(X_test_reduced)

best_Tree_recall = recall_score(y_test, y_pred)
best_Tree_precision = precision_score(y_test, y_pred)
best_Tree_f1_score = f1_score(y_test, y_pred)
best_Tree_roc_macro = roc_auc_score(y_test, best_Tree.predict_proba(X_test_reduced)[:,1])


print(f'recall: {best_Tree_recall}')
print(f'precision: {best_Tree_precision}')
print(f'f1_score: {best_Tree_f1_score}')
print(f"roc_macro: {best_Tree_roc_macro}") 

draw_confusion_matrix(best_Tree)

N, train_score, test_score = learning_curve(best_Tree, X_train_reduced, y_train,cv=4,
                                            scoring='f1',train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(8,8))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()

#

# In[4]:
# *Random Forest*
RandomPipeline = make_pipeline(RandomForestClassifier(random_state=0))
hyper_params = {
    'randomforestclassifier__n_estimators':[100,150,250,400,600],
    'randomforestclassifier__criterion':['gini','entropy'],
    'randomforestclassifier__min_samples_split':[2,6,12],
    'randomforestclassifier__min_samples_leaf':[1,4,6,10],
    'randomforestclassifier__max_features':['auto','srqt','log2',int,float],
    'randomforestclassifier__verbose':[2],
    'randomforestclassifier__class_weight':['balanced','balanced_subsample'],
   'randomforestclassifier__n_jobs':[-1]
}

           
RF_grid = GridSearchCV(RandomPipeline,hyper_params,scoring='f1')
RF_grid.fit(X_train_reduced,y_train)

print(RF_grid.best_params_)


best_forest = (RF_grid.best_estimator_)
best_forest.fit(X_train_reduced,y_train)

y_pred = best_forest.predict(X_test_reduced)

best_forest_recall = recall_score(y_test, y_pred)
best_forest_precision = precision_score(y_test, y_pred)
best_forest_f1_score = f1_score(y_test, y_pred)
best_forest_roc_macro = roc_auc_score(y_test, best_forest.predict_proba(X_test_reduced)[:,1])

print(f'recall: {best_forest_recall}')
print(f'precision: {best_forest_precision}')
print(f'f1_score: {best_forest_f1_score}')
print(f"roc_macro: {best_forest_roc_macro}")

draw_confusion_matrix(best_forest)

N, train_score, test_score = learning_curve(best_forest, X_train_reduced, y_train,cv=4,
                                            scoring='f1',train_sizes=np.linspace(0.1,1,10))


plt.figure(figsize=(8,8))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()


print(classification_report(y_test,y_pred))
best_forest_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'balanced_accuracy: {best_forest_balanced_accuracy}')

    
# *RF 2  plus sophistiquée*

    # In[164]:

# Random Forest 2

best_RF_feat_selection = make_pipeline(SelectKBest(f_classif,k=35),PolynomialFeatures(2),best_forest)

best_RF_feat_selection.fit(X_train_reduced,y_train)

y_pred = best_RF_feat_selection.predict(X_test_reduced)

best_RF_feat_selection_recall = recall_score(y_test, y_pred)
best_RF_feat_selection_precision = precision_score(y_test, y_pred)
best_RF_feat_selection_f1_score = f1_score(y_test, y_pred)
best_RF_feat_selection_roc_macro = roc_auc_score(y_test, best_RF_feat_selection.predict_proba(X_test_reduced)[:,1])

print(f'recall: {best_RF_feat_selection_recall}')
print(f'precision: {best_RF_feat_selection_precision}')
print(f'f1_score: {best_RF_feat_selection_f1_score}')
print(f"roc_macro: {best_RF_feat_selection_roc_macro}")

print(classification_report(y_test,y_pred))
best_RF_feat_selection_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'balanced_accuracy: {best_RF_feat_selection_balanced_accuracy}')


N,train_score, test_score = learning_curve(best_RF_feat_selection, X_train_reduced, y_train, cv=4, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()

# *XGBOOST*

# In[115]:


XGBPipeline = make_pipeline(XGBClassifier(random_state=0))


# In[116]:
XGB_param = {
              'xgbclassifier__learning_rate': [0.3],
              'xgbclassifier__max_depth': [10],
              'xgbclassifier__n_estimators': [100],
              'xgbclassifier__subsample': [0.5, 0.8, 1],
              'xgbclassifier__colsample_bytree': [0.5,1],
              'xgbclassifier__gamma': [0, 0.5, 1],
              'xgbclassifier__scale_pos_weight':[2.5, 3.5, 4.5]

            }


XGB_grid = GridSearchCV(XGBPipeline,XGB_param,scoring='f1',n_jobs=-1)
XGB_grid.fit(X_train_reduced,y_train)

best_XGB = (XGB_grid.best_estimator_)
XGB_grid.best_params_
best_XGB.fit(X_train_reduced,y_train)
y_pred = best_XGB.predict(X_test_reduced)
best_XGB_recall = recall_score(y_test, y_pred)
best_XGB_precision = precision_score(y_test, y_pred)
best_XGB_f1_score = f1_score(y_test, y_pred)
best_XGB_roc_macro = roc_auc_score(y_test, best_XGB.predict_proba(X_test_reduced)[:,1])

print(f'recall: {best_XGB_recall}')
print(f'precision: {best_XGB_precision}')
print(f'f1_score: {best_XGB_f1_score}')
print(f"roc_macro: {best_XGB_roc_macro}")

N, train_score, test_score = learning_curve(best_XGB, X_train_reduced, y_train, 
                                           cv=4, scoring='f1', 
                                           train_sizes=np.linspace(0.1,1,10))


plt.figure(figsize=(8,8))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()


draw_confusion_matrix(best_XGB)

print(classification_report(y_test,y_pred))
best_XGB_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'balanced_accuracy: {best_XGB_balanced_accuracy}')

# In[ ]:


# In[ ]:
# XGBOOST 2
best_XGB_feature_selection = make_pipeline(SelectKBest(f_classif,k=35),PolynomialFeatures(2),
                                            best_XGB)

best_XGB_feature_selection.fit(X_train_reduced,y_train)

y_pred = best_XGB_feature_selection.predict(X_test_reduced)

best_XGB_feature_selection_recall = recall_score(y_test, y_pred)
best_XGB_feature_selection_precision = precision_score(y_test, y_pred)
best_XGB_feature_selection_f1_score = f1_score(y_test, y_pred)
best_XGB_feature_selection_roc_macro = roc_auc_score(y_test, best_XGB_feature_selection.predict_proba(X_test_reduced)[:,1])

print(f'recall: {best_XGB_feature_selection_recall}')
print(f'precision: {best_XGB_feature_selection_precision}')
print(f'f1_score: {best_XGB_feature_selection_f1_score}')
print(f"roc_macro: {best_XGB_feature_selection_roc_macro}")

N,train_score, test_score = learning_curve(best_XGB_feature_selection, X_train_reduced, y_train, cv=4, scoring='f1',
    
                                               train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(5,5))
plt.plot(N,train_score.mean(axis=1),label='train score')
plt.plot(N,test_score.mean(axis=1),label='test score')
plt.legend()

print(classification_report(y_test,y_pred))
best_XGB_feat_selection_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'balanced_accuracy: {best_XGB_feat_selection_balanced_accuracy}')




# In[ ]:
    


# Récap des modèles best


best_LR = pd.DataFrame(data=[best_LR_recall,
                             best_LR_precision, 
                             best_LR_f1_score,
                             best_LR_roc_macro], 
             columns=['Logistic Regression'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_Tree = pd.DataFrame(data=[best_Tree_recall,
                               best_Tree_precision, 
                               best_Tree_f1_score,
                               best_Tree_roc_macro], 
             columns=['Decision Tree'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_forest = pd.DataFrame(data=[best_forest_recall,
                                 best_forest_precision, 
                                 best_forest_f1_score,
                                 best_forest_roc_macro], 
             columns=['Random Forest'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])
 


best_XGB = pd.DataFrame(data=[best_XGB_recall,
                              best_XGB_precision, 
                              best_XGB_f1_score,
                              best_XGB_roc_macro], 
             columns=['XGBOOST'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])

best_RF_feature_selection = pd.DataFrame(data=[best_RF_feat_selection_recall,
                                                     best_RF_feat_selection_precision, 
                                                     best_RF_feat_selection_f1_score,
                                                     best_RF_feat_selection_roc_macro], 
             columns=['Random Forest Feat Selectkbest'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_XGB_feat_selection = pd.DataFrame(data=[best_XGB_feature_selection_recall,
                                                     best_XGB_feature_selection_precision, 
                                                     best_XGB_feature_selection_f1_score,
                                                     best_XGB_feature_selection_roc_macro], 
             columns=['XGBOOST Feat Selectkbest'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])



df_models = round(pd.concat([best_LR,
                             best_Tree,
                             best_forest,
                             best_XGB,
                             best_RF_feature_selection,
                             best_XGB_feat_selection
                             ], axis=1),6)

#import matplotlib
#import matplotlib.pyplot as plt
colour_palette = ["lightgray","lightgray","#0f4c81"]
colormap = mcolors.LinearSegmentedColormap.from_list("", colour_palette)

background_color = "#fbfbfb"

figrecap_modeles = plt.figure(figsize=(10,8))
gs = figrecap_modeles.add_gridspec(4, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = figrecap_modeles.add_subplot(gs[0, :])

sb.heatmap(df_models.T, cmap=colormap,annot=True,fmt=".1%",vmin=0,vmax=0.95, linewidths=2.5,cbar=False,ax=ax0,annot_kws={"fontsize":12})
figrecap_modeles.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-2.15,'Model Comparison',fontsize=18,fontweight='bold',fontfamily='serif')
#ax0.text(0,-0.9,'Random Forest performs the best for overall Accuracy,\nbut is this enough? Is Recall more important in this case?',fontsize=14,fontfamily='serif')
ax0.tick_params(axis=u'both', which=u'both',length=0)
plt.show()





# In[ ]:
# ### *3.5 Feature Selection*

# In[109]:


from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, RFECV 
from mlxtend.evaluate import feature_importance_permutation
from sklearn.inspection import permutation_importance


# *Feature Selection sur Random Forest*

# In[292]:

best_forest.fit(X_train_reduced,y_train)
# Variables Importance
best_forest_importances_std = np.std([tree.feature_importances_ for tree in best_forest.named_steps["randomforestclassifier"].estimators_], axis=0)
best_forest_importances = pd.Series(best_forest.named_steps["randomforestclassifier"].feature_importances_, index=X_train_reduced.columns)
best_forest_importances_df = pd.DataFrame(best_forest_importances, columns=['Importance'])
best_forest_importances_df['Std'] = best_forest_importances_std
best_forest_importances_df.sort_values('Importance', ascending=True, inplace=True)
fig, ax = plt.subplots(figsize=(15,45))
best_forest_importances_df['Importance'].plot.barh(xerr=best_forest_importances_df['Std'], color='cornflowerblue', ax=ax)
ax.set_title("Feature importances Random Forest", fontsize = 22)
ax.set_xlabel("Mean decrease in impurity")
fig.tight_layout()


# In[370]:


best_forest.fit(X_train_reduced,y_train)
# SelectFromModel
sel_best_forest = SelectFromModel(estimator=best_forest.named_steps["randomforestclassifier"], prefit=True,threshold='mean')
X_train_sfm_RF = sel_best_forest.transform(X_train_reduced)
cols = sel_best_forest.get_support(indices=True)
features_df_new = X_train_reduced.iloc[:,cols]
features_df_new.columns

# In[372]:

    

# RFECV
sel_rfecv_best_forest = RFECV(estimator=best_forest.named_steps["randomforestclassifier"], min_features_to_select=10, step=1,cv=4)
X_train_rfecv_RF = sel_rfecv_best_forest.fit_transform(X_train_reduced, y_train)
print(sel_rfecv_best_forest.get_support())
cols = sel_rfecv_best_forest.get_support(indices=True)
features_df_new = X_train_reduced.iloc[:,cols]
features_df_new.columns
features_df_new.shape

# In[122]:
best_forest.fit(X_train_reduced,y_train)
# Permutation importance
r = permutation_importance(best_forest, X_test_reduced, y_test,
                            n_repeats=5,
                            random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X_train_reduced.columns[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")




# Random forest with feature selection 1 (selectfromModel), 2 (RFECV)  et 3 (PI)

# Feature Selection 1 SelectfromModel
X_train_best_forest_featSelFM_selection = X_train_reduced.iloc[:,sel_best_forest.get_support(indices=True)]
X_test_best_forest_featSelFM_selection = X_test_reduced.iloc[:,sel_best_forest.get_support(indices=True)]
forest_final=best_forest.fit(X_train_best_forest_featSelFM_selection,y_train)
forest_finalSelFM_predict = best_forest.predict(X_test_best_forest_featSelFM_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_forest, X_test_best_forest_featSelFM_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()

best_forest_featSelFM_selection_recall = recall_score(y_test,forest_finalSelFM_predict)
best_forest_featSelFM_selection_precision = precision_score(y_test,forest_finalSelFM_predict)
best_forest_featSelFM_selection_f1_score = f1_score(y_test,forest_finalSelFM_predict)
best_forest_featSelFM_selection_roc_macro = roc_auc_score(y_test, forest_finalSelFM_predict)


print(best_forest_featSelFM_selection_recall)
print(best_forest_featSelFM_selection_precision)
print(best_forest_featSelFM_selection_f1_score)
print(best_forest_featSelFM_selection_roc_macro)


# feature Selection 2 RFECV
X_train_best_forest_featRFECV_selection = X_train_reduced.iloc[:,sel_rfecv_best_forest.get_support(indices=True)]
X_test_best_forest_featRFECV_selection = X_test_reduced.iloc[:,sel_rfecv_best_forest.get_support(indices=True)]
forest_final=best_forest.fit(X_train_best_forest_featRFECV_selection,y_train)
forest_finalRFECV_predict = best_forest.predict(X_test_best_forest_featRFECV_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_forest, X_test_best_forest_featRFECV_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()
best_forest_featRFECV_selection_recall = recall_score(y_test,forest_finalRFECV_predict)
best_forest_featRFECV_selection_precision = precision_score(y_test,forest_finalRFECV_predict)
best_forest_featRFECV_selection_f1_score = f1_score(y_test,forest_finalRFECV_predict)
best_forest_featRFECV_selection_roc_macro = roc_auc_score(y_test, forest_finalRFECV_predict)

print(best_forest_featRFECV_selection_recall)
print(best_forest_featRFECV_selection_precision)
print(best_forest_featRFECV_selection_f1_score)
print(best_forest_featRFECV_selection_roc_macro)



# feature Selection 3 Permutation Importance (top 13)


#secu_11 0.012 +/- 0.000
#catv_CATV_DROU0.007 +/- 0.000
#catr_Route Départementale0.006 +/- 0.000
#age     0.004 +/- 0.000
#Gpe_dep_10.004 +/- 0.000
#lat     0.004 +/- 0.000
#secu_21 0.003 +/- 0.000
#catu_Piéton0.003 +/- 0.000
#long    0.003 +/- 0.000
#POP     0.003 +/- 0.000
#agg_En agglomération0.003 +/- 0.000
#agg_Hors agglomération0.002 +/- 0.000
#obs_OBS_NI0.002 +/- 0.000
#annee   0.001 +/- 0.000
#POP15P_CS10.001 +/- 0.000
#TAUX_P  0.001 +/- 0.000
#MED_VIE 0.001 +/- 0.000
#Gpe_dep_20.001 +/- 0.000
#Gpe_dep_00.001 +/- 0.000
#trajet_Promenade loisirs0.001 +/- 0.000
#heure   0.001 +/- 0.000
#plan_Partie rectiligne0.001 +/- 0.000
#manv_MANV_DEPO0.001 +/- 0.000
#circ_Bidirectionnelle0.001 +/- 0.000
#PC1     0.001 +/- 0.000
#Gpe_dep_60.001 +/- 0.000
#prof_Plat0.001 +/- 0.000
#secu_93 0.001 +/- 0.000
#SUPERF  0.001 +/- 0.000
#secu_13 0.001 +/- 0.000
#jour    0.001 +/- 0.000
#minute  0.001 +/- 0.000
#trajet_Utilisation professionnelle0.000 +/- 0.000
#catu_Passager0.000 +/- 0.000
#choc_CHOC_AVANT0.000 +/- 0.000
#Gpe_dep_30.000 +/- 0.000
#PC3     0.000 +/- 0.000
#obs_OBS_VEH_STAT0.000 +/- 0.000
#Gpe_dep_50.000 +/- 0.000
#lum_Plein jour0.000 +/- 0.000
#catr_Route nationale0.000 +/- 0.000
#catv_CATV_PL0.000 +/- 0.000
#manv_MANV_SCHG0.000 +/- 0.000
#Gpe_dep_90.000 +/- 0.000
#Gpe_dep_40.000 +/- 0.000
#lum_Nuit sans éclairage public0.000 +/- 0.000
#atm_Normale0.000 +/- 0.000
#catr_Autoroute0.000 +/- 0.000
#secu_3  0.000 +/- 0.000
#weekday_vendredi0.000 +/- 0.000
#surf_Normale0.000 +/- 0.000
#catv_CATV_TC0.000 +/- 0.000
#obsm_9  0.000 +/- 0.000
#catv_CATV_ENG0.000 +/- 0.000
#weekday_dimanche0.000 +/- 0.000
#manv_MANV_DIV0.000 +/- 0.000
#classe age_(70.0, 80.0_0.000 +/- 0.000
#manv_MANV_CONTR0.000 +/- 0.000
#trajet_Autre0.000 +/- 0.000
#secu_31 0.000 +/- 0.000
#classe age_(-0.001, 10.0_0.000 +/- 0.000
#secu_12 0.000 +/- 0.000
#choc_CHOC_NI0.000 +/- 0.000
#obs_OBS_ARB0.000 +/- 0.000
#Gpe_dep_80.000 +/- 0.000
#plan_En S0.000 +/- 0.000
#surf_Non renseigné0.000 +/- 0.000
#situ_Non renseigné0.000 +/- 0.000
#int_Giratoire0.000 +/- 0.000
#obs_OBS_GLIS0.000 +/- 0.000
#manv_MANV_NI0.000 +/- 0.000
#trajet_Domicile Travail0.000 +/- 0.000
#manv_MANV_E2F0.000 +/- 0.000
#choc_CHOC_MULTI0.000 +/- 0.000
#catu_Piéton en mouvement sur roller/trottinette0.000 +/- 0.000
#trajet_Domicile Ecole0.000 +/- 0.000
#secu_2  0.000 +/- 0.000
#catv_CATV_AUTR0.000 +/- 0.000




select_PI = ['secu_11','catv_CATV_DROU','lat','Gpe_dep_1','age','long',
                 'agg_En agglomération','secu_21','POP','obs_OBS_NI',
                 'catu_Piéton','agg_Hors agglomération','catr_Route Départementale']
X_train_best_forest_featPI_selection = X_train_reduced[select_PI]
X_test_best_forest_featPI_selection = X_test_reduced[select_PI]
forest_final=best_forest.fit(X_train_best_forest_featPI_selection,y_train)
forest_finalPI_predict = best_forest.predict(X_test_best_forest_featPI_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_forest, X_test_best_forest_featPI_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()
best_forest_featPI_selection_recall = recall_score(y_test,forest_finalPI_predict)
best_forest_featPI_selection_precision = precision_score(y_test,forest_finalPI_predict)
best_forest_featPI_selection_f1_score = f1_score(y_test,forest_finalPI_predict)
best_forest_featPI_selection_roc_macro = roc_auc_score(y_test, forest_finalPI_predict)

print(best_forest_featPI_selection_recall)
print(best_forest_featPI_selection_precision)
print(best_forest_featPI_selection_f1_score)
print(best_forest_featPI_selection_roc_macro)







# *Feature Selection sur XGBOOST*

# In[292]:

# Variables importance
best_XGB.fit(X_train_reduced,y_train)
sorted_idx = best_XGB.named_steps["xgbclassifier"].feature_importances_.argsort()
plt.subplots(figsize=(15,45))
plt.barh(X_train_reduced.columns[sorted_idx], best_XGB.named_steps["xgbclassifier"].feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")



# In[1]:


# SelectFromModel

best_XGB.fit(X_train_reduced,y_train)
sel_best_XGB = SelectFromModel(estimator=best_XGB.named_steps["xgbclassifier"], prefit=True, threshold='mean')  
X_train_sfm_XGB = sel_best_XGB.transform(X_train_reduced)
cols = sel_best_XGB.get_support(indices=True)
features_XGB_df_new = X_train_reduced.iloc[:,cols]
features_XGB_df_new.columns


# In[372]:


# RFECV
best_XGB.fit(X_train_reduced,y_train)
sel_rfecv_best_XGB = RFECV(estimator=best_XGB.named_steps["xgbclassifier"], min_features_to_select=10, step=2,cv=3)
X_train_rfecv_XGB = sel_rfecv_best_XGB.fit_transform(X_train_reduced, y_train)
print(sel_rfecv_best_XGB.get_support())
pd.set_option('display.max_rows',None)
cols = sel_rfecv_best_XGB.get_support(indices=True)
features_XGB_df_new = X_train_reduced.iloc[:,cols]
features_XGB_df_new.columns.tolist()



# In[352]:

# Permutation importance
best_XGB.fit(X_train_reduced,y_train)
r = permutation_importance(best_XGB, X_test_reduced, y_test,
                            n_repeats=10,
                            random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X_train_reduced.columns[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")

#secu_11 0.023 +/- 0.000
#PC1     0.018 +/- 0.000
#catv_CATV_DROU0.010 +/- 0.000
#lat     0.009 +/- 0.000
#age     0.009 +/- 0.000
#long    0.006 +/- 0.000
#secu_21 0.005 +/- 0.000
#obs_OBS_NI0.004 +/- 0.000
#annee   0.004 +/- 0.000
#POP     0.004 +/- 0.000
#Gpe_dep_10.003 +/- 0.000
#Gpe_dep_00.003 +/- 0.000
#obsm_OBSM_AUCUN0.003 +/- 0.000
#heure   0.002 +/- 0.000
#catu_Passager0.002 +/- 0.000
#PC3     0.002 +/- 0.000
#secu_13 0.002 +/- 0.000
#TAUX_P  0.002 +/- 0.000
#MED_VIE 0.001 +/- 0.000
#catr_Route Départementale0.001 +/- 0.000
#catu_Conducteur0.001 +/- 0.000
#catv_CATV_TC0.001 +/- 0.000
#POP15P_CS10.001 +/- 0.000
#secu_93 0.001 +/- 0.000
#catv_CATV_PL0.001 +/- 0.000
#agg_Hors agglomération0.001 +/- 0.000
#catr_Autoroute0.001 +/- 0.000
#situ_Sur chaussée0.001 +/- 0.000
#choc_CHOC_AVANT0.001 +/- 0.000
#col_Trois véhicules et plus - collisions multiples0.001 +/- 0.000
#plan_Partie rectiligne0.001 +/- 0.000
#Gpe_dep_20.001 +/- 0.000
#jour    0.001 +/- 0.000
#catv_CATV_ENG0.001 +/- 0.000
#trajet_Promenade loisirs0.001 +/- 0.000
#prof_Plat0.001 +/- 0.000
#minute  0.001 +/- 0.000
#Gpe_dep_30.001 +/- 0.000
#col_Sans collision0.001 +/- 0.000
#surf_Normale0.001 +/- 0.000
#Gpe_dep_60.001 +/- 0.000
#manv_MANV_TOUR0.000 +/- 0.000
#obs_OBS_GLIS0.000 +/- 0.000
#catr_Route nationale0.000 +/- 0.000
#obsm_9  0.000 +/- 0.000
#obs_OBS_VEH_STAT0.000 +/- 0.000
#int_Giratoire0.000 +/- 0.000
#trajet_Utilisation professionnelle0.000 +/- 0.000
#secu_3  0.000 +/- 0.000
#choc_CHOC_MULTI0.000 +/- 0.000
#circ_Bidirectionnelle0.000 +/- 0.000
#plan_Non renseigné0.000 +/- 0.000
#Gpe_dep_80.000 +/- 0.000
#weekday_dimanche0.000 +/- 0.000
#secu_31 0.000 +/- 0.000
#manv_MANV_SCHG0.000 +/- 0.000
#manv_MANV_NI0.000 +/- 0.000
#obs_OBS_ARB0.000 +/- 0.000
#Gpe_dep_50.000 +/- 0.000
#secu_92 0.000 +/- 0.000
#int_Hors intersection0.000 +/- 0.000
#atm_Normale0.000 +/- 0.000
#plan_En S0.000 +/- 0.000
#manv_MANV_DIV0.000 +/- 0.000
#Gpe_dep_40.000 +/- 0.000
#Gpe_dep_70.000 +/- 0.000
#manv_MANV_CONTR0.000 +/- 0.000
#secu_12 0.000 +/- 0.000
#situ_Non renseigné0.000 +/- 0.000
#situ_Sur piste cyclable0.000 +/- 0.000
#classe age_(50.0, 60.0_0.000 +/- 0.000
#secu_22 0.000 +/- 0.000
#classe age_(-0.001, 10.0_0.000 +/- 0.000



# In[ ]:
# XGBOOST with feature selection 1 (selectfromModel) et 2 (RFECV)
#features_XGB_df_new.head()

# feature selection 1 SelectfromModel
X_train_best_XGB_featSelFM_selection = X_train_reduced.iloc[:,sel_best_XGB.get_support(indices=True)]
X_test_best_XGB_featSelFM_selection = X_test_reduced.iloc[:,sel_best_XGB.get_support(indices=True)]
XGB_final=best_XGB.fit(X_train_best_XGB_featSelFM_selection,y_train)
XGB_finalSelFM_predict = best_XGB.predict(X_test_best_XGB_featSelFM_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_XGB, X_test_best_XGB_featSelFM_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()
best_XGB_featSelFM_selection_recall = recall_score(y_test,XGB_finalSelFM_predict)
best_XGB_featSelFM_selection_precision = precision_score(y_test,XGB_finalSelFM_predict)
best_XGB_featSelFM_selection_f1_score = f1_score(y_test,XGB_finalSelFM_predict)
best_XGB_featSelFM_selection_roc_macro = roc_auc_score(y_test, XGB_finalSelFM_predict)

print(best_XGB_featSelFM_selection_recall)
print(best_XGB_featSelFM_selection_precision)
print(best_XGB_featSelFM_selection_f1_score)
print(best_XGB_featSelFM_selection_roc_macro)


# feature selection 2 RFECV
X_train_best_XGB_featRFECV_selection = X_train_reduced.iloc[:,sel_rfecv_best_XGB.get_support(indices=True)]
X_test_best_XGB_featRFECV_selection = X_test_reduced.iloc[:,sel_rfecv_best_XGB.get_support(indices=True)]
XGB_final=best_XGB.fit(X_train_best_XGB_featRFECV_selection,y_train)
XGB_finalRFECV_predict = best_XGB.predict(X_test_best_XGB_featRFECV_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_XGB, X_test_best_XGB_featRFECV_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()
best_XGB_featRFECV_selection_recall = recall_score(y_test,XGB_finalRFECV_predict)
best_XGB_featRFECV_selection_precision = precision_score(y_test,XGB_finalRFECV_predict)
best_XGB_featRFECV_selection_f1_score = f1_score(y_test,XGB_finalRFECV_predict)
best_XGB_featRFECV_selection_roc_macro = roc_auc_score(y_test, XGB_finalRFECV_predict)

print(best_XGB_featRFECV_selection_recall)
print(best_XGB_featRFECV_selection_precision)
print(best_XGB_featRFECV_selection_f1_score)
print(best_XGB_featRFECV_selection_roc_macro)




# feature Selection 3 Permutation Importance (top 13)
select_PI = ['secu_11','PC1','catv_CATV_DROU','age',
                 'lat','long','secu_21','obsm_OBSM_AUCUN',
                 'obs_OBS_NI','annee','POP','Gpe_dep_0',
                 'Gpe_dep_1']
X_train_best_XGB_featPI_selection = X_train_reduced[select_PI]
X_test_best_XGB_featPI_selection = X_test_reduced[select_PI]
XGB_final=best_XGB.fit(X_train_best_XGB_featPI_selection,y_train)
XGB_finalPI_predict = best_XGB.predict(X_test_best_XGB_featPI_selection)
fig, ax = plt.subplots(figsize=(8, 5))
ConfusionMatrixDisplay.from_estimator(best_XGB, X_test_best_XGB_featPI_selection, y_test, cmap=plt.cm.Blues, normalize=None, ax=ax)
ax.set_title("Confusion Matrix", fontsize = 15)
plt.show()
best_XGB_featPI_selection_recall = recall_score(y_test,XGB_finalPI_predict)
best_XGB_featPI_selection_precision = precision_score(y_test,XGB_finalPI_predict)
best_XGB_featPI_selection_f1_score = f1_score(y_test,XGB_finalPI_predict)
best_XGB_featPI_selection_roc_macro = roc_auc_score(y_test, XGB_finalPI_predict)

print(best_XGB_featPI_selection_recall)
print(best_XGB_featPI_selection_precision)
print(best_XGB_featPI_selection_f1_score)
print(best_XGB_featPI_selection_roc_macro)




# Récap des modèles best Random Forest et XGBOOST


best_forest_featSelFM_selection = pd.DataFrame(data=[best_forest_featSelFM_selection_recall,
                                                     best_forest_featSelFM_selection_precision, 
                                                     best_forest_featSelFM_selection_f1_score,
                                                     best_forest_featSelFM_selection_roc_macro], 
             columns=['Random Forest Score SelFM'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_forest_featRFECV_selection = pd.DataFrame(data=[best_forest_featRFECV_selection_recall,
                                                     best_forest_featRFECV_selection_precision, 
                                                     best_forest_featRFECV_selection_f1_score,
                                                     best_forest_featRFECV_selection_roc_macro], 
             columns=['Random Forest Score RFECV'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_forest_featPI_selection = pd.DataFrame(data=[best_forest_featPI_selection_recall,
                                                     best_forest_featPI_selection_precision, 
                                                     best_forest_featPI_selection_f1_score,
                                                     best_forest_featPI_selection_roc_macro], 
             columns=['Random Forest Score PI'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])
 


best_XGB_featSelFM_selection = pd.DataFrame(data=[best_XGB_featSelFM_selection_recall,
                                                     best_XGB_featSelFM_selection_precision, 
                                                     best_XGB_featSelFM_selection_f1_score,
                                                     best_XGB_featSelFM_selection_roc_macro], 
             columns=['XGBOOST Score SelFM'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])

best_XGB_featREFCV_selection = pd.DataFrame(data=[best_XGB_featRFECV_selection_recall,
                                                     best_XGB_featRFECV_selection_precision, 
                                                     best_XGB_featRFECV_selection_f1_score,
                                                     best_XGB_featRFECV_selection_roc_macro], 
             columns=['XGBOOST Score RFECV'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])


best_XGB_featPI_selection = pd.DataFrame(data=[best_XGB_featPI_selection_recall,
                                                     best_XGB_featPI_selection_precision, 
                                                     best_XGB_featPI_selection_f1_score,
                                                     best_XGB_featPI_selection_roc_macro], 
             columns=['XGBOOST Score PI'],
             index=["Recall", "Precision", "F1", "ROC AUC Score"])



df_models = round(pd.concat([best_forest_featSelFM_selection,
                             best_forest_featRFECV_selection,
                             best_forest_featPI_selection,
                             best_XGB_featSelFM_selection,
                             best_XGB_featREFCV_selection,
                             best_XGB_featPI_selection
                             ], axis=1),6)
import matplotlib.colors as mcolors


colors = ["lightgray","lightgray","#0f4c81"]
colormap = mcolors.LinearSegmentedColormap.from_list("", colors)

background_color = "#fbfbfb"

fig = plt.figure(figsize=(8,8)) # create figure
gs = fig.add_gridspec(4, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sb.heatmap(df_models.T, cmap=colormap,annot=True,fmt=".1%",vmin=0,vmax=0.95, linewidths=2.5,cbar=False,ax=ax0,annot_kws={"fontsize":12})
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-2.15,'Model Comparison',fontsize=18,fontweight='bold',fontfamily='serif')
#ax0.text(0,-0.9,'Random Forest performs the best for overall Accuracy,\nbut is this enough? Is Recall more important in this case?',fontsize=14,fontfamily='serif')
ax0.tick_params(axis=u'both', which=u'both',length=0)


plt.show()



# In[ ]:


# *Precision recall curve : ajustement du threshold pour le Best XGBoost Sel REFCV*

# In[ ]:


from sklearn.metrics import precision_recall_curve


best_XGB.fit(X_train_best_XGB_featRFECV_selection,y_train)
y_score = best_XGB.predict_proba(X_test_best_XGB_featRFECV_selection)[:, 1]

precision, recall, threshold = precision_recall_curve(y_test, best_XGB.predict_proba(X_test_best_XGB_featRFECV_selection)[:, 1])

# In[ ]:


plt.plot(threshold, precision[:-1],label='precision')
plt.plot(threshold, recall[:-1],label='recall')
plt.legend()


# In[ ]:


def model_final(model,X,threshold=0):
    return model.predict_proba(X)[:, 1] > threshold

y_pred = model_final(best_XGB,X_test_best_XGB_featRFECV_selection,threshold=0.44)


print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


print(recall_score(y_test,XGB_finalRFECV_predict))
print(precision_score(y_test,XGB_finalRFECV_predict))
print(f1_score(y_test,XGB_finalRFECV_predict))
print(roc_auc_score(y_test,XGB_finalRFECV_predict))



#### Visualisation des prédictions avec le modèle retenu : XGBoost



# In[236]:

df_dep.shape
df_dep_y = df_dep['grav']
df_dep_X = df_dep.drop(columns=['grav'])
X_train_dep, X_test_dep, y_train, y_test = train_test_split(df_dep_X, df_dep_y, test_size=0.3, 
                                                    random_state=0,
                                                    stratify = df_dep_y)

print(X_train_dep.shape[0])
print(X_test_dep.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])
X_train_dep.reset_index(drop = True, inplace = True)
X_test_dep.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)


# Accidents graves et prédits graves
y_pred = pd.DataFrame(model_final(best_XGB,X_test_best_XGB_featRFECV_selection,threshold=0.44))

df_accidents_grav_obs = pd.concat([X_test_dep,y_test],axis=1,ignore_index=False)
df_accidents_grav_pred = pd.concat([X_test_dep,y_pred],axis=1,ignore_index=False)
df_accidents_grav_pred.rename(columns={0 : 'grav'},inplace=True)


df_accidents_grav_obs.head()
df_accidents_grav_pred.head()

dep_accident_grav = df_accidents_grav_obs[df_accidents_grav_obs['grav']==1].dep.value_counts().to_frame().reset_index()
dep_accident_grav['indice_de_gravite']=np.log10(dep_accident_grav['dep'])
print(dep_accident_grav.head())

dep_accident_pgrav = df_accidents_grav_pred[df_accidents_grav_pred['grav']==1].dep.value_counts().to_frame().reset_index()
dep_accident_pgrav['indice_de_gravite']=np.log10(dep_accident_pgrav['dep'])

###### dep_accident_grav et dep_accident_pgrav à exporter et à travailler sur jupyter notebook pour la visualisation

dep_accident_grav_data = dep_accident_grav.to_csv('dep_accident_grav.csv', index = True)
print('\nCSV String:\n', dep_accident_grav_data)

dep_accident_pgrav_data = dep_accident_pgrav.to_csv('dep_accident_pgrav.csv', index = True)
print('\nCSV String:\n', dep_accident_pgrav_data)


#import matplotlib.pyplot as plt
#sb.set_style('whitegrid')
#import plotly.express as px
#import geopandas as gpd
#import json
#import geojson
#import folium

#with open('departements.geojson',encoding='UTF-8') as dep:
#    departement = geojson.load(dep)

# In[237]:


#for feature in departement['features']:
#    feature['id']= feature['properties']['code']

# In[238]:

#fig_grav = px.choropleth_mapbox(dep_accident_grav, locations = 'index',
#                            geojson= departement,
#                            color='indice_de_gravite',
#                            color_continuous_scale=["green","orange","red"],
#                            range_color=[2,3.5],
#                            hover_name='index',
#                            hover_data=['dep'],
#                            title="Répartition des accidents graves observés en France",
#                            mapbox_style="open-street-map",
#                            center= {'lat':46, 'lon':2},
#                            zoom =4, 
#                            opacity= 0.8)



#fig_gravp = px.choropleth_mapbox(dep_accident_gravp, locations = 'index',
#                            geojson= departement,
#                            color='indice_de_gravite',
#                            color_continuous_scale=["green","orange","red"],
#                            range_color=[2,3.5],
#                            hover_name='index',
#                            hover_data=['dep'],
#                            title="Répartition des accidents graves prédits en France",
#                            mapbox_style="open-street-map",
#                            center= {'lat':46, 'lon':2},
#                            zoom =4, 
#                            opacity= 0.8)

#fig_grav.show()
#fig_gravp.show()
#fig_grav.savefig(os.path.join('Accidents graves observés des données Test.png'), dpi=300, format='png', bbox_inches='tight') 
#fig_gravp.savefig(os.path.join('Accidents graves prédits des données Test.png'), dpi=300, format='png', bbox_inches='tight') 

# In[ ]:



# ### *3.9 Modèle choisi*

# In[ ]:



# ## Partie 4 - Interprétabilité des résultats ##

# ### 4.1 Valeurs de SHAP 

# In[ ]:
# Evaluation globale 
import shap
import fasttreeshap



#X_train_best_XGB_featRFECV_selection.rename(columns={'POP15P_CS1':'POP_CS1',
#                                                       'secu_11':'secu_onze',
#                                                       'secu_12':'secu_douze',
#                                                       'secu_13':'secu_treize',
#                                                       'secu_21':'secu_vintgetun',
#                                                       'secu_93':'secu_quatrevingttreize',
#                                                       'classe age_(20.0, 30.0_':'classe_agevingttrentre',
#                                                       'classe age_(30.0, 40.0_':'classe_agetrentrequanrante',
#                                                       'classe age_(40.0, 50.0_':'classe_agequanrantecinquante',
#                                                       'classe age_(50.0, 60.0_':'classe_agecinquantesoixante',
#                                                       'classe age_(60.0, 70.0_':'classe_agesoixantesoixantedix',
#                                                       'classe age_(80.0, 90.0_':'classe_agequatrevingtquatrevingtdix',
#                                                       'Gpe_dep_0':'Gpe_dep_zero',
#                                                       'Gpe_dep_1':'Gpe_dep_un',
#                                                       'Gpe_dep_2':'Gpe_dep_deux',
#                                                       'Gpe_dep_3':'Gpe_dep_trois',
#                                                       'Gpe_dep_5':'Gpe_dep_cinq',
#                                                       'Gpe_dep_6':'Gpe_dep_six',
#                                                       'PC1':'PCun',
#                                                       'PC3':'PCtrois'},inplace=True)
#X_train_best_XGB_featRFECV_selection.head()



X_train_shapval, X_train_val_shapval = train_test_split(X_train_best_XGB_featRFECV_selection,
                                                    train_size=15000,
                                                    random_state=0,stratify=y_train)

X_test_shapval, X_test_val_shapval, y_test_shapval, y_test_val_shapval  = train_test_split(X_test_best_XGB_featRFECV_selection,y_test,
                                                    train_size=2000,
                                                    random_state=0,stratify=y_test)
                                                   
print(X_train_shapval.shape)                             
print(X_test_shapval.shape)
X_train_shapval.reset_index(drop = True, inplace = True)
X_test_shapval.reset_index(drop = True, inplace = True)
y_test_shapval.reset_index(drop = True, inplace = True)

 
shap_explain = fasttreeshap.TreeExplainer(best_XGB.named_steps["xgbclassifier"], 
                                  data=X_train_shapval,
                                  algorithm="v2",
                                  n_jobs=-1,
                                  feature_perturbation="interventional"
                                  )

shap_values = shap_explain(X_test_shapval).values

# In[ ]:
    
# Evaluation globale du modèle
from matplotlib.colors import LinearSegmentedColormap
shap.summary_plot(shap_values,X_test_shapval,
                  feature_names=X_test_shapval.columns.tolist())

# SHAP Feat Importance
shap.summary_plot(
    shap_values, X_test_shapval, feature_names=X_test_shapval.columns, plot_type="bar",
    cmap='hsv',alpha=0.4
)
# In[ ]:


# Dependence plot 

ax2 = fig.add_subplot(224)
shap.dependence_plot('secu_11', shap_values, X_test_shapval, interaction_index=None)
shap.dependence_plot("secu_11", shap_values, X_test_shapval, interaction_index="agg_En agglomération")
shap.dependence_plot("secu_11", shap_values, X_test_shapval, interaction_index="PC1")
shap.dependence_plot("secu_11", shap_values, X_test_shapval, interaction_index="catv_CATV_DROU")


shap.dependence_plot('obsm_OBSM_AUCUN', shap_values, X_test_shapval, interaction_index=None)

shap.dependence_plot('PC1', shap_values, X_test_shapval, interaction_index=None)


shap.dependence_plot('PC1', shap_values, X_test_shapval, interaction_index="catv_CATV_DROU",show=False)
plt.title("Zoom PC1 dependence plot with catv_drou")
plt.xlim(left=1.5,right=2)
plt.ylim(bottom=-0.2,top=0.5)
plt.show()


shap.dependence_plot('PC1', shap_values, X_test_shapval, interaction_index="secu_11",show=False)
plt.title("Zoom PC1 dependence plot with secu_11")
plt.xlim(left=-5,right=-2)
plt.ylim(bottom=-1.5,top=-0.5)
plt.show()




# Evaluation locale pour une instance

y_pred = model_final(best_XGB,X_test_shapval,threshold=0.44)

df_y_pred = pd.DataFrame(y_pred)
df_y_pred.rename(columns={0:'grav_pred'},inplace=True)
df_y_pred.grav_pred=df_y_pred.grav_pred.replace('False',0)
df_y_pred.grav_pred=df_y_pred.grav_pred.replace('True',1)
pd.set_option('display.max_rows',None)
# Faux négatifs
print('#------- Indexes of FNs from the test data :\n')
print(np.where((np.array(df_y_pred).ravel()==0) & (np.array(y_test_shapval).ravel()==1)),'\n')
# Faux positifs
print('#------- Indexes of FPs from the test data :\n')
print(np.where((np.array(df_y_pred).ravel()==1) & (np.array(y_test_shapval).ravel()==0)),'\n')



# Un Faux négatif 94
print('Probability obtained :',y_pred[94])
shap.force_plot(shap_explain.expected_value, shap_values[94,:],features=X_test_shapval.iloc[94],matplotlib=True, show=False,feature_names=X_test_shapval.columns)
plt.savefig(os.path.join('force plot FN94.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show() 



# Un Faux négatif 953

print('Probability obtained :',y_pred[953])
shap.force_plot(shap_explain.expected_value, shap_values[953,:],features=X_test_shapval.iloc[953],matplotlib=True, show=False,feature_names=X_test_shapval.columns)
plt.savefig(os.path.join('force plot FN953.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()



# Un Faux positif 230

print('Probability obtained :',y_pred[230])
shap.force_plot(shap_explain.expected_value, shap_values[230,:],features=X_test_shapval.iloc[230],matplotlib=True, show=False,feature_names=X_test_shapval.columns)
plt.savefig(os.path.join('force plot FP230.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show() 


# Un Faux positif 419

print('Probability obtained :',y_pred[419])
shap.force_plot(shap_explain.expected_value, shap_values[419,:],features=X_test_shapval.iloc[419],matplotlib=True, show=False,feature_names=X_test_shapval.columns)
plt.savefig(os.path.join('force plot FP419.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show() 


# Evaluation des secu_11=1

indexsecu = X_test_shapval[X_test_shapval.secu_11==1]
shap_values_secu11 = shap_explain(X_test_shapval.iloc[indexsecu.index,:]).values
shap.decision_plot(shap_explain.expected_value, shap_values_secu11,feature_names = X_test_shapval.columns.tolist())   
plt.savefig(os.path.join('decision plot secu_11.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()

# Evaluation des obsm_OBSM_AUCUN=1

indexobsmaucun = X_test_shapval[X_test_shapval.obsm_OBSM_AUCUN==1]
shap_values_obsmaucun = shap_explain(X_test_shapval.iloc[indexobsmaucun.index,:]).values
shap.decision_plot(shap_explain.expected_value, shap_values_obsmaucun,feature_names = X_test_shapval.columns.tolist())   
plt.savefig(os.path.join('decision plot obsm_aucun.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()

# Evaluation des agg_En agglomération=1

indexaggEnagg = X_test_shapval[X_test_shapval['agg_En agglomération']==1]
shap_values_aggEnagg = shap_explain(X_test_shapval.iloc[indexaggEnagg.index,:]).values
shap.decision_plot(shap_explain.expected_value, shap_values_aggEnagg,feature_names = X_test_shapval.columns.tolist())   
plt.savefig(os.path.join('decision plot agg_En aggl.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()


# Evaluation des catv_CATV_DROU=1

indexcatvdrou = X_test_shapval[X_test_shapval['catv_CATV_DROU']==1]
shap_values_catvdrou = shap_explain(X_test_shapval.iloc[indexcatvdrou.index,:]).values
shap.decision_plot(shap_explain.expected_value, shap_values_catvdrou,feature_names = X_test_shapval.columns.tolist())   
plt.savefig(os.path.join('decision plot catv_drou.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()


# Evaluation des catr_Route Departementale=1

indexcatrdepart = X_test_shapval[X_test_shapval['catr_Route Départementale']==1]
shap_values_catrdepart = shap_explain(X_test_shapval.iloc[indexcatrdepart.index,:]).values
shap.decision_plot(shap_explain.expected_value, shap_values_catrdepart,feature_names = X_test_shapval.columns.tolist())   
plt.savefig(os.path.join('decision plot catrdepart.png'), dpi=300, format='png', bbox_inches='tight') 
plt.show()








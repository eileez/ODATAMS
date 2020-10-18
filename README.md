# ODATAMS
Objects Detection and Tracking Algorithm in MesoNH Simulations

L'objectif de ce projet est le développement d'une approche globale de détection et de suivi automatique d'objets (structures météorologiques) au sein de simulations atmosphériques du modèle de recherche Méso-NH.

Par approche "globale", nous souhaitons que l'algorithme développé puisse être appliqué à tout type d'objets (nuages, cellules convectives etc.) et à des résolutions variables. De plus, en vue de réduire la quantités de données des simulations atmosphériques, nous souhaitons que l'algorithme soit intégrable en tant que traitement online au sein de Méso-NH. Si possible, le code pourrait être parallélisable.
In fine, cet algorithme permettrait d’extraire et retranscrire clairement les informations utiles d’une simulation atmosphérique. L’utilisation répandue d’un tel algorithme parmi la communauté atmosphérique permettrait un gain de temps, lisibilité, compréhension et de quantité de données liés aux simulations atmosphériques.

## Méthodologie
1. méthode de seuillage sur des champs de données de sorties Méso-NH pour détecter des objets en 2D ou 3D. 
2. suivi spatio-temporel des objets à l'aide des champs de « traceurs lagrangiens » implémentés dans Méso-NH.

## Avancée
### Stage juin 2020 - sept. 2020, Elizabeth Fu 
Détection et suivi de cellules convectives au sein de la simulation Méso-NH de la tempête Adrian (Corse, 29 octobre 2018)
- Etat de l'art (modèle Méso-NH, traceurs lagrangiens, méthodes de détection et de suivi d'objets, méthode classique d'analyse de données)
- Algorithme DETECT_LGZT
- Algorithme DETECT_TRACK_WT

Perspectives
- amélioration marquage des particules selon leur comportement (cohérence, convergence ou divergence de trajectoire)
- choix du seuil de particules conservées pour considérer 2 objets comme étant le même temporellement
- application à d'autres objets (différente échelle spatiale et temporelle)

## Contact
- Elizabeth Fu, Etudiante-Ingénieure en Mathématiques Appliquées et Méthodes Numériques à l'INSA Toulouse : elizabeth.wt.fu@gmail.com
- Florian Pantillon, chercheur CNRS au Laboratoire d'Aérologie UMR 5560 : florian.pantillon@aero.obs-mip.fr

## Références
- Modèle Méso-NH et traceurs lagrangiens : http://mesonh.aero.obs-mip.fr/mesonh/dir_doc/lag_m46_22avril2005/lagrangian46/
- Projet Objects, CNRM : https://gitlab.com/tropics/objects

# author : Elizabeth Fu
# elizabeth.wt.fu@gmail.com
# Institut National Sciences Appliquées (INSA) Toulouse - Laboratoire d'Aérologie, Toulouse
# june 2020 - sept 2020

(*) : dossier/fichier/script important

### DOSSIERS
* DETECT_TRACK_CONV_CELLS_WT/
* DETECT_CONV_CELLS_LGZT/
    -> résultats finaux

OLD_RESULTS/
old_scripts/
    -> /!\ pas bonne manipulation des traceurs
    
traceurs/
    -> plus de détail sur les traceurs, plots etc

### SCRIPTS
read_multi_ncdf.py
    -> script qui ouvre plusieurs fichiers netCDF 1 à 1 et lit certaines variables (ex : time)
read_traceurs.py

* detect_track_convective_cells_WT.py
    -> script qui donne résultats dans DETECT_TRACK_CONV_CELLS_WT/
    fonctionnement détaillé du script est disponible dans "GUIDE_detect_track_convective_cells_WT.txt"

* detect_convective_cells_LGZT.py
    -> script qui donne résultats dans DETECT_TRACK_CONV_CELLS_LGZT/

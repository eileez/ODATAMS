# author : Elizabeth Fu
# elizabeth.wt.fu@gmail.com
# Institut National Sciences Appliquées (INSA) Toulouse - Laboratoire d'Aérologie, Toulouse
# june 2020 - sept 2020

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import sys
import glob

####################################
# LECTURE TRACEURS - UPDATED 27/08
####################################

# choix simulation
EXP = 'CNTRL'

# choix point de maillage
i = 323
j = 328
k = 36 # env. 5174m pour level

print("Point de maillage (i=" + str(i) + " ; j=" + str(j) + " ; k=" + str(k) + ")")

# charge data
list_of_paths = glob.glob("/tmpdir/fu/OUT/new_5_6_UTC/CNTRL.1.SEG01.OUT.*.nc")
if len(list_of_paths) != 0:
    list_of_paths.sort()
    # read time and coordinates
    for path in list_of_paths:
        ncfile = xr.open_dataset(path, mask_and_scale=True, decode_times=True, decode_coords=True)
        time = str(ncfile.coords['time'][0].values)
        if (time.find(":00:00.000000000") != -1):
            print("réinitialisation coordonnées")
        print("TIME", time)
        
        lgzt = ncfile.data_vars['LGZT']
        print("LGZT", lgzt[0,k,j,i].values)

#        lgyt = ncfile.data_vars['LGYT']
#        print("LGYT", lgyt[0,k,j,i].values)

#        lgxt = ncfile.data_vars['LGXT']
#        print("LGXT", lgxt[0,k,j,i].values)

        print("- - -")

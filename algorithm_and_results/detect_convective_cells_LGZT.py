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
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import wrf
import math
from mpl_toolkits import mplot3d
import glob
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
import time

print("##################################")
print("### OPEN FILE AND LOAD DATA... ###")
print("##################################")

### simulation
EXP = 'CNTRL'

### variable of interest
VAR_name = 'LGZT'

# numero of output
# 5-6 UTC : 061-072
# 15-16 UTC : 181-192
N1 = '061'
N2 = '072'

hour_oclock = 'False'

if (N1 == '061') | (N1 == '181'):
    hour_oclock = 'True'

### open file
ncfile1 = xr.open_dataset("/tmpdir/fu/OUT/new_5_6_UTC/"+EXP+".1.SEG01.OUT."+N1+".nc", mask_and_scale=True, decode_times=True, decode_coords=True)
ncfile2 = xr.open_dataset("/tmpdir/fu/OUT/new_5_6_UTC/"+EXP+".1.SEG01.OUT."+N2+".nc", mask_and_scale=True, decode_times=True, decode_coords=True)

list_of_paths = ["/tmpdir/fu/OUT/new_5_6_UTC/"+EXP+".1.SEG01.OUT."+N1+".nc", "/tmpdir/fu/OUT/new_5_6_UTC/"+EXP+".1.SEG01.OUT."+N2+".nc"]
nb_time_stamps = len(list_of_paths)

# orography and altitude
ncfile3 = xr.open_dataset("/tmpdir/fu/"+EXP+".1.SEG01.006DIA.nc", mask_and_scale=True, decode_times=True, decode_coords=True) #pour ZS et ALT

# read time
time1 = ncfile1.coords['time']
time2 = ncfile2.coords['time']
print("time of object detection :", time1.values, " to ", time2.values)

# read orography
ZS =  ncfile3.data_vars['ZS'][:,:]
# read altitude
ALT =  ncfile3.data_vars['ALT'][:,:,:]
# read all coordinates
longitude = ncfile3.coords['longitude'][0,:].values                     # shape (602,)
latitude = ncfile3.coords['latitude'][:,0].values                       # shape (502,)

print("...OK")

############################################################
print("initial geographical domain for particule selection")
############################################################

ZOOM_loc = 'NORD_OUEST_ITALIE'

# BALEARES - AFRIQUE DU NORD (100kmx100km)
left = 3.
right = 4.
bottom = 37.
top = 38.

# NORD_OUEST_ITALIE
left = 9.
right = 10.
bottom = 43.
top = 44.

# define zoom
x0 = np.searchsorted( longitude, left ) # on cherche les indices pour insérer l'array left dans l'array longitude
x1 = np.searchsorted( longitude, right )
y0 = np.searchsorted( latitude, bottom )
y1 = np.searchsorted( latitude, top )

print("...OK")

#####################
print("load data...")
#####################

VAR = ncfile1.data_vars[VAR_name][0,:,:,:] #3D in space (72,502,602)
var = np.zeros((VAR.shape[0], VAR.shape[1], VAR.shape[2])) 
lgxt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 
lgyt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 
lgzt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 

time_ind = 0
for path in list_of_paths:
    ncfile = xr.open_dataset(path, mask_and_scale=True, decode_times=True, decode_coords=True)
    time_file = ncfile.coords['time'][0].values
    print(time_file)
    print('reading LGXT, LGYT, LGZT')
    lgzt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGZT'][0,:,y0:y1,x0:x1]
    lgxt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGXT'][0,:,y0:y1,x0:x1]
    lgyt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGYT'][0,:,y0:y1,x0:x1]
    print('...OK')
    time_ind+=1

level_array = ncfile1.level 
level_array = np.asarray(level_array)

print("###########################################")
print("### DETECT OBJECTS FOR EACH TIME STAMP ###")
print("###########################################")

# get indices of zoom domain
var[:,y0:y1,x0:x1] = np.ones((np.shape(var[:,y0:y1,x0:x1])[0], np.shape(var[:,y0:y1,x0:x1])[1], np.shape(var[:,y0:y1,x0:x1])[2] )) # met des valeurs arbitrairement
indices_zoom_domain = list(zip(*np.where(var != 0)))

### apply threshold on lgzt
low_thrs = 3300

labeled_array = np.zeros((VAR.shape[0], VAR.shape[1], VAR.shape[2]), dtype = 'int')
nb_features_solo = 0
num_features_solo = []

for ind in indices_zoom_domain:
    if (ind[0] > 0) & (ind[0] < 71):
        var[ind[0], ind[1], ind[2]] = level_array[ind[0]] - lgzt[1, ind[0], ind[1], ind[2]]

cond = (var >= low_thrs)
var = np.where(cond, 1, 0) # modified array (nb_time_stamps,72,502,602)	# can't apply the threshold floor by floor. 

### detect and label objects
print("detect and label objects...")
labeled_array = label(var)[0] 				# (72,502,602)
nb_features = label(var)[1] 
print("number of objects found : ", nb_features)
print("...OK")

### find location of objects
print("find locations of objects...")
objects_location = ndimage.find_objects(labeled_array)
obj_nb = 0 
for obj in objects_location:
    vertical_extension = np.shape(labeled_array[obj])[0]
    y_width = np.shape(labeled_array[obj])[1]
    x_width = np.shape(labeled_array[obj])[2]
    if (vertical_extension == 1) & (y_width == 1) & (x_width == 1):
        nb_features_solo += 1
        num_features_solo.append(obj_nb+1)
    obj_nb += 1

print("nombre d'objets composés d'une seule particule :", nb_features_solo)
print("numéro des objets composés d'une seule particule :", num_features_solo)

print("...OK")

########################################################
print("plot objets identifiés")
########################################################

# get indices objets
indices_obj = []
nb_particules_obj = []

for obj_nb in range(nb_features):
    print("obj nb #", obj_nb+1)
    indices_obj += [list(zip(*np.where(labeled_array[:,:,:] == obj_nb+1)))] 
    nb_particules_obj.append(len(indices_obj[obj_nb]))

print("nomdre de particules pour chaque objet identifié à t+deltat : ", nb_particules_obj)

# get real positions of particules during time frame
lgxt_obj, lgyt_obj, lgzt_obj = [], [], []

for time_ind in range(nb_time_stamps):
    tmp_lgxt_time_ind, tmp_lgyt_time_ind, tmp_lgzt_time_ind = [], [], []

    for obj_nb in range(nb_features):
        tmp_lgxt_obj_nb, tmp_lgyt_obj_nb, tmp_lgzt_obj_nb = [], [], []
        for ind in indices_obj[obj_nb]:
            if (time_ind == 0):
                tmp_lgxt_obj_nb.append(lgxt[1,ind[0], ind[1], ind[2]])    
                tmp_lgyt_obj_nb.append(lgyt[1,ind[0], ind[1], ind[2]])    
                tmp_lgzt_obj_nb.append(lgzt[1,ind[0], ind[1], ind[2]])    
            else:
                tmp_lgxt_obj_nb.append(ind[2]*3000)    
                tmp_lgyt_obj_nb.append(ind[1]*3000)    
                tmp_lgzt_obj_nb.append(level_array[ind[0]])    
    
        tmp_lgxt_time_ind += [tmp_lgxt_obj_nb]
        tmp_lgyt_time_ind += [tmp_lgyt_obj_nb]
        tmp_lgzt_time_ind += [tmp_lgzt_obj_nb]

    lgxt_obj += [tmp_lgxt_time_ind]
    lgyt_obj += [tmp_lgyt_time_ind]
    lgzt_obj += [tmp_lgzt_time_ind]

lgxt_obj = np.array(lgxt_obj)
lgyt_obj = np.array(lgyt_obj)
lgzt_obj = np.array(lgzt_obj)

#print(len(lgxt_obj))           # 2 : ok
#print(len(lgxt_obj[0]))        # 8 : ok
#print(len(lgxt_obj[0][0]))     # nb particules objet #1 : ok
#print(lgxt_obj[0][0][0])       # single value : ok 
 
np.seterr(divide='ignore', invalid='ignore')  # for 3d plot, ignore 'nan'
color_sequence = ['black','red','orange','gold','green','lime','cyan','blue','indigo','magenta',\
		'maroon','pink','grey','deeppink','dodgerblue','chocolate','coral','purple','navy','darkorange',\
		'teal','violet','yellowgreen','slateblue','olive','goldenrod', 'slategrey','darkkhaki', 'lightcoral', 'brown', \
		'mediumpurple', 'lightgrey']



fig = plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.set_xlim(x0*3000,x1*3000)
ax.set_ylim(y0*3000,y1*3000)
ax.set_zlim(0,14000)

for obj_nb in range(nb_features):
    color_pick = color_sequence[obj_nb]
    ax.scatter3D(lgxt_obj[1][obj_nb], lgyt_obj[1][obj_nb], lgzt_obj[1][obj_nb], c=color_pick, marker='o', label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))

#    for time_ind in range(nb_time_stamps):
#        if time_ind == 0:
#            ax.scatter3D(lgxt_obj[time_ind][obj_nb], lgyt_obj[time_ind][obj_nb], lgzt_obj[time_ind][obj_nb], c=color_pick, marker='^', label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
#        else:
#            ax.scatter3D(lgxt_obj[time_ind][obj_nb], lgyt_obj[time_ind][obj_nb], lgzt_obj[time_ind][obj_nb], c=color_pick, marker='o',label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
#
#        if time_ind + 2 <= nb_time_stamps:
#            # trajectories of each particule of obj_nb (time_ind - time_ind+1)
#            for a,b,c,d,e,f in zip(lgxt_obj[time_ind][obj_nb], lgxt_obj[time_ind+1][obj_nb],\
#                                        lgyt_obj[time_ind][obj_nb], lgyt_obj[time_ind+1][obj_nb],\
#                                        lgzt_obj[time_ind][obj_nb], lgzt_obj[time_ind+1][obj_nb]):
#
#                ax.plot([a,b], [c,d], [e,f], color = 'grey')

ax.legend()
### add geographical coordinates
ax.grid(True)
ax.set_xlabel('LGXT (m)')
ax.set_ylabel('LGYT (m)')
ax.set_zlabel('LGZT (m)')
### add title
ax.set_title('Position at 05:55 of objects identified during time frame 05:00-05:55')
### save as png/pdf
plt.savefig( 'figure_' + EXP + '_time_frame_' +  N1 + '_' + N2 + '_' + ZOOM_loc + '_obj_identifies.png')

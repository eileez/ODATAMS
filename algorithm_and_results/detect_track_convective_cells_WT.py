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

print("beginning of script... outputs are put into txt file")

### simulation
EXP = 'CNTRL'

# numero of output 		# 5-6 UTC : 061-072 // 15-16 UTC : 181-192
N1 = '181'
N2 = '192'

### file for print redirecting, saves results of this algorithm
f = open('output_' + N1 + '_' + N2, "a")

# variable of interest
VAR_name = 'WT'

# name of object
if VAR_name == 'WT':
    object = 'CONVECTIVE CELLS'

print("#######################")
print("OPEN FILE AND LOAD DATA")
print("#######################")
print("#######################", file=f)
print("OPEN FILE AND LOAD DATA", file=f)
print("#######################", file=f)

### open file
ncfile1 = xr.open_dataset("/tmpdir/fu/OUT/15_16_UTC/"+EXP+".1.SEG01.OUT."+N1+".nc", mask_and_scale=True, decode_times=True, decode_coords=True)
ncfile2 = xr.open_dataset("/tmpdir/fu/OUT/15_16_UTC/"+EXP+".1.SEG01.OUT."+N2+".nc", mask_and_scale=True, decode_times=True, decode_coords=True)

### list of outputs
# manual selection
list_of_paths = ["/tmpdir/fu/OUT/15_16_UTC/"+EXP+".1.SEG01.OUT."+N1+".nc", "/tmpdir/fu/OUT/15_16_UTC/"+EXP+".1.SEG01.OUT."+N2+".nc"]
# selection during full hour
#list_of_paths = glob.glob("/tmpdir/fu/OUT/15_16_UTC/CNTRL.1.SEG01.OUT.???.nc")
#list_of_paths.sort()

nb_time_stamps = len(list_of_paths)

# orography and altitude
ncfile3 = xr.open_dataset("/tmpdir/fu/"+EXP+".1.SEG01.006DIA.nc", mask_and_scale=True, decode_times=True, decode_coords=True) #pour ZS et ALT

# read times
time1 = ncfile1.coords['time']
time2 = ncfile2.coords['time']
print(object, "DETECTION AT T0 : ", time1.values, ", AND T+DELTAT : ", time2.values, file=f)

# read orography
ZS =  ncfile3.data_vars['ZS'][:,:]

# read altitude
ALT =  ncfile3.data_vars['ALT'][:,:,:]

# read all coordinates
longitude = ncfile3.coords['longitude'][0,:].values
latitude = ncfile3.coords['latitude'][:,0].values

print("...OK")
print("...OK", file=f)

print("---------------------------------------------------")
print("INITIAL GEOGRAPHICAL DOMAIN FOR PARTICULE SELECTION")
print("---------------------------------------------------")
print("---------------------------------------------------", file=f)
print("INITIAL GEOGRAPHICAL DOMAIN FOR PARTICULE SELECTION", file=f)
print("---------------------------------------------------", file=f)

ZOOM_loc = 'NORTH-CORSICA_ITALIA'
print("Selected domain : ", ZOOM_loc)
print("Selected domain : ", ZOOM_loc, file=f)

# BALEARES - NORTH OF AFRICA
left = 3.
right = 4.
bottom = 37.
top = 38.

# NORTH OF CORSICA - ITALIA
left = 9.
right = 10.
bottom = 43.
top = 44.

# define zoom
x0 = np.searchsorted( longitude, left )
x1 = np.searchsorted( longitude, right )
y0 = np.searchsorted( latitude, bottom )
y1 = np.searchsorted( latitude, top )

print("...OK")
print("...OK", file=f)

print("---------------------------")
print("INITIALISATION OF VARIABLES")
print("---------------------------")
print("---------------------------", file=f)
print("INITIALISATION OF VARIABLES", file=f)
print("---------------------------", file=f)

VAR =  ncfile1.data_vars[VAR_name][0,:,:,:] #3D in space (72,502,602)
var = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 			# var[0,:,:,:] is shape (72,502,602)
lgxt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 
lgyt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 
lgzt = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2])) 

time_ind = 0
for path in list_of_paths:
    ncfile = xr.open_dataset(path, mask_and_scale=True, decode_times=True, decode_coords=True)
    time = ncfile.coords['time'][0].values
    print(time, file=f)
    print('reading', VAR_name, ', LGXT, LGYT, LGZT', file=f)
    VAR = ncfile.data_vars[VAR_name][0,:,:,:]
    var[time_ind,:,y0:y1,x0:x1] = VAR[:,y0:y1,x0:x1]
    lgxt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGXT'][0,:,y0:y1,x0:x1]
    lgyt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGYT'][0,:,y0:y1,x0:x1]
    lgzt[time_ind,:,y0:y1,x0:x1] = ncfile.data_vars['LGZT'][0,:,y0:y1,x0:x1]
    time_ind+=1

level_array = ncfile1.level 
level_array = np.asarray(level_array)

print('...OK')
print(" ")
print('...OK', file=f)
print(" ", file=f)

print("#####################################", file=f)
print("1. DETECT OBJECTS FOR EACH TIME STAMP", file=f)
print("#####################################", file=f)
print("#####################################")
print("1. DETECT OBJECTS FOR EACH TIME STAMP")
print("#####################################")

### apply threshold on VAR
labeled_array = np.zeros((nb_time_stamps, VAR.shape[0], VAR.shape[1], VAR.shape[2]), dtype = 'int')
nb_features = []
nb_features_solo = []
num_features_solo = []
objects_location = []

low_thrs = 1
var = np.where(var >= low_thrs, 1, 0)

for time_ind in range(nb_time_stamps):
    print("time stamp : ", time_ind, file=f)
    ### detect and label objects
    print("detect and label objects...", file=f)
    labeled_array[time_ind,:,:,:] = label(var[time_ind,:,:,:])[0] 				# shape (72,502,602)
    nb_features.append(label(var[time_ind,:,:,:])[1]) 
    print("number of objects found : ", nb_features[time_ind], file=f)
    print("...OK", file=f)

    ### find location of objects
    print("find locations of objects...", file=f)
    objects_location += [ndimage.find_objects(labeled_array[time_ind,:,:,:])]
    obj_nb = 0
    nb_obj_solo = 0
    num_features_solo_time_ind = []

    ### find objects composed of only 1 particle    
    for obj in objects_location[time_ind]:
        vertical_extension = np.shape(labeled_array[time_ind,:,:,:][obj])[0]
        y_width = np.shape(labeled_array[time_ind,:,:,:][obj])[1]
        x_width = np.shape(labeled_array[time_ind,:,:,:][obj])[2]
        if (vertical_extension == 1) & (y_width == 1) & (x_width == 1):
            num_features_solo_time_ind.append(obj_nb+1) 
            nb_obj_solo += 1
        obj_nb += 1
    
    nb_features_solo.append(nb_obj_solo)
    num_features_solo += [num_features_solo_time_ind]
         
    print(" ", file=f)

print("...OK")
print("...OK", file=f)

indices_obj_all_time_stamps = []
nb_particles_all_time_stamps = [] 								# nb_particles_all_time_stamps[0] is list of number of particles of each object detected at t0
												# len(nb_particles_all_time_stamps[0]) == nb_features[0]
												# nb_particles_all_time_stamps[1] is list of nb of particles of each objet detected at t+deltat

lgxt_obj, lgyt_obj, lgzt_obj = [], [], []

for time_ind in range(nb_time_stamps):
    indices_obj = [] 		
    nb_particles_obj = []	
    lgxt_obj_time_ind, lgyt_obj_time_ind, lgzt_obj_time_ind = [], [], []
   							
    for obj_nb in range(nb_features[time_ind]):
        indices_obj += [list(zip(*np.where(labeled_array[time_ind,:,:,:] == obj_nb+1)))] 
        nb_particles_obj.append(len(indices_obj[obj_nb]))

        ### get position of objects
        tmp_lgxt_obj_nb, tmp_lgyt_obj_nb, tmp_lgzt_obj_nb = [], [], []
        for ind in indices_obj[obj_nb]:
            tmp_lgxt_obj_nb.append(ind[2]*3000)    
            tmp_lgyt_obj_nb.append(ind[1]*3000)    
            tmp_lgzt_obj_nb.append(level_array[ind[0]])    
    
        lgxt_obj_time_ind += [tmp_lgxt_obj_nb]
        lgyt_obj_time_ind += [tmp_lgyt_obj_nb]
        lgzt_obj_time_ind += [tmp_lgzt_obj_nb]
 
    lgxt_obj += [lgxt_obj_time_ind]
    lgyt_obj += [lgyt_obj_time_ind]
    lgzt_obj += [lgzt_obj_time_ind]

    indices_obj_all_time_stamps += [indices_obj]
    nb_particles_all_time_stamps += [nb_particles_obj]

lgxt_obj = np.array(lgxt_obj)
lgyt_obj = np.array(lgyt_obj)
lgzt_obj = np.array(lgzt_obj)

print("---------------")
print(">>> RESULTS <<<")
print("---------------")
print("---------------", file=f)
print(">>> RESULTS <<<", file=f)
print("---------------", file=f)

nb_mesh_point_zoom_domain = ((x1-x0)+1)*((y1-y0)+1)*72
nb_mesh_point_full_domain = 502*602*72 

for time_ind in range(nb_time_stamps):
    print("time stamp : ", time_ind, file=f)
    for obj_nb in range(nb_features[time_ind]):
        print("object #", obj_nb+1, file=f)
        ratio_zoom_domain = (nb_particles_all_time_stamps[time_ind][obj_nb] * 100) / nb_mesh_point_zoom_domain
        ratio_full_domain = (nb_particles_all_time_stamps[time_ind][obj_nb] * 100) / nb_mesh_point_full_domain
        if (obj_nb+1 in num_features_solo[time_ind]):
            print("    NUMBER OF PARTICLES : ", nb_particles_all_time_stamps[time_ind][obj_nb], "/!\ composed of only 1 particle !", file=f)
        else:
            print("    NUMBER OF PARTICLES : ", nb_particles_all_time_stamps[time_ind][obj_nb], file=f)            
        print("        (", str(ratio_zoom_domain), "% of zoom domain)", file=f)
        print("        (", str(ratio_full_domain), "% of full domain)", file=f)
        
        print(" ", file=f)
        print("    SHAPE : ", file=f)
        print("        width1 : ", max(lgxt_obj[time_ind][obj_nb]) - min(lgxt_obj[time_ind][obj_nb]) +1, "(m)", file=f)
        print("        width2 : ", max(lgyt_obj[time_ind][obj_nb]) - min(lgyt_obj[time_ind][obj_nb]) +1, "(m)",file=f)
        print("        height : ", max(lgzt_obj[time_ind][obj_nb]) - min(lgzt_obj[time_ind][obj_nb]) +1, "(m)",file=f)

        # for each coordinate, interpol on MesoNH mesh
        ind_lon1 = (np.rint(min(lgxt_obj[time_ind][obj_nb])/3000)).astype(int)                             # horizontal resolution is 3km
        ind_lon2 = (np.rint(max(lgxt_obj[time_ind][obj_nb])/3000)).astype(int)                             # horizontal resolution is 3km
        ind_lat1 = (np.rint(min(lgyt_obj[time_ind][obj_nb])/3000)).astype(int)                             # horizontal resolution is 3km
        ind_lat2 = (np.rint(max(lgyt_obj[time_ind][obj_nb])/3000)).astype(int)                             # horizontal resolution is 3km

        print(" ", file=f)
        print("    LOCALISATION : ", file=f)
        print("        LGXT : ", min(lgxt_obj[time_ind][obj_nb]), " - ", max(lgxt_obj[time_ind][obj_nb]), "(m)", file=f)
        print("        longitude : ", longitude[ind_lon1], " - ", longitude[ind_lon2], "(°E)", file=f)
        print("        LGYT : ", min(lgyt_obj[time_ind][obj_nb]), " - ", max(lgyt_obj[time_ind][obj_nb]), "(m)", file=f)
        print("        latitude : ", latitude[ind_lat1], " - ", latitude[ind_lat2], "(°N)", file=f)
        print("        LGZT : ", min(lgzt_obj[time_ind][obj_nb]), "- ", max(lgzt_obj[time_ind][obj_nb]), "(m)", file=f)
        
        print(" ", file=f)
    print("- - -", file=f)

print("number of objects detected at t0 that are composed of only 1 particle : ", num_features_solo[0], ", TOTAL :", nb_features_solo[0], file=f)
print("number of objects detected at t+deltat that are composed of only 1 particle : ", num_features_solo[1], ", TOTAL :", nb_features_solo[1], file=f)

np.seterr(divide='ignore', invalid='ignore')  # for 3d plot, ignore 'nan'
color_sequence = ['black','red','orange','gold','green','lime','cyan','blue','indigo','magenta',\
		'maroon','pink','grey','deeppink','dodgerblue','chocolate','coral','purple','navy','darkorange',\
		'teal','violet','yellowgreen','slateblue','olive','goldenrod', 'slategrey','darkkhaki']

plot1 = 'True'
obj_ = 'convective_cells'

if plot1 == 'True':
    print("--------------------------------------------")
    print("PLOT OBJETS DETECTES A CHAQUE PAS DE TEMPS")
    print("--------------------------------------------")

    for time_ind in range(nb_time_stamps):
        fig = plt.figure(figsize=(14,10))
        ax = plt.axes(projection='3d')
        ax.set_xlim(x0*3000,x1*3000)
        ax.set_ylim(y0*3000,y1*3000)
        ax.set_zlim(0,14000)

        for obj_nb in range(nb_features[time_ind]):
            color_pick = color_sequence[obj_nb]
            ax.scatter3D(lgxt_obj[time_ind][obj_nb], lgyt_obj[time_ind][obj_nb], lgzt_obj[time_ind][obj_nb], c=color_pick, marker='o',label = 'objet' + str(obj_nb+1))

        ax.legend()
        ### add geographical coordinates
        ax.grid(True)
        ax.set_xlabel('LGXT (m)')
        ax.set_ylabel('LGYT (m)')
        ax.set_zlabel('LGZT (m)')

        ### add title
        ax.set_title('Detected ' + object + ' at time stamp t' + str(time_ind))

        ### save as png/pdf
        plt.savefig( 'figure_' + EXP + '_' + N1 + '_' + N2 + '_' + ZOOM_loc + '_detected_' + obj_ + '_t' + str(time_ind) + '.png')

plot_2D = 'True'
lvl = 26
alt = int(np.rint(level_array[lvl]))
ALT_zoom =  ncfile3.data_vars['ALT'][:,y0:y1,x0:x1]

if plot_2D == 'True':
    print("----------------------------------------------------------")
    print("PLOT 2D OBJETS DETECTES + WT OBSERVE A CHAQUE PAS DE TEMPS")
    print("----------------------------------------------------------")

    for time_ind in range(nb_time_stamps):
        fig = plt.figure()
        ax = plt.axes()

        # plot observed WT
        if time_ind == 0:
            WT = ncfile1.data_vars[VAR_name][0,lvl,y0:y1,x0:x1]
        if time_ind == 1:
            WT = ncfile2.data_vars[VAR_name][0,lvl,y0:y1,x0:x1]

        WT.plot.contourf( cmap='RdYlBu', levels=range(-2,3)) # diverging colormap

        for obj_nb in range(nb_features[time_ind]):
            color_pick = color_sequence[obj_nb]
            num_ind = 0
            for ind in indices_obj_all_time_stamps[time_ind][obj_nb]:
                if ind[0] == 26:
                    ax.scatter(lgxt_obj[time_ind][obj_nb][num_ind], lgyt_obj[time_ind][obj_nb][num_ind], c=color_pick, marker='o')
                num_ind += 1

        ### add geographical coordinates
        ax.grid(True)
        ax.set_xlabel('LGXT (m)')
        ax.set_ylabel('LGYT (m)')
 
        ### add title
        ax.set_title('Detected ' + object + ' at time stamp t' + str(time_ind) + ' (alt:' + str(alt) +'m)')

        ### save as png/pdf
        plt.savefig( 'figure_' + EXP + '_' + N1 + '_' + N2 + '_' + ZOOM_loc + '_' + obj_ + '_t' + str(time_ind) +'_' + str(alt) + 'm.png')

print("-------------------------------------------------------------------------------------")
print("FIND TRAJECTORIES DURING [T0 ; T+DELTAT] OF PARTICLES OF OBJECTS DETECTED AT T+DELTAT")
print("-------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------", file=f)
print("FIND TRAJECTORIES DURING [T0 ; T+DELTAT] OF PARTICLES OF OBJECTS DETECTED AT T+DELTAT", file=f)
print("-------------------------------------------------------------------------------------", file=f)

### get real positions of particles during the 2 selected time stamps
print("get real positions of all particles from objects detected at t+deltat", file=f)
good_lgxt, good_lgyt, good_lgzt = [], [], []

for time_ind in range(nb_time_stamps):
    print("get positions at t", time_ind, file=f)
    good_lgxt_time_ind, good_lgyt_time_ind, good_lgzt_time_ind = [], [], []

    for obj_nb in range(nb_features[1]):
        tmp_lgxt_obj_nb, tmp_lgyt_obj_nb, tmp_lgzt_obj_nb = [], [], []
        for ind in indices_obj_all_time_stamps[1][obj_nb]:
            if time_ind == 0:
                tmp_lgxt_obj_nb.append(lgxt[time_ind+1,ind[0],ind[1],ind[2]])    
                tmp_lgyt_obj_nb.append(lgyt[time_ind+1,ind[0],ind[1],ind[2]])    
                tmp_lgzt_obj_nb.append(lgzt[time_ind+1,ind[0],ind[1],ind[2]])        
            else:
                tmp_lgxt_obj_nb.append(ind[2]*3000)    
                tmp_lgyt_obj_nb.append(ind[1]*3000)    
                tmp_lgzt_obj_nb.append(level_array[ind[0]])        

        good_lgxt_time_ind += [tmp_lgxt_obj_nb]
        good_lgyt_time_ind += [tmp_lgyt_obj_nb]
        good_lgzt_time_ind += [tmp_lgzt_obj_nb]
 
    good_lgxt += [good_lgxt_time_ind]
    good_lgyt += [good_lgyt_time_ind]
    good_lgzt += [good_lgzt_time_ind]

good_lgxt = np.array(good_lgxt)
good_lgyt = np.array(good_lgyt)
good_lgzt = np.array(good_lgzt)

print("...OK", file=f)

plot2 = 'True'

if plot2 == 'True':
    print("-------")
    print("PLOT 3D")
    print("-------")

    fig = plt.figure(figsize=(14,10))
    ax = plt.axes(projection='3d')
    ax.set_xlim(x0*3000,x1*3000)
    ax.set_ylim(y0*3000,y1*3000)
    ax.set_zlim(0,14000)

    for obj_nb in range(nb_features[1]):
        print("object #", obj_nb+1)    
        color_pick = color_sequence[obj_nb]
        for time_ind in range(nb_time_stamps):
            if time_ind == 0:
                ax.scatter3D(good_lgxt[time_ind][obj_nb], good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='^', label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
            else:
                ax.scatter3D(good_lgxt[time_ind][obj_nb], good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='o',label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
       
            if time_ind + 2 <= nb_time_stamps:
                # trajectories of each particule of obj_nb (during t0 and t+deltat)
                for a,b,c,d,e,g in zip(good_lgxt[time_ind][obj_nb], good_lgxt[time_ind+1][obj_nb],\
					good_lgyt[time_ind][obj_nb], good_lgyt[time_ind+1][obj_nb],\
					good_lgzt[time_ind][obj_nb], good_lgzt[time_ind+1][obj_nb]):

                    ax.plot([a,b], [c,d], [e,g], color = 'grey')

    ax.legend()
    ### add geographical coordinates
    ax.grid(True)
    ax.set_xlabel('LGXT (m)')
    ax.set_ylabel('LGYT (m)')
    ax.set_zlabel('LGZT (m)')
    ### add title
    ax.set_title('Trajectories during 15:00-15:55 of particles from detected ' + object + ' at 15:55')
    ### save as png/pdf
    plt.savefig( 'figure_' + EXP + '_' +  N1 + '_' + N2 + '_' + ZOOM_loc + '_all_trajectories.png')


plot2_each_obj = 'True'

if plot2_each_obj == 'True':
    print("----------------------")
    print("PLOT 3D OF EACH OBJECT")
    print("----------------------")

    for obj_nb in range(nb_features[1]):
        print("object #", obj_nb+1)    
        color_pick = color_sequence[obj_nb]
        fig = plt.figure(figsize=(14,10))
        ax = plt.axes(projection='3d')
        ax.set_xlim(x0*3000,x1*3000)
        ax.set_ylim(y0*3000,y1*3000)
        ax.set_zlim(0,14000)
        for time_ind in range(nb_time_stamps):
            if time_ind == 0:
                ax.scatter3D(good_lgxt[time_ind][obj_nb], good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='^', label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
            else:
                ax.scatter3D(good_lgxt[time_ind][obj_nb], good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='o',label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
       
            if time_ind + 2 <= nb_time_stamps:
                # trajectories of each particule of obj_nb (during t0 and t+deltat)
                for a,b,c,d,e,g in zip(good_lgxt[time_ind][obj_nb], good_lgxt[time_ind+1][obj_nb],\
					good_lgyt[time_ind][obj_nb], good_lgyt[time_ind+1][obj_nb],\
					good_lgzt[time_ind][obj_nb], good_lgzt[time_ind+1][obj_nb]):

                    ax.plot([a,b], [c,d], [e,g], color = 'grey')

        ax.legend()
        ### add geographical coordinates
        ax.grid(True)
        ax.set_xlabel('LGXT (m)')
        ax.set_ylabel('LGYT (m)')
        ax.set_zlabel('LGZT (m)')
        ### add title
        ax.set_title('Trajectories during 15:00-15:55 of particles from detected ' + object + ' #' + str(obj_nb+1) + ' at 15:55')
        ### save as png/pdf
        plt.savefig( 'figure_' + EXP + '_' +  N1 + '_' + N2 + '_' + ZOOM_loc + '_obj' + str(obj_nb+1) + '_trajectory.png')


plot3 = 'True'

if plot3 == 'True':
    print("----------------")
    print("VERTICAL SECTION")
    print("----------------")
    for obj_nb in range(nb_features[1]):
        print("object #", obj_nb+1)
        fig = plt.figure(figsize=(14,10))
        ax = plt.axes()
        ax.set_xlim(800000,1000000)
        ax.set_ylim(0,16000)
    
        color_pick = color_sequence[obj_nb]
        for time_ind in range(nb_time_stamps):
            if time_ind == 0:
                ax.scatter(good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='^', label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
            else:
                ax.scatter(good_lgyt[time_ind][obj_nb], good_lgzt[time_ind][obj_nb], c=color_pick, marker='o',label = 'objet' + str(obj_nb+1) + ' t' + str(time_ind))
       
            if time_ind + 2 <= nb_time_stamps:
                # trajectories of each particule of obj_nb (time_ind - time_ind+1)
                for a,b,c,d in zip(good_lgyt[time_ind][obj_nb], good_lgyt[time_ind+1][obj_nb],\
				good_lgzt[time_ind][obj_nb], good_lgzt[time_ind+1][obj_nb]):

                    ax.plot([a,b], [c,d], color = 'grey')

        ax.legend()
        ### add geographical coordinates
        ax.grid(True)
        ax.set_xlabel('LGYT (m)')
        ax.set_ylabel('LGZT (m)')
        ### add title
        ax.set_title('Trajectories during 15:00-15:55 of particles from detected ' + object + ' #' + str(obj_nb+1) + ' at 15:55 (vertical section)')
        ### save as png/pdf
        plt.savefig( 'figure_' + EXP + '_' +  N1 + '_' + N2 + '_' + ZOOM_loc + '_obj' + str(obj_nb+1) + '_trajectory_vertical_section.png')


print(" ", file=f)
print("##########################################################")
print("2. OBJET IDENTIFICATION BETWEEN THE 2 SELECTED TIME STAMPS")
print("##########################################################")
print("##########################################################", file=f)
print("2. OBJET IDENTIFICATION BETWEEN THE 2 SELECTED TIME STAMPS", file=f)
print("##########################################################", file=f)
### aim of this part is to identify which structures represent the same object but at different time stamps

print("--------------------------------------------------------------------------")
print("GET ORIGINAL POSITION (AT T0) OF PARTICLES OF OBJECTS DETECTED AT T+DELTAT")
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------", file=f)
print("GET ORIGINAL POSITION (AT T0) OF PARTICLES OF OBJECTS DETECTED AT T+DELTAT", file=f)
print("--------------------------------------------------------------------------", file=f)
### for each object detected at t+deltat and for each of its particle, we get the position of the particle at t0 
### a new 3D variable called "ratio" is created
### with this variables, we mark the origin position (at t0) of all particles and how many particles are found on each mesh point 
### lastly, for each particle we check if its position at t0 is within an object detected at t0

nb_particles_IN = []
indices_positive_ratios = []
indices_double_cond = []
num_origin_all_objects = [] 										# list of lists.
													# if the ith particule of object #a (numerotation at t+deltat) came from object #b (numerotation at t0),
													# num_obj[a-1][i-1] returns b-1

num_obj_with_origin = [] 										# list of numeros of objects at t+deltat that have origin at t0
ratios_array = np.zeros((nb_features[1], VAR.shape[0], VAR.shape[1], VAR.shape[2]), dtype = 'int')   	# 27 72 502 602

lgxt_positive_ratios, lgyt_positive_ratios, lgzt_positive_ratios = [], [], []

for obj_nb in range(nb_features[1]):
    tmp_lgxt_ratios, tmp_lgyt_ratios, tmp_lgzt_ratios = [], [], [] 
    num_origin_obj = []
    nb_particles_IN_obj = []

    ### SEE WHERE PARTICLES OF OBJECTS DETECTED AT T+DELTAT ARE COMING FROM AT T0
    ### because their position at t0 is not on a mesh point, we need to interpolate the position on the closest mesh point
    for a,b,c in zip(good_lgxt[0][obj_nb], good_lgyt[0][obj_nb], good_lgzt[0][obj_nb]):
        # horizontal interpolation
        x_ind = (np.rint(a/3000)).astype(int)                             # horizontal resolution is 3km
        y_ind = (np.rint(b/3000)).astype(int)    
        # vertical interpolation
        idx_level = (np.abs(level_array - c)).argmin()
        z_ind = idx_level

        #### we count how many particles are found for each mesh point
        ratios_array[obj_nb, z_ind, y_ind, x_ind] += 1									# ON A BIEN np.sum(ratios_array[obj_nb,:,:,:]) == nb_particules_all_time_stamps[1][obj_nb] 
        
	### we convert the position (given by index) to lgxt/lgyt/lgzt positions (meters)
        tmp_lgxt_ratios.append(x_ind*3000)
        tmp_lgyt_ratios.append(y_ind*3000)
        tmp_lgzt_ratios.append(level_array[idx_level])

    ### we save the lgxt/lgyt/lgzt positions
    lgxt_positive_ratios += [tmp_lgxt_ratios]
    lgyt_positive_ratios += [tmp_lgyt_ratios]
    lgzt_positive_ratios += [tmp_lgzt_ratios]

    ### WE CHECK IF THE ORIGIN POSITION (AT T0) OF PARTICLES IS FOUND IN OBJECTS DETECTED AT T0
    indices_positive_ratios += [list(zip(*np.where(ratios_array[obj_nb,:,:,:] > 0)))]
    indices_double_cond += [list(zip(*np.where((ratios_array[obj_nb,:,:,:] > 0) & (labeled_array[0,:,:,:] >= 1))))]     

    ### if yes, we get the number of the origin object for each mesh point and get how many particles were interpolated on this mesh point
    if (len(indices_double_cond[obj_nb]) != 0):
        num_obj_with_origin.append(obj_nb+1)
        for ind in indices_double_cond[obj_nb]:
            num_origin_obj.append(labeled_array[0,ind[0], ind[1], ind[2]])
            nb_particles_IN_obj.append(ratios_array[obj_nb, ind[0], ind[1], ind[2]])

    nb_particles_IN += [nb_particles_IN_obj]
    num_origin_all_objects += [num_origin_obj]

print("...OK")
print("...OK", file=f)

print("---------------")
print(">>> RESULTS <<<")
print("---------------")
print("---------------", file=f)
print(">>> RESULTS <<<", file=f)
print("---------------", file=f)
print("number of objets detected at t+deltat that have particles originating from objets detected at t0 : ", num_obj_with_origin, file=f)

for obj_nb in num_obj_with_origin:
    print("object #", obj_nb, " detected at t+deltat", file=f)
    print("    number of particles within object : ", nb_particles_all_time_stamps[1][obj_nb-1], file=f)
    ratio_old_particles = (sum(nb_particles_IN[obj_nb-1]) * 100) / nb_particles_all_time_stamps[1][obj_nb-1]
    print("    number of particles originating from objects detected at t0 :", sum(nb_particles_IN[obj_nb-1]), "(", str(ratio_old_particles), "%)", file=f)

    for obj_origin_nb in range(nb_features[1]):
        nb_particles_each_origin = 0

        if obj_origin_nb+1 in num_origin_all_objects[obj_nb-1]: ### first check if the object number if found in list of origin objects for #obj_nb detected at t+deltat
            idx = [i for i, e in enumerate(num_origin_all_objects[obj_nb-1]) if e == obj_origin_nb+1]
            for ind in idx:
                nb_particles_each_origin += nb_particles_IN[obj_nb-1][ind]
                ratio_each_origin = (nb_particles_each_origin*100) / nb_particles_all_time_stamps[1][obj_nb-1]
            print("        ", nb_particles_each_origin, " particle(s) originate from object # ", obj_origin_nb+1, "detected at t0, (", ratio_each_origin  ,"% of #" , obj_nb, ")", file=f)

    print(" ", file=f)

### close file for output redirecting
f.close()

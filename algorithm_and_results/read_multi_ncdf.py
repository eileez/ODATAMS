# author : Elizabeth Fu
# elizabeth.wt.fu@gmail.com
# Institut National Sciences Appliquées (INSA) Toulouse - Laboratoire d'Aérologie, Toulouse
# june 2020 - sept 2020

import xarray as xr
import netCDF4 as nc
import glob

list_of_paths = glob.glob('/tmpdir/fu/OUT/5_6_UTC/CNTRL.1.SEG01.OUT.0??.nc')
list_of_paths.sort()

for path in list_of_paths:
    print(path)
    ncfile = xr.open_dataset(path, mask_and_scale=True, decode_times=True, decode_coords=True)
    time = ncfile.coords['time'][0].values
    print(time)

#############################Import of packages######################################################
import pandas as pd
import pygrib
from scipy.io import netcdf
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap  # import Basemap matplotlib toolkit
import numpy as np
import cdsapi
from math import sin, cos, sqrt, atan2, radians

import scipy.spatial
import time

import os

from numpy import deg2rad, rad2deg


from datetime import datetime, time, timedelta, timezone



########################Download & Data Wrangling#####################################################
def download_era(era_path, start_year, end_year, grid_pts):
    """
    Download of era5 weather data 
    Parameters
    ----------
    era_path:       path of weather data if available/path where weather data should be saved
    start_year:     start of time period's year which should be calculated
    end_year:       end of time period's year which should be calculated   
    grid_pts:       grid points of the european transmission network
    Returns
    -------
        
    """
    
    if os.path.exists(era_path):
        return

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': ['100m_v_component_of_wind', '100m_u_component_of_wind', '2m_temperature', 
                        'forecast_albedo', 'forecast_surface_roughness', 'surface_solar_radiation_downwards', 
                        'total_sky_direct_solar_radiation_at_surface'],           
            'year': list(range(start_year, end_year+1)),
            'month': [str(month) for month in range(1, 13)],
            'day': [str(day) for day in range(1, 31)],
            'time': [f"{str(time).rjust(2, '0')}:00" for time in range(24)],
            'area': [grid_pts.latitude.max() + 1, grid_pts.longitude.min() - 1, grid_pts.latitude.min() - 1,
                     grid_pts.longitude.max() + 1]
        },
        era_path)



def convert_era(relevant_era_path, era_path, gebco_file_path, start_year, end_year, grid_pts, sea_depth):
    """
    
    Converts the downloaded weather data from grib-format and lists to an dataframe 
    
    
    Parameters
    ----------
    relevant_era_path   path where converted weather data should be saved
    era_path:           path of weather data if available/path where weather data should be saved
    gebco_file_path     path of gebco data, containing sea depth information
    start_year:         start of time period's year which should be calculated
    end_year:           end of time period's year which should be calculated   
    grid_pts:           grid points of the european transmission network
    sea_depth:          Water depth limiting the possible places to install offshore wind turbines
        
    Returns
    -------
    relevant_era5_df:       Dataframe, containing all relevant data of the downloaded weather data
        
    """
    
    global relevant_era5_df

    download_era(era_path, start_year, end_year, grid_pts)
    era5_data = pygrib.open(era_path)
    era5_data.seek(0)

    era5_lats, era5_lons = era5_data.select()[0].latlons()
    era5_lons[era5_lons > 180] = era5_lons[era5_lons > 180] - 360 #Transform 0 - 360 to -180 to 180
    v_wind_list = era5_data.select(name = '100 metre V wind component')
    u_wind_list = era5_data.select(name = '100 metre U wind component')
    surface_list = era5_data.select(name = 'Forecast surface roughness')
    
    temperature_list = era5_data.select(name = '2 metre temperature')
    albedo_list = era5_data.select(name = 'Forecast albedo')
    solar_down_list = era5_data.select(name = 'Surface solar radiation downwards')
    solar_direct_list = era5_data.select(name = 'Total sky direct solar radiation at surface')
    print('lists selected')
    
    # Transform data
    file2read = netcdf.NetCDFFile(gebco_file_path, 'r')
    lat = file2read.variables['lat'][:] * 1 # select lats (array)
    lon = file2read.variables['lon'][:] * 1 # select lons (array)
    elevation = file2read.variables['elevation'][:] * 1 # select elevations (matrix)
    elevation_df = pd.DataFrame(elevation, columns=np.round(lon, 2), index=np.round(lat, 2)) # elevation matrix to dataframe, column names are lons, index names are lats (rounded)
    col_gr = elevation_df.groupby(elevation_df.columns, axis=1).mean() # group by columns, then take mean
    elevation_grouped = col_gr.groupby(col_gr.index, axis=0).mean() # group by index, so that both columns and indexes are grouped

    water_depth = sea_depth # water depth, can be adjusted

    elevation_mask = elevation_grouped >= -water_depth # create mask for grouped dataframe
    elevation_mask_fit = elevation_mask.loc[np.round(era5_lats[:, 0], 2), np.round(era5_lons[0, :], 2)] # select only the needed columns (for era values also rounded by 2 digits)
    elevation_mask_reshaped = elevation_mask_fit.values.reshape(
        (elevation_mask_fit.shape[0] * elevation_mask_fit.shape[1],)) # reshape the elevation mask into array

    lonlat = np.array([era5_lats.reshape((era5_lats.shape[0] * era5_lats.shape[1],)), era5_lons.reshape((era5_lons.shape[0] * era5_lons.shape[1],))]).T
    bm = Basemap()
    island = pd.DataFrame(lonlat, columns=['latitude', 'longitude']).apply(lambda x:bm.is_land(x.longitude, x.latitude), axis=1).map(int)
    island = np.array(island).reshape((-1,))

    v_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in v_wind_list))
    u_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in u_wind_list))
    wind_values = np.sqrt(np.square(v_values) + np.square(u_values)).astype('float32')

    solar_down_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in solar_down_list))
    solar_direct_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in solar_direct_list))
    temperature_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in temperature_list))
    surface_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in surface_list))
    albedo_values = np.concatenate(list(x.values.astype('float32').reshape((-1,)) for x in albedo_list))

    date = np.array(list(np.datetime64(str(x.validDate)) for x in v_wind_list)).repeat(
        (v_wind_list[0].values.shape[0] * v_wind_list[0].values.shape[1]))
    elevation_mask_reshaped_complete = np.tile(elevation_mask_reshaped, (len(v_wind_list),))
    island_complete = np.tile(island, len(v_wind_list))
    lats_complete = np.tile(era5_lats.reshape((-1,)), (len(v_wind_list),))
    lons_complete = np.tile(era5_lons.reshape((-1,)), (len(v_wind_list),))
    print('values tranformed')
    
    
    
    lats_reshaped = era5_lats.reshape((era5_lats.shape[0] * era5_lats.shape[1]), )
    lats_fit = lats_reshaped[elevation_mask_reshaped]
    lons_reshaped = era5_lons.reshape((era5_lons.shape[0] * era5_lons.shape[1]), )
    lons_fit = lons_reshaped[elevation_mask_reshaped]
    dist_matrix = scipy.spatial.distance.cdist(np.array([lats_fit, lons_fit]).T,
                                               grid_pts[['latitude', 'longitude']].values,
                                               get_distance)
    dist_grid = scipy.spatial.distance.cdist(grid_pts[['latitude', 'longitude']].values,
                                      grid_pts[['latitude', 'longitude']].values,
                                      get_distance)
    dist_grid[dist_grid == 0] = np.inf
    dist_argmin = dist_grid.min(axis=1)

    max_min_dist_era5 = max(dist_argmin)

    dist_bool = dist_matrix.min(axis=1) < max_min_dist_era5

    dist_argmin = dist_matrix.argmin(axis=1)[dist_bool]
    num_periods = len(v_wind_list)

    dist_bool = np.tile(dist_bool, num_periods)
    dist_argmin = np.tile(dist_argmin, num_periods)

    lats_complete_masked = lats_complete[elevation_mask_reshaped_complete]
    lons_complete_masked = lons_complete[elevation_mask_reshaped_complete]

    wind_values_masked = wind_values[elevation_mask_reshaped_complete]
    solar_down_values_masked = solar_down_values[elevation_mask_reshaped_complete]
    solar_direct_values_masked = solar_direct_values[elevation_mask_reshaped_complete]
    temperature_values_masked = temperature_values[elevation_mask_reshaped_complete]
    surface_values_masked = surface_values[elevation_mask_reshaped_complete]
    albedo_values_masked = albedo_values[elevation_mask_reshaped_complete]
    date_masked = date[elevation_mask_reshaped_complete]
    island_complete_masked = island_complete[elevation_mask_reshaped_complete]

    lats_complete_masked_relevant = lats_complete_masked[dist_bool]
    lons_complete_masked_relevant = lons_complete_masked[dist_bool]

    wind_values_masked_relevant = wind_values_masked[dist_bool]
    solar_down_values_masked_relevant = solar_down_values_masked[dist_bool]
    solar_direct_values_masked_relevant = solar_direct_values_masked[dist_bool]
    temperature_values_masked_relevant = temperature_values_masked[dist_bool]
    surface_values_masked_relevant = surface_values_masked[dist_bool]
    albedo_values_masked_relevant = albedo_values_masked[dist_bool]
    date_masked_relevant = date_masked[dist_bool]
    island_complete_masked_relevant = island_complete_masked[dist_bool]

    grid_pts_lat = grid_pts[['latitude']].values[dist_argmin].reshape((-1,))
    grid_pts_lon = grid_pts[['longitude']].values[dist_argmin].reshape((-1,))

    ## get missing grid points
    missing_grids = [x for x in np.linspace(0, dist_grid.shape[0] -1, dist_grid.shape[0], dtype=int) if x not in dist_argmin]
    missing_lat = grid_pts.loc[missing_grids, 'latitude'].values
    missing_lon = grid_pts.loc[missing_grids, 'longitude'].values

    for i in range(len(missing_grids)):
        lon_missing = lons_complete_masked_relevant[dist_matrix[:, missing_grids].argmin(axis=0)][i]
        lat_missing = lats_complete_masked_relevant[dist_matrix[:, missing_grids].argmin(axis=0)][i]

        lats_complete_masked_part = lats_complete_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        lons_complete_masked_part = lons_complete_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        wind_values_masked_part = wind_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        solar_down_values_masked_part = solar_down_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        solar_direct_values_masked_part = solar_direct_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]    
        temperature_values_masked_part = temperature_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        surface_values_masked_part = surface_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        albedo_values_masked_part = albedo_values_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        date_masked_part = date_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]
        island_complete_masked_part = island_complete_masked_relevant[
            (lons_complete_masked_relevant == lon_missing) & (lats_complete_masked_relevant == lat_missing)]

        array_lat_grid = np.full((lats_complete_masked_part.shape[0],), missing_lat[i])
        array_lon_grid = np.full((lons_complete_masked_part.shape[0],), missing_lon[i])

        if i == 0:
            missing_array_lat_grid = array_lat_grid
            missing_array_lon_grid = array_lon_grid
            missing_lats_complete_masked_part = lats_complete_masked_part
            missing_lons_complete_masked_part = lons_complete_masked_part
            missing_wind_values_masked_part = wind_values_masked_part
            missing_solar_down_values_masked_part = solar_down_values_masked_part
            missing_solar_direct_values_masked_part = solar_direct_values_masked_part
            missing_temperature_values_masked_part = temperature_values_masked_part
            missing_surface_values_masked_part = surface_values_masked_part
            missing_albedo_values_masked_part = albedo_values_masked_part
            missing_date_masked_part = date_masked_part
            missing_island_complete_masked_part = island_complete_masked_part

            # missing_array = np.hstack([part_array, np.full((part_array.shape[0], 1), missing_lat[i]), np.full((part_array.shape[0], 1), missing_lon[i])])
        else:

            missing_array_lat_grid = np.concatenate([missing_array_lat_grid, array_lat_grid])
            missing_array_lon_grid = np.concatenate([missing_array_lon_grid, array_lon_grid])
            missing_lats_complete_masked_part = np.concatenate(
                [missing_lats_complete_masked_part, lats_complete_masked_part])
            missing_lons_complete_masked_part = np.concatenate(
                [missing_lons_complete_masked_part, lons_complete_masked_part])
            missing_wind_values_masked_part = np.concatenate(
                [missing_wind_values_masked_part, wind_values_masked_part])
            missing_solar_down_values_masked_part = np.concatenate(
                [missing_solar_down_values_masked_part, solar_down_values_masked_part])
            missing_solar_direct_values_masked_part = np.concatenate(
                [missing_solar_direct_values_masked_part, solar_direct_values_masked_part])
            missing_temperature_values_masked_part = np.concatenate(
                [missing_temperature_values_masked_part, temperature_values_masked_part])
            missing_surface_values_masked_part = np.concatenate(
                [missing_surface_values_masked_part, surface_values_masked_part])
            missing_albedo_values_masked_part = np.concatenate(
                [missing_albedo_values_masked_part, albedo_values_masked_part])
            missing_date_masked_part = np.concatenate([missing_date_masked_part, date_masked_part])
            missing_island_complete_masked_part = np.concatenate(
                [missing_island_complete_masked_part, island_complete_masked_part])

    grid_pts_lat_missed = np.concatenate([grid_pts_lat, missing_array_lat_grid])
    grid_pts_lon_missed = np.concatenate([grid_pts_lon, missing_array_lon_grid])
    lats_complete_masked_relevant_missed = np.concatenate(
        [lats_complete_masked_relevant, missing_lats_complete_masked_part])
    lons_complete_masked_relevant_missed = np.concatenate(
        [lons_complete_masked_relevant, missing_lons_complete_masked_part])
    wind_values_masked_relevant_missed = np.concatenate(
        [wind_values_masked_relevant, missing_wind_values_masked_part])
    solar_down_values_masked_relevant_missed = np.concatenate(
        [solar_down_values_masked_relevant, missing_solar_down_values_masked_part])
    solar_direct_values_masked_relevant_missed = np.concatenate(
        [solar_direct_values_masked_relevant, missing_solar_direct_values_masked_part])
    temperature_values_masked_relevant_missed = np.concatenate(
        [temperature_values_masked_relevant, missing_temperature_values_masked_part])
    surface_values_masked_relevant_missed = np.concatenate(
        [surface_values_masked_relevant, missing_surface_values_masked_part])
    albedo_values_masked_relevant_missed = np.concatenate(
        [albedo_values_masked_relevant, missing_albedo_values_masked_part])
    date_masked_relevant_missed = np.concatenate([date_masked_relevant, missing_date_masked_part])
    island_complete_masked_relevant_missed = np.concatenate(
        [island_complete_masked_relevant, missing_island_complete_masked_part])

    print('missing grids added')

    del era5_data, grid_pts_lat, grid_pts_lon, missing_array_lat_grid, lats_complete_masked_relevant, missing_lats_complete_masked_part, lons_complete_masked_relevant, missing_lons_complete_masked_part
    del wind_values_masked_relevant, missing_wind_values_masked_part, solar_down_values_masked_relevant, missing_solar_down_values_masked_part, solar_direct_values_masked_relevant, missing_solar_direct_values_masked_part
    del temperature_values_masked_relevant, missing_temperature_values_masked_part, surface_values_masked_relevant, missing_surface_values_masked_part, albedo_values_masked_relevant, missing_albedo_values_masked_part
    del date_masked_relevant, missing_date_masked_part, island_complete_masked_relevant, missing_island_complete_masked_part
    del array_lat_grid, array_lon_grid, lats_complete_masked_part, lons_complete_masked_part, wind_values_masked_part, solar_down_values_masked_part, solar_direct_values_masked_part
    del temperature_values_masked_part, surface_values_masked_part, albedo_values_masked_part, date_masked_part, island_complete_masked_part

    del era5_lats, era5_lons, v_wind_list, u_wind_list, solar_down_list, temperature_list, surface_list, albedo_list, solar_direct_list, file2read, lat, lon, elevation, elevation_df
    del elevation_grouped, elevation_mask, elevation_mask_fit, elevation_mask_reshaped, lonlat, island, v_values, u_values, wind_values, solar_down_values, solar_direct_values
    del temperature_values, surface_values, albedo_values, elevation_mask_reshaped_complete, island_complete, lats_complete, lons_complete
    
    del wind_values_masked, solar_down_values_masked, solar_direct_values_masked, temperature_values_masked, surface_values_masked, albedo_values_masked, date_masked, island_complete_masked
    del lats_complete_masked, lons_complete_masked, dist_argmin, dist_bool, date
    columns = ['date', 'longitude', 'latitude', 'wind_velocity_100', 'solardown_radiation', 'solardirect_radiation', 'temperature', 'surface',
                   'albedo', 'is_land', 'grid_point_lat', 'grid_point_lon']
    relevant_era5_df = pd.DataFrame(np.array(
            [date_masked_relevant_missed, lons_complete_masked_relevant_missed, lats_complete_masked_relevant_missed,
             wind_values_masked_relevant_missed, solar_down_values_masked_relevant_missed,
             solar_direct_values_masked_relevant_missed, temperature_values_masked_relevant_missed, 
             surface_values_masked_relevant_missed, albedo_values_masked_relevant_missed,
             island_complete_masked_relevant_missed, grid_pts_lat_missed, grid_pts_lon_missed]).T, columns=columns)
    print('dataframe built')
    del date_masked_relevant_missed, lons_complete_masked_relevant_missed, lats_complete_masked_relevant_missed, wind_values_masked_relevant_missed, solar_down_values_masked_relevant_missed
    del solar_direct_values_masked_relevant_missed, temperature_values_masked_relevant_missed, surface_values_masked_relevant_missed, albedo_values_masked_relevant_missed
    del island_complete_masked_relevant_missed, grid_pts_lat_missed, grid_pts_lon_missed
    
    relevant_era5_df.to_csv(relevant_era_path, index=False, sep = ';')
    print('dataframe stored') 
    
    return relevant_era5_df


# Distance calculation
def get_distance(point1, point2):
    """
    Calculation of the distance between two points, which are given in degree
    """
    
    R = 6370
    lat1 = radians(point1[0])  #insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance



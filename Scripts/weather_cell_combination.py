#############################Import of packages######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
#from geopy.distance import geodesic

import time

import matplotlib.colors as mcolors
import matplotlib
matplotlib.style.use('default')
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, timezone
                        

#############################Cell to node conversion######################################################
def era_to_grid_grouper(relevant_era5_df, grid_pts, start_year, save_signal_path):
    """
    
    Groups weather cells by grid points: 
        - every weather cell is allocated to its nearest grid point
        - for each grid point, the signals of the weather cells are averaged
    
    
    Parameters
    ----------
    relevant_era5_df:   Dataframe containing all relevant information (weather data, relative and absolute signals, grid points)
    grid_pts:           latitude and longitude of transmission grid nodes
    start_year:         Year of first observed time point 
    save_signal_path:   Path for saving the grouped signals 
        
    Returns
    -------
    grids_wide:             Signals (wind, solar) for every grid point 
    grids_wide_onshore      Signals (onshore) for grid points on land
    grids_wide_offshore     Signals (offshore) for grid points only in sea
    grids_wide_solar_land   Signals (solar) for grid points on land
    relevant_era5_df        Dataframe containing all relevant information (weather data, relative and absolute signals, grid points)
    """
    
    global grids
    global grids_wide
    global grids_wide_onshore
    global grids_wide_offshore
    global grids_wide_solar_land
    global grids_wide_solar_land01
    global grids_wide_solar_land02
    global grids_wide_solar_land005
    global relevant_era5_df

    relevant_era5_df['solar_power_abs'].fillna(0)
    relevant_era5_df['solar_power_rel'].fillna(0)
    
    #relevant_era5_df['solar_power_abs'] = 0
    #relevant_era5_df['solar_power_rel'] = 0
    relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'solar_power_abs'] = np.nan
    relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'solar_power_rel'] = np.nan

    relevant_grids = relevant_era5_df.groupby(['grid_point_lat', 'grid_point_lon', 'date']).mean()[['wind_power_rel',  'solar_power_rel']].reset_index()#'wind_power_abs','solar_power_abs',
        
    relevant_era5_df[relevant_era5_df.date == relevant_era5_df.date[0]].is_land.value_counts()
    relevant_era5_df[relevant_era5_df.date == relevant_era5_df.date[0]]
    
    relevant_grids.fillna(0, inplace= True)
    

    grids_all = relevant_grids.merge(grid_pts[['latitude', 'longitude', 'country']], left_on=['grid_point_lat', 'grid_point_lon'], right_on=['latitude', 'longitude'], how = 'outer')
    #grids.to_csv('grids.csv', sep = ';', index = False)
    
    # Grids all data
    grids = grids_all[['country', 'date', 'grid_point_lat', 'grid_point_lon', 'wind_power_rel', 'solar_power_rel']]
    grids_wide = pd.pivot(grids, index=['date', 'country'], columns=['grid_point_lat', 'grid_point_lon'], values = ['wind_power_rel', 'solar_power_rel']).reset_index()
    grids_wide.date = pd.to_datetime(grids_wide.date, utc=True)
    
    ############################################################################################
    # Grids separated matrices for wind onshore, wind offshore, solar - export data in csv
    
    relevant_grids_landsea = relevant_era5_df.groupby(['is_land', 'grid_point_lat', 'grid_point_lon', 'date']).mean()[['wind_power_rel',  'solar_power_rel']].reset_index()#'wind_power_abs','solar_power_abs',
    relevant_grids_landsea.fillna(0, inplace= True)
    grids_landsea_all = relevant_grids_landsea.merge(grid_pts[['latitude', 'longitude', 'country']], left_on=['grid_point_lat', 'grid_point_lon'], right_on=['latitude', 'longitude'], how = 'outer')
    grids_land = grids_landsea_all[grids_landsea_all.is_land == 1]
    grids_land = grids_land[['country', 'date', 'grid_point_lat', 'grid_point_lon', 'wind_power_rel', 'solar_power_rel']]
    grids_sea = grids_landsea_all[grids_landsea_all.is_land == 0]
    grids_sea = grids_sea[['country', 'date', 'grid_point_lat', 'grid_point_lon', 'wind_power_rel']]
    grids_wide_onshore = pd.pivot(grids_land, index=['date'], columns=['country','grid_point_lat', 'grid_point_lon'], values = ['wind_power_rel']).reset_index()
    grids_wide_onshore.date = pd.to_datetime(grids_wide_onshore.date, utc=True)
    grids_wide_offshore = pd.pivot(grids_sea, index=['date'], columns=['country','grid_point_lat', 'grid_point_lon'], values = ['wind_power_rel']).reset_index()
    grids_wide_offshore.date = pd.to_datetime(grids_wide_offshore.date, utc=True)
    grids_wide_solar_land = pd.pivot(grids_land, index=['date'], columns=['country','grid_point_lat', 'grid_point_lon'], values = ['solar_power_rel']).reset_index()
    grids_wide_solar_land.date = pd.to_datetime(grids_wide_solar_land.date, utc=True)
    
    grids_wide_onshore.to_csv(save_signal_path + f'grids_wide_{start_year}_onshore_power_rel.csv', sep = ';', index=False)
    grids_wide_offshore.to_csv(save_signal_path + f'grids_wide_{start_year}_offshore_power_rel.csv', sep = ';', index=False)
    grids_wide_solar_land.to_csv(save_signal_path + f'grids_wide_{start_year}_solar_power_rel_land.csv', sep = ';', index=False)
    
    del grids_wide_onshore, grids_wide_offshore,grids_wide_solar_land
    
    # Grids separated matrices and data for wind onshore, wind offshore, solar - preparing for elastic net
    grids_wide_onshore = pd.pivot(grids_land, index=['date','country'], columns=['grid_point_lat', 'grid_point_lon'], values = ['wind_power_rel']).reset_index()
    grids_wide_onshore.date = pd.to_datetime(grids_wide_onshore.date, utc=True)
    grids_wide_offshore = pd.pivot(grids_sea, index=['date','country'], columns=['grid_point_lat', 'grid_point_lon'], values = ['wind_power_rel']).reset_index()
    grids_wide_offshore.date = pd.to_datetime(grids_wide_offshore.date, utc=True)
    grids_wide_solar_land = pd.pivot(grids_land, index=['date','country'], columns=['grid_point_lat', 'grid_point_lon'], values = ['solar_power_rel']).reset_index()
    grids_wide_solar_land.date = pd.to_datetime(grids_wide_solar_land.date, utc=True)
    
    # save the special format for predictions in the elastic net part
    #grids_wide_onshore.to_csv(f'grids_wide_onshore_{start_year}.csv', sep = ';', index=False)
    #grids_wide_offshore.to_csv(f'grids_wide_offshore_{start_year}.csv', sep = ';', index=False)
    #grids_wide_solar_land.to_csv(f'grids_wide_solar_land_{start_year}.csv', sep = ';', index=False)
    #grids_wide.to_csv(f'grids_wide_{start_year}.csv', sep = ';', index=False)

    del grids, grids_all, relevant_grids_landsea, grids_landsea_all, grids_land, grids_sea
    
    return grids_wide, grids_wide_onshore, grids_wide_offshore, grids_wide_solar_land, relevant_era5_df
    


def grid_visualization(relevant_era5_df, grids_path):
    """
    
    visualization of grid points and weather cells
    
    Parameters
    ----------
    relevant_era5_df    Dataframe containing relevant information
    grids_path          path for saving figure
        
    Returns
    -------
        
    """
    
    global relevant_era5_df
    colors = list(mcolors.CSS4_COLORS.keys())

    colors.remove('white')
    colors.remove('whitesmoke')
    colors.remove('grey')
    colors.remove('lightgrey')
    colors.remove('lightgray')
    colors.remove('gray')

    for i in colors:
        if (i.find('grey') or i.find('gray') or i.find('white'))  != -1:
            colors.remove(i)

    ax_x = 8.27
    ax_y = ax_x * (relevant_era5_df.latitude.max() - relevant_era5_df.latitude.min()) / (relevant_era5_df.longitude.max() - relevant_era5_df.longitude.min())

    fig = plt.figure(figsize=(ax_x,ax_y))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    ax.set_extent([relevant_era5_df.longitude.min()-0.5, relevant_era5_df.longitude.max()+0.5, relevant_era5_df.latitude.min()-0.5, relevant_era5_df.latitude.max()+0.5], crs=ccrs.PlateCarree())

    ax.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
    ax.add_feature(cartopy.feature.LAND, color='lightgray', zorder=0,
                   linewidth=0.5, edgecolor='black')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                 xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10))

    ax.coastlines(resolution='50m', linewidth=0.3, color='black')
    gl.xlabels_top = False
    gl.ylabels_right = False

    for i, data in relevant_era5_df.groupby(['grid_point_lat', 'grid_point_lon']):
        co = np.random.choice(np.array(colors))
        #ax.scatter(data.grid_point_lon, data.grid_point_lat, alpha=0.1, c=co)
        ax.scatter(data.grid_point_lon, data.grid_point_lat, s=7, c = co, marker = "D")


        ax.scatter(data.longitude, data.latitude, s=0.2, c = co)

    plt.savefig(grids_path, bbox_inches='tight',
                        pad_inches=0.1)
                            
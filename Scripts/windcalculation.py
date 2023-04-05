import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians

import plotly.express as px
import plotly

import scipy.spatial
import time
import configparser

import wind_turbine as wt

import matplotlib
matplotlib.style.use('default')
from numpy import exp, log
from numpy import sqrt, pi, cos, sin, tan, arcsin, arccos
from numpy import deg2rad, rad2deg

import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, timezone



########################Wind Calculation#####################################################
def wind_calculation(relevant_era5_df, onshore_path, offshore_path):
    """
    
    Parameters
    ----------
    onshore_path
    offshore_path
    relevant_era5_df
        
    Returns
    -------
    relevant_ra5_df:    pandas.DataFrame
    
        columns:
        # date = utcTime timestamps
        # ...    
        # wind_power_abs
        # wind_power_rel
        # power_out
       
    """

    global relevant_era5_df
    global onshore_turbine
    global offshore_turbine

    cfp_onshore = configparser.RawConfigParser()
    cfp_onshore.read(onshore_path)

    cfp_offshore = configparser.RawConfigParser()
    cfp_offshore.read(offshore_path)

    onshore_hub = int(cfp_onshore.get('windcfg', 'HUB_HEIGHT'))
    offshore_hub = int(cfp_offshore.get('windcfg', 'HUB_HEIGHT'))

    # setting hub heigh according to weather cells, different for onshore and offshore
    relevant_era5_df.loc[relevant_era5_df.is_land == 1, 'hub_height'] = onshore_hub
    relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'hub_height'] = offshore_hub

    # wind speed in hub heigh for every timepoint and every weather cell    
    relevant_era5_df['wind_velocity'] = speed_hub(relevant_era5_df.wind_velocity_100.astype(float), relevant_era5_df.hub_height.astype(float), relevant_era5_df.surface.astype(float))

    onshore_curve = pd.DataFrame([cfp_onshore.get('windcfg', 'V').split(','), cfp_onshore.get('windcfg', 'POW').split(', ')]).T
    onshore_curve.columns = ['V', 'POW']
    onshore_curve.V = onshore_curve.V.astype(int)
    onshore_curve.POW = onshore_curve.POW.astype(float)
    
    offshore_curve = pd.DataFrame([cfp_offshore.get('windcfg', 'V').split(','), cfp_offshore.get('windcfg', 'POW').split(', ')]).T
    offshore_curve.columns = ['V', 'POW']
    offshore_curve.V = offshore_curve.V.astype(int)
    offshore_curve.POW = offshore_curve.POW.astype(float)

    onshore_turbine = wt.Turbine('Siemens', 'Siemens SWT 107', 3.6, dict(zip(onshore_curve['V'], onshore_curve['POW'])))
    onshore_turbine.load_turbine()

    offshore_turbine = wt.Turbine('Siemens', 'MHI Vestas V164', 9.5, dict(zip(offshore_curve['V'], offshore_curve['POW'])))
    offshore_turbine.load_turbine()

    ## calculation of turbine power
    relevant_era5_df.loc[relevant_era5_df.is_land == 1, 'wind_power_abs'] = onshore_turbine.compute_turbine_power(relevant_era5_df.loc[relevant_era5_df.is_land == 1, 'wind_velocity'].values)
    relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'wind_power_abs'] = offshore_turbine.compute_turbine_power(relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'wind_velocity'].values)
    relevant_era5_df.loc[relevant_era5_df.is_land == 1, 'wind_power_rel'] = relevant_era5_df.loc[relevant_era5_df.is_land == 1, 'wind_power_abs'] / onshore_curve.POW.max()
    relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'wind_power_rel'] = relevant_era5_df.loc[relevant_era5_df.is_land == 0, 'wind_power_abs'] / offshore_curve.POW.max()

    return relevant_era5_df


def speed_hub(speed_100, h_hub, z_0):
    """
    
    Parameters
    ----------
    speed_100:      wind speed in 100 meters
    h_hub:          hub heigh of turbine
    z_0:            surface roughness
        
    Returns
    -------
    speed:          wind speed in hub heigh
        
    """
    
    speed_100 = speed_100
    h_hub = h_hub
    z_0 = z_0

    speed = speed_100 * ((np.log(h_hub) - np.log(z_0)) / (np.log(100) - np.log(z_0)))
    return speed   


def wind_visualization(onshore_turbine, offshore_turbine):
    """
    Plots the power curve of the onshore and the offshore wind turbine
    
    Parameters
    ----------
    onshore_turbine
    offshore_turbine
        
    """
    
    onshore_turbine.plot_turbine_curve()
    offshore_turbine.plot_turbine_curve()


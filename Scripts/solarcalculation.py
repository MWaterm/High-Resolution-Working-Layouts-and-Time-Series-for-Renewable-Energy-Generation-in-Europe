############################Solar Coversion Functions################################  





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('default')
#from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians
import plotly.express as px
import plotly
import scipy.spatial
import time
import json
import pv_module as pv
import os
from numpy import exp, log
from numpy import sqrt, pi, cos, sin, tan, arcsin, arccos
from numpy import deg2rad, rad2deg

import matplotlib.pyplot as plt


def solar_calculation(weather_df, latitude, longitude, config_fname, sp_slope, sp_azimuth, sp_azimuth_a1, sp_azimuth_a2):

    """
    Number of day in year starting at 1 up to 365 / 366 if leap yr
    
    Parameters
    ----------
    weather_df : pandas.DataFrame
        
        columns:
        # date = utcTime timestamps
        # G_bn
        # G_d
        # fcast_albedo
        # T_a
    
    latitude
    longitude
    
    config_fname : string
    
    sp_slope
    sp_azimuth
        
    Returns
    -------
    relevant_era5_df : pandas.DataFrame
    
        columns:
        # date = utcTime timestamps
        # ...
        # solar_power_rel
        # solar_power_abs
        
    """

    global relevant_era5_df
   

    config_dict = json.load(open(config_fname))
    pv_mod = pv.pv_module(config_dict)

    
    weather_df['dayNum'] = weather_df.date.dt.day_of_year
    weather_df['utcHour'] = weather_df.date.dt.hour
    weather_df['utcMin'] = weather_df.date.dt.minute 
    weather_df['totalDays'] = np.where(weather_df.date.dt.is_leap_year, 366, 365)
    
    totalDays = weather_df['totalDays'].to_numpy()
    N = weather_df['dayNum'].to_numpy()
    utcHour = weather_df['utcHour'].to_numpy()
    utcMin = weather_df['utcMin'].to_numpy()
    
    G_bn = (weather_df['solardirect_radiation'].to_numpy())/3600 # Transform J/m^2 to W/m^2
    G_d =( weather_df['solardown_radiation'].to_numpy()- weather_df['solardirect_radiation'].to_numpy())/3600 # Transform J/m^2 to W/m^2
    T_a = weather_df['temperature'].to_numpy()-273.15 # Transform Kelvin to Degrees Celcius
    albedo = weather_df['albedo'].to_numpy()
    
    latitude = latitude.to_numpy()
    latitude = latitude.astype(float)
    longitude = longitude.to_numpy()
    longitude = longitude.astype(float)
    
    G_t, term1, term2, term3, G_b, G_on = reindl(G_bn, G_d, latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth, albedo)
    
    output_df = weather_df.copy().drop(['dayNum', 'utcHour', 'utcMin', 'totalDays'], axis=1)
    output_df['G_t'] = G_t
    output_df['T_c'] = pv_mod.get_cell_temp(output_df.G_t, T_a)
    output_df['eta'] = pv_mod.get_eta_mp(output_df.T_c)
    
     # Area of Solar-Panel
    # 1038mm x 20194 mm = 2173572mm**2 ~ 2.1735M**2
    panel_size = 2.1735
    output_df['power_out'] = output_df.eta * output_df.G_t*panel_size
    solar_out = output_df['date']
    solar_out['power_out_south'] = output_df['power_out']
    del output_df


    G_t, term1, term2, term3, G_b, G_on = reindl(G_bn, G_d, latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth_a1, albedo)

    output_df = weather_df.copy().drop(['dayNum', 'utcHour', 'utcMin', 'totalDays'], axis=1)
    output_df['G_t'] = G_t
    output_df['T_c'] = pv_mod.get_cell_temp(output_df.G_t, T_a)
    output_df['eta'] = pv_mod.get_eta_mp(output_df.T_c)
    
     # Area of Solar-Panel
    # 1038mm x 20194 mm = 2173572mm**2 ~ 2.1735M**2
    panel_size = 2.1735
    output_df['power_out'] = output_df.eta * output_df.G_t*panel_size
    solar_out['power_out_east'] = output_df['power_out']
    del output_df

    
    G_t, term1, term2, term3, G_b, G_on = reindl(G_bn, G_d, latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth_a2, albedo)
    
    output_df = weather_df.copy().drop(['dayNum', 'utcHour', 'utcMin', 'totalDays'], axis=1)
    output_df['G_t'] = G_t
    output_df['T_c'] = pv_mod.get_cell_temp(output_df.G_t, T_a)
    output_df['eta'] = pv_mod.get_eta_mp(output_df.T_c)
    
     # Area of Solar-Panel
    # 1038mm x 20194 mm = 2173572mm**2 ~ 2.1735M**2
    panel_size = 2.1735
    output_df['power_out'] = output_df.eta * output_df.G_t*panel_size
    solar_out['power_out_west'] = output_df['power_out']
    del output_df

    relevant_era5_df['solar_power_abs'] = 0.5*solar_out['power_out_south'] + 0.25*solar_out['power_out_east'] + 0.25*solar_out['power_out_west']
    relevant_era5_df['solar_power_rel'] = relevant_era5_df['solar_power_abs']/440 # Nominal power of Solar Cell

    return relevant_era5_df





def dayNum(utcTime):
    
    """
    Number of day in year starting at 1 up to 365 / 366 if leap yr
    
    Parameters
    ----------
    utcTime : datetime.datetime
        
    Returns
    -------
    dayOfYear : int
    totalDays : int
    
    """
    
    startOfYear = datetime(utcTime.year,1,1, tzinfo=utcTime.tzinfo)
    endOfYear = datetime(utcTime.year,12,31, tzinfo=utcTime.tzinfo)
    
    dayOfYear = (utcTime - startOfYear).days + 1
    totalDays = (endOfYear - startOfYear).days + 1

    return (dayOfYear, totalDays)

def equationOfTime(N, totalDays):
    
    """
    Compute the equation of time in minutes for a given day in year
    
    Parameters
    ----------
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
        
    Returns
    -------
    et : float
        equation of time in minutes
        
    """
    
    # eq 2.2 on p.51, Kalogirou

    b = (N - 81) * (2 * pi / totalDays) # use radians instead of degrees

    # eq 2.1 on p.51, Kalogirou
    et = 9.87 * sin(2*b) - 7.53 * cos(b) - 1.5 * sin(b) # in minutes
    
    return et

def apparentSolarTime(longitude, N, totalDays, utcHour, utcMin):
    
    """
    Compute the apparentSolarTime
    
    # AST = LST + ET ± 4(SL - LL) - DS
    # eq 2.3 on p.51, Kalogirou
    # AST = apparent solar time
    # LST = local std time
    # ET = equation of time
    # SL = standard longitude (of time zone)
    # LL = local longitude
    # DS = daylight saving (ie either 0 or 60 min)
    
    # since we use utcTime SL = LL ??
    
    Parameters
    ----------
    longitude : array_like
        longitude in degrees
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
        
    Returns
    -------
    ast : float
        apparentSolarTime in decimal hours
        
    """

    # calc std long from tz info
    std_long = np.round(longitude/15)*15
    loc_long = longitude
    
    et = equationOfTime(N, totalDays)
    
    ast = np.full((np.size(longitude),),np.nan)
    
    ast[loc_long < 0] = utcHour[loc_long < 0] + utcMin[loc_long < 0]/60 + et[loc_long < 0]/60 + 4 * (std_long[loc_long < 0] - loc_long[loc_long < 0])/60
    ast[loc_long >= 0] = utcHour[loc_long >= 0] + utcMin[loc_long >= 0]/60 + et[loc_long >= 0]/60 - 4 * (std_long[loc_long >= 0] - loc_long[loc_long >= 0])/60
    
    return ast

def declination(N, totalDays): 
    
    """
    returns solar declination in radians
    
    Parameters
    ----------
    N : array_like
        Day number in year starting from 1
        
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    
    Returns
    -------
    decl : float
        solar declination in radians
        
    """
    
    # eq 2.7 on p.55, Kalogirou
    g = 2 * pi * (N - 1) / (totalDays+1) # adjusted such that div by 366 in leap years
    
    # eq 2.6 on p.55
    decl = 0.006918 - 0.399912*cos(g) + 0.070257*sin(g) - \
           0.006758*cos(2*g) + 0.000907*sin(2*g) - \
           0.002697*cos(3*g) + 0.00148*sin(3*g)
    
    return decl

def hourAngle(longitude, N, totalDays, utcHour, utcMin):
    
    """
    returns solar hour angle in radians
    
    Parameters
    ----------
    longitude : array_like
        longitude in degrees
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    utcHour : array_like
    utcMin : array_like
        
    Returns
    -------
    ha : array_like
        solar hour angle in radians
        
    """
    
    ast = apparentSolarTime(longitude, N, totalDays, utcHour, utcMin) # in decimal hours
    
    # eq 2.9 on p.59, Kalogirou
    ha = 15 * (ast - 12)
    ha = deg2rad(ha)
    
    return ha

def solarAltitude(latitude, longitude, N, totalDays, utcHour, utcMin):
    
    """
    returns solar altitude angle in radians
    
    Parameters
    ----------
    longitude : array_like
        longitude in degrees
    latitude : array_like
        latitude in degrees    
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    utcHour : array_like
    utcMin : array_like
        
    Returns
    -------
    alpha : float
        solar altitude angle in radians
        
    """
    
    lat = deg2rad(latitude)
    decl = declination(N, totalDays)
    ha = hourAngle(longitude, N, totalDays, utcHour, utcMin)
    
    # eq 2.12 on p.60, Kalogirou
    sin_alpha = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(ha)
    alpha = arcsin(sin_alpha)
    
    return alpha

def solarZenith(latitude, longitude, N, totalDays, utcHour, utcMin):
    
    """
    returns solar zenith angle in radians
    
    # eq 2.11 on p.60, Kalogirou
    # phi + alpha = pi/2
    
    # eq 2.12 on p.60, Kalogirou
    # sin(alpha) = cos(Phi)
    
    Parameters
    ----------
    latitude : array_like
        latitude in degrees  
    longitude : array_like
        longitude in degrees
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    utcHour : array_like
    utcMin : array_like
        
    Returns
    -------
    phi : float
        solar zenith angle in radians
        
    """
    
    alpha = solarAltitude(latitude, longitude, N, totalDays, utcHour, utcMin)
    phi = pi/2 - alpha
    
    return phi

def solarAzimuth(latitude, longitude, N, totalDays, utcHour, utcMin): 
    
    """
    returns solar azimuth angle in radians
    
    Parameters
    ----------
    latitude : array_like
        latitude in degrees  
    longitude : array_like
        longitude in degrees
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    utcHour : array_like
    utcMin : array_like
        
    Returns
    -------
    z : float
        solar altitude angle in radians
        
    """
    
    decl = declination(N, totalDays)
    ha = hourAngle(longitude, N, totalDays, utcHour, utcMin)
    alpha = solarAltitude(latitude, longitude, N, totalDays, utcHour, utcMin)
    
    # eq 2.13 p.60, Kalogirou
    sin_z = cos(decl) * sin(ha) / cos(alpha)
    z = arcsin(sin_z)
    
    lat = deg2rad(latitude)    
    cond = cos(ha) > tan(decl) / tan(lat) # what if div by 0 ?? lat = 0° / 90° / 270°
    noon_bool = apparentSolarTime(longitude, N, totalDays, utcHour, utcMin) < 12

    if (not cond) & noon_bool:
        z = -pi + abs(z)
    if (not cond) & (not noon_bool):
        z = pi - z

    return z

def incidenceAngle(latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth):
    
    """
    returns incidence angle in radians
    
    # eq 2.18 on p.62, Kalogirou
    # cos_theta = sin(L)*sin(d)*cos(b) - cos(L)*sin(d)*sin(b)*cos(z)
                + cos(L)*cos(d)*cos(h)*cos(b) 
                + sin(L)*cos(d)*cos(h)*sin(b)*cos(z)
                + cos(d)*sin(h)*sin(b)*sin(z)
    
    # eq 2.18 is a general relationship for the angle of incidence on a surface of any orientation
    # for specific cases it can be reduced to much simpler forms 
    # eg if beta = 0 then incidence angle theta = zenith angle Phi
    
    Parameters
    ----------
    latitude : array_like
        latitude in degrees  
    longitude : array_like
        longitude in degrees
    N : array_like
        Day number in year starting from 1
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
    utcHour : array_like
    utcMin : array_like
    sp_slope : float
        solar panel surface tilt angle from the horizontal = beta
    sp_azimuth : float
        solar panel azimuth = z
        angle between the normal to the surface from true south
        westward is designated as positive
  
    Returns
    -------
    theta : float
        incidence angle in radians
        
    """
  
    lat = deg2rad(latitude)
    decl = declination(N, totalDays)
    ha = hourAngle(longitude, N, totalDays, utcHour, utcMin)
    beta = deg2rad(sp_slope)
 
    z = deg2rad(sp_azimuth) 
    
    cos_theta = sin(lat)*sin(decl)*cos(beta) - cos(lat)*sin(decl)*sin(beta)*cos(z) \
                + cos(lat)*cos(decl)*cos(ha)*cos(beta) \
                + sin(lat)*cos(decl)*cos(ha)*sin(beta)*cos(z) \
                + cos(decl)*sin(ha)*sin(beta)*sin(z)
    theta = arccos(cos_theta)
    
    return theta


def reindl(G_bn, G_d, latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth, albedo):
    
    """
    Calculates G_t = total radiation on a tilted surface based on the Reindl model (1990)
    
    # eq 2.104 from p.103, Kalogirou
    G_t = (G_b + G_d * A) * R_b
        + G_d * (1-A) * 0.5*(1+cos(beta)) * (1+sqrt(G_b / (G_b + G_d)) * sin(beta/2)**3)
        + (G_b + G_d) * rho * 0.5*(1-cos(beta))
    
    Notes:
    ----------
    # 1st term = Beam radiation 
        G_b,t   = G_b * R_b
        
    # 2nd term = Circumsolar diffuse radiation + Isotropic diffuse + Horizon brightening diffuse radiation 
        G_d,t = G_d * A * R_b                # circumsolar diffuse
                + G_d
                * (1-A) * 0.5*(1+cos(beta))  # isotropic diffuse
                * (1 + f * sin(beta/2)**3)   # horizon brightening diffuse
        
        where f is the correction factor for horizon brighening effects under cloudy & clear skies
        f = sqrt(G_b / (G_b + G_d))
        
    # 3rd term = Ground reflected radiation 
        G_g,t = (G_b + G_d) * rho * 0.5*(1-cos(beta))
    
    Symbols:
    ----------
    # G = Irradiance [W/m^2 or J/m^2 hr]
        # rate of energy falling on surface per unit area
        
    # Subscripts:
        # b = beam / direct radiation
        # d = diffuse radiation
        # g = ground reflected radiation
        # n = radiation at normal incidence
        # o = extraterrestrial
        # t = radiation on tilted surface
    
    Variables:
    ----------
    # G_t  = total radiation on a tilted plane [W/m^2 or J/m^2 hr]
    # G_b  = beam radiation on a horizontal surface [..]
    # G_d  = diffuse radiation on a horizontal surface [..]
    
    # G_on = extraterrestrial radiation at normal incidence [..]
        # can be calculated using eq. 2.77 Kalogirou
        
        # G_on = G_sc * (1 + 0.033 * cos(360 * N / totalDays))
        
        # G_sc is measured by satellite as being 1.361 kilowatts per square meter (kW/m2) at solar minimum 
        # (the time in the 11-year solar cycle when the number of sunspots is minimal) 
        # and approximately 0.1% greater (roughly 1.362 kW/m2) at solar maximum.
    
    # A = anisotropy index 
        # defines a portion of the diffuse radiation to be treated as circumsolar 
        # with the remaining portion considered isotropic
        
        # A = G_b,n / G_o,n # eq 2.102 p.103, Kalogirou
        # G_bn = beam radiation at normal incidence 
        # G_on = extraterrestrial radiation at normal incidence
        
    # R_b = beam radiation tilt factor 
        # ratio of beam radiation on a tilted surface to the  beam radiation on a horizontal surface
 
        # R_b = G_b,t / G_b = cos(theta) / cos(Phi) # eq 2.88 on p.100, Kalogirou
        # G_b,t = tilted surface beam radiation
        # G_b   = horiz. beam radiation
        
    # beta = solar panel surface tilt angle [rad]
    # rho = ground albedo / reflectance
    
    Parameters
    ----------
    G_bn : array_like
        beam radiation at normal incidence
        
    G_d : array_like
        diffuse radiation on a horizontal surface
        
    latitude : array_like
        latitude in degrees  
        
    longitude : array_like
        longitude in degrees
        
    N : array_like
        Day number in year starting from 1
        
    totalDays : array_like
        Total days in year 365 for normal years, 366 for leap years
        
    utcHour : array_like
    utcMin : array_like
    
    sp_slope : array_like
        solar panel surface tilt angle from the horizontal = beta
        
    sp_azimuth : array_like
        solar panel azimuth = z = angle between the normal to the surface from true south
        westward is designated as positive
        
    albedo : array_like
        forecasted albedo
        
    Returns
    -------
    G_t : float
        G_t = total radiation on a tilted surface [W/m^2]
        
    """

    beta = deg2rad(sp_slope)
 
    
    Phi = solarZenith(latitude, longitude, N, totalDays, utcHour, utcMin)
    theta = incidenceAngle(latitude, longitude, N, totalDays, utcHour, utcMin, sp_slope, sp_azimuth)

    # eq 2.87 on p.100, Kalogirou
    G_b = np.where(((Phi >= deg2rad(85))| (theta >= deg2rad(85))), 0, np.maximum(G_bn * np.cos(Phi), 0))
    
    
    # eq 2.88 on p.100, Kalogirou
    R_b = np.where(((Phi >= deg2rad(85))| (theta >= deg2rad(85))), 0, np.cos(theta) / np.cos(Phi))

    rho = albedo
    
    # eq 2.77 p.92, Kalogirou
    G_sc = 1366.1 # [W/m^2]
    G_on = G_sc * (1 + 0.033 * np.cos(360 * N / totalDays))
    
     # eq 2.102 p.103, Kalogirou
    A = np.where(((Phi >= deg2rad(85))| (theta >= deg2rad(85))), 0, G_bn / G_on)

    term1 = G_b * R_b
    term2 = (G_d * A * R_b) + (G_d * (1-A) * 0.5*(1+np.cos(beta)) * (1+np.sqrt(G_b / (G_b + G_d)) * np.sin(beta/2)**3))
    term3 = (G_b + G_d) * rho * 0.5*(1-np.cos(beta))


    G_t = term1 + term2 + term3   
    
    return G_t, term1, term2, term3, G_b, G_on

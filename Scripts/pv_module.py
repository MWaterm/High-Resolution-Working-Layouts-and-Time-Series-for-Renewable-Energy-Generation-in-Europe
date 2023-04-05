#############################PV Module######################################################
import pandas as pd
import numpy as np
#from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians
import time
from datetime import datetime, time, timedelta, timezone
import matplotlib
matplotlib.style.use('default')
from numpy import exp, log
from numpy import sqrt, pi, cos, sin, tan, arcsin, arccos
from numpy import deg2rad, rad2deg



class pv_module:
    """ 
    Class of the PV-Module
    Including:
                - _init_: Characteristics of PV-Module given by config file
                - get_cell_temp: Calulates cell temperature in a recursive way
                - get_eta_mp: Calculates Efficiency for given Cell-Temperature
    """

    
    k = 1.381 * 10**(-23) # boltzmann constant
    q = 1.602 * 10**(-19) # standard charge of an electron
    
    T_ref = 298.2 # [Kelvin] = 25°C
    T_a_NOCT = T_ref
    
    G_t_NOCT = 800
    
    tau_alpha = 0.9 # rough estimate Duffie, Beckham Chpt 23.3 p.774
    
    def __init__(self, config_dict):
        
        """
        Parameters
        ----------
        N_s         number of cells on solar panel
        
        I_sc        short circuit (sc) current
        V_oc        open circuit (oc) voltage
        I_mp        current at max power (mp)
        V_mp        voltage at max power (mp)
        
        mu_I_sc     temp coeff of I_sc [% per °C]
        mu_V_oc     temp coeff of V_oc [% per °C]
        
        eta_mp_ref  reference efficiency at ref temp
        
        T_ref       reference temp at which data is provided [Kelvin]
        T_c_NOCT    cell temperature at NOCT conditions [Kelvin]
        
        """
        self.__dict__.update(config_dict)
        
        self.mu_I_sc = self.mu_I_sc * self.I_sc # convert [% per °C] to [A/°C]
        self.mu_V_oc = self.mu_V_oc * self.V_oc # convert [% per °C] to [V/°C]
        # source: https://unmethours.com/question/36669/is-ak-or-vk-same-as-c-on-temperature-coefficient
        
        # Duffie, Beckham eq. 23.2.18
        self.mu_eta_mp = self.eta_mp_ref * self.mu_V_oc / self.V_mp
    
    def get_cell_temp(self, G_t, T_a, eta=None):
        
        """
        get cell temperature based on ambient temperature
        based on Kalogirou eq. 9.36
        
        (alternative formula in Duffie, Beckham eq. 23.2.16 including wind speed but more complicated ...)
        Parameters
        ----------
        G_t : float
            total radiation on a tilted plane [W/m^2 or J/m^2 hr]
        T_a : float
            ambient temperature in Celsius
        
        Returns
        -------
        T_c : float
            solar cell temperature in Celsius
        
        """
        
        if eta is None:
                    
            # more accurate eq. 9.35
            # with solar cell reference efficiency at NOCT
            T_c = (self.T_c_NOCT - self.T_a_NOCT) * (G_t / self.G_t_NOCT) * (1 - self.eta_mp_ref / self.tau_alpha) + T_a

            # use it to estimate eta
            eta = self.get_eta_mp(T_c)

            # use eta to calc more accurate T_c
            T_c = self.get_cell_temp(G_t, T_a, eta)
        
        elif eta is not None:
            # more accurate eq. 9.35
            T_c = (self.T_c_NOCT - self.T_a_NOCT) * (G_t / self.G_t_NOCT) * (1 - eta / self.tau_alpha) + T_a
        
        return T_c
    
    def get_eta_mp(self, T_c):
        
        """
        get temp dependent max power efficiency
        based on Duffie, Beckham eq. 23.2.16
        Parameters
        ----------
        T_c : float
            solar cell temperature in Celsius
            
        Returns
        -------
        eta_mp : float
            solar cell efficiency based on temperature
        
        """
        
        
        eta_mp = self.eta_mp_ref + self.mu_eta_mp * (T_c - self.T_ref)
        
        return eta_mp

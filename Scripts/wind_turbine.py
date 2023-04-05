#############################Wind Turbine######################################################

import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import os.path
from matplotlib.ticker import FormatStrFormatter

class Turbine:
    """ 
    Class of the wind turbine
    Including:
                - _init_: Characteristics of turbine given by config file
    """
    
    turbine_hersteller = None

    turbine_bezeichner = None

    turbine_kennlinie = None

    turbine_rated_power = None

    turbine_polynom_lower = None

    turbine_polynom_upper = None

    turbine_minimum_windspeed = None

    turbine_split_windspeed = None

    turbine_rated_windspeed = None

    def __init__(self, hersteller=None, bezeichner=None, rated_power=None, kennlinie=None):
        """
        Parameters
        ----------
        hersteller:    manufacturer of the turbine
        bezeichner:    name of turbine
        kennlinie:     discrete power curve of turbine
        rated_power:   rated power of the turbine
        
        """

        self.turbine_hersteller = hersteller

        self.turbine_bezeichner = bezeichner

        self.turbine_kennlinie = kennlinie

        self.turbine_rated_power = rated_power


    def load_turbine(self):
        """
        load power curve based on a combination of third-order polynomials

        """

        self.turbine_polynom_lower = self.polynom_fitting()['Polynom_lower']

        self.turbine_polynom_upper = self.polynom_fitting()['Polynom_upper']

        return


    def polynom_fitting(self):
        """
        Fit of two third-order polynomials to approaximate turbine's power curve
        
        Parameters
        ----------
        turbine_rated_power:    rated power of the turbine
        turbine_kennlinie:      discrete power curve
        
        Returns
        -------
        dict:       dictionary with lower and upper polynomial 
        
        """
        
        # minimum_windspeed

        for key, value in self.turbine_kennlinie.items():

            if value == 0:

                minimum_windspeed = int(key)

                self.turbine_minimum_windspeed = minimum_windspeed

            else:

                break

        # rated_windspeed

        for key, value in self.turbine_kennlinie.items():

            if value >= self.turbine_rated_power:
                rated_windspeed = int(key)

                self.turbine_rated_windspeed = rated_windspeed

                break

        # split_windspeed = turning point in "kennlinie" 
        curve = np.polyfit(np.arange(minimum_windspeed, rated_windspeed + 1, 1),

                           list(self.turbine_kennlinie.values())[minimum_windspeed:rated_windspeed + 1],

                           3)  # Polynomial fitting

        x = sy.Symbol('x')

        polynom = curve[0] * x ** 3 + curve[1] * x ** 2 + curve[2] * x + curve[3]

        polynom_2nd_derivation = polynom.diff(x, 2)  # Derivate fitted polynomial to calculate turning point
       
        split_windspeed = int(round(sy.solve(sy.Eq(polynom_2nd_derivation, 0))[

                                        0]))  # Determine the inflection point of the polynomial via the second derivative

        self.turbine_split_windspeed = split_windspeed

        # Calculate polynom_lower and polynom_upper

        polynom_lower = np.polyfit(np.arange(minimum_windspeed, split_windspeed + 1, 1),

                                   list(self.turbine_kennlinie.values())[minimum_windspeed:split_windspeed + 1], 3)


        polynom_upper = np.polyfit(np.arange(split_windspeed, rated_windspeed + 1, 1),

                                   list(self.turbine_kennlinie.values())[split_windspeed: rated_windspeed + 1], 3)

        return dict(zip(['Polynom_lower', 'Polynom_upper'], [polynom_lower, polynom_upper]))


    def plot_turbine_curve(self):
        """
        Plot the approximated turbine's power curve
        
        """

        plt.rc('xtick', labelsize=10)

        plt.rc('ytick', labelsize=10)

        plt.rc('axes', labelsize=10)

        plt.rc('legend', fontsize=10)

        fig, ax = plt.subplots(figsize=(4.135, 3.2))  # Größe des Plots festlegen
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.grid(b=True, which='both', axis='both', alpha=0.2)

        x = [float(x) for x in list(self.turbine_kennlinie.keys())]

        x_lower = np.arange(self.turbine_minimum_windspeed, self.turbine_split_windspeed + 1, 1)

        x_upper = np.arange(self.turbine_split_windspeed, self.turbine_rated_windspeed + 1, 1)

        y = [float(x) for x in list(self.turbine_kennlinie.values())]

        plt.scatter(x, y, c='darkblue', marker='x')

        plt.hlines(y=0,

                   xmin=0,

                   xmax=self.turbine_minimum_windspeed,

                   colors='darkblue')

        plt.plot(np.linspace(x_lower[0], x_lower[-1]),

                 np.poly1d(self.turbine_polynom_lower)(np.linspace(x_lower[0], x_lower[-1])),

                 label='\u03B1_1$=' + str(round(self.turbine_polynom_lower[0], 4))

                       + ', \u03B2_1$=' + str(round(self.turbine_polynom_lower[1], 4))

                       + ',\n\u03B3_1$=' + str(round(self.turbine_polynom_lower[2], 4))

                       + ', \u03B4_1$=' + str(round(self.turbine_polynom_lower[3], 4)),

                 color='darkblue')

        plt.plot(np.linspace(x_upper[0], x_upper[-1]),

                 np.poly1d(self.turbine_polynom_upper)(np.linspace(x_upper[0], x_upper[-1])),

                 label='\u03B1_2$=' + str(round(self.turbine_polynom_upper[0], 4))

                       + ', \u03B2_2$=' + str(round(self.turbine_polynom_upper[1], 4))

                       + ',\n\u03B3_2$=' + str(round(self.turbine_polynom_upper[2], 4))

                       + ', \u03B4_2$=' + str(round(self.turbine_polynom_upper[3], 4)),

                 color='darkblue')

        plt.hlines(y=self.turbine_rated_power,

                   xmin=self.turbine_rated_windspeed,

                   xmax=25,

                   colors='darkblue')

        plt.xlabel('Wind speed in [m/s]', fontname = 'Times New Roman')

        plt.ylabel('Power in [MW]', fontname = 'Times New Roman')

        plt.xticks()

        plt.xticks(rotation=0)

        plt.title(self.turbine_bezeichner + ' (' + str(self.turbine_rated_power) + ' MW)', fontsize=12, fontname = 'Times New Roman')

        plt.text(x=0.43,

                 #y=0.175,
                 y=0.225,

                 s='$\u03B1_1=$' + str(round(self.turbine_polynom_lower[0], 3))

                   + ', $\u03B2_1=$' + str(round(self.turbine_polynom_lower[1], 3))

                   + '\n$\u03B3_1=$' + str(round(self.turbine_polynom_lower[2], 3))

                   + ', $\u03B4_1=$' + str(round(self.turbine_polynom_lower[3], 3)),

                 fontsize=12, transform=ax.transAxes, fontname = 'Times New Roman')

        plt.text(x=0.43,  # x=13,

                 y=0.475,  # y=3,

                 s='$\u03B1_2=$' + str(round(self.turbine_polynom_upper[0], 3))

                   + ', $\u03B2_2=$' + str(round(self.turbine_polynom_upper[1], 3))

                   + '\n$\u03B3_2=$' + str(round(self.turbine_polynom_upper[2], 3))

                   + ', $\u03B4_2=$' + str(round(self.turbine_polynom_upper[3], 3)),

                 fontsize=12, transform=ax.transAxes, fontname = 'Times New Roman')

        plt.text(x=0.43,

                 y=0.075,

                 #s='\u03C4=' + str(self.turbine_split_windspeed),
                 s='$v_{split}=$' + str(self.turbine_split_windspeed),
                 
                 fontsize=12, transform=ax.transAxes, fontname = 'Times New Roman')

        plt.savefig('Kennlinien-Plots/Kennlinie_' + self.turbine_bezeichner + '.pdf', bbox_inches='tight',

                    pad_inches=0.1, dpi=300)  # Plot speichern unter angegebenen Namen

        plt.close()

        print(' > Power curve of ' + self.turbine_bezeichner + ' is plotted.')

        return


    def compute_turbine_power(self, scaled_windspeed):
        """
        Function of the approximated turbine's power curve, which computes the turbine's generation for a given windspeed
        
        Parameters
        ----------
        turbine (class)
        scaled_windspeed:       windspeed in hub heigh
        
        Returns
        -------
        scaled_windspeed:       Turbine's power for the given windspeed 
        
        
        """
        

        values = scaled_windspeed

        result_upper = self.turbine_polynom_upper[0] * values ** 3 \
         \
                         + self.turbine_polynom_upper[1] * values ** 2 \
         \
                         + self.turbine_polynom_upper[2] * values \
         \
                         + self.turbine_polynom_upper[3]
        
        result_lower = self.turbine_polynom_lower[0] * values ** 3 \
 \
                         + self.turbine_polynom_lower[1] * values ** 2 \
 \
                         + self.turbine_polynom_lower[2] * values \
 \
                         + self.turbine_polynom_lower[3]
        
        scaled_windspeed[values < self.turbine_minimum_windspeed] = 0
        scaled_windspeed[(self.turbine_minimum_windspeed < values) & (values < self.turbine_split_windspeed)] = result_lower[(self.turbine_minimum_windspeed < values) & (values < self.turbine_split_windspeed)]
        scaled_windspeed[(self.turbine_split_windspeed < values) & (values < self.turbine_rated_windspeed)] = result_upper[(self.turbine_split_windspeed < values) & (values < self.turbine_rated_windspeed)]
        scaled_windspeed[(self.turbine_rated_windspeed < values) & (values < 25)] = self.turbine_rated_power
        scaled_windspeed[values > 25] = 0
        scaled_windspeed[scaled_windspeed > self.turbine_rated_power] = 0
        
        #scaled_windspeed[value < self.turbine_split_windspeed]
        
        return scaled_windspeed




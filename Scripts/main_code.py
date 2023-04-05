"""
This code estimates the working layouts presented in the work: 
    "High-Resolution Working Layouts and Time Series for Renewable Energy 
    Generation in Europe: A Data-Driven Approach for Accurate Fore- 
    and Nowcasting" by Oliver Grothe, Fabian Kächele and Mira Watermeyer
    
    The estimation includes the following steps: 
        - data import: 
            "weather_data()"
            "feedin_data()"
        - calculation of relative signals for onshore wind, offshore wind and 
            PV generation with weather-to-energy conversions: 
            "windcalculation()"
            "solarcalculation()"
        - weather cell combination: 
            "weather_cell_combination()"
        - layout estimation: 
            "layout_estimation()"
    The functions and scripts for the steps are outlined. 
    
By setting the parameter, this code can be used to estimate working layouts 
with installed capacities and feed-in of renewable energy sources for various 
time periods and locations

"""

import pandas as pd
import pickle

import weather_data
import feedin_data
import pv_module
import wind_turbine
import windcalculation
import solarcalculation
import weather_cell_combination
import layout_estimation

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v



"""
Parameter setting: 
"""
start_year = 2019
end_year = 2019
sea_depth = 70
entsoe_token = 'cf34733d-ec5f-4508-89f1-16eb81da8f1f'
onshore_config_path = '01_data/power_curves/Onshore.cfg'
offshore_config_path = '01_data/power_curves/Offshore.cfg'
grid_pts_path = '01_data/network_nodes_grid.xlsx'
gebco_file_path = '01_data/gebco2021.nc'

era_path = f'01_data/{start_year}/era_{start_year}_{end_year}.grib'
entsoe_path = f'01_data/{start_year}/entsoe_{start_year}_{end_year}.csv'
relevant_era_path = f'01_data/{start_year}/relevant_era_{start_year}_{end_year}.csv'
grids_path = '01_data/{start_year}/grid_pts.pdf'

save_data_path = f'01_data/{start_year}/'
save_signal_path = ''
save_layout_path = ''
save_feedin_path = ''


grid_pts = pd.read_excel(grid_pts_path, engine='openpyxl')

relevant_era5_df = None
grids = None
entsoe_data = None
output_df = None


# Import weather data:
relevant_era5_df = weather_data.convert_era(relevant_era_path = relevant_era_path, era_path=era_path, gebco_file_path=gebco_file_path, start_year=start_year, end_year=end_year, grid_pts=grid_pts, sea_dept=sea_depth)
# Calculate wind singnals for weather cells
relevant_era5_df = windcalculation.wind_calculation(relevant_era5_df=relevant_era5_df, onshore_path=onshore_config_path, offshore_path=offshore_config_path)
# Calculate solar signals for weather cells
relevant_era5_df = solarcalculation.solar_calculation(relevant_era5_df[['date','solardirect_radiation','solardown_radiation','albedo','temperature']], relevant_era5_df['latitude'], relevant_era5_df['longitude'], config_fname = 'LR4_72HBD_440M_config.yml', sp_slope = 45, sp_azimuth = 0, sp_azimuth_a1 = -90, sp_azimuth_a2 = 90)
# Combine weather cells in allocating to grid nodes
grids_wide, grids_wide_onshore, grids_wide_offshore, grids_wide_solar_land, relevant_era5_df = weather_cell_combination.era_to_grid_grouper(relevant_era5_df, grid_pts, start_year=start_year, save_signal_path='')

# Download feed-in data:
entsoe_data, entsoe_data_grouped = feedin_data.download_entsoe(entsoe_path=entsoe_path, entsoe_token=entsoe_token, start_year=start_year, end_year=end_year)

# Prepare prediction
entsoe_data_grouped_yp1 = entsoe_data_grouped.copy()
grids_wide_yp1 = grids_wide.copy()
grids_wide_offshore_yp1 = grids_wide_offshore.copy()
grids_wide_onshore_yp1 = grids_wide_onshore.copy()
grids_wide_solar_land_yp1 = grids_wide_solar_land.copy()    
save(f'predvars_{start_year}', 'entsoe_data_grouped_yp1', 'grids_wide_yp1', 'grids_wide_offshore_yp1', 'grids_wide_onshore_yp1', 'grids_wide_solar_land_yp1')
del entsoe_data_grouped_yp1, grids_wide_yp1, grids_wide_offshore_yp1, grids_wide_onshore_yp1, grids_wide_solar_land_yp1

pred = 0
if start_year<2022:
    load(f'predvars_{start_year+1}')
    pred = 1

# Layout estimation, generation calculation and generation prediction
layout_estimation.elastic_net(type='offshore', start_year=start_year, l1ratio=0.7, pred = pred, entsoe_data_grouped=entsoe_data_grouped, grid_signals=grids_wide_offshore, entsoe_data_grouped_yp1=entsoe_data_grouped_yp1, grid_signals_yp1=grids_wide_offshore_yp1, save_layout_path='', save_feedin_path='')
print('elnet offshore')

layout_estimation.elastic_net(type='onshore', start_year=start_year, l1ratio=0.7, pred = pred, entsoe_data_grouped=entsoe_data_grouped, grid_signals=grids_wide_onshore, entsoe_data_grouped_yp1=entsoe_data_grouped_yp1, grid_signals_yp1=grids_wide_onshore_yp1, save_layout_path='', save_feedin_path='')
print('elnet onshore')

layout_estimation.elastic_net(type='solar', start_year=start_year, l1ratio=0.7, pred = pred, entsoe_data_grouped=entsoe_data_grouped, grid_signals=grids_wide_solar, entsoe_data_grouped_yp1=entsoe_data_grouped_yp1, grid_signals_yp1=grids_wide_solar_yp1, save_layout_path='', save_feedin_path='')
print('elnet solar')

layout_estimation.elastic_net(type='wind', start_year=start_year, l1ratio=0.7, pred = pred, entsoe_data_grouped=entsoe_data_grouped, grid_signals=grids_wide, entsoe_data_grouped_yp1=entsoe_data_grouped_yp1, grid_signals_yp1=grids_wide_yp1, save_layout_path='', save_feedin_path='')
print('elnet wind')

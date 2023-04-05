###################################Layout Estimation##################################################

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.style.use('default')
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta, timezone
from math import sin, cos, sqrt, atan2, radians


###################################Elastic Net##################################################
def elastic_net(type, start_year, l1ratio, pred, entsoe_data_grouped, grid_signals, grid_signals_yp1, save_layout_path, save_feedin_path):
    """
    
    Country-wise layout estimation: 
        - signals are standardised
        - standardised signals of all relevant grid nodes for one country are regressed to the country's feed-in
        - estimation of optimal weights is done with an regularised regression: elastic-net
        -retransform standardisation results in capacity estimations
        
        - the estimation is done for each generator type separately
        
    Parameters
    ----------
    type:                       generator type (wind, onshore, offshore, solar)
    start_year:                 
    l1ratio:                    ration between L1 and L2 in the elastic-net
    pred:                       only estimation or prediction as well (if, pred = 1)
    entsoe_data_grouped:        sorted feed-in  
    grid_signals:        
            grids_wide:                 aggregated signals for all grid nodes
            grids_wide_offshore:        aggregated offshore wind signals for all grid nodes in the sea
            grids_wide_onshore:         aggregated onshore wind signals for all onshore grid nodes
            grids_wide_solar_land:      aggregated PV wind signals for all onshore grid nodes in the sea
    grid_signals_yp1:
            entsoe_data_grouped_yp1:    aggregated offshore wind signals for all grid nodes in the sea for the next year
            grids_wide_yp1:             aggregated signals for all grid nodes for the next year
            grids_wide_offshore_yp1:    aggregated offshore wind signals for all grid nodes in the sea for the next year
            grids_wide_onshore_yp1:     aggregated onshore wind signals for all onshore grid nodes for the next year
            grids_wide_solar_land_yp1:  aggregated PV wind signals for all onshore grid nodes for the next year
    save_layout_path
    save_feedin_path
    
    Returns
    -------
        
    """    
    
    if type == 'wind':
        df_merged = entsoe_data_grouped.merge(grid_signals, on = ['country', 'date'], how='inner')
        if pred == 1:
            df_merged_yp1 = entsoe_data_grouped_yp1.merge(grid_signals_yp1, on = ['country', 'date'], how='inner')

        for state in pd.unique(df_merged['country']):
            y_wind_state = df_merged.Wind[df_merged.country == state]
            y_wind_state.iloc[1:] = y_wind_state.iloc[0:-1]
            X_wind_state = df_merged.wind_power_rel[df_merged.country == state]
            X_wind_cols_state = X_wind_state.columns
            X_wind_state.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_state.columns]
            X_wind_state.fillna(0, inplace= True)
            scaler = StandardScaler().fit(X_wind_state)
            standardized_data = scaler.transform(X_wind_state)
            standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state.columns)
            X_wind_state = standardized_df
            
            regr_state = ElasticNetCV(l1_ratio=l1ratio, eps=0.0001, n_alphas=100, alphas=None, fit_intercept=True,
                                precompute='auto', max_iter=10000, tol=0.0001, cv=10, copy_X=True,
                                verbose=0, n_jobs=None, positive=True, random_state=None, selection='cyclic')
            regr_state.fit(X_wind_state, y_wind_state)

            beta_wind_state = regr_state.coef_ / scaler.scale_

            X_wind_lats_state = X_wind_cols_state.get_level_values(0)
            X_wind_lons_state = X_wind_cols_state.get_level_values(1)
            
            #beta_wind_state.to_csv(f'Layout_{state}.csv')
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_state_df = pd.DataFrame(np.array([X_wind_lons_state, X_wind_lats_state, beta_wind_state, scaler.scale_]).T, columns = ColumnsLayout)
            layout_state_df.to_csv(save_layout_path + f'layout_wind_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            yhat_wind_state = regr_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_wind_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_wind_state]).T, columns = ColumnsLayoutFeedin)
            yhat_wind_state_df.to_csv(save_feedin_path + f'layout_feedin_wind_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            print('state: ' + state + ' l1ratio: ' + str(regr_state.l1_ratio_) + ' alpha: ' + str(regr_state.alpha_) + ' RMSE: ' + str(sqrt(((y_wind_state - yhat_wind_state)**2).mean())))

# Linear Regression:            
            regr_lin_state = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=True)
            regr_lin_state.fit(X_wind_state, y_wind_state)
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            print(' RMSE OLS: ' + str(sqrt(((y_wind_state - yhat_lin_wind_state)**2).mean())))
            beta_lin_wind_state = regr_lin_state.coef_ / scaler.scale_

            #beta_wind_state.to_csv(f'Layout_{state}.csv')
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_lin_state_df = pd.DataFrame(np.array([X_wind_lons_state, X_wind_lats_state, beta_lin_wind_state, scaler.scale_]).T, columns = ColumnsLayout)
            layout_lin_state_df.to_csv(save_layout_path + f'layout_lin_wind_{state}_{start_year}.csv', sep = ';', index=False)
            
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_lin_wind_state]).T, columns = ColumnsLayoutFeedin)
            yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_wind_{state}_{start_year}.csv', sep = ';', index=False)

# Prediction:
            if pred==1:
                try:
# Prediction elastic net:                     
                    y_wind_state_yp1 = df_merged_yp1.Wind[df_merged_yp1.country == state]
                    y_wind_state_yp1.iloc[1:] = y_wind_state_yp1.iloc[0:-1]
                    X_wind_state_yp1 = df_merged_yp1.wind_power_rel[df_merged_yp1.country == state]
                    X_wind_cols_state_yp1 = X_wind_state_yp1.columns
                    X_wind_state_yp1.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_cols_state_yp1]
                    X_wind_state.fillna(0, inplace= True)
                    scaler = StandardScaler().fit(X_wind_state_yp1)
                    standardized_data = scaler.transform(X_wind_state_yp1)
                    standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state_yp1.columns)
                    X_wind_state_yp1 = standardized_df
                    yhat_wind_state_yp1 = regr_lin_state.predict(X_wind_state_yp1)
                    yhat_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    yhat_wind_state_df.to_csv(save_feedin_path + f'layout_feedin_pred_wind_{state}_{start_year+1}_{l1ratio}.csv', sep = ';', index=False)
# Prediction linear regression:                    
                    yhat_lin_wind_state_yp1 = regr_state.predict(X_wind_state_yp1)
                    yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_lin_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_pred_wind_{state}_{start_year+1}.csv', sep = ';', index=False)
                    
                    
                    pred = 1
                except:
                    print('No prediction for ' + state)

    if type == 'offshore':
        df_merged = entsoe_data_grouped.merge(grid_signals, on = ['country', 'date'], how='inner')
        if pred == 1:
            df_merged_yp1 = entsoe_data_grouped_yp1.merge(grid_signals_yp1, on = ['country', 'date'], how='inner')

        for state in pd.unique(df_merged['country']):
            y_wind_state = df_merged.Wind_Offshore[df_merged.country == state]
            y_wind_state.iloc[1:] = y_wind_state.iloc[0:-1]
            X_wind_state = df_merged.wind_power_rel[df_merged.country == state]
            X_wind_cols_state = X_wind_state.columns
            X_wind_state.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_state.columns]
            X_wind_state.fillna(0, inplace= True)
            scaler = StandardScaler().fit(X_wind_state)
            standardized_data = scaler.transform(X_wind_state)
            standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state.columns)
            X_wind_state = standardized_df
            
            regr_state = ElasticNetCV(l1_ratio=l1ratio, eps=0.0001, n_alphas=100, alphas=None, fit_intercept=True,
                                precompute='auto', max_iter=10000, tol=0.0001, cv=10, copy_X=True,
                                verbose=0, n_jobs=None, positive=True, random_state=None, selection='cyclic')
            regr_state.fit(X_wind_state, y_wind_state)

            beta_wind_offshore_state = regr_state.coef_ / scaler.scale_

            X_wind_offshore_lats_state = X_wind_cols_state.get_level_values(0)
            X_wind_offshore_lons_state = X_wind_cols_state.get_level_values(1)
            
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_offshore_state_df = pd.DataFrame(np.array([X_wind_offshore_lons_state, X_wind_offshore_lats_state, beta_wind_offshore_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_offshore_state_df.to_csv(save_layout_path + f'layout_offshore_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            yhat_wind_offshore_state = regr_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_wind_offshore_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_wind_offshore_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_wind_offshore_state_df.to_csv(save_feedin_path + f'layout_feedin_offshore_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)


# Linear Regression:            
            regr_lin_state = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=True)
            regr_lin_state.fit(X_wind_state, y_wind_state)
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            print(' RMSE OLS: ' + str(sqrt(((y_wind_state - yhat_lin_wind_state)**2).mean())))
            beta_lin_wind_state = regr_lin_state.coef_ / scaler.scale_

            X_wind_lats_state = X_wind_cols_state.get_level_values(0)
            X_wind_lons_state = X_wind_cols_state.get_level_values(1)
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_lin_state_df = pd.DataFrame(np.array([X_wind_lons_state, X_wind_lats_state, beta_lin_wind_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_lin_state_df.to_csv(save_layout_path + f'layout_lin_offshore_{state}_{start_year}.csv', sep = ';', index=False)
            
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_lin_wind_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_offshore_{state}_{start_year}.csv', sep = ';', index=False)
                        


            if pred == 1:
                try:
                    y_wind_state_yp1 = df_merged_yp1.Wind_Offshore[df_merged_yp1.country == state]
                    y_wind_state_yp1.iloc[1:] = y_wind_state_yp1.iloc[0:-1]
                    X_wind_state_yp1 = df_merged_yp1.wind_power_rel[df_merged_yp1.country == state]
                    X_wind_cols_state_yp1 = X_wind_state_yp1.columns
                    X_wind_state_yp1.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_cols_state_yp1]
                    X_wind_state_yp1.fillna(0, inplace= True)
                    scaler = StandardScaler().fit(X_wind_state_yp1)
                    standardized_data = scaler.transform(X_wind_state_yp1)
                    standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state_yp1.columns)
                    X_wind_state_yp1 = standardized_df
                    yhat_wind_state_yp1 = regr_state.predict(X_wind_state_yp1)
                    yhat_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    #yhat_wind_state_df.to_csv(save_feedin_path + f'layout_feedin_pred_offshore_{state}_{start_year+1}_{l1ratio}.csv', sep = ';', index=False)
# Prediction linear regression:                    
                    yhat_lin_wind_state_yp1 = regr_lin_state.predict(X_wind_state_yp1)
                    yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_lin_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_pred_offshore_{state}_{start_year+1}.csv', sep = ';', index=False)

                    pred = 1
                except:
                    print('No prediction for ' + state)
                    
            print('state: ' + state + ' l1ratio: ' + str(regr_state.l1_ratio_) + ' alpha: ' + str(regr_state.alpha_) + ' RMSE: ' + str(sqrt(((y_wind_state - yhat_wind_offshore_state)**2).mean())))
            
    
    if type == 'onshore':
        df_merged = entsoe_data_grouped.merge(grid_signals, on = ['country', 'date'], how='inner')
        if pred == 1:
            df_merged_yp1 = entsoe_data_grouped_yp1.merge(grid_signals_yp1, on = ['country', 'date'], how='inner')

        for state in pd.unique(df_merged['country']):
            y_wind_state = df_merged.Wind_Onshore[df_merged.country == state]
            y_wind_state.iloc[1:] = y_wind_state.iloc[0:-1]
            X_wind_state = df_merged.wind_power_rel[df_merged.country == state]
            X_wind_cols_state = X_wind_state.columns
            X_wind_state.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_state.columns]
            X_wind_state.fillna(0, inplace= True)
            scaler = StandardScaler().fit(X_wind_state)
            standardized_data = scaler.transform(X_wind_state)
            standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state.columns)
            X_wind_state = standardized_df
            
            regr_state = ElasticNetCV(l1_ratio=l1ratio, eps=0.0001, n_alphas=100, alphas=None, fit_intercept=True,
                                precompute='auto', max_iter=10000, tol=0.0001, cv=10, copy_X=True,
                                verbose=0, n_jobs=None, positive=True, random_state=None, selection='cyclic')
            regr_state.fit(X_wind_state, y_wind_state)

            beta_wind_onshore_state = regr_state.coef_ / scaler.scale_

            X_wind_onshore_lats_state = X_wind_cols_state.get_level_values(0)
            X_wind_onshore_lons_state = X_wind_cols_state.get_level_values(1)
            
            #beta_wind_state.to_csv(f'Layout_{state}.csv')
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_onshore_state_df = pd.DataFrame(np.array([X_wind_onshore_lons_state, X_wind_onshore_lats_state, beta_wind_onshore_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_onshore_state_df.to_csv(save_layout_path + f'layout_onshore_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            yhat_wind_onshore_state = regr_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_wind_onshore_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_wind_onshore_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_wind_onshore_state_df.to_csv(save_feedin_path + f'layout_feedin_onshore_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)

# Linear Regression:            
            regr_lin_state = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=True)
            regr_lin_state.fit(X_wind_state, y_wind_state)
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            print(' RMSE OLS: ' + str(sqrt(((y_wind_state - yhat_lin_wind_state)**2).mean())))
            beta_lin_wind_state = regr_lin_state.coef_ / scaler.scale_

            X_wind_lats_state = X_wind_cols_state.get_level_values(0)
            X_wind_lons_state = X_wind_cols_state.get_level_values(1)
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_lin_state_df = pd.DataFrame(np.array([X_wind_lons_state, X_wind_lats_state, beta_lin_wind_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_lin_state_df.to_csv(save_layout_path + f'layout_lin_onshore_{state}_{start_year}.csv', sep = ';', index=False)
            
            yhat_lin_wind_state = regr_lin_state.predict(X_wind_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_wind_state, yhat_lin_wind_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_onshore_{state}_{start_year}.csv', sep = ';', index=False)


            if pred == 1:
                try:
                    y_wind_state_yp1 = df_merged_yp1.Wind_Onshore[df_merged_yp1.country == state]
                    y_wind_state_yp1.iloc[1:] = y_wind_state_yp1.iloc[0:-1]
                    X_wind_state_yp1 = df_merged_yp1.wind_power_rel[df_merged_yp1.country == state]
                    X_wind_cols_state_yp1 = X_wind_state_yp1.columns
                    X_wind_state_yp1.columns = [f'{(pair[0], pair[1])}' for pair in X_wind_cols_state_yp1]
                    X_wind_state_yp1.fillna(0, inplace= True)
                    scaler = StandardScaler().fit(X_wind_state_yp1)
                    standardized_data = scaler.transform(X_wind_state_yp1)
                    standardized_df = pd.DataFrame(standardized_data, columns=X_wind_state_yp1.columns)
                    X_wind_state_yp1 = standardized_df
                    yhat_wind_state_yp1 = regr_state.predict(X_wind_state_yp1)
                    yhat_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    #yhat_wind_state_df.to_csv(save_feedin_path + f'layout_feedin_pred_onshore_{state}_{start_year+1}_{l1ratio}.csv', sep = ';', index=False)
# Prediction linear regression:                    
                    yhat_lin_wind_state_yp1 = regr_lin_state.predict(X_wind_state_yp1)
                    yhat_lin_wind_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_wind_state_yp1, yhat_lin_wind_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    yhat_lin_wind_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_pred_onshore_{state}_{start_year+1}.csv', sep = ';', index=False)

                    pred = 1
                except:
                    print('No prediction for ' + state)
                    
            print('state: ' + state + ' l1ratio: ' + str(regr_state.l1_ratio_) + ' alpha: ' + str(regr_state.alpha_) + ' RMSE: ' + str(sqrt(((y_wind_state - yhat_wind_onshore_state)**2).mean())))
            
    if type == 'solar':
        df_merged = entsoe_data_grouped.merge(grid_signals, on = ['country', 'date'], how='inner')
        if pred == 1:
            df_merged_yp1 = entsoe_data_grouped_yp1.merge(grid_signals_yp1, on = ['country', 'date'], how='inner')

        for state in pd.unique(df_merged['country']):
            y_solar_state = df_merged.Solar[df_merged.country == state]
            y_solar_state.iloc[1:] = y_solar_state.iloc[0:-1]
            X_solar_state = df_merged.solar_power_rel[df_merged.country == state]
            #X_solar_state = df_merged.solar_power_rel[df_merged.country == state]
            X_solar_cols_state = X_solar_state.columns
            X_solar_state.columns = [f'{(pair[0], pair[1])}' for pair in X_solar_state.columns]
            X_solar_state.fillna(0, inplace= True)
            scaler = StandardScaler().fit(X_solar_state)
            standardized_data = scaler.transform(X_solar_state)
            standardized_df = pd.DataFrame(standardized_data, columns=X_solar_state.columns)
            X_solar_state = standardized_df
        
            regr_state = ElasticNetCV(l1_ratio= l1ratio, eps=0.0001, n_alphas=100, alphas=None, fit_intercept=True, 
                                      precompute='auto', max_iter=10000, tol=0.0001, cv=10, copy_X=True,
                                verbose=0, n_jobs=None, positive=True, random_state=None, selection='cyclic')
            regr_state.fit(X_solar_state, y_solar_state)
        
            beta_solar_state = regr_state.coef_ / scaler.scale_
        
            X_solar_lats_state = X_solar_cols_state.get_level_values(0)
            X_solar_lons_state = X_solar_cols_state.get_level_values(1)
            
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_state_df = pd.DataFrame(np.array([X_solar_lons_state, X_solar_lats_state, beta_solar_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_state_df.to_csv(save_layout_path + f'layout_solar_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            yhat_solar_state = regr_state.predict(X_solar_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_solar_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_solar_state, yhat_solar_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_solar_state_df.to_csv(save_feedin_path + f'Test45_layout_feedin_solar_{start_year}_{state}_{l1ratio}_02.csv', sep = ';', index=False)
            #yhat_solar_state_df.to_csv(save_feedin_path + f'layout_feedin_solar_{state}_{start_year}_{l1ratio}.csv', sep = ';', index=False)
            
            print('state: ' + state + ' l1ratio: ' + str(regr_state.l1_ratio_) + ' alpha: ' + str(regr_state.alpha_) + ' RMSE: ' + str(sqrt(((y_solar_state - yhat_solar_state)**2).mean())))

            regr_lin_state = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=True)
            regr_lin_state.fit(X_solar_state, y_solar_state)
            yhat_lin_solar_state = regr_lin_state.predict(X_solar_state)
            print('RMSE OLS: ' + str(sqrt(((y_solar_state - yhat_lin_solar_state)**2).mean())))
            beta_lin_solar_state = regr_lin_state.coef_ / scaler.scale_

            X_solar_lats_state = X_solar_cols_state.get_level_values(0)
            X_solar_lons_state = X_solar_cols_state.get_level_values(1)
            ColumnsLayout = ['longitude', 'latitude', 'beta', 'sigma']
            layout_lin_state_df = pd.DataFrame(np.array([X_solar_lons_state, X_solar_lats_state, beta_lin_solar_state, scaler.scale_]).T, columns = ColumnsLayout)
            #layout_lin_state_df.to_csv(save_layout_path + f'layout_lin_solar_{state}_{start_year}.csv', sep = ';', index=False)
            
            yhat_lin_solar_state = regr_lin_state.predict(X_solar_state)
            ColumnsLayoutFeedin = ['country', 'date', 'actual', 'elnet']
            yhat_lin_solar_state_df = pd.DataFrame(np.array([df_merged.country[df_merged.country == state], df_merged.date[df_merged.country == state], y_solar_state, yhat_lin_solar_state]).T, columns = ColumnsLayoutFeedin)
            #yhat_lin_solar_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_solar_{state}_{start_year}.csv', sep = ';', index=False)
            
            
# Prediction:             
            if pred == 1:
                try:
                    y_solar_state_yp1 = df_merged_yp1.Solar[df_merged_yp1.country == state]
                    y_solar_state_yp1.iloc[1:] = y_solar_state_yp1.iloc[0:-1]
                    X_solar_state_yp1 = df_merged_yp1.solar_power_rel[df_merged_yp1.country == state]
                    X_solar_cols_state_yp1 = X_solar_state_yp1.columns
                    X_solar_state_yp1.columns = [f'{(pair[0], pair[1])}' for pair in X_solar_cols_state_yp1]
                    X_solar_state_yp1.fillna(0, inplace= True)
                    scaler = StandardScaler().fit(X_solar_state_yp1)
                    standardized_data = scaler.transform(X_solar_state_yp1)
                    standardized_df = pd.DataFrame(standardized_data, columns=X_solar_state_yp1.columns)
                    X_solar_state_yp1 = standardized_df
                    yhat_solar_state_yp1 = regr_state.predict(X_solar_state_yp1)
                    yhat_solar_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_solar_state_yp1, yhat_solar_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    #yhat_solar_state_df.to_csv(save_feedin_path + f'layout_feedin_pred_solar_{state}_{start_year+1}_{l1ratio}.csv', sep = ';', index=False)
# Prediction linear regression:                    
                    yhat_lin_solar_state_yp1 = regr_lin_state.predict(X_solar_state_yp1)
                    yhat_lin_solar_state_df = pd.DataFrame(np.array([df_merged_yp1.country[df_merged_yp1.country == state], df_merged_yp1.date[df_merged_yp1.country == state], y_solar_state_yp1, yhat_lin_solar_state_yp1]).T, columns = ColumnsLayoutFeedin)
                    yhat_lin_solar_state_df.to_csv(save_feedin_path + f'layout_lin_feedin_pred_solar_{state}_{start_year+1}.csv', sep = ';', index=False)
                    pred = 1
                except:
                    print('No prediction for ' + state)
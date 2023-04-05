import os.path
import os
import pandas as pd
import numpy as np
import requests
import xmltodict

#############################Entso-e data######################################################

def download_entsoe(entsoe_path, entsoe_token, start_year, end_year):
    """
    Download and sort actual generation data from ENTSO-E via API
    
    Parameters
    ----------
    entsoe_path
    entsoe_token
    start_year
    end_year
    
    Returns
    -------
    entsoe_data:            Dataframe, actual generation data for onshore and offshore wind and PV 
    entsoe_data_grouped:    Dataframe, sorted actual generation data for onshore and offshore wind and PV 
    """
    
    if os.path.exists(entsoe_path):
        return
    global entsoe_data
    print('downloading entsoe...')
    domain_map = {
        'POR': '10YPT-REN------W',
        'ESP': '10YES-REE------0',
        'FRA': '10YFR-RTE------C',
        'BEL': '10YBE----------2',
        'CHE': '10YCZ-CEPS-----N',
        'LUX': '10YLU-CEGEDEL-NQ',
        'NLD': '10YNL----------L',
        'ITA': '10YIT-GRTN-----B',
        'DEU': '10Y1001A1001A83F',
        'AUT': '10YAT-APG------L',
        'DNK': '10Y1001A1001A65H',
        'CZE': '10YCZ-CEPS-----N',
        'POL': '10YPL-AREA-----S',
        'HUN': '10YHU-MAVIR----U',
        'SVK': '10YSK-SEPS-----K',
        'SVN': '10YSI-ELES-----O',
        'HRV': '10YHR-HEP------M',
        'GRC': '10YGR-HTSO-----Y',
        'ALB': '10YAL-KESH-----5',
        'MKD': '10YMK-MEPSO----8',
        'BGR': '10YCA-BULGARIA-R',
        'MNE': '10YCS-CG-TSO---S',
        'BIH': '10YBA-JPCC-----D',
        'SRB': '10YCS-SERBIATSOV',
        'ROU': '10YRO-TEL------P'
    }

    #https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_generation_domain
    #section 4.4.8

    entsoe_url = 'https://web-api.tp.entsoe.eu/api?'

    token = entsoe_token
    document_type = 'A74'
    process_type = 'A16'

    periods_start = [str(year) + '01010000' for year in list(range(start_year, end_year+2))]

    psr_type_wind_on = 'B19'
    psr_type_wind_off = 'B18'
    psr_type_solar = 'B16'


    psr_types = [psr_type_wind_off, psr_type_solar, psr_type_wind_on, ]


    entsoe_data = pd.DataFrame()
    print(periods_start)
    for domain_key, domain_value in domain_map.items():
        print(1)
        for psr_type in psr_types:
            print(2)
            for i in range(len(periods_start)-1):
                print(3)
                period_start = periods_start[i]
                period_end = periods_start[i+1]

                params = {
                    'securityToken': token,
                    'documentType': document_type,
                    'processType': process_type,
                    'periodStart': period_start,
                    'periodEnd': period_end,
                    'In_Domain': domain_value,
                    'psrType': psr_type
                }

                print(params)
                r= requests.get(entsoe_url, params= params)

                if r.ok:

                    r_dict = xmltodict.parse(r.content)
                    try:
                        df = pd.Series(r_dict['GL_MarketDocument']['TimeSeries']['Period']['Point']).apply(pd.Series)

                        start = r_dict['GL_MarketDocument']['TimeSeries']['Period']['timeInterval']['start']
                        end = r_dict['GL_MarketDocument']['TimeSeries']['Period']['timeInterval']['end']

                        period_resolution = r_dict['GL_MarketDocument']['TimeSeries']['Period']['resolution']

                    except:
                        try:
                            df = pd.Series(r_dict['GL_MarketDocument']['TimeSeries'][0]['Period']['Point']).apply(pd.Series)

                            start = r_dict['GL_MarketDocument']['TimeSeries'][0]['Period']['timeInterval']['start']
                            end = r_dict['GL_MarketDocument']['TimeSeries'][0]['Period']['timeInterval']['end']

                            period_resolution = r_dict['GL_MarketDocument']['TimeSeries'][0]['Period']['resolution']
                        except:
                            print('Error!!!')
                            print('URL: ', r.url)
                            continue
                    period_len = None
                    if period_resolution == 'PT15M':
                        period_len = '15min'
                    elif period_resolution == 'PT60M':
                        period_len = '60min'

                    df['date'] = pd.date_range(start = start, end = end, freq=period_len, inclusive='left')
                    df['country'] = domain_key
                    df['psrType'] = psr_type
                    df.drop('position', axis=1, inplace=True)

                    entsoe_data = pd.concat([entsoe_data, df])

                else:
                    print('response not ok!')
                    print('URL: ', r.url)
    entsoe_data['date'] = pd.to_datetime(entsoe_data.date)
    entsoe_data.to_csv(entsoe_path, index=False, sep=';')
    
    
    entsoe_data['date'] = pd.to_datetime(entsoe_data.date)

     
    entsoe_data_wide = pd.pivot(entsoe_data, index=['country','date'], columns=['psrType'], values= 'quantity').reset_index()
    entsoe_data_wide.fillna(0, inplace=True)
    entsoe_data_wide['Wind'] = entsoe_data_wide.B18.astype(int) + entsoe_data_wide.B19.astype(int)

    entsoe_data_wide.rename({'B16': 'Solar', 'B18': 'Wind_Offshore', 'B19': 'Wind_Onshore'}, axis=1, inplace=True)
    entsoe_data_wide.Solar = entsoe_data_wide.Solar.astype(int)
    entsoe_data_wide.Wind_Offshore = entsoe_data_wide.Wind_Offshore.astype(int)
    entsoe_data_wide.Wind_Onshore = entsoe_data_wide.Wind_Onshore.astype(int)

      #entsoe_data_grouped = entsoe_data_wide.groupby(['country', pd.Grouper(key='date', axis=0, freq='h')])[['Wind', 'Solar']].mean().reset_index()
      #entsoe_data_grouped.date = pd.to_datetime(entsoe_data_grouped.date)
     
    entsoe_data_grouped = entsoe_data_wide.groupby(['country', pd.Grouper(key='date', axis=0, freq='h')])[['Wind_Offshore', 'Wind_Onshore', 'Wind', 'Solar']].mean().reset_index()
    entsoe_data_grouped.date = pd.to_datetime(entsoe_data_grouped.date)
     
      #entsoe_data_grouped.to_csv(save_data_path + f'entsoe_data_grouped_{start_year}_{end_year}.csv', sep = ';', index=False)
     
    #entsoe_data_grouped = pd.read_csv(save_data_path + f'entsoe_data_grouped_{start_year}_{end_year}.csv', sep = ';')
    #entsoe_data_grouped.date = pd.to_datetime(entsoe_data_grouped.date)

    entsoe_data_grouped.columns = pd.MultiIndex.from_product([entsoe_data_grouped.columns, [''], ['']])
    del entsoe_data_wide #, relevant_era5_df#, grids, relevant_era5_df, relevant_grids, grids_all
     

    return entsoe_data, entsoe_data_grouped
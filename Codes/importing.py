"""
This code create a common database depending of the country of the original dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
import os
import datetime
from sys import exit



def importing(catchment:str, country:str):
    """Load variables (PP, PET, Q) from US or CL datset
    Parameters
    ----------
    catchment : string
        ID of the basin
    country : string
        Selection of the country (US: USA or CL: Chile)

    Returns
    -------
    pd.DataFrame
        PP: Precipitation in mm/day
        PET: Potential evapotranspiration in mm/day (by Hargreaves & Samani equation)
        Q: Streamflow in mm/day

    """

    # Checking the format of the catchment name
    if country == "CL":
        catchment = str(int(catchment))
    elif country == "US":
        if isinstance(catchment, str):
            if len(catchment) == len(str(int(catchment))) and int(catchment)<10000000:
                catchment = '0' + catchment
        else:
            if catchment<10000000:
                catchment = '0' + str(catchment)
            else:
                catchment = str(catchment)

    #Definition of the path
    if country == "CL":
        #path = './data/CL/'
        print('Hydro-LSTM has not been tested in chilean catchments')
        exit(0)
        
    elif country == "US":
        current_path = os.getcwd()
        path = os.path.abspath(os.path.join(current_path, os.pardir)) + '/data/'
        #path = './data/'
    path = Path(path)

    # Loading data
    # Path
    if country == "US":
        HUC_number_path = path / 'camels_attributes_v2.0' / 'camels_name.txt' # os.getcwd() / 
        col_names = ['gauge_id','huc_02','gauge_name']
        HUC_number = pd.read_csv(HUC_number_path, sep=';', header=1, names=col_names)
        HUC_number.index = HUC_number.gauge_id.values        

        HUC = HUC_number.huc_02[int(catchment)]
        if HUC<10:
            HUC = '0' + str(HUC)
        else:
            HUC = str(HUC)
      
        forcing_file = catchment + '_lump_maurer_forcing_leap.txt'
        forcing_path = path / 'basin_mean_forcing' / 'maurer_extended' / HUC # I have to generalize that
        forcing_path = os.getcwd() / forcing_path / forcing_file
        
        Q_file = catchment + '_streamflow_qc.txt'
        Q_path = path / 'usgs_streamflow' / HUC # I have to generalize that
        Q_path = os.getcwd() / Q_path / Q_file

        Topo_file = 'camels_topo.txt'
        Topo_path = path / 'camels_attributes_v2.0'
        Topo_path = os.getcwd() / Topo_path / Topo_file


    elif country == "CL":
        PP_forcing_path = path / '4_CAMELScl_precip_cr2met' / '4_CAMELScl_precip_cr2met.txt'
        Q_forcing_path = path / '3_CAMELScl_streamflow_mm' / '3_CAMELScl_streamflow_mm.txt'
        PET_forcing_path = path / '12_CAMELScl_pet_hargreaves' / '12_CAMELScl_pet_hargreaves.txt'

    #Reading the files
    if country == "US":
        col_names = ['gauge_id','gauge_lat','gauge_lon','elev_mean','slope_mean','area_gages2','area_geospa_fabric']
        Topo_df = pd.read_csv(Topo_path, sep=';', header=1, names=col_names)
        Topo_df.index = Topo_df.gauge_id
        lat = Topo_df.gauge_lat[int(catchment)]*2*np.pi/360

        forcing_df = pd.read_csv(forcing_path, sep='\s+', header=3)
        dates = (forcing_df.Year.map(str) + "/" + forcing_df.Mnth.map(str) + "/" + forcing_df.Day.map(str))
        jul = pd.to_datetime(forcing_df.Year.map(str), format="%Y/%m/%d")
        forcing_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
        forcing_df = forcing_df.drop(['Year','Mnth','Day','Hr','dayl(s)'], axis=1)
        forcing_df['basin'] = int(catchment)

        PP_df = forcing_df.loc[:,{'basin', 'prcp(mm/day)'}]
        PP_df.rename(columns={"prcp(mm/day)": 'PP'}, inplace=True)

        forcing_df['PET'] = 0.408*0.0023*(forcing_df['tmax(C)'] - forcing_df['tmin(C)'])**0.5*(0.5*forcing_df['tmax(C)'] + 0.5*forcing_df['tmin(C)'] + 17.8)
        forcing_df['julian'] = pd.DatetimeIndex(forcing_df.index).to_julian_date() - pd.DatetimeIndex(jul).to_julian_date() + 1
        forcing_df['gamma'] = 0.4093*np.sin(2*np.pi*forcing_df.julian/365 - 1.405)
        forcing_df['hs'] = np.arccos(-np.tan(lat)*np.tan(forcing_df.gamma))
        forcing_df['PET'] = 3.7595*10*(forcing_df.hs*np.sin(lat)*np.sin(forcing_df.gamma)+np.cos(lat)*np.cos(forcing_df.gamma)*np.sin(forcing_df.hs))*forcing_df.PET

        forcing_df['basin'] = int(catchment)

        PP_df = forcing_df.loc[:, {'basin', 'prcp(mm/day)'}]
        PP_df.rename(columns={'prcp(mm/day)': 'PP'}, inplace=True)
        PET_df = forcing_df.loc[:, {'basin', 'PET'}]

        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])
        col_names = ['basin', 'Year', 'Mnth', 'Day', 'Q_obs', 'flag']
        Q_df = pd.read_csv(Q_path, sep='\s+', header=None, names=col_names)
        dates = (Q_df.Year.map(str) + "/" + Q_df.Mnth.map(str) + "/" + Q_df.Day.map(str))
        Q_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
        Q_df = Q_df.drop(['Year', 'Mnth', 'Day', 'flag'], axis=1)
        Q_df.Q_obs = 28316846.592 * Q_df.Q_obs * 86400 / (area * 10 ** 6)

    elif country == "CL":
        PP_df = pd.read_csv(PP_forcing_path, sep='\t')
        PP_df.rename(columns={'gauge_id': 'date'}, inplace=True)
        PP_df.index = pd.to_datetime(PP_df['date'], format="%Y/%m/%d")
        PP_df['basin'] = int(catchment)
        PP_df = PP_df.loc[:,{'basin', catchment}]
        PP_df.rename(columns={catchment: 'PP'}, inplace=True)

        PET_df = pd.read_csv(PET_forcing_path, sep='\t')
        PET_df.rename(columns={'gauge_id': 'date'}, inplace=True)
        PET_df.index = pd.to_datetime(PET_df['date'], format="%Y/%m/%d")
        PET_df['basin'] = int(catchment)
        PET_df = PET_df.loc[:,{'basin', catchment}]
        PET_df.rename(columns={catchment: 'PET'}, inplace=True)

        Q_df = pd.read_csv(Q_forcing_path, sep='\t', low_memory=False)
        Q_df.rename(columns={'gauge_id': 'date'}, inplace=True)
        Q_df.index = pd.to_datetime(Q_df['date'], format="%Y/%m/%d")
        Q_df['basin'] = int(catchment)
        Q_df = Q_df.loc[:,{'basin', catchment}]
        Q_df.rename(columns={catchment: 'Q_obs'}, inplace=True)

    return PP_df, PET_df, Q_df


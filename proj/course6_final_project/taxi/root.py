import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import course6_final_project
from course6_final_project.taxi.preprocess import preprocess_files, get_region


def header():
    return 'NEW YORK TAXI: Predict Demand Time Series';


def run():

    def week1():
        fname = 'yellow_tripdata_2016-05.csv'
        preprocess_files([fname])

        fname = 'reg.'+fname
        regstat_df = pd.read_csv(utils.PATH.STORE_FOR(6, fname, 'taxi'),
                                 index_col=['hour'],
                                 parse_dates=['hour'])

        print(regstat_df.head())

        # 40.748425, -73.985568   - Empire State Building
        empst_reg = get_region({'pickup_latitude': 40.748425, 'pickup_longitude': -73.985568})
        t = regstat_df.index.values
        s = regstat_df[str(empst_reg)]
        plt.plot(t, s)
        plt.show()

        zero_cnt = (regstat_df.values == 0).sum()
        print(zero_cnt, regstat_df.shape[0]*regstat_df.shape[1])

        return
    #week1()
	
    def week2():
		
        fname = 'yellow_tripdata_2016-05.csv'
        preprocess_files([fname])

        fname = 'reg.'+fname
        regstat_df = pd.read_csv(utils.PATH.STORE_FOR(6, fname, 'taxi'),
                                 index_col=['hour'],
                                 parse_dates=['hour'])
        
        regstat_aggr = regstat_df.sum()
        
        print(sum(regstat_aggr == 0))
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
	    
        west, south, east, north = -74.26, 40.50, -73.70, 40.92
 
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)
 
        m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
                    llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
        x, y = m(uber_data['Lon'].values, uber_data['Lat'].values)
        m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd_r);
		
        return
	
    week2()
	
	
    return



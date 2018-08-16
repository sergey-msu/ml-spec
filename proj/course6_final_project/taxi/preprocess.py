import os
import utils
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import binned_statistic_2d


def preprocess_files(fnames):

    for fname in fnames:

        # 1. clean data

        fpath = utils.PATH.STORE_FOR(6, fname, 'taxi')

        if not os.path.exists(fpath):
            print('read file \'{0}\' contents...'.format(fpath))
            df = pd.read_csv(utils.PATH.COURSE_FILE(6, 'yellow_tripdata_2016-05.csv', 'taxi'),
                             parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

            print('cleaning data...')
            df = clean_data(df)

            print('saving file...')
            df.to_csv(fpath, header=True, index=None)

        # 2. fill region/hour table

        fpath = utils.PATH.STORE_FOR(6, 'reg.'+fname, 'taxi')

        if not os.path.exists(fpath):
            print('read file \'{0}\' contents...'.format(fpath))
            df = pd.read_csv(utils.PATH.STORE_FOR(6, fname, 'taxi'),
                             parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

            print('filling region/hour data...')
            regstat_df = fill_region_table(df)

            print('saving file...')
            regstat_df.to_csv(fpath, header=True)

        print('preprocessing finished')
    return


def clean_data(df):

    df = df[(df['passenger_count'] > 0) &
            (df['trip_distance'] > 0) &
            (df['tpep_pickup_datetime'] < df['tpep_dropoff_datetime']) &
            (df['pickup_longitude'] >= -74.25559) &
            (df['pickup_longitude'] <= -73.70001) &
            (df['pickup_latitude'] >= 40.49612) &
            (df['pickup_latitude'] <= 40.91553)]

    df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].map(lambda x: x.replace(second=0, minute=0))

    df.dropna(inplace=True)

    #df['region'] = df.apply(get_region, axis=1)

    return df


def fill_region_table(df):

    dtime = df['tpep_pickup_datetime'][0]
    hours = get_hour_range(dtime)
    total = len(hours)

    result_data = np.zeros((total, 2500))

    for i, hour in enumerate(hours):
        hour_df = df[df['tpep_pickup_datetime'] == hour]
        lons = hour_df['pickup_longitude']
        lats = hour_df['pickup_latitude']
        H = binned_statistic_2d(lons, lats, None,
                                range=[[-74.25559, -73.70001], [40.49612, 40.91553]],
                                bins=50,
                                statistic='count')

        result_data[i, :] = H.statistic.reshape((-1,))

    regstat_df = pd.DataFrame(result_data, columns=list(range(1, 2501)), dtype=int)
    regstat_df['hour'] = hours
    regstat_df.set_index('hour', inplace=True)

    return regstat_df


def get_hour_range(date):
    start_date = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date   = start_date + relativedelta(months=1)
    tot_hours = (end_date - start_date).days*24

    return [ start_date + relativedelta(hours=i) for i in range(tot_hours) ]


def get_region(row, reg_df = pd.read_csv(utils.PATH.COURSE_FILE(6, 'regions.csv', 'taxi'), sep=';')):
    lat, long = row['pickup_latitude'], row['pickup_longitude']
    return reg_df.loc[(reg_df['west'] <= long) &
                      (long < reg_df['east']) &
                      (reg_df['south'] <= lat) &
                      (lat < reg_df['north']), 'region'].item()

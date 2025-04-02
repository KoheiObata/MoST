import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
import os
import datetime
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data
import random

def time_features(dates, freq='h'):
    dates['month'] = dates.date.apply(lambda row:row.month,1)
    dates['day'] = dates.date.apply(lambda row:row.day,1)
    dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
    dates['hour'] = dates.date.apply(lambda row:row.hour,1)
    dates['minute'] = dates.date.apply(lambda row:row.minute,1)
    dates['minute'] = dates.minute.map(lambda x:x//15)
    freq_map = {
        'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
        'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
        't':['month','day','weekday','hour','minute'],
    }
    return dates[freq_map[freq.lower()]].values

def load_google_data(data_dir,st,ed,interval,top_10):
    file_list=glob.glob(f'{data_dir}/*')
    df=pd.DataFrame()
    dates=pd.date_range(start=st, end=ed-interval, freq='D')
    # dates=pd.DataFrame(dates, columns={'date'})
    dates=pd.DataFrame(dates)
    dates=dates.rename(columns={0:'date'})
    for filepath in file_list:
        name=filepath.split('/')[-1].replace('.csv','')
        df_temp=pd.read_csv(filepath)
        df_temp=df_temp[[*top_10,'date']]
        df_temp['name']=name
        df_temp['date']=pd.to_datetime(df_temp['date'])
        df_temp=df_temp.loc[(df_temp['date']>=st)&(df_temp['date']<ed)]
        df_temp=pd.merge_asof(dates, df_temp, on='date')
        df_temp=df_temp.set_index('date')
        df_temp=df_temp[~df_temp.index.duplicated(keep='first')]
        print(name,len(df_temp))
        df=pd.concat([df,df_temp],axis=0)

    name_list=df['name'].unique()
    data=np.empty((int(len(df)/len(name_list)),len(top_10),len(name_list)))
    for i,name in enumerate(name_list):
        X=df.loc[df['name']==name].drop(['name'],axis=1)
        data[:,:,i]=X
    return data, dates

def get_area(area):
    if area=='country':
        top_10=["US", "CN", "JP", "DE", "IN", "GB", "FR", "BR", "IT", "CA", "KR", "AU", "MX", "ES", "ID", "RU", "NL", "CH", "SA", "TR", "NG", "SE", "PL", "AR", "BE", "TH", "IR", "AT", "NO", "IL", "AE", "ZA", "IE", "DK", "SG", "MY", "PH", "CL", "CO", "BD", "FI", "PK", "VN", "EG", "CZ", "RO", "UA", "NG", "KW", "PT"]
    elif area=='region':
        top_10=['US-AL', 'US-AK', 'US-AZ', 'US-AR', 'US-CA', 'US-CO', 'US-CT', 'US-DE', 'US-FL', 'US-GA', 'US-HI', 'US-ID', 'US-IL', 'US-IN', 'US-IA', 'US-KS', 'US-KY', 'US-LA', 'US-ME', 'US-MD', 'US-MA', 'US-MI', 'US-MN', 'US-MS', 'US-MO', 'US-MT', 'US-NE', 'US-NV', 'US-NH', 'US-NJ', 'US-NM', 'US-NY', 'US-NC', 'US-ND', 'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI', 'US-SC', 'US-SD', 'US-TN', 'US-TX', 'US-UT', 'US-VT', 'US-VA', 'US-WA', 'US-WV', 'US-WI', 'US-WY']
    return top_10


def load_air_data(DATADIR,st,ed):
    file_list=glob.glob(f'{DATADIR}/*')
    d=len(file_list)
    df=pd.DataFrame()
    for file_path in file_list:
        location=file_path.split('/')[-1].split('_')[2]
        df_temp=pd.read_csv(file_path)
        df_temp['date']=df_temp['year'].astype('str')
        df_temp['date']=df_temp['date'].str.cat([df_temp['month'].astype('str').str.zfill(2), df_temp['day'].astype('str').str.zfill(2), df_temp['hour'].astype('str').str.zfill(2)],sep='-')
        df_temp['date']=pd.to_datetime(df_temp['date'],format='%Y-%m-%d-%H')

        df_temp=df_temp.loc[(df_temp['date']>=st) & (df_temp['date']<ed)]
        dates=df_temp[['date']]

        df_temp=df_temp[['PM2.5','PM10','SO2','NO2','CO','O3','station']]

        df_temp=df_temp.interpolate()
        df_temp=df_temp.fillna(method='bfill')
        df_temp=df_temp.fillna(method='pad')
        df=pd.concat([df,df_temp],axis=0)

    data=np.empty((int(len(df)/d),d,6))
    for i,file_path in enumerate(file_list):
        location=file_path.split('/')[-1].split('_')[2]
        X=df.loc[df['station']==location].drop(['station'],axis=1).to_numpy()
        data[:,i:(i+1),:]=X[:,np.newaxis,:]

    return data, dates


class HazeData(data.Dataset):
    def __init__(self, data_dir, data_start_time, data_end_time, start_time, end_time):
        self.knowair_fp = data_dir
        self.data_start = self._get_time(data_start_time)
        self.data_end = self._get_time(data_end_time)
        self.start_time = self._get_time(start_time)
        self.end_time = self._get_time(end_time)

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()

        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)


    def _process_feature(self):
        metero_var = ['100m_u_component_of_wind', '100m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'boundary_layer_height', 'k_index', 'relative_humidity+950', 'relative_humidity+975', 'specific_humidity+950', 'surface_pressure', 'temperature+925', 'temperature+950', 'total_precipitation', 'u_component_of_wind+950', 'v_component_of_wind+950', 'vertical_velocity+950', 'vorticity+950']
        metero_use = ['2m_temperature', 'boundary_layer_height', 'k_index', 'relative_humidity+950', 'surface_pressure', 'total_precipitation', 'u_component_of_wind+950', 'v_component_of_wind+950',]
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], 184 ,axis=1)
        w_arr = np.repeat(w_arr[:, None], 184, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

        print('feature',self.feature.shape)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

        print('time_arr',self.time_arr.shape)
        print('time_arrow',len(self.time_arrow))

    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

        print('time_arr',self.time_arr.shape)

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

        print('knowair',self.knowair.shape)
        print('feature',self.feature.shape)
        print('pm25',self.pm25.shape)

    def _get_idx(self, t):
        t0 = self.data_start
        # return int((t.timestamp - t0.timestamp) / (60 * 60 * 3))
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime.datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]

    def get_feature_dates(self):
        dates=[]
        for i in range(len(self.time_arrow)):
            dates.append(self.time_arrow[i].datetime)
        # dates=pd.DataFrame(dates, columns={'date'})
        dates=pd.DataFrame(dates)
        dates=dates.rename(columns={0:'date'})
        dates['date'] = dates['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
        dates['date']=pd.to_datetime(dates['date'], format='%Y-%m-%d %H:%M')
        return self.feature, dates

def load_forecast_tensor(name, args):
    if name in ('nyccb'):
        data = np.load(f'datasets/nyccb/data.npy')
        dates = pd.read_csv(f'datasets/nyccb/date.csv')
        dates['date'] = pd.to_datetime(dates['date'])

        st=datetime.datetime(year=2015,month=1,day=1)
        train_ed=datetime.datetime(year=2016,month=1,day=1)
        valid_ed=datetime.datetime(year=2017,month=1,day=1)
        test_ed=datetime.datetime(year=2018,month=1,day=1)
        pred_lens = [8, 56, 112, 224, 448] #1,7,14,28,56day (-,1,2,4,8week)

        train_length=len(dates[(dates['date']>=st) & (dates['date']<train_ed)])
        valid_length=len(dates[(dates['date']>=train_ed) & (dates['date']<valid_ed)])
        test_length=len(dates[(dates['date']>=valid_ed) & (dates['date']<test_ed)])
        freq='h'
        dates=time_features(dates, freq=freq)
        print('data',data.shape)
        print('dates',dates.shape)
        print('nan',np.count_nonzero(np.isnan(data)))

        t,d1,d2 = data.shape # time, location, keywords
        data = np.reshape(data, (t, d1*d2))

        print('train,valid,test',train_length,valid_length,test_length)
        train_slice = slice(None, train_length)
        valid_slice = slice(train_length, train_length+valid_length)
        test_slice = slice(train_length+valid_length, train_length+valid_length+test_length)
        n_covariate_cols = dates.shape[-1]

    if name in ('knowairweek1', 'knowairweek2', 'knowairweek3'):
        DATADIR=f'datasets/KnowAir.npy'
        data_start_time = [[2015, 1, 1, 0, 0], 'GMT']
        data_end_time = [[2018, 12, 31, 21, 0], 'GMT']
        haze=HazeData(DATADIR, data_start_time, data_end_time, data_start_time, data_end_time)
        data, dates = haze.get_feature_dates()
        if name=='knowairweek1':
            data=data[:,:50,:]
            pred_lens = [56, 112, 224, 448] #7,14,28,56day (1,2,4,8week)
        elif name=='knowairweek2':
            data=data[:,:100,:]
            pred_lens = [56, 112, 224, 448] #7,14,28,56day (1,2,4,8week)
        elif name=='knowairweek3':
            data=data
            pred_lens = [56, 112, 224, 448] #7,14,28,56day (1,2,4,8week)

        st=datetime.datetime(year=2015,month=1,day=1)
        train_ed=datetime.datetime(year=2016,month=1,day=1)
        valid_ed=datetime.datetime(year=2017,month=1,day=1)
        test_ed=datetime.datetime(year=2018,month=1,day=1)

        train_length=len(dates[(dates['date']>=st) & (dates['date']<train_ed)])
        valid_length=len(dates[(dates['date']>=train_ed) & (dates['date']<valid_ed)])
        test_length=len(dates[(dates['date']>=valid_ed) & (dates['date']<test_ed)])
        freq='h'
        dates=time_features(dates, freq=freq)
        print('data',data.shape)
        print('dates',dates.shape)
        print('nan',np.count_nonzero(np.isnan(data)))

        t,d1,d2 = data.shape # time, location, keywords
        data = np.reshape(data, (t, d1*d2))

        print('train,valid,test',train_length,valid_length,test_length)
        train_slice = slice(None, train_length)
        valid_slice = slice(train_length, train_length+valid_length)
        test_slice = slice(train_length+valid_length, train_length+valid_length+test_length)
        n_covariate_cols = dates.shape[-1]

    if name in ('SNS', 'music', 'apparel', 'e_commerce', 'vod', 'sweets', 'facilities'):
        if name in ('SNS', 'music', 'apparel'):
            area='country'
        elif name in ('e_commerce', 'vod', 'sweets', 'facilities'):
            area='region'
        DATADIR=f'datasets/{area}/{name}'
        st=datetime.datetime(year=2011,month=1,day=1)
        train_ed=datetime.datetime(year=2018,month=1,day=1)
        valid_ed=datetime.datetime(year=2019,month=1,day=1)
        test_ed=datetime.datetime(year=2020,month=1,day=1)
        interval=datetime.timedelta(days=1)
        freq='h'
        top_10=get_area(area)
        data, dates=load_google_data(DATADIR,st,test_ed,interval,top_10)
        dates=time_features(dates, freq=freq)
        print('data',data.shape)
        print('dates',dates.shape)
        print('nan',np.count_nonzero(np.isnan(data)))

        t,d1,d2 = data.shape # time, location, keywords
        data = np.reshape(data, (t, d1*d2))

        train_length = (train_ed - st).days
        valid_length = (valid_ed - train_ed).days
        test_length = (test_ed - valid_ed).days
        print('train,valid,test',train_length,valid_length,test_length)
        train_slice = slice(None, train_length)
        valid_slice = slice(train_length, train_length+valid_length)
        test_slice = slice(train_length+valid_length, train_length+valid_length+test_length)
        # pred_lens = [7, 14, 28, 56, 112, 224]
        pred_lens = [7]
        n_covariate_cols = dates.shape[-1]


    if name=='nyccb':
        scaler = StandardScaler().fit(data)
    else:
        scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    data = np.reshape(data, (t, d1, d2))
    data = np.expand_dims(data, 0)

    dates = np.expand_dims(dates, -2)
    dates = np.repeat(dates, data.shape[-2], axis=-2)
    dates = np.expand_dims(dates, 0)

    data = np.concatenate((dates,data),axis=-1)
    print('data',data.shape)

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, freq


def load_encode_tensor(name):
    if 'synthetic' in name:
        DATADIR=f'datasets/synthetic/{name}.npy'
        data = np.load(DATADIR)
        b,t,d1,d2 = data.shape # batch, time, location, pollutants
        data = np.reshape(data, (b*t,d1,d2))
        freq='h'
        dates = np.zeros((t,4))
        print('data',data.shape)
        print('dates',dates.shape)
        print('nan',np.count_nonzero(np.isnan(data)))

        data = np.reshape(data, (b*t, d1*d2))

        train_length = t
        valid_length = 0
        test_length = 0
        print('train,valid,test',train_length,valid_length,test_length)
        train_slice = slice(None, train_length)
        valid_slice = slice(train_length, train_length+valid_length)
        test_slice = slice(train_length+valid_length, train_length+valid_length+test_length)
        pred_lens = []
        n_covariate_cols = dates.shape[-1]

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.reshape(data, (b*t, d1, d2))
    data = np.reshape(data, (b, t, d1, d2))

    dates = np.expand_dims(dates, -2)
    dates = np.repeat(dates, data.shape[-2], axis=-2)
    dates = np.expand_dims(dates, 0)
    dates = np.repeat(dates, data.shape[0], axis=0)

    data = np.concatenate((dates,data),axis=-1)
    print('data',data.shape)
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, freq





def load_classification_tensor(name, args):
    PATH = f'datasets/{name}'

    if args.random_dims:
        if 'daily' in name: d=45
        if 'realdisp' in name: d=117
        l = list(range(d))
        l = random.sample(l, len(l))
    tr_filenames = sorted(glob.glob(f'{PATH}/train/labels/*'))
    tr_X, tr_Y = [], []
    for filename in tr_filenames:
        x_filename = filename.replace('labels','samples')
        y = np.loadtxt(filename)
        tr_Y.append(y)
        x = np.loadtxt(x_filename)
        if args.random_dims: x = x[:,l]
        if not args.non_tensor_dims: x = convert_tensor(x, name)
        tr_X.append(x)

    te_filenames = sorted(glob.glob(f'{PATH}/test/labels/*'))
    te_X, te_Y = [], []
    for filename in te_filenames:
        x_filename = filename.replace('labels','samples')
        y = np.loadtxt(filename)
        te_Y.append(y)
        x = np.loadtxt(x_filename)
        if args.random_dims: x = x[:,l]
        if not args.non_tensor_dims: x = convert_tensor(x, name)
        te_X.append(x)

    tr_Y = np.stack(tr_Y, axis=0)
    tr_X = np.stack(tr_X, axis=0)
    te_Y = np.stack(te_Y, axis=0)
    te_X = np.stack(te_X, axis=0)

    tr_X = np.concatenate((np.zeros((*tr_X.shape[:-1],4)),tr_X),axis=-1)
    te_X = np.concatenate((np.zeros((*te_X.shape[:-1],4)),te_X),axis=-1)

    n_covariate_cols = 4
    print('data',tr_X.shape)
    return tr_Y, tr_X, te_Y, te_X, n_covariate_cols


def convert_tensor(X, name):
    if 'daily' in name:
        n_units = 5
        n_sensors = 9
    elif 'realdisp' in name:
        n_units = 9
        n_sensors = 13

    tensor = np.zeros((X.shape[0], n_units, n_sensors))
    for i in range(n_units):
        tensor[:, i, :] = X[:, i*n_sensors:(i+1)*n_sensors]
    return tensor

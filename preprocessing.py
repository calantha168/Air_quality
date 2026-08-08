import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self, dataFile=None, header=True):
        self.scaler = StandardScaler()
        if dataFile is not None:
            self.raw_input = pd.read_csv(dataFile)

    def sum_missing(self, data):
        return data.isnull().sum()

    def treat_missing(self, data):
        data = data.drop(columns=['No'], errors='ignore')
        data = data.sort_values(['station', 'year', 'month', 'day', 'hour'])

        cols_to_fill = [
            'PM2.5', 'PM10', 'SO2', 'NO2', 'CO',
            'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM'
        ]

        for col in cols_to_fill:
            data[f'{col}_was_missing'] = data[col].isna().astype(int)

            # small gaps: linear interpolation within each station
            data[col] = data.groupby('station')[col].transform(
                lambda x: x.interpolate(method='linear', limit=6, limit_direction='both')
            )

            # remaining: fill in the regional avg
            regional_fill = data.groupby(['year', 'month', 'day', 'hour'])[col].transform('mean')
            data[col] = data[col].fillna(regional_fill)

            # extra
            data[col] = data.groupby('station')[col].transform(lambda x: x.ffill().bfill())

        data['RAIN_was_missing'] = data['RAIN'].isna().astype(int)
        data['RAIN'] = data['RAIN'].fillna(0)

        data['wd'] = data.groupby('station')['wd'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
        )

        return data

    def treat_outliers(self, data):
        pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        for col in pollutant_cols:
            lower = data[col].quantile(0.01)
            upper = data[col].quantile(0.99)
            data[col] = data[col].clip(lower=lower, upper=upper)
        return data

    def add_datetime(self, data):
        data = data.copy()
        data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
        return data

    def encode_categories(self, data, fit=False):
        data = data.copy()
        data['raw_station'] = data['station']
        data = pd.get_dummies(data, columns=['wd', 'station'])

        dummy_cols = [c for c in data.columns if c.startswith('wd_') or c.startswith('station_')]
        data[dummy_cols] = data[dummy_cols].astype(int)

        if fit:
            self.feature_columns = data.columns
        else:
            data = data.reindex(columns=self.feature_columns, fill_value=0)

        return data

    def scale_data(self, data, fit=False):
        numeric_cols = [
            'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
            'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
            'year', 'month', 'day', 'hour'
        ]
        if fit:
            self.scaler.fit(data[numeric_cols])
        data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        return data

    def fit(self, data):
        data = self.treat_missing(data)
        data = self.add_datetime(data)
        data = self.treat_outliers(data)
        data = self.encode_categories(data, fit=True)
        data = self.scale_data(data, fit=True)
        return data

    def transform(self, data):
        data = self.treat_missing(data)
        data = self.add_datetime(data)
        data = self.treat_outliers(data)
        data = self.encode_categories(data)
        data = self.scale_data(data, fit=False)
        return data

import glob
import os

import numpy as np
import pandas as pd
import torch
import talib

from torch.utils.data import Dataset
from typing import List, Optional, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from ..utils.data_processing import Normalizer


cur_dir = Path(__file__).parent.parent.parent

@dataclass
class StocksDatasetConfig:

    load_ensemble: Optional[bool] = field(
        default=True,
        metadata='Load data from csv files.'
    )

    dataset_dir: Optional[str] = field(
        default=os.path.join(cur_dir, 'data', 'raw'),
        metadata='Directory contains csv files.'
    )

    pre_fields: Optional[List[str]] = field(
        default_factory=lambda: ['Ticker', 'Date/Time', 'Open', 'Close', 'Low', 'High', 'Volume'],
        metadata='Needing fields to pre-process.',
    )

    post_fields: Optional[List[str]] = field(
        default_factory=lambda: ['open', 'low', 'high', 'EMA', 'SMA',  'volume'],
        metadata="Needing field take to train model."
    )

    normal_cols: Optional[List[str]] = field(
        default_factory=lambda: ['open', 'high', 'low', 'EMA', 'SMA']
    )

    outlier_cols: Optional[List[str]] = field(
        default_factory=lambda: ['volume']
    )

    target_field: Optional[str] = field(
        default_factory=lambda: ['close'],
        metadata='Field used to as target.'
    )

    averaging_fields: Optional[List[str]] = field(
        default_factory=lambda: ['EMA', 'SMA'],
        metadata='Additional averaging indices.'
    )

    separate_ticker: Optional[bool] = field(
        default=True,
        metadata='Separate stock price per ticker.'
    )

    unit: Optional['str'] = field(
        default='min',
        metadata='Unit of time being consider, \'second\', \'min\', \'hour\', \'day\'.',
    )

    timeperiod: Optional[int] = field(
        default=15,
        metadata="Number of last 'unit' price at current time."
    )

    full_dataset: Optional[bool] = field(
        default=True,
        metadata='Indicate that dataset is full, which is split into train, val, test set,'
    )

    data_raw: Optional[bool] = field(
        default=True,
        metadata='Indicate that dataset is raw, which is loaded from files.'
    )

    dtype: Optional[torch.dtype] = field(
        default=torch.float32,
        metadata='Dtype for dataset.'
    )

class StockDataset(Dataset):
    def __init__(self, config: StocksDatasetConfig, df: pd.DataFrame=None):
        super().__init__()
        self.config = config
        self.df: pd.DataFrame = None
        self.sample_df: pd.DataFrame = None
        self.normalizer = Normalizer(config=self.config)

        if not self.config.data_raw:
            self.sample_df = df
        else:
            if self.config.load_ensemble:
                self.df: pd.DataFrame = self.load_ensemble_data_csv()
            else:
                self.df = df
            self.pre_process()

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx: int):
        sample = self.sample_df.loc[idx, :].values
        return torch.tensor(sample[0]).to(self.config.dtype), torch.tensor(sample[1]).to(self.config.dtype)

    def load_data_csv(
            self,
            csv_path: str,
    ):
        df = pd.read_csv(
            filepath_or_buffer=csv_path,
            header=0,
            delimiter=',',
        )

        columns = df.columns
        common_fields = set(columns).intersection(self.config.pre_fields)
        assert common_fields, "No any needing field."

        return df.loc[:, list(common_fields)]

    def load_ensemble_data_csv(self):
        csv_paths = glob.glob(f"{self.config.dataset_dir}/*.csv")
        dfs = [self.load_data_csv(csv_path) for csv_path in csv_paths]
        return pd.concat(dfs, axis=0, ignore_index=True,)

    def pre_process(self):
        # Rename columns to lower case
        self.df.columns = self.df.columns.map(mapper=str.lower)

        # Date form YYYY/MM/DD HH:mm:SS
        self.df['date_time'] = pd.to_datetime(self.df['date/time'])
        # Timestamp in second
        self.df['timestamp'] = self.df['date_time'].apply(lambda x: x.timestamp())

        if self.config.unit == 'min':
            duration = 60
            unit = 'm'
        elif self.config.unit == 'hour':
            duration = 3600
            unit = 'h'
        elif self.config.unit == 'day':
            duration = 24*3600
            unit = 'D'
        else:
            duration = 1
            unit = 's'

        # Timestamp form unit
        self.df[f'timestamp_{self.config.unit}'] = (self.df['timestamp'] // duration).convert_dtypes('int')
        # Date form unit
        self.df['date/time'] = self.df[f'timestamp_{self.config.unit}'].apply(lambda x: pd.to_datetime(x, unit=unit))

        ticker_dfs = []
        for ticker in self.df['ticker'].unique():
            index = self.df['ticker'] == ticker
            ticker_df: pd.DataFrame = self.df.loc[index, :]
            ticker_df = ticker_df.sort_values(
                by=[f"timestamp_{self.config.unit}", "timestamp"],
                ascending=True, ignore_index=True, inplace=False
            )

            ticker_df['unit_return'] = ticker_df['close'].pct_change(periods=1)

            if self.config.averaging_fields:
                self._add_averaging_index(df=ticker_df)
                ticker_df.fillna(value=0, inplace=True)

            self.normalizer.fit(
                df=ticker_df,
                ticker=ticker,
            )
            ticker_dfs.append(ticker_df)

        self.df = pd.concat(ticker_dfs, ignore_index=True, axis=0)
        self.df.index = self.df.loc[:, 'date_time']
        self.df.drop(columns=['date_time'], inplace=True, axis=1)

    def split(
            self,
            train_ratio: float,
            val_ratio: float
    ) -> Iterable[Dataset]:
        if not self.config.data_raw:
            raise ValueError('Data is already sub-set, no need split.')

        train_dfs, val_dfs, test_dfs, sample_dfs = [], [], [], []
        for ticker in self.df['ticker'].unique():
            index = self.df['ticker'] == ticker
            ticker_df: pd.DataFrame = self.df.loc[index, :]

            ticker_df = self.normalizer(
                df=ticker_df,
                ticker=ticker
            )

            if self.config.data_raw:
                ticker_df = self._generate_timeseries(df=ticker_df)
            sample_dfs.append(ticker_df)

            ticker_train_df, ticker_val_df, ticker_test_df = self._split_df(
                ticker_df,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            train_dfs.append(ticker_train_df)
            val_dfs.append(ticker_val_df)
            test_dfs.append(ticker_test_df)

        if self.config.full_dataset:
            self.sample_df = pd.concat(sample_dfs, ignore_index=True, axis=0)

        train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
        test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

        self.config.data_raw = False
        sub_dataset_config = StocksDatasetConfig(
            load_ensemble=False,
            data_raw=False,
            full_dataset=False,
        )
        train_dataset = StockDataset(config=sub_dataset_config, df=train_df)
        val_dataset = StockDataset(config=sub_dataset_config, df=val_df)
        test_dataset = StockDataset(config=sub_dataset_config, df=test_df)

        return train_dataset, val_dataset, test_dataset

    def _generate_timeseries(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        fields = set(df.columns).intersection(set(self.config.post_fields))
        assert fields
        fields = list(fields)
        values = df.loc[:, fields].values
        values = np.pad(array=values, pad_width=((self.config.timeperiod-1, 0), (0, 0)), constant_values='0')
        targets = df.loc[:, self.config.target_field].values
        windows = []
        for i in range(0, len(values) - self.config.timeperiod,1):
            window = values[i:i+self.config.timeperiod, ...]
            windows.append((window, targets[i]))

        sample_df = pd.DataFrame(data=windows, columns=['input', 'output'])

        return sample_df

    def _split_df(
            self,
            df: pd.DataFrame,
            train_ratio: float = 0.0,
            val_ratio: float = 0.0
    ) -> Iterable[pd.DataFrame]:
        df = df.sample(frac=1).reset_index(drop=True)
        train_length: int = int(train_ratio * len(df))
        val_length: int = int(val_ratio * len(df))
        test_length: int = len(df) - train_length - val_length

        if train_length > 0:
            train_df: pd.DataFrame = df.iloc[:train_length, :].reset_index(drop=True)
        else:
            train_df = None

        if val_length > 0:
            val_df: pd.DataFrame = df.iloc[train_length:-test_length, :].reset_index(drop=True)
        else:
            val_df = None

        if test_length > 0 and val_length > 0:
            test_df: pd.DataFrame = df.iloc[-test_length:, :].reset_index(drop=True)
        else:
            test_df = None

        return train_df, val_df, test_df

    def _add_averaging_index(
            self,
            df: pd.DataFrame
    ):
        if "EMA" in self.config.averaging_fields:
            df['EMA'] = talib.EMA(df.loc[:, 'close'], timeperiod=self.config.timeperiod)
        if "SMA" in self.config.averaging_fields:
            df['SMA'] = talib.SMA(df.loc[:, 'close'], timeperiod=self.config.timeperiod)
        if "RSI" in self.config.averaging_fields:
            df['RSI'] = talib.RSI(df.loc[:, 'close'], timeperiod=self.config.timeperiod)
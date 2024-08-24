import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Mapping
from copy import copy


class OutlierClipper:
    def __init__(self, config):
        self.config = config
        self.lower_bound: float = None
        self.upper_bound: float = None
        self.scalers: Mapping[str, StandardScaler] = dict()

    def fit(
        self,
        df: pd.DataFrame
    ):

        for col in self.config.outlier_cols:
            q0, q1, q2, q3, q4 = np.percentile(
                df.loc[:, [col]].values,
                method='midpoint', axis=0, q=[0, 25, 50, 75, 100],
            )

            IQR = q3 - q1
            self.lower_bound = q1 - 1.5 * IQR
            self.upper_bound = q3 + 1.5 * IQR

            df.loc[:, self.config.outlier_cols] = np.clip(
                a=df.loc[:, [col]],
                a_min=self.lower_bound, a_max=self.upper_bound,
            )
            scaler = StandardScaler(with_std=True, with_mean=True)
            scaler.fit(df.loc[:, [col]])
            self.scalers[col] = scaler

    def __call__(self, df: pd.DataFrame):
        for col in self.config.outlier_cols:
            tmp_value = self.scalers[col].transform(df.loc[:, [col]])
            df.drop(columns=[col], inplace=True)
            df.loc[:, [col]] = tmp_value


class Normalizer:
    def __init__(self, config):
        self.config = config
        self.normalizers: Mapping[(str, str), StandardScaler] = dict()
        self.clippers: Mapping[str, OutlierClipper] = dict()

    def fit(self, df: pd.DataFrame, ticker: str):
        clipper = OutlierClipper(config=self.config)
        clipper.fit(df=df)
        self.clippers[ticker] = clipper
        for col in self.config.standard_cols:
            scaler = StandardScaler(with_std=True, with_mean=True)
            scaler.fit(X=df.loc[:, [col]])
            self.normalizers[(ticker, col)] = scaler

    def __call__(
            self,
            df: pd.DataFrame,
            ticker: str,
            **kwargs
    ):
        copy_df: pd.DataFrame = copy(df)
        self.clippers[ticker](copy_df)
        for col in self.config.standard_cols:
            scaler = self.normalizers[(ticker, col)]
            copy_df.loc[:, col] = scaler.transform(copy_df.loc[:, [col]])

        return copy_df
import abc
import time
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np 
import pandas as pd 
import tushare as ts 

from .utils import DataConfig, format_api_dt, format_local_dt, format_local_code, format_api_code


class FinDataDownloadBase(object):
    DATE_FIELD_NAME = "trade_date"
    def __init__(self, config: DataConfig):
        self.start_datetime = pd.to_datetime(config.start_datetime)
        self.data_dir = Path(config.data_dir).expanduser().resolve()

        self._calendars = []
        self._instruments = []
        self._market_data = {}
        self._financial_data = {}
    
    @abc.abstractmethod
    def _get_calendars(self) -> List[pd.Timestamp]:
        pass
    
    @abc.abstractmethod
    def _get_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        pass
    
    @abc.abstractmethod
    def _get_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def _get_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def _update_calendars(self) -> List[pd.Timestamp]:
        pass

    @abc.abstractmethod
    def _update_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        pass

    @abc.abstractmethod
    def _update_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def _update_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass
    
    def _dump_calendars(self, calendars: List[pd.Timestamp]):
        pass

    def _dump_instruments(self, instruments: List[Tuple[str, pd.Timestamp, pd.Timestamp]]):
        pass

    def _dump_market_data(self, market_data: Dict[str, pd.DataFrame]):
        pass

    def _dump_financial_data(self, financial_data: Dict[str, pd.DataFrame]):
        pass

    def download(self):
        pass

    def update(self):
        pass


class FinDataDownloadTushare(FinDataDownloadBase):
    DATA_SOURCE = "tushare"
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self._api = ts.pro_api(token=config.ts_token)
    
    def _get_calendars(self) -> List[pd.Timestamp]:
        calendars_df = self._api.query(
            "trade_cal", start_date=format_api_dt(self.start_datetime)
        )
        calendars = pd.to_datetime(calendars_df[calendars_df.is_open==1].cal_date).tolist()
        return calendars
    
    def _get_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        # 读tushare中所有上市、退市和暂停上市的个股
        instruments = []
        for status in ["L", "D", "P"]:
            df = self._api.query(
                "stock_basic", 
                exchange="", 
                list_status=status, 
                fields="ts_code,list_date,delist_date"
            )
            instruments.append(df)
        instruments_df = pd.concat(instruments)
        instruments_df.list_date = pd.to_datetime(instruments_df.list_date)
        instruments_df.delist_date = pd.to_datetime(instruments_df.delist_date)
        instruments_df.ts_code = instruments_df.ts_code.apply(format_local_code)
        instruments = list(instruments_df.itertuples(index=False, name=None))
        return instruments

    def _get_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        instrument = format_api_code(instrument, "tushare")
        lastest_datetime = format_api_dt(self._calendars[-1])
        start_datetime = format_api_dt(self.start_datetime)

        # 循环读取日线行情数据，用drop_duplicates()删去重复的temp_dt行
        temp_dt = start_datetime
        market_df = []
        while temp_dt < lastest_datetime:
            temp_market_df = ts.pro_bar(
                ts_code=instrument, 
                adj="hfq", 
                start_date=temp_dt
            )
            if len(temp_market_df) == 0:
                break
            market_df.append(temp_market_df)
            temp_dt = temp_market_df.trade_date[0]
            time.sleep(0.1)
        market_df = pd.concat(market_df).drop_duplicates()

        # 循环读取每日行情指标
        temp_dt = start_datetime
        market_indicator_df = []
        while temp_dt < lastest_datetime:
            temp_indicator_df = self._api.query(
                "daily_basic",
                ts_code=instrument,
                start_date=temp_dt
            )
            if len(temp_indicator_df) == 0:
                break
            market_indicator_df.append(temp_indicator_df)
            temp_dt = temp_indicator_df.trade_date[0]
            time.sleep(0.1)
        market_indicator_df = pd.concat(market_indicator_df).drop_duplicates()

        # 格式转换并输出
        market_df.trade_date = pd.to_datetime(market_df.trade_date)
        market_indicator_df.trade_date = pd.to_datetime(market_indicator_df.trade_date)

        market_df.rename(columns={"trade_date": self.DATE_FIELD_NAME}, inplace=True)
        market_indicator_df(columns={"trade_date": self.DATE_FIELD_NAME}, inplace=True)

        market_df = market_df.drop("ts_code", axis=1)
        market_indicator_df = market_indicator_df.drop("ts_code", axis=1)
        
        # 按交易日历对齐
        market_df = self._merge_calendar(market_df, self._calendars)
        market_indicator_df = self._merge_calendar(market_indicator_df, self._calendars)

        # 转换为{字段名: pd.Dataframe}的格式
        market_fields = market_df.columns.tolist()
        indicator_fields = market_indicator_df.columns.tolist()

        market_data = {}
        for field in market_fields:
            if field not in market_data.keys():
                market_data[field] = market_df.loc[:,[field]]

        for field in indicator_fields:
            if field not in market_data.keys():
                market_data[field] = market_indicator_df.loc[:,[field]]
        
        return market_data
    
    def _merge_calendar(self, df: pd.DataFrame, calendars: List[pd.Timestamp]):
        calendars_df = pd.DataFrame(data=calendars, columns=[self.DATE_FIELD_NAME])
        cal_df = calendars_df[
            (calendars_df[self.DATE_FIELD_NAME]>=df.index.min())&(calendars_df[self.DATE_FIELD_NAME]<=df.index.max())
        ]
        cal_df = cal_df.set_index(self.DATE_FIELD_NAME).sort_index()
        df = df.set_index(self.DATE_FIELD_NAME)
        merged_df = df.reindex(cal_df.index)
        return merged_df

    def _get_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        instrument = format_api_code(instrument, "tushare")
        lastest_datetime = format_api_dt(self._calendars[-1])
        start_datetime = format_api_dt(self.start_datetime)
  
    def _update_calendars(self) -> List[pd.Timestamp]:
        pass
   
    def _update_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        pass
    
    def _update_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    def _update_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass
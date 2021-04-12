import abc
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np 
import pandas as pd 
import tushare as ts 

from .utils import DataConfig


class FinDataDownloadBase(object):
    def __init__(self, config: DataConfig):
        self.start_datetime = config.start_datetime
        self.data_dir = Path(config.data_dir).expanduser().resolve()
    
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

    def load_exist_data(self):
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
    
    def _get_calendars(self) -> List[pd.Timestamp]:
        pass
    
    def _get_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        pass
    
    def _get_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    def _get_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass
  
    def _update_calendars(self) -> List[pd.Timestamp]:
        pass
   
    def _update_instruments(self) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        pass
    
    def _update_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass

    def _update_financial_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        pass
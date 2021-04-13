from typing import Union

import pandas as pd 
import numpy as np 

class DataConfig:
    start_datetime = "2007-01-01"
    data_dir = "./findata/"
    calendars_name = "calendars.csv"
    basic_data_dir = "basic"
    instruments_name = "instruments.csv"


def format_api_dt(datetime: Union[str, pd.Timestamp], api="tushare") -> str:
    if api == "tushare":
        if isinstance(datetime, str):
            datetime = pd.Timestamp(datetime)
        datetime = datetime.strftime(r"%Y%m%d")
    else:
        raise ValueError("Api :{} not supported.".format(api))
    return datetime

def format_local_dt(datetime: Union[str, int], api="tushare") -> pd.Timestamp:
    if api == "tushare":
        if isinstance(datetime, int):
            datetime = str(datetime)
        datetime = pd.to_datetime(datetime)
    else:
        raise ValueError("Api :{} not supported.".format(api))
    return datetime

def format_api_code(code: str, api: str = "tushare") -> str:
    if api == "tushare":
        code = ".".join([code[6:], code[:6]])
    else:
        raise ValueError("Api :{} not supported.".format(api))
    return code

def format_local_code(code: str, api: str = "tushare") -> str:
    if api == "tushare":
        code = "".join(code.split('.')[::-1])
    else:
        raise ValueError("Api :{} not supported.".format(api))
    return code

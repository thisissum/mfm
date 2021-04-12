from typing import Union

import pandas as pd 
import numpy as np 

class DataConfig:
    start_datetime = "2007-01-01"
    data_dir = "./findata/"


def format_dt(datetime: Union[str, pd.Timestamp], api="tushare") -> str:
    if api == "tushare":
        if isinstance(datetime, str):
            datetime = pd.Timestamp(datetime)
        datetime = datetime.strftime(r"%Y%m%d")
    else:
        raise ValueError("Api :{} not supported.".format(api))
    return datetime
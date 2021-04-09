from typing import *

import pandas as pd 
import numpy as np 

class BarraRegression(object):
    def __init__(
        self, 
        ret: Union[np.ndarray, pd.DataFrame, pd.Series], 
        market_cap: Union[np.ndarray, pd.DataFrame, pd.Series], 
        industry_factor: Union[np.ndarray, pd.DataFrame, pd.Series], 
        style_factor: Union[np.ndarray, pd.DataFrame, pd.Series]
    ):
        ret, market_cap, industry_factor, style_factor = self._process_inputs(ret, market_cap, industry_factor, style_factor)
        self._ret = ret
        self._mkt_cap = market_cap
        self._ind_factor = industry_factor
        self._style_factor = style_factor

    def _process_inputs(self, ret, mkt_cap, ind, style):
        """transform inputs to 2D-array and drop null value if exist"""
        if isinstance(ret, (pd.DataFrame, pd.Series,)):
            ret = ret.values
        if isinstance(mkt_cap, (pd.DataFrame, pd.Series,)):
            mkt_cap = mkt_cap.values
        if isinstance(ind, (pd.DataFrame, pd.Series,)):
            ind = ind.values
        if isinstance(style, (pd.DataFrame, pd.Series,)):
            style = style.values

        # drop null
        ret_null = np.isnan(ret.flatten())
        ori_num = len(ret_null)
        mkt_cap_null = np.isnan(mkt_cap.flatten())
        ind_null = np.isnan(ind.reshape(ori_num, -1).sum(axis=1))
        style_null = np.isnan(style.reshape(ori_num, -1).sum(axis=1))
        rows_to_drop = ret_null | mkt_cap_null | ind_null | style_null

        ret = ret.reshape(ori_num, 1)
        mkt_cap = mkt_cap.reshape(ori_num, 1)
        ind = ind.reshape(ori_num, -1)
        style = ind.reshape(ori_num, -1)

        ret_selected = ret[rows_to_drop==False]
        mkt_cap_selected = mkt_cap[rows_to_drop==False]
        ind_selected = ind[rows_to_drop==False]
        style_selected = style[rows_to_drop==False]

        return ret_selected, mkt_cap_selected, ind_selected, style_selected
    
    def _eval_wls_weight(self):
        weight = np.sqrt(self._mkt_cap) / np.sum(np.sqrt(self._mkt_cap))
        weight = np.diag(weight)
        return weight
    
    def _eval_industry_constrain_matrix(self):
        ind_num = self._ind_factor.shape[1]
        style_num = self._style_factor.shape[1]
        constraint_matrix = np.eye(1 + ind_num + style_num)
        

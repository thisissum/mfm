from typing import Union

import pandas as pd 
import numpy as np 
from statsmodels.stats.weightstats import DescrStatsW


class BarraRegression(object):
    """
    Multifactor model regression in barra CNE5  
    Example: 
        >>> model = BarraRegression(ret_tplus1, market_cap_t, industry_factor_t, style_factor_t)
        >>> model.fit()
        >>> factor_ret = model.factor_ret
        >>> stock_specific_ret = model.stock_specific_ret
    """
    def __init__(
        self, 
        ret: Union[np.ndarray, pd.DataFrame, pd.Series], 
        market_cap: Union[np.ndarray, pd.DataFrame], 
        industry_factor: Union[np.ndarray, pd.DataFrame], 
        style_factor: Union[np.ndarray, pd.DataFrame]
    ):
        ret, market_cap, industry_factor, style_factor = self._process_inputs(ret, market_cap, industry_factor, style_factor)
        self._ret = ret
        self._mkt_cap = market_cap
        self._ind_factor = industry_factor
        self._style_factor = self._normalize(style_factor)
        self._country_factor = np.ones(ret.shape)

        self._pure_factor_portfolio_weight = None
        self._factor_ret = None
        self._stock_specific_ret = None
        self._pure_factor_portfolio_exposure = None
        self._r_square = None
    
    @property
    def pure_factor_portfolio_weight(self):
        return self._pure_factor_portfolio_weight
    
    @property
    def pure_factor_portfolio_exposure(self):
        return self._pure_factor_portfolio_exposure
    
    @property
    def factor_ret(self):
        return self._factor_ret
    
    @property
    def stock_specific_ret(self):
        return self._stock_specific_ret
    
    @property
    def r_square(self):
        return self._r_square

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
    
    def _eval_wls_weight(self) -> np.ndarray:
        weight = np.sqrt(self._mkt_cap) / np.sum(np.sqrt(self._mkt_cap))
        weight = np.diag(weight)
        return weight
    
    def _eval_industry_constrain_matrix(self) -> np.ndarray:
        ind_num = self._ind_factor.shape[1]
        style_num = self._style_factor.shape[1]
        # compute industry cap
        ind_cap = np.array([np.sum(self._ind_factor[:, i] * self._mkt_cap) for i in range(ind_num)])
        # build constrain matrix shape = (K, K-1), K = 1 + ind_num + style_num
        constrain_mat = np.eye(1 + ind_num + style_num)
        constrain_mat[ind_num, 1:(1+ind_num)] = -1 * ind_cap / np.sum(ind_cap)
        constrain_mat = np.delete(constrain_mat, ind_num, axis=1)
        return constrain_mat
    
    def _normalize(self, style_factor: np.ndarray):
        weighted_stats = DescrStatsW(style_factor, weights=self._mkt_cap.flatten())
        weighted_mu = weighted_stats.mean
        factor_std = np.std(style_factor, axis=0, ddof=1)
        return (style_factor - weighted_mu) / factor_std
    
    
    def fit(self):
        wls_weight = self._eval_wls_weight()
        constrain = self._eval_industry_constrain_matrix()
        # factor = [1, industry_factor, style_factor], shape = (N, K)
        factor = np.hstack([self._country_factor, self._ind_factor, self._style_factor])

        # pure_factor_portfolio_weight.shape = (K, N), K = 1 + ind_num + style_num, N = stock_num
        pure_factor_portfolio_weight = np.linalg.multi_dot(
            [constrain, 
            np.linalg.inv(
                np.linalg.multi_dot(
                    [constrain.T, 
                    factor.T, 
                    wls_weight, 
                    factor, 
                    constrain]
                )
            ), 
            constrain.T, 
            factor.T, 
            wls_weight]
        )
        factor_ret = np.matmul(pure_factor_portfolio_weight, self._ret) # (K, 1)
        stock_specific_ret = self._ret - np.matmul(factor, factor_ret) # (N, 1)
        pure_factor_portfolio_exposure = np.matmul(pure_factor_portfolio_weight, factor) # (K, K)

        self._pure_factor_portfolio_weight = pure_factor_portfolio_weight
        self._factor_ret = factor_ret
        self._stock_specific_ret = stock_specific_ret
        self._pure_factor_portfolio_exposure = pure_factor_portfolio_exposure
        self._r_square = 1 - np.var(stock_specific_ret, ddof=1) / np.var(self._ret, ddof=1)

        return None

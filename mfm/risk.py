import pandas as pd 
import numpy as np 


class CovEstimator(object):
    def __init__(
        self, 
        T: int, 
        factor_q: int = 15, 
        specific_q: int = 10,
        ema_alpha: float=0.5
    ):
        assert T > max(factor_q, specific_q), "T must bigger than q"
        
        self._T = T
        self._factor_ret_q = factor_q
        self._specific_ret_q = specific_q
        self._ema_alpha = ema_alpha
    
    def _demean(self, array, axis=0):
        return array - np.mean(array, axis=axis, keepdims=True)
    
    def _ema(self, array, alpha=0.05, axis=0):
        length = array.shape[axis]
        axis_to_expand = [i for i in range(len(array.shape))]
        axis_to_expand.remove(axis)
        weight = np.array([(1-alpha)**length-1] + [(1-alpha)**i*alpha for i in range(length-1)])
        weight = np.expand_dims(weight, axis_to_expand)
        return np.sum(array * weight, axis=axis)

    def _newey_west(self, ret: np.ndarray, q: int):
        # 输入矩阵形状检查
        assert len(ret.shape) == 2 and ret.shape[0] > max(self._factor_ret_q, self._specific_ret_q), \
            "Time window must longer than {}".format(max(self._factor_ret_q, self._specific_ret_q))
        
        # ret.shape = [timestep(t), factor_num(fnum)]
        t = ret.shape[0]
        ret = self._demean(ret, axis=0)

        expand_ret = np.expand_dims(ret, axis=2)
        cov_per_t = np.matmul(expand_ret, expand_ret.transpose(0,2,1)) # (t, fnum, fnum)
        
        # 指数加权平均估计样本协方差
        cov_0 = self._ema(cov_per_t, self._ema_alpha, axis=0) # (fnum, fnum)

        # 调整协方差矩阵
        adj_cov = np.zeros(cov_0.shape) # (fnum, fnum)
        for i in range(1, q+1):
            cov_lag_i = np.matmul(ret[:t-i].T, ret[i:]) / (t-i)
            cov_i = cov_lag_i + cov_lag_i.T
            weight_i = 1 - (i/(1+q))
            adj_cov += weight_i * cov_i
        
        return cov_0 + adj_cov



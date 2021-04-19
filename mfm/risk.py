import pandas as pd 
import numpy as np 


def ema(array: np.ndarray, alpha: float = 0.05, axis: int = 0):
    if axis > len(array.shape):
        raise IndexError("axis: {} out of range: {}".format(axis, len(array.shape)))
    
    length = array.shape[axis]
    axis_to_expand = [i for i in range(len(array.shape))]
    axis_to_expand.remove(axis)
    weight = np.array([(1-alpha)**length-1] + [(1-alpha)**i*alpha for i in range(length-1)])
    
    if axis_to_expand:
        weight = np.expand_dims(weight, axis_to_expand)
    
    return np.sum(array * weight, axis=axis)


class CovEstimator(object):
    def __init__(
        self, 
        T: int, 
        factor_q: int = 15, 
        specific_q: int = 10,
        ema_alpha: float=0.05,
    ):
        assert T > max(factor_q, specific_q), "T must bigger than q"
        
        self._T = T
        self._factor_ret_q = factor_q
        self._specific_ret_q = specific_q
        self._ema_alpha = ema_alpha
    
    def _demean(self, array: np.ndarray, axis: int = 0):
        return array - np.mean(array, axis=axis, keepdims=True)

    def _newey_west_adj(self, ret: np.ndarray, q: int) -> np.ndarray:
        # 输入矩阵形状检查
        assert len(ret.shape) == 2 and ret.shape[0] > q, "Time window must longer than {}".format(q)
        
        # ret.shape = [timestep(t), factor_num(fnum)]
        t = ret.shape[0]
        ret = self._demean(ret, axis=0)

        expand_ret = np.expand_dims(ret, axis=2)
        cov_per_t = np.matmul(expand_ret, expand_ret.transpose(0,2,1)) # (t, fnum, fnum)
        
        # 指数加权平均估计样本协方差
        cov_0 = ema(cov_per_t, self._ema_alpha, axis=0) # (fnum, fnum)

        # 调整协方差矩阵
        adj_cov = np.zeros(cov_0.shape) # (fnum, fnum)
        for i in range(1, q+1):
            cov_lag_i = np.matmul(ret[:t-i].T, ret[i:]) / (t-i)
            cov_i = cov_lag_i + cov_lag_i.T
            weight_i = 1 - (i/(1+q))
            adj_cov += weight_i * cov_i
        
        return cov_0 + adj_cov
    
    def _eigenfactor_adj(self, factor_cov: np.ndarray, T: int, m: int = 100, alpha: float = 1.5):
        # 因子收益率样本协方差矩阵分解
        D, U = np.linalg.eig(factor_cov) # D.shape=(fnum), U.shape=(fnum, fnum)
        K = factor_cov.shape[0] # fnum
        # 模拟M次
        V = np.zeros(K)
        for i in range(m):
            eigen_factor_ret = np.zeros((K, T))
            for i in range(T):
                eigen_factor_ret[:, i] = np.random.normal(0, np.sqrt(D))
            simulated_factor_ret = np.matmul(U, eigen_factor_ret) # (fnum, T)
            simulated_factor_cov = np.cov(simulated_factor_ret)
            simulated_D, simulated_U = np.linalg.eig(simulated_factor_cov)
            eigenfactor_cov = np.linalg.multi_dot([simulated_U.T, factor_cov, simulated_U])
            eigenfactor_var = np.diag(eigenfactor_cov) # shape=(fnum)
            V += np.sqrt(simulated_D / eigenfactor_var)
        V = V / m
        V_s = alpha * (V - 1) + 1
        D_adj = np.matmul(np.diag(V_s**2), D)
        factor_cov_adj = np.linalg.multi_dot([U, D_adj, U.T])
        return factor_cov_adj
    
    def _bayesian_shinkage(self, specific_std: np.ndarray, market_cap: np.ndarray, q: float, group_num: int = 10):
        # 按市值分组
        mkt_cap_groups = np.array_split(np.sort(market_cap.flatten()), group_num)
        # 每组的布尔索引
        group_conditions = [np.isin(market_cap, mkt_cap_groups[i]) for i in range(group_num)]

        adj_std = np.zeros(specific_std.shape)
        for _, g in enumerate(group_conditions):
            # 每组计算先验特质波动率
            weight = market_cap[g] / market_cap[g].sum()
            prior_std = weight * specific_std[g]
            # 先验波动率的权重
            prior_std_weight = (q * np.abs(specific_std[g] - prior_std)) / (
                np.sqrt((1/len(g)*np.sum(specific_std[g]-prior_std))) + \
                (q * np.abs(specific_std[g] - prior_std))
            )
            adj_std[g] = prior_std_weight * prior_std + specific_std[g] * (1-prior_std_weight)
        return adj_std






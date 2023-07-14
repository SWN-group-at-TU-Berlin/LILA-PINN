#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:12:04 2023

@author: ivodaniel
"""

import sys
sys.path.append('..')
sys.path.append('../..')
import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import wntr
import torch
from torch import Tensor
from sklearn.model_selection import train_test_split

from models import LILA

# import for typing
from typing import Iterable, Union, List
from torch._C import _TensorMeta
from sklearn.preprocessing._data import MaxAbsScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.size']= 16
from mpl_axes_aligner import align

def simulate_leakage_data(
        n_ir: int,
        inp_file: str,
        noise: bool = False,
        leaks = None,
        ) -> DataFrame :

    u_path = '../_utils/'
    leak_signals = pd.read_csv(u_path+'Data/leak_ground_truth/2019_Leakages.csv',sep=';',decimal=',',parse_dates=['Timestamp'])
    leak_signals.index = leak_signals['Timestamp'].values
    leak_signals = leak_signals.drop(['Timestamp'],axis=1)
    leak_signals /= 3600

    wn = wntr.network.WaterNetworkModel(inp_file)
    s = 'nonoise'
    if noise:
        s = 'noise'
        pat_res = wn.get_pattern('P-Residential').multipliers
        pat_com = wn.get_pattern('P-Commercial').multipliers
        #pat_ind = wn.get_pattern('P-Industrial').multipliers
        for j_name,junction in wn.junctions():
            for pattern in junction.demand_timeseries_list:
                if pattern.base_value>0:
                    if pattern.pattern_name.startswith('P-Res'):
                        pattern.pattern.multipliers = pat_res
                    elif pattern.pattern_name.startswith('P-Com'):
                        pattern.pattern.multipliers = pat_com

    generate_data = True if not os.path.isfile(u_path+'Data/SIM_data/scada_pressure_{}_d{}.pkl'.format(s,n_ir)) else False
    if generate_data or (not leaks is None):
        save = True
        #create leaks
        if not leaks is None:
            leak_signals = leaks
            save = False
        for leak_pipe in leak_signals.columns:
            n_leak_name = 'n_leak_{}'.format(leak_pipe)
            wn = wntr.morph.link.split_pipe(wn,leak_pipe,'{}_1'.format(leak_pipe),n_leak_name)
            n_leak = wn.get_node(n_leak_name)
            pattern = leak_signals[leak_pipe].values
            wn.add_pattern('leak_{}'.format(leak_pipe),pattern)
            n_leak.add_demand(1,'leak_{}'.format(leak_pipe))

        #regulate industrial flows
        for name,junction in wn.junctions():
            for pattern in junction.demand_timeseries_list:
                if pattern.pattern_name in ['P-Indutr-{}'.format(i) for i in list(np.arange(n_ir+1,5))]:
                    pattern.base_value = 0

        #simulate
        wn.options.time.duration = 31536000-1
        sim = wntr.sim.EpanetSimulator(wn)
        res = sim.run_sim()
        scada_pressure = res.node['pressure']
        scada_pressure.index = pd.date_range(start='2019',end='2019-12-31 23:55', freq='5T')
        if save:
            scada_pressure.to_pickle(u_path+'Data/SIM_data/scada_pressure_{}_d{}.pkl'.format(s,n_ir))
    else:
        scada_pressure = pd.read_pickle(u_path+'Data/SIM_data/scada_pressure_{}_d{}.pkl'.format(s,n_ir))

    return scada_pressure

def extract_irregular_flows(
        n_ir: int,
        inp_file: str,
        ) -> DataFrame :
    wn = wntr.network.WaterNetworkModel(inp_file)
    Q_ir = pd.DataFrame(index=pd.date_range(start='2019-01-01',end='2019-12-31 23:55',freq='5T'),dtype=float)
    for i,(name,pattern) in enumerate(wn.patterns()):
        if pattern.multipliers.shape[0]==105120 and name.startswith('P-I') and not name.endswith('4'):
            Q_ir[name] = pattern.multipliers
    if n_ir < Q_ir.shape[1]:
        Q_ir = Q_ir.iloc[:,:n_ir]
    return Q_ir

def preprocess2torch(
        df: DataFrame,
        sensor_list: list = None,
        train_start: str = None,
        train_end: str = None,
        val_size: float = None,
        test_start: str = None,
        test_end: str = None,
        scaler: MaxAbsScaler = None,
        device: str = 'cpu',
        ) ->  Iterable[Union[_TensorMeta,_TensorMeta,_TensorMeta]] :
    # scaling
    scaled = False if scaler is None else True
    if scaled:
        scaler.fit(df.loc[train_start:train_end])
    df_scaled = pd.DataFrame(
        index = df.index,
        columns=df.columns,
        data = scaler.transform(df) if scaled else df
        )
    if sensor_list:
        df_scaled = df_scaled[sensor_list]
    # segmenting
    df_train   = df_scaled.loc[train_start:train_end] if not (train_start is None) & (train_end is None) else pd.DataFrame()
    df_test    = df_scaled.loc[test_start:test_end] if not (test_start is None) & (test_end is None) else pd.DataFrame()
    # validation split
    if val_size is None or val_size==0:
        x_train,x_val = df_train,None
    else:
        x_train,x_val,_,_ = train_test_split(df_train,df_train,test_size=val_size,random_state=42)
    # pushing to Tensor
    x_train_tensor    = torch.from_numpy(x_train.values).float().to(device)
    x_val_tensor      = torch.from_numpy(x_val.values).float().to(device) if x_val is not None else None
    x_test_tensor     = torch.from_numpy(df_test.values).float().to(device)

    return x_train_tensor,x_val_tensor,x_test_tensor

def calibrate_LILA(
        P_train,
        sensor_list: List,
        hyperparams,
        Q_train = None,
        verbose: bool = False,
        **kwargs,
        ):
    # model setup
    n_sensors   = len(sensor_list)
    model = LILA(n_sensors)

    # fit initial linear regression
    if Q_train is None:
        losses = model.fit(P_train,P_val=P_train,num_epochs=500,verbose=verbose)
    else:
        losses = model.fit(P_train,Q_train,P_val=P_train,Q_val=Q_train,num_epochs=500,verbose=verbose)

    if hyperparams==False:
        if verbose:
            print('train_losses: {:.2E} // val_losses: {:.2E}'.format(losses[0][-1],losses[1][-1]))
        return model
    elif hyperparams=='opt':
        # optimise QNET hyperparameters
        #   based on MSE so test data still remains unseen
        VALUES_n_add    = list(range(1,n_sensors))
        VALUES_n_layers = list(range(3))
        VALUES_n_cells  = [1,3,6,12]
        VALUES_act_fns  = [None,nn.ReLU(),nn.Tanh(),nn.SELU(),nn.SiLU()]
        opt_losses = model.optimise_QNET(P_train,Q_train,VALUES_n_add=VALUES_n_add,VALUES_n_layers=VALUES_n_layers,VALUES_n_cells=VALUES_n_cells,VALUES_act_fns=VALUES_act_fns,verbose=verbose)
        return opt_losses
    else:
        # select best parameters and fit PINN across k-fold
        model_qnet = model.fit_QNET_kfold(P_train,n_add=hyperparams['n_add'],Q=Q_train,k_fold_splits=hyperparams['n_splits'],num_epochs=500,verbose=verbose,**kwargs)
        return model_qnet

def plot_flows(
        i,
        model,
        n_ir,
        P_test: Tensor,
        scada_flows,
        test_start: str,
        test_end: str,
        sensor_list: List = None,
        ):
    q_pred = model.lin_reg.qnet(P_test).detach().numpy()
    q_true = pd.DataFrame(MaxAbsScaler().fit_transform(scada_flows),columns=scada_flows.columns,index=scada_flows.index)
    n_add = q_pred.shape[1]

    f,axs = plt.subplots(n_add,figsize=(15,5*n_add),sharex=True)
    for k in range(n_add):
        ax = axs[k] if n_add>1 else axs
        # plot model predicted flow
        z = k if sensor_list is None else sensor_list[k]
        index = scada_flows.loc[test_start:test_end].index
        pd.Series(q_pred[:index.shape[0],z],index=scada_flows.loc[test_start:test_end].index).plot(ax=ax,label='Predicted flow {}'.format(k+1))
        # plot ground truth
        if k<n_ir:
            (q_true.iloc[:,k].loc[test_start:test_end]).plot(ax=ax,label='Ground truth: {}'.format(q_true.iloc[:,k].name))

        ax.legend(loc=1)
        ax.set_xlim(test_start,test_end)
        ax.set_ylabel('Norm. flow rate (-)')
    f.suptitle(i)
    f.tight_layout()

def cusum(df, delta=4, C_thr=3, est_length='3 days'):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318
    df        :  data to analyze
    delta     :  parameter to calculate slack value K =>  K = (delta/2)*sigma
    K         :  reference value, allowance, slack value for each pipe
    C_thr     :  threshold for raising flag
    est_length:  Window for estimating distribution parameters mu and sigma
    ---
    df_cs:  pd.DataFrame containing cusum-values for each df column
    """

    est_data = df.loc[:(df.index[0] + pd.Timedelta(est_length))]
    ar_mean = est_data.mean(axis=0).values
    ar_sigma = est_data.std(axis=0).values
    ar_K = (delta/2) * ar_sigma

    cumsum_p = np.zeros(df.shape)
    cumsum_n = np.zeros(df.shape)

    for i in range(1, df.shape[0]):
        cumsum_p[i, :] = np.maximum(0, df.iloc[i, :] - ar_mean + cumsum_p[i-1, :] - ar_K)
        cumsum_n[i, :] = np.maximum(0, -df.iloc[i, :] + ar_mean + cumsum_n[i-1, :] - ar_K)

    df_cs_p = pd.DataFrame(cumsum_p / ar_sigma, columns=df.columns, index=df.index)
    df_cs_n = pd.DataFrame(cumsum_n / ar_sigma, columns=df.columns, index=df.index)

    df_cs = pd.concat([df_cs_p, df_cs_n]).groupby(level=0).max()

    leak_det = {}

    for column in df_cs:
        if any(df_cs[column] > C_thr):
            s_cs = df_cs[column]
            leak_det[column] = s_cs[s_cs > C_thr].index[0]

    leak_det = pd.Series(leak_det)

    return leak_det, df_cs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:36:42 2023

@author: ivodaniel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import List
from collections import OrderedDict
from itertools import product


class QNET_FCNN(nn.Module):
    '''
    A customisable feed-forward Neural Network:
        - utilised to predict Qd from P
        - nn.Sequential() takes an OrderedDict() to set up the network
        - Parameters for the OrderedDict are passed in the __init__() function
    '''
    def __init__(
            self,
            input_shape: int,
            output_shape: int,
            n_cells: int = None,
            n_layers: int = 0,
            activation: nn.Module = None,
            out_act: bool = True,
            batch_norm: bool = True,
            last_bias: bool = False,
            device: str = 'cpu',
            ) -> None:
        '''
        Parameters
        ----------
        input_shape : int -> Must equal the number of pressure sensors.
        output_shape : int -> Must be set to the number of expected irregular flows.
        n_cells : int, optional -> Number of cells in each hidden layer. The default is equal to input_shape.
        n_layers : int, optional -> Number of hidden layers. The default is 1.
        activation : nn.Module, optional -> The activation function within each hidden layer. The default is nn.SiLU().
        out_act : bool, optional -> The activation function after the last layer (e.g., to restrain the output to a certain range). The default is True.
        batch_norm : bool, optional -> Using a BatchNorm2D layer to enhance training with mini-batches. The default is True.
        last_bias : bool, optional -> Allowing for a bias in the last layer. The default is False.
        device : str, optional -> The default is 'cpu'.
        '''
        super().__init__()
        self.device = device
        if n_cells is None:
            n_cells = input_shape

        od = OrderedDict()
        od['in'] = nn.Linear(input_shape,n_cells)
        if batch_norm:
            od['norm'] = nn.BatchNorm1d(n_cells)
        if not activation is None:
            od['in_act'] = activation
        for i in range(n_layers):
            od['lin{}'.format(i)] = nn.Linear(n_cells,n_cells)
            if batch_norm:
                od['norm{}'.format(i)] = nn.BatchNorm1d(n_cells)
            if not activation is None:
                od['act{}'.format(i)] = activation
        od['out'] =  nn.Linear(n_cells,output_shape,bias=last_bias)
        if out_act:
            od['out_act'] = nn.ReLU()

        self.linear_stack = nn.Sequential(od)

    def forward(
            self,
            x: Tensor,
            ) -> Tensor:
        '''
       Parameters
        ----------
        x : Tensor -> P

        Returns
        -------
        Tensor -> Qd
        '''
        return self.linear_stack(x)

class Linear2D(nn.Module):
    '''
    NxN Linear Regression layer:
        P_j = k0_ij + P_i * k1_ij + sum_d(kd_ij * Qd)
        i,j \in N -> Number of sensors
        d   \in D -> Number of flows

    Attributes:
        K0      -> general bias (N)
        K1      -> weights for P (N)
        Kd      -> weigths for all Qd (DxNxN)
        qnet    -> feed-forward ANN:
                    - predicting Qd from P (f: P -> Qd)
                    - implemented as PINN through customised layer and loss function
    '''
    def __init__(
            self,
            n_sensors: int,
            ) -> None:
        '''
        Parameters
        ----------
        n_sensors : int -> Number of sensors.
        '''
        super().__init__()
        self.N      = n_sensors
        self.k0     = nn.Parameter(torch.zeros((self.N)))
        self.k1     = nn.Parameter(torch.ones(self.N))
        self.qnet   = None

    def register_flows(
            self,
            n_flows: int,
            **kwargs,
            ) -> None:
        '''
        Parameters
        ----------
        n_flows : int -> Number of irregular flows.
        **kwargs : dict or keywords -> see QNET_FCNN.__init__()
        '''
        self.D      = n_flows
        self.kd     = nn.Parameter(torch.zeros((self.D,self.N)))
        self.qnet   = QNET_FCNN(self.N,self.D,**kwargs)

    def register_known_flows(
            self,
            n_flows: int,
            ) -> None:
        self.D_known    = n_flows
        self.kd_known   = nn.Parameter(torch.zeros((self.D_known,self.N)))

    def forward(
            self,
            P: Tensor,
            Q: Tensor = None,
            ) -> Tensor:
        '''
        Parameters
        ----------
        P : Tensor -> Pressure
        Q : Tensor -> knwon flow
        Raises
        ------
        ValueError -> checks if P.shape[1]==N

        Returns
        -------
        Tensor -> Predicted values for pressure P_pred.
        '''
        T = P.shape[0]
        if not P.shape[1]==self.N:
            raise ValueError(
                'Shape of P ({} is not equal to the predefined number of sensors: {})'.format(P.shape[1],self.N)
                )

        K1 = torch.matmul(self.k1.reshape(-1,1),1/self.k1.reshape(1,-1))
        K0 = (self.k0[:,None] - self.k0[None,:])/self.k1[None,:]
        if not self.qnet is None:
            Kd = (self.kd[:,:,None] - self.kd[:,None,:])/self.k1[None,None,:]
            flows = (self.qnet(P)[:,:,None,None]*Kd[None,...]).sum(axis=1)
        else:
            flows = 0
        if not Q is None:
            Kd_known = (self.kd_known[:,:,None] - self.kd_known[:,None,:])/self.k1[None,None,:]
            known_flows = (Q[:,:,None,None]*Kd_known[None,...]).sum(axis=1)
        else:
            known_flows = 0

        P_pred = (
            (K0[None,:,:] * torch.ones(T)[:,None,None])
            + flows
            + known_flows
            + (K1[None,:,:] * torch.ones(T)[:,None,None])
            * (P[:,None,:] * torch.ones(self.N)[None,:,None])
        )
        return P_pred

class LILA(nn.Module):
    '''
    '''
    def __init__(
            self,
            n_sensors: int,
            ) -> None:
        '''
        Parameters
        ----------
        n_sensors : int -> Number of sensors.
        '''
        super().__init__()
        self.lin_reg = Linear2D(n_sensors)

    def register_flows(
            self,
            n_flows: int,
            **kwargs,
            ) -> None:
        '''
        Parameters
        ----------
        n_flows : int -> Number of irregular flows.
        **kwargs : dict or keywords -> Arguments to the constructor of the feed-forward ANN.
        '''
        self.lin_reg.register_flows(n_flows,**kwargs)

    def optimise_QNET(
            self,
            P: Tensor,
            Q: Tensor = None,
            k_fold_splits: int      = 10,
            VALUES_n_add: List      = None,
            VALUES_n_layers: List   = None,
            VALUES_n_cells: List    = None,
            VALUES_act_fns: List    = None,
            num_epochs: int         = 200,
            verbose: bool           = False,
            **kwargs,
            ) -> pd.DataFrame:
        '''
        Parameters
        ----------
        P_train : Tensor -> Pressure.
        k_fold_splits : int, optional -> Number of folds for k-fold validation. The default is 10.
        VALUES_n_add : List, optional -> List of values for the total number of irregular flows. The default is None.
        VALUES_n_layers : List, optional -> List of values for the number of hidden layers in the PINN. The default is None.
        VALUES_n_cells : List, optional -> List of values for the number of hidden cells in the PINN. The default is None.
        VALUES_act_fns : List, optional -> List of activation functions. The default is None.
        verbose : bool, optional -> Printing option. The default is False.

        Returns
        -------
        opt_losses : pd.DataFrame -> Table with the results of the hyperparameter study.
        '''
        kf = KFold(n_splits=k_fold_splits,shuffle=True,random_state=42)
        if VALUES_n_add is None:
            VALUES_n_add = list(range(1,self.lin_reg.N))
        if VALUES_n_layers is None:
            VALUES_n_layers = [0]
        if VALUES_n_cells is None:
            VALUES_n_cells = [self.lin_reg.N]
        if VALUES_act_fns is None:
            VALUES_act_fns = [None]

        N_TOT = (
            len(VALUES_n_add)
            *len(VALUES_n_layers)
            *len(VALUES_n_cells)
            *len(VALUES_act_fns)
            )

        opt_losses = list()
        for i,(n_add,n_layers,n_cells,act_fn)in enumerate(product(VALUES_n_add,
                                                                  VALUES_n_layers,
                                                                  VALUES_n_cells,
                                                                  VALUES_act_fns,
                                                                  )):
            if verbose:
                print('Run {} of {}: A:{},L:{},C:{},ACT:{}'.format(i+1,N_TOT,n_add,n_layers,n_cells,act_fn))
            kf_losses = list()
            for i_k, (train_index, val_index) in enumerate(kf.split(P)):
                P_tr = P[train_index]
                P_val = P[val_index]
                if not Q is None:
                    Q_tr = Q[train_index]
                    Q_val = Q[val_index]
                self.register_flows(n_add,n_cells=n_cells,n_layers=n_layers,activation=act_fn,**kwargs)
                if Q is None:
                    losses = self.fit(P_tr,P_val=P_val,num_epochs=num_epochs,verbose=False)
                else:
                    losses = self.fit(P_tr,Q=Q_tr,P_val=P_val,Q_val=Q_val,num_epochs=num_epochs,verbose=False)
                kf_losses.append([losses[0][-1],losses[1][-1]])
            kf_losses = np.array(kf_losses).min(axis=0)
            if verbose:
                print('{:.2E} // {:.2E}'.format(kf_losses[0],kf_losses[1]))
            opt_losses.append([n_add,n_layers,n_cells,act_fn,kf_losses[0],kf_losses[1]])

        opt_losses = pd.DataFrame(
            opt_losses,
            columns=['n_add','n_layers','n_cells','act_fn',
                     'train_loss','val_loss']
            )

        return opt_losses

    def fit_QNET_kfold(
            self,
            P: Tensor,
            n_add: int,
            Q: Tensor = None,
            k_fold_splits: int = 10,
            num_epochs: int = 200,
            verbose: bool = False,
            **kwargs,
            ) -> nn.Module:
        '''
        Finds the best fit for the PINN across a k-fold set.
        See above for parameter description.
        '''
        kf = KFold(n_splits=k_fold_splits,shuffle=True,random_state=42)
        for i_k, (train_index, val_index) in enumerate(kf.split(P)):
            P_tr = P[train_index]
            P_val = P[val_index]
            if not Q is None:
                Q_tr = Q[train_index]
                Q_val = Q[val_index]
            self.register_flows(n_add,**kwargs)
            if Q is None:
                losses = self.fit(P_tr,P_val=P_val,num_epochs=num_epochs,verbose=False)
            else:
                losses = self.fit(P_tr,Q=Q_tr,P_val=P_val,Q_val=Q_val,num_epochs=num_epochs,verbose=False)
            if i_k > 0:
                if losses[0][-1] < losses_opt:
                    losses_opt = losses[0][-1]
                    losses_opt_tr = losses[1][-1]
                    model_opt = copy.deepcopy(self)
            else:
                losses_opt = losses[0][-1]
                losses_opt_tr = losses[1][-1]
                model_opt = copy.deepcopy(self)
        if verbose:
            print('train_losses: {:.2E} // val_losses: {:.2E}'.format(losses_opt,losses_opt_tr))
        return model_opt

    def forward(
            self,
            P: Tensor,
            Q: Tensor = None,
            ) -> Tensor:
        '''
        Parameters
        ----------
        P : Tensor -> Pressure
        Q : Tensor -> known flow

        Returns
        -------
        Tensor -> Predicted pressure
        '''
        return self.lin_reg(P,Q)

    def fit(
            self,
            P: Tensor,
            Q: Tensor = None,
            P_val: Tensor = None,
            Q_val: Tensor = None,
            loss_fn: nn.Module = nn.MSELoss(),
            optimizer: optim.Optimizer = optim.Adam,
            reduce_LR: bool = True,
            num_epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = False,
            ) -> List[float]:
        '''
        Parameters
        ----------
        P : Tensor -> Pressure training data.
        Q: Tensor -> Flow (irregular) training data.
        P_val : Tensor, optional -> Validation data. The default is None.
        Q_val : Tensor, optional -> Validation data. The default is None.
        loss_fn : nn.Module, optional -> Instance of loss function. The default is nn.MSELoss().
        optimizer : optim.Optimizer, optional -> Instance of optimizer. The default is optim.Adam.
        reduce_LR : bool, optional -> Include learining rate scheduler during training. The default is True.
        num_epochs : int, optional -> Number of training epochs. The default is 100.
        batch_size : int, optional -> Size of mini-batches during training. The default is 32.
        verbose : bool, optional -> Printing option. The default is False.

        Returns
        -------
        List[float] -> train losses & val losses.
        '''
        train_losses = []
        val_losses = []

        if not Q is None:
            self.lin_reg.register_known_flows(Q.shape[1])

        train_dataset = TensorDataset(P,P) if Q is None else TensorDataset(P,Q)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if not P_val is None:
            P_val       = P_val
            val_dataset = TensorDataset(P_val, P_val) if Q_val is None else TensorDataset(P_val, Q_val)
            val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optimizer(self.parameters())
        if reduce_LR:
            lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=verbose)
        for epoch in range(num_epochs):
            self.train()
            train_epoch_loss = 0
            for P_batch, Q_batch in train_loader:
                optimizer.zero_grad()
                if Q is None:
                    outputs = self(P_batch)
                else:
                    outputs = self(P_batch,Q_batch)
                targets = (P_batch[:,:,None] * torch.ones(P_batch.shape[1])[None,None,:])
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            train_losses.append(train_epoch_loss / len(train_loader))

            self.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                for P_batch, Q_batch in val_loader:
                    if Q_val is None:
                        outputs = self(P_batch)
                    else:
                        outputs = self(P_batch,Q_batch)
                    targets = (P_batch[:,:,None] * torch.ones(P_batch.shape[1])[None,None,:])
                    loss = loss_fn(outputs, targets)
                    val_epoch_loss += loss.item()
            val_losses.append(val_epoch_loss / len(val_loader))
            if reduce_LR:
                lr_scheduler.step(val_losses[-1])

            if verbose:
                print('EPOCH {}: loss: {} / val_loss: {}'.format(epoch,train_losses[-1],val_losses[-1]))

        return train_losses, val_losses

    def predict_MRE(
            self,
            P: Tensor,
            Q: Tensor = None,
            ) -> Tensor:
        '''
        Parameters
        ----------
        P : Tensor -> Pressure
        Q : Tensor -> known flow

        Returns
        -------
        Tensor -> Model reconstruction error (MRE).
        '''
        e = (self(P,Q)- (P[:,:,None] * torch.ones(P.shape[1])[None,None,:])).detach()
        return e

    def detect(
            self,
            e: Tensor,
            algo: str = 'cusum',
            **kwargs,
            ):
        pass

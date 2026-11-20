#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for advection equation solver
# Author: Wenhan Tang

import numpy as np
from scipy import sparse

from pdb import set_trace

class AdvectionSolver(object):

    def __init__(self, objMesh, objData, varNameList, uName, vName, H = 1):
        self.objMesh = objMesh
        self.objData = objData
        self.nE = self.objMesh.nE
        self.nT = self.objMesh.nT
        self.lon_c, self.lat_c = self.objMesh.lon_c, self.objMesh.lat_c
        self.lon_s, self.lat_s = self.objMesh.lon_s, self.objMesh.lat_s
        self.lon_c_rad, self.lat_c_rad = self.objMesh.lon_c_rad, self.objMesh.lat_c_rad
        self.lon_s_rad, self.lat_s_rad = self.objMesh.lon_s_rad, self.objMesh.lat_s_rad

        self.varNameList = varNameList
        self.uName = uName
        self.vName = vName
        self.H = H
        self.var_dict = {}
        self.var_ext_dict = {}
        self.data_initialization()
        self.intensive_to_extensive()

        self.already_set_dt = False

    def set_dt(self, dt = None, courant = None):
        if not(dt is None):
            self.dt = dt
        else:
            h_min = self.objMesh.neighbor_distance.min()
            VV_max = np.sqrt(self.u**2 + self.v**2).max()
            print("courant = ", courant, " h_min = ", h_min, " VV_max = ", VV_max)
            self.dt = courant * h_min / VV_max
        self.already_set_dt = True
        print("Set dt = " + str(self.dt) + " s")


    def data_initialization(self):
        
        for varName in self.varNameList: 
            self.var_dict[varName] = self.objData.get_data(time = 0, varName = varName)
        self.u = self.objData.get_data(time = 0, varName = self.uName)
        self.v = self.objData.get_data(time = 0, varName = self.vName)
        for varName in self.varNameList:
            assert len(self.var_dict[varName]) == self.nT
        assert len(self.u) == self.nE
        assert len(self.u) == len(self.v)

    
    def intensive_to_extensive(self):
        for varName in self.varNameList:
            self.var_ext_dict[varName] = self.var_dict[varName] * self.H * self.objMesh.grid_area

    def extensive_to_intensive(self):
        for varName in self.varNameList:
            self.var_dict[varName] = self.var_ext_dict[varName] / self.H / self.objMesh.grid_area

    def spherical_to_local(self, lon1, lat1, lon2, lat2, R):
        """
            lon1, lat1, lon2, lat2 are in rad unit.
        """
        lon1_deg, lon2_deg = np.rad2deg(lon1), np.rad2deg(lon2)
        check_sign = lon1_deg * lon2_deg
        check_index = (check_sign < 0) & (np.abs(lon1_deg) > 90) & (np.abs(lon2_deg) > 90)
        lon1_deg[check_index] = np.where(lon1_deg[check_index] < 0, lon1_deg[check_index] + 360, lon1_deg[check_index])
        lon2_deg[check_index] = np.where(lon2_deg[check_index] < 0, lon2_deg[check_index] + 360, lon2_deg[check_index])
        lon1, lon2 = np.deg2rad(lon1_deg), np.deg2rad(lon2_deg)
        dx = (lon2 - lon1) * R * np.cos( (lat1 + lat2) / 2 )
        dy = (lat2 - lat1) * R
        
        return dx, dy

    def compute_source(self, varName, time):
        #source_data = self.objData.get_source(time = time, varName = varName) * self.H * self.objMesh.grid_area
        source_data = self.objData.get_source(time = time, varName = varName) * self.objMesh.grid_area
        return source_data
        
    def compute_flux(self, varName):
        E_neighbor = self.objMesh.edge2neighbor
        index_T1, index_T2 = E_neighbor[:, 0], E_neighbor[:, 1]
        var_T1, var_T2 = self.var_dict[varName][index_T1], self.var_dict[varName][index_T2]
        lon_rad_T1, lat_rad_T1 = self.lon_c_rad[index_T1], self.lat_c_rad[index_T1]
        lon_rad_T2, lat_rad_T2 = self.lon_c_rad[index_T2], self.lat_c_rad[index_T2]

        
        toward_vector_x, toward_vector_y = self.spherical_to_local(lon_rad_T1, lat_rad_T1, lon_rad_T2, lat_rad_T2, self.objMesh.R)
        index_V1, index_V2 = self.objMesh.E[:, 0], self.objMesh.E[:, 1]
        edge_vector_x, edge_vector_y =  self.spherical_to_local(self.objMesh.V[index_V1, 0], self.objMesh.V[index_V1, 1], self.objMesh.V[index_V2, 0], self.objMesh.V[index_V2, 1], self.objMesh.R)
        ref1_x, ref1_y = edge_vector_y, -edge_vector_x
        ref2_x, ref2_y = -edge_vector_y, edge_vector_x
        ref_x = np.full( (self.objMesh.nE), np.nan )
        ref_y = np.full( (self.objMesh.nE), np.nan )
        
        choose1 = np.where(ref1_x * toward_vector_x + ref1_y * toward_vector_y > 0)[0]
        choose2 = np.where(ref2_x * toward_vector_x + ref2_y * toward_vector_y > 0)[0]
        
        ref_x[choose1] = ref1_x[choose1]
        ref_y[choose1] = ref1_y[choose1]
        ref_x[choose2] = ref2_x[choose2]
        ref_y[choose2] = ref2_y[choose2]
        assert np.sum(np.isnan(ref_x) == 0)
        assert np.sum(np.isnan(ref_y) == 0)
        assert len(choose1) + len(choose2) == self.objMesh.nE

        ref_length = np.sqrt(ref_x**2 + ref_y**2)
        
        UU = (ref_x * self.u + ref_y * self.v) / ref_length
        flux = UU * np.where(UU >= 0, var_T1, var_T2) # Kg/m^3 * m/s --> Kg/m^2/s
        flux_extensive = flux * self.objMesh.h * self.H # Kg/m^2/s --> Kg/s
        flux_data = np.concatenate( (-flux_extensive.copy(), flux_extensive.copy()) )
        flux_Matrix_ii = np.concatenate( (np.arange(self.nE), np.arange(self.nE)) )
        flux_Matrix_jj = np.concatenate( (index_T1, index_T2) )
        flux_Matrix = sparse.coo_matrix( (flux_data, (flux_Matrix_ii, flux_Matrix_jj)), shape = (self.nE, self.nT) )

        return flux_Matrix, UU

    def check_flux_for_tail(self, varName, flux_Matrix_csr, UU, ind_T):
        E_neighbor = self.objMesh.edge2neighbor
        index_T1, index_T2 = E_neighbor[:, 0], E_neighbor[:, 1]
        access_Vindex = self.objMesh.T[ind_T, :]
        access_Vindex = access_Vindex[access_Vindex != self.objMesh.undef_ind]
        access_Vindex = np.concatenate( (access_Vindex, np.array([access_Vindex[0]])) )
        print()
        print("Current state of tail #%d: %.4f; %.4f" % (ind_T, self.var_dict[varName][ind_T], self.var_ext_dict[varName][ind_T]))
        net_flux = 0.0
        net_UU4i = 0.0
        for ind_V1, ind_V2 in zip(access_Vindex[:-1], access_Vindex[1:]):
            ind_E = self.objMesh.edge2index[ (ind_V1, ind_V2) if ind_V1 < ind_V2 else (ind_V2, ind_V1) ] 
            UU4i = -UU[ind_E] if ind_T == index_T1[ind_E] else UU[ind_E]
            ind_temp = np.where(self.objMesh.edge2neighbor[ind_E] != ind_T)[0]
            assert len(ind_temp) == 1
            neighbor_ind_T = self.objMesh.edge2neighbor[ind_E, int(ind_temp)]
            print("UU4i = %10.4f, UUL = %20.4f flux = %20.4f from tail #%10d: %20.4f; %20.4f" % (UU4i, UU4i * self.objMesh.h[ind_E], flux_Matrix_csr[ind_E, ind_T], neighbor_ind_T, self.var_dict[varName][neighbor_ind_T], self.var_ext_dict[varName][neighbor_ind_T]))
            net_flux += flux_Matrix_csr[ind_E, ind_T]
            net_UU4i += UU4i * self.objMesh.h[ind_E]
        print("Net flux = ", net_flux)
        print("Net UU = ", net_UU4i)
        print()
        

    
    def one_time_step_forward(self, time):
        assert self.already_set_dt, "Please set dt first."

        self.u = self.objData.get_data(time = time, varName = self.uName)
        self.v = self.objData.get_data(time = time, varName = self.vName)

        for varName in self.varNameList:

            flux_Matrix, UU = self.compute_flux(varName)

            flux_Matrix_csr = flux_Matrix.tocsr()
            flux_to_each_tail = flux_Matrix_csr.sum(axis=0)[0].A[0]

            self.var_ext_dict[varName] += self.compute_source(varName, time) * self.dt
            self.var_ext_dict[varName] += flux_to_each_tail * self.dt
            
        self.extensive_to_intensive()
       


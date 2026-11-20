#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for ideal scenarios
# Author: Wenhan Tang

import numpy as np
from .data_loading import Dataset

from pdb import set_trace

class IdealScenario(object):

    def __init__(self, objMesh):
        self.mesh = objMesh
        self.lon_c = self.mesh.lon_c
        self.lat_c = self.mesh.lat_c
        self.lon_s = self.mesh.lon_s
        self.lat_s = self.mesh.lat_s
        self.Nc = len(self.lon_c)
        self.Ns = len(self.lon_s)
        self.objData = Dataset(data_type = "ideal", objScn = self)

    def get_data(self, varName, time):
        if ((varName == self.uName) or (varName == self.vName)) and self.stagger_wind:
            lon, lat = self.lon_s, self.lat_s
        else:
            lon, lat = self.lon_c, self.lat_c
        data = self.retrieve_data_by_lonlat(varName, time, lon, lat)
        if ((varName != self.uName) and (varName != self.vName)):
            data = self.preprocess(varName = varName, data = data)
        return data

    def get_source(self, varName, time):
        assert varName in self.varNameList, "Sould be in " + str(self.varNameList)
        lon, lat = self.lon_c, self.lat_c
        return self.retrieve_source_by_lonlat(varName, time, lon, lat)


    
    def retrieve_data_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )
    
    def preprocess(self, varName, data):
        return data.copy()

    def postprocess(self, varName, data):
        return data.copy()
    
    

class Scenario_Ractangle_Block(IdealScenario):

    def __init__(self, objMesh, lon1, lon2, lat1, lat2, pollution_conc = 1, wind = 100):
        
        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2

        self.varNameList = ["pollution"]
        self.uName = "u"
        self.vName = "v"
        self.pollution_conc = pollution_conc
        self.wind = wind
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            data = np.full( (Ndata), 0.0 )
            focus_domain = (( lon >= self.lon1) & (lon <= self.lon2) & ( lat >= self.lat1 ) & (lat <= self.lat2))
            data[focus_domain] = self.pollution_conc
            
        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            data = -self.wind * np.cos(np.deg2rad(lat))

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )

        else:
            assert False

        return data
    

class Scenario_OnePoint(IdealScenario):

    def __init__(self, objMesh, plon, plat, pollution_conc = 1., wind = 100.):
        
        
        self.plon = plon
        self.plat = plat

        self.varNameList = ["pollution"]
        self.uName = "u"
        self.vName = "v"
        self.pollution_conc = pollution_conc
        self.wind = wind
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            data = np.full( (Ndata), 0.0 )
            
            nearest_ind = np.argmin(np.sqrt((lon - self.plon)**2 + (lat - self.plat)**2))
            data[nearest_ind] = self.pollution_conc

        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            data = -self.wind * np.cos(np.deg2rad(lat))
            

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )
        else:
            assert False

        return data

class Scenario_Ractangle_Block_multitests(IdealScenario):

    def __init__(self, objMesh, lon1, lon2, lat1, lat2, wind = 100):
        
        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2

        self.varNameList = ["pollution1", "pollution2", "pollution3"]
        self.uName = "u"
        self.vName = "v"
        self.wind = wind
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            if varName == "pollution1":
                background_conc = 0.0
                pollution_conc = 1
            elif varName == "pollution2":
                background_conc = 0.0
                pollution_conc = 2
            elif varName == "pollution3":
                background_conc = 10.0
                pollution_conc = 0.1
            data = np.full( (Ndata), background_conc )
            focus_domain = (( lon >= self.lon1) & (lon <= self.lon2) & ( lat >= self.lat1 ) & (lat <= self.lat2))
            data[focus_domain] = pollution_conc
            
        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            data = -self.wind * np.cos(np.deg2rad(lat))

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )

        else:
            assert False

        return data

    def preprocess(self, varName, data):
        if varName == "pollution3":
            return 1 / data
        else:
            return data.copy()
        
    def postprocess(self, varName, data):
        if varName == "pollution3":
            return 1 / data
        else:
            return data.copy()

class Scenario_OnePointSource(IdealScenario):

    def __init__(self, objMesh, plon, plat, source_flux = 1., wind = 100.):
        
        
        self.plon = plon
        self.plat = plat

        self.varNameList = ["pollution"]
        self.uName = "u"
        self.vName = "v"
        self.source_flux = source_flux
        self.wind = wind
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            data = np.full( (Ndata), 0.0 )
            
        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            data = -self.wind * np.cos(np.deg2rad(lat))
            

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )
        else:
            assert False

        return data

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        assert varName in self.varNameList
        data = np.full( (Ndata), 0.0 )
        
        nearest_ind = np.argmin(np.sqrt((lon - self.plon)**2 + (lat - self.plat)**2))
        data[nearest_ind] = self.source_flux

        return data

class Scenario_Divergence_Block(IdealScenario):

    def __init__(self, objMesh, **kwargs):
        
        self.lon1 = -10
        self.lon2 = 10
        self.lat1 = -10
        self.lat2 = 10

        self.varNameList = ["pollution"]
        self.uName = "u"
        self.vName = "v"
        self.pollution_conc = 1
        self.wind = 463.31
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            data = np.full( (Ndata), 0.0 )
            focus_domain = (( lon >= self.lon1) & (lon <= self.lon2) & ( lat >= self.lat1 ) & (lat <= self.lat2))
            data[focus_domain] = self.pollution_conc
            
        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            in_eastern = np.where(lon >= 0)[0]
            in_western = np.where(lon < 0)[0]
            data[in_eastern] = self.wind * np.cos(np.deg2rad(lat[in_eastern]))
            data[in_western] = -self.wind * np.cos(np.deg2rad(lat[in_western]))

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )

        else:
            assert False

        return data

class Scenario_Ractangle_Block_CrossPolar(IdealScenario):

    def __init__(self, objMesh, lon1, lon2, lat1, lat2, pollution_conc = 1, wind = 100):
        
        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2

        self.varNameList = ["pollution"]
        self.uName = "u"
        self.vName = "v"
        self.pollution_conc = pollution_conc
        self.wind = wind
        self.stagger_wind = True
        IdealScenario.__init__(self, objMesh)

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):

        assert len(lon) == len(lat)
        Ndata = len(lon)

        if varName in self.varNameList:
            data = np.full( (Ndata), 0.0 )
            focus_domain = (( lon >= self.lon1) & (lon <= self.lon2) & ( lat >= self.lat1 ) & (lat <= self.lat2))
            data[focus_domain] = self.pollution_conc
            
        elif varName == "u":
            data = np.full( (Ndata), 0.0 )
            data = -self.wind * np.cos(np.deg2rad(lon)) * np.sin(np.deg2rad(lat))

        elif varName == "v":
            data = np.full( (Ndata), 0.0 )
            data = self.wind * np.sin(np.deg2rad(lon))

        else:
            assert False

        return data
    


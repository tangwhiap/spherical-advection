#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for real scenarios
# Author: Wenhan Tang

import numpy as np
from .data_loading import Dataset
import netCDF4 as nc
from scipy.interpolate import griddata

from pdb import set_trace


class RealScenario(object):

    def __init__(self, objMesh):

        self.objMesh = objMesh
        self.lon_c = self.objMesh.lon_c
        self.lat_c = self.objMesh.lat_c
        self.lon_s = self.objMesh.lon_s
        self.lat_s = self.objMesh.lat_s
        self.Nc = len(self.lon_c)
        self.Ns = len(self.lon_s)
        self.objData = Dataset(data_type = "real", objScn = self)
        self.init_netcdf()

    def init_netcdf(self):
        self.varFileName_to_ncf = {}
        self.varFileName_to_lon = {}
        self.varFileName_to_lat = {}
        self.fileName_to_ncf = {}
        for ind, fileName in enumerate(self.fileName_to_varFileName):
            ncf = nc.Dataset(fileName)
            self.fileName_to_ncf[fileName] = ncf
            flon  = ncf.variables["longitude"][:].filled(np.nan)
            flat  = ncf.variables["latitude"][:].filled(np.nan)
            for varFileName in self.fileName_to_varFileName[fileName]:
                self.varFileName_to_ncf[varFileName] = ncf
                self.varFileName_to_lon[varFileName] = flon.copy()
                self.varFileName_to_lat[varFileName] = flat.copy()

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

    def get_netcdf_data(self, ncf, i_time, varFileName, objMesh, lon, lat):
        vardata = ncf.variables[varFileName][i_time, :, :].filled(np.nan)
        lon_old = self.varFileName_to_lon[varFileName]
        lat_old = self.varFileName_to_lat[varFileName]
        interpolated_data = self.interpolate(vardata, lon_old, lat_old, lon, lat)

        return interpolated_data

    def get_netcdf_data_anyTime(self, time, varFileName, objMesh, lon, lat):

        ncf = self.varFileName_to_ncf[varFileName]
        varFile_timeList = self.varFileName_to_timeList[varFileName]
        assert (time >= varFile_timeList[0]) #and (time <= varFile_timeList[-1])
        time_diff = varFile_timeList - time
        index_0 = np.where(time_diff == 0)[0]
        if len(index_0) > 0:
            assert len(index_0) == 1
            i_time = index_0[0]
            if i_time in self.data_lib[varFileName]:
                print("Get single value from lib, at time = ", varFile_timeList[i_time])
                data = self.data_lib[varFileName][i_time].copy()
            else:
                print("Get single value from file, at time = ", varFile_timeList[i_time])
                data = self.get_netcdf_data(ncf, i_time, varFileName, self.objMesh, lon = lon, lat = lat)
                self.data_lib[varFileName][i_time] = data.copy()
        elif time > varFile_timeList[-1]:
            i_time = len(varFile_timeList) - 1
            if i_time in self.data_lib[varFileName]:
                print("Get single value from lib, at time = ", varFile_timeList[i_time])
                data = self.data_lib[varFileName][i_time].copy()
            else:
                print("Get the last value from file, at time = ", varFile_timeList[i_time])
                data = self.get_netcdf_data(ncf, i_time, varFileName, self.objMesh, lon = lon, lat = lat)
                self.data_lib[varFileName][i_time] = data.copy()
        else:
            index_change = np.where(time_diff[1:] * time_diff[:-1] < 0)[0]
            assert len(index_change) == 1
            index_change = index_change[0]
            i_time = index_change
            j_time = index_change + 1
            if i_time in self.data_lib[varFileName]:
                print("Retrieve %s from lib, at time = %d." % (varFileName, varFile_timeList[i_time]))
                data_1 = self.data_lib[varFileName][i_time].copy()
            else:
                print("Retrieve %s from file, at time = %d." % (varFileName, varFile_timeList[i_time]))
                data_1 = self.get_netcdf_data(ncf, i_time, varFileName, self.objMesh, lon = lon, lat = lat)
                self.data_lib[varFileName][i_time] = data_1.copy()
            if j_time in self.data_lib[varFileName]:
                print("Retrieve %s from lib, at time = %d." % (varFileName, varFile_timeList[j_time]))
                data_2 = self.data_lib[varFileName][j_time].copy()
            else:
                print("Retrieve %s from file, at time = %d." % (varFileName, varFile_timeList[j_time]))
                data_2 = self.get_netcdf_data(ncf, j_time, varFileName, self.objMesh, lon = lon, lat = lat)
                self.data_lib[varFileName][j_time]  = data_2.copy()
                
            time_1 = varFile_timeList[i_time]
            time_2 = varFile_timeList[j_time]
            assert (time_1 < time) and (time < time_2)
            F = (time - time_1) / (time_2 - time_1)
            data = data_1 * (1 - F) + data_2 * F
            print("Get value for %s between %d and %d, F = %f" % (varFileName, varFile_timeList[i_time], varFile_timeList[j_time], F))
        return data

    
    def interpolate(self, vardata, lon_old, lat_old, lon_new, lat_new):

        LON, LAT = np.meshgrid(lon_old, lat_old)
        vardata_ = np.concatenate( (vardata[:, -1].reshape(-1,1), vardata, vardata[:, 0].reshape(-1,1)), axis = 1) 
        LON_ = np.concatenate( (LON[:, -1].reshape(-1,1) - 360, LON, LON[:, 0].reshape(-1,1) + 360), axis = 1) 
        LAT_ = np.concatenate( (LAT[:, -1].reshape(-1,1), LAT, LAT[:, 0].reshape(-1,1)), axis = 1) 
        interpolated_data = griddata( (LON_.flatten(), LAT_.flatten()), vardata_.flatten(), (lon_new, lat_new) )

        return interpolated_data

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )
    
    def preprocess(self, varName, data):
        return data.copy()

    def postprocess(self, varName, data):
        return data.copy()
 

    def __del__(self):
        for fileName in self.fileName_to_ncf:
            self.fileName_to_ncf[fileName].close()

class Scenario_ECMWF(RealScenario):

    def __init__(self, objMesh, dataDir):
        self.fileName_to_varFileName = {dataDir + "/ERA5-pl-global.nc": ["t", "u", "v"]}
        self.varNameList = ["t"]
        self.uName = "u"
        self.vName = "v" 
        #self.stagger_wind = False
        self.stagger_wind = True
        self.Pressure = 50000
        self.R_air = 287
        self.init_data_lib()
        RealScenario.__init__(self, objMesh)

    def init_data_lib(self):
        self.data_lib = {}
        for fileName in self.fileName_to_varFileName:
            for varFileName in self.fileName_to_varFileName[fileName]:
                self.data_lib[varFileName] = {}

    def init_netcdf(self):
        RealScenario.init_netcdf(self)
        self.varFileName_to_timeList = {}
        for varFileName in self.varFileName_to_ncf:
            ncf = self.varFileName_to_ncf[varFileName]
            timeNum = ncf.variables["time"][:].filled(np.nan)
            timeUnit = ncf.variables["time"].units
            timeCalendar = ncf.variables["time"].calendar
            timeList = nc.num2date(timeNum, units = timeUnit, calendar = timeCalendar)
            timeDiff = timeList - timeList[0]
            timeSeconds = np.array([time.seconds for time in timeDiff])
            self.varFileName_to_timeList[varFileName] = timeSeconds.copy()


    def retrieve_data_by_lonlat(self, varName, time, lon, lat):
        varFileName = varName
        data = self.get_netcdf_data_anyTime(time = time, varFileName = varFileName, objMesh = self.objMesh, lon = lon, lat = lat)
        
        return data    

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )

    def preprocess(self, varName, data):
        # Convert from temperature to density
        return self.Pressure / self.R_air / data

    def postprocess(self, varName, data):
        # Convert from density to temperature
        return self.Pressure / self.R_air / data

class Scenario_Temperature_Advection(RealScenario):

    def __init__(self, objMesh, dataDir):
        self.fileName_to_varFileName = {dataDir + "/ERA5-pl-global.nc": ["t", "u", "v"]}
        self.varNameList = ["densT", "dens"]
        self.uName = "u"
        self.vName = "v" 
        #self.stagger_wind = False
        self.stagger_wind = True
        self.Pressure_init = 50000
        self.R_air = 287
        self.init_data_lib()
        RealScenario.__init__(self, objMesh)

    def init_data_lib(self):
        self.data_lib = {}
        for fileName in self.fileName_to_varFileName:
            for varFileName in self.fileName_to_varFileName[fileName]:
                self.data_lib[varFileName] = {}

    def init_netcdf(self):
        RealScenario.init_netcdf(self)
        self.varFileName_to_timeList = {}
        for varFileName in self.varFileName_to_ncf:
            ncf = self.varFileName_to_ncf[varFileName]
            timeNum = ncf.variables["time"][:].filled(np.nan)
            timeUnit = ncf.variables["time"].units
            timeCalendar = ncf.variables["time"].calendar
            timeList = nc.num2date(timeNum, units = timeUnit, calendar = timeCalendar)
            timeDiff = timeList - timeList[0]
            timeSeconds = np.array([time.seconds for time in timeDiff])
            self.varFileName_to_timeList[varFileName] = timeSeconds.copy()

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):
        if varName in self.varNameList:
            assert time == 0, "Can only access for time = 0, time = %d is not allowed." % (time)
            if varName == "dens":
                data_t = self.get_netcdf_data_anyTime(time = time, varFileName = "t", objMesh = self.objMesh, lon = lon, lat = lat)
                data = self.Pressure_init / self.R_air / data_t
            else:
                data = self.Pressure_init / self.R_air * np.full( (len(lon)), 1.0 )
                
        else:
            assert varName in [self.uName, self.vName]
            varFileName = varName
            data = self.get_netcdf_data_anyTime(time = time, varFileName = varFileName, objMesh = self.objMesh, lon = lon, lat = lat)


        return data    

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):
        return np.full( (len(lon)), 0.0 )

    def preprocess(self, varName, data):
        return data.copy()

    def postprocess(self, varName, data):
        return data.copy()
    


   
class Scenario_Airplane_Emission(RealScenario):

    def __init__(self, objMesh, dataDir, lon1, lon2, lat1, lat2, time1, time2, co2_emission):
        self.fileName_to_varFileName = {dataDir + "/ERA5-pl-global.nc": ["u", "v"]}
        self.varNameList = ["co2"]
        self.uName = "u"
        self.vName = "v" 
        #self.stagger_wind = False
        self.stagger_wind = True
        self.init_data_lib()
        self.lon1 = lon1
        self.lat1 = lat1
        self.lon2 = lon2
        self.lat2 = lat2

        self.x1, self.y1, self.z1 = objMesh.spherical_to_cartesian(np.deg2rad(self.lon1), np.deg2rad(self.lat1), objMesh.R)
        self.x2, self.y2, self.z2 = objMesh.spherical_to_cartesian(np.deg2rad(self.lon2), np.deg2rad(self.lat2), objMesh.R)

        assert time2 >= time1
        self.time1 = time1
        self.time2 = time2

        self.co2_emission = co2_emission

        RealScenario.__init__(self, objMesh)

    def compute_great_circle_path(self, time):
        if time <= self.time1:
            lon, lat = self.lon1, self.lat1
        elif time >= self.time2:
            lon, lat = self.lon2, self.lat2
        else:
            F = (time - self.time1) / (self.time2 - self.time1)
            plane_x = self.x1 * (1 - F) + self.x2 * F
            plane_y = self.y1 * (1 - F) + self.y2 * F
            plane_z = self.z1 * (1 - F) + self.z2 * F
            lon_rad, lat_rad, _ = self.objMesh.cartesian_to_spherical(plane_x, plane_y, plane_z)
            lon = np.rad2deg(lon_rad)
            lat = np.rad2deg(lat_rad)
        return lon, lat

    def get_great_circle_path(self, N=1000):
        lonList = []
        latList = []
        for time in np.linspace(self.time1, self.time2, N):
            lon, lat = self.compute_great_circle_path(time)
            lonList.append(lon)
            latList.append(lat)
        lonList = np.array(lonList)
        latList = np.array(latList)
        return lonList, latList


    def get_great_circle_path_until_time(self, time, N=1000):
        lonList = []
        latList = []
        for time in np.linspace(self.time1, time, N):
            lon, lat = self.compute_great_circle_path(time)
            lonList.append(lon)
            latList.append(lat)
        lonList = np.array(lonList)
        latList = np.array(latList)
        return lonList, latList
    
    def init_data_lib(self):
        self.data_lib = {}
        for fileName in self.fileName_to_varFileName:
            for varFileName in self.fileName_to_varFileName[fileName]:
                self.data_lib[varFileName] = {}

    def init_netcdf(self):
        RealScenario.init_netcdf(self)
        self.varFileName_to_timeList = {}
        for varFileName in self.varFileName_to_ncf:
            ncf = self.varFileName_to_ncf[varFileName]
            timeNum = ncf.variables["time"][:].filled(np.nan)
            timeUnit = ncf.variables["time"].units
            timeCalendar = ncf.variables["time"].calendar
            timeList = nc.num2date(timeNum, units = timeUnit, calendar = timeCalendar)
            timeDiff = timeList - timeList[0]
            timeSeconds = np.array([time.seconds for time in timeDiff])
            self.varFileName_to_timeList[varFileName] = timeSeconds.copy()

    def retrieve_data_by_lonlat(self, varName, time, lon, lat):
        if varName in self.varNameList:
            data = np.full( (len(lon)), 0.0 )
        else:
            assert varName in [self.uName, self.vName]
            varFileName = varName
            data = self.get_netcdf_data_anyTime(time = time, varFileName = varFileName, objMesh = self.objMesh, lon = lon, lat = lat)
        return data    

    def retrieve_source_by_lonlat(self, varName, time, lon, lat):

        assert varName == "co2"
        data = np.full( (len(lon)), 0.0 )
        plane_lon, plane_lat = self.compute_great_circle_path(time)
        nearest_ind = np.argmin(np.sqrt((lon - plane_lon)**2 + (lat - plane_lat)**2))
        data[nearest_ind] = self.co2_emission / self.objMesh.grid_area[nearest_ind] # Kg/s --> Kg/m^2/s

        return data

    def preprocess(self, varName, data):
        return data.copy()

    def postprocess(self, varName, data):
        return data.copy()
    


   






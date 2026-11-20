#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for loading the initial/boundary data
# Author: Wenhan Tang

import numpy as np
from pdb import set_trace

class Dataset(object):
    def __init__(self, data_type, **kwargs):
        self.data_type = data_type
        if data_type == "real":
            self.init_real(**kwargs)
        elif data_type == "ideal":
            self.init_ideal(**kwargs)
        else:
            assert False, "Incorrect initialization type: " + data_type + "."



    def init_real(self, objScn, **kwargs):
        self.objScn = objScn
 
    def init_ideal(self, objScn, **kwargs):
        self.objScn = objScn
        
    def get_data(self, **kwargs):
        if self.data_type == "real":
            data = self.get_real_data(**kwargs)
            
        elif self.data_type == "ideal":
            data = self.get_ideal_data(**kwargs)
        else:
            assert False
        return data

    def get_source(self, **kwargs):
        if self.data_type == "real":
            data = self.get_real_source(**kwargs)
            
        elif self.data_type == "ideal":
            data = self.get_ideal_source(**kwargs)
        else:
            assert False
        return data

    def get_ideal_data(self, time, varName):
        data = self.objScn.get_data(varName = varName, time = time) 
        return data

    def get_ideal_source(self, time, varName):
        data = self.objScn.get_source(varName = varName, time = time) 
        return data

    
    def get_real_data(self, time, varName):
        data = self.objScn.get_data(varName = varName, time = time) 
        return data

    def get_real_source(self, time, varName):
        data = self.objScn.get_source(varName = varName, time = time) 
        return data


    

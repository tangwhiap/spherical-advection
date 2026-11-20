#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for integrating experimental scenario settings, solver settings, parameter settings, etc., and provides a time integration algorithm
# Author: Wenhan Tang

from .sphere_mesh import SphereTriMesh, SphereHexMesh
import numpy as np
 
class Experiment(object):
    def __init__(self, scenario_name, scenario_type, meshFile, solver_name, kwargs_Scenario = {}):
        self.scenario_name = scenario_name
        self.scenario_type = scenario_type
        self.meshFile = meshFile
        self.solver_name = solver_name
        
        assert self.scenario_type in ["real", "ideal"]
        
        self.objMesh = SphereHexMesh(from_file = True, fileName = meshFile)
        
        loc = locals()
        exec("from ." + scenario_type + "_scenarios import Scenario_" + scenario_name + " as Scenario")
        Scenario = loc["Scenario"]
        
        self.objScn = Scenario(objMesh = self.objMesh, **kwargs_Scenario)
        self.objData = self.objScn.objData
        self.varNameList = self.objScn.varNameList
        self.uName = self.objScn.uName
        self.vName = self.objScn.vName
        
        loc = locals()
        exec("from .advection import " + solver_name + " as Solver")
        Solver = loc["Solver"]
        
        self.objSolver = Solver(self.objMesh, self.objData, self.varNameList, self.uName, self.vName, H = 1)
        
    def postprocess(self):
        var_show_dict = {}
        for varName in self.varNameList:
            var_show_dict[varName] = self.objScn.postprocess(varName = varName, data = self.objSolver.var_dict[varName])
        return var_show_dict
    
    def simulation(self, total_time, timeUnit = "s", courant = 0.5):

        assert timeUnit in ["s", "min", "hr"]
        timeUnitRatio = {"s": 1, "min": 60, "hr": 3600}

        self.objSolver.set_dt(courant = courant)

        total_time = total_time * timeUnitRatio[timeUnit]

        var_dict = {}
        var_ext_dict = {}
        var_show_dict = {}
        total_mass_dict = {}
        var_show_dict_init = self.postprocess()
        for varName in self.varNameList:
            var_dict[varName] = [self.objSolver.var_dict[varName].copy()]
            var_ext_dict[varName] = [self.objSolver.var_ext_dict[varName].copy()]
            var_show_dict[varName] = [var_show_dict_init[varName].copy()]
            total_mass_dict[varName] = [self.objSolver.var_ext_dict[varName].sum()]

        step = 0
        time = 0
        stepList = [step]
        timeList = [time]
        dt = self.objSolver.dt
      
        while time <= total_time + dt:
            self.objSolver.one_time_step_forward(time = time)
            var_show_dict_thisStep = self.postprocess()
            for varName in self.varNameList:
                var_dict[varName].append(self.objSolver.var_dict[varName].copy())
                var_ext_dict[varName].append(self.objSolver.var_ext_dict[varName].copy())
                var_show_dict[varName].append(var_show_dict_thisStep[varName].copy())
                total_mass_dict[varName].append(self.objSolver.var_ext_dict[varName].sum())

            step += 1
            time += dt

            print("Step = %d, Time =  %.2f %s" % (step, time/timeUnitRatio[timeUnit], timeUnit) )
            stepList.append(step)
            timeList.append(time)
           
        for varName in self.varNameList:
            var_dict[varName]  = np.array(var_dict[varName])
            var_ext_dict[varName] = np.array(var_ext_dict[varName])
            var_show_dict[varName] = np.array(var_show_dict[varName])
            total_mass_dict[varName] = np.array(total_mass_dict[varName])

        stepList = np.array(stepList)
        timeList = np.array(timeList)
        results = {
            "stepList": stepList,
            "timeList": timeList,
            "total_mass_dict": total_mass_dict,
            "var_dict": var_dict,
            "var_ext_dict": var_ext_dict,
            "var_show_dict": var_show_dict,
        }
        return results
                                                            

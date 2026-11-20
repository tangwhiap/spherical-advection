#!/usr/bin/env python

# CS555 spring 2024 - Final project
# Module for generating the triangle or hexagon mesh grids on a sphere
# Author: Wenhan Tang

import numpy as np
import copy
from scipy.sparse import coo_matrix
import plotly.graph_objects as go
import netCDF4 as nc
from .show_progress import show_progress
#import shapefile
#from shapely.geometry import Point, Polygon
import json
from pdb import set_trace

class SphereMesh(object):
    def __init__(self):
        """
            Should define
            R, V, T, E
            nV, nT, nE
        """
        self.undef_ind = -99999999
        self.default_lighting =dict(
                ambient=1, 
                diffuse=0, 
                fresnel=0, 
                specular=0,  
                roughness=1,  
                facenormalsepsilon=1, 
                vertexnormalsepsilon=1,
        )
        self.landocean_implemented = False
        self.landocean = None
        self.coastlines_implemented = False
        self.coastlines = None

    def info_show(self, meshName = ""):
        if len(meshName) > 0:
            meshName = meshName + " "
        print("Spherical " + meshName + "mesh grid created.")
        print("nV = %d, nT = %d, nE = %d" % (self.nV, self.nT, self.nE))
        print("h_ave = %.4f, h_max = %.4f, h_min = %.4f" % (np.mean(self.h), np.max(self.h), np.min(self.h)))

    def spherical_to_cartesian(self, lon, lat, r):
        phi = lon * 1
        theta = np.pi/2 - lat
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # 极角
        phi = np.arctan2(y, x)  # 方位角
        lon = phi * 1
        lat = np.pi/2 - theta
        return lon, lat, r

    def update_cartesian(self):
        X, Y, Z = self.spherical_to_cartesian(self.V[:, 0], self.V[:, 1], self.R)
        self.X = X.copy()
        self.Y = Y.copy()
        self.Z = Z.copy()
        
        
    def mid_point(self, x1, y1, z1, x2, y2, z2):
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        z_mid = (z1 + z2) / 2

        lon_mid, lat_mid, r_mid = self.cartesian_to_spherical(x_mid, y_mid, z_mid)
        return lon_mid, lat_mid, r_mid
    

    def compute_h(self):

        LL = []
        for i,j in self.E:
            L = np.sqrt( (self.X[i] - self.X[j])**2 + (self.Y[i] - self.Y[j])**2 + (self.Z[i] - self.Z[j])**2 )
            LL.append(L)
        LL = np.array(LL)
        return LL

    def compute_lonlat_stagger_grid(self):
        index_1, index_2 = self.E[:, 0], self.E[:, 1]
        x1, y1, z1 = self.X[index_1], self.Y[index_1], self.Z[index_1]
        x2, y2, z2 = self.X[index_2], self.Y[index_2], self.Z[index_2]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        z_center = (z1 + z2) / 2
        lon_rad, lat_rad, _ = self.cartesian_to_spherical(x_center, y_center, z_center)
        lon_deg = np.rad2deg(lon_rad)
        lat_deg = np.rad2deg(lat_rad)

        self.lon_s = lon_deg
        self.lat_s = lat_deg

        self.lon_s_rad = lon_rad
        self.lat_s_rad = lat_rad

    def compute_edge2index(self):
        nE = self.E.shape[0]
        self.edge2index = {((i,j) if i <= j else (j,i)): ind for i, j, ind in zip(self.E[:,0], self.E[:,1], np.arange(nE))}

    def compute_edge2neighbor(self):
        edge2neighbor = np.full( (self.nE, 2), self.undef_ind )
        for ind_E in range(self.nE):
            T_index = np.where(((self.E[ind_E, 0] == self.T) | (self.E[ind_E, 1] == self.T)).sum(axis=1) == 2)[0]
            assert len(T_index) == 2
            edge2neighbor[ind_E, :] = T_index
        self.edge2neighbor = edge2neighbor

    def compute_neighbor_distance(self):
        ind_T1 = self.edge2neighbor[:, 0]
        ind_T2 = self.edge2neighbor[:, 1]
        lon1, lat1 = self.lon_c_rad[ind_T1], self.lat_c_rad[ind_T1]
        x1, y1, z1 = self.spherical_to_cartesian(lon1, lat1, self.R)
        lon2, lat2 = self.lon_c_rad[ind_T2], self.lat_c_rad[ind_T2]
        x2, y2, z2 = self.spherical_to_cartesian(lon2, lat2, self.R)
        self.neighbor_distance = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )

    def compute_grid_area(self):
        N_edges = self.T.shape[1]
        edge_list = np.arange(N_edges).astype("int")
        ii = 0
        grid_area = np.full( (self.nT), 0.0 )
        for jj, kk in zip(edge_list[1:-1], edge_list[2:]):
            d_grid_area = np.full( (self.nT), np.nan )
            ind_M = np.column_stack((self.T[:, ii], self.T[:, jj], self.T[:, kk]))
            no_area_ind = np.where(ind_M == self.undef_ind)[0]
            have_area_ind = np.where(np.sum(ind_M != self.undef_ind, axis=1) == 3)[0]
            assert len(no_area_ind) + len(have_area_ind) == self.nT
            if len(no_area_ind) > 0:
                d_grid_area[no_area_ind] = 0.0
            if len(have_area_ind) > 0:
                ind_T1 = ind_M[have_area_ind, 0]
                ind_T2 = ind_M[have_area_ind, 1]
                ind_T3 = ind_M[have_area_ind, 2]
                lon1, lat1 = self.V[ind_T1, 0], self.V[ind_T1, 1]
                x1, y1, z1 = self.spherical_to_cartesian(lon1, lat1, self.R)
                lon2, lat2 = self.V[ind_T2, 0], self.V[ind_T2, 1]
                x2, y2, z2 = self.spherical_to_cartesian(lon2, lat2, self.R)
                lon3, lat3 = self.V[ind_T3, 0], self.V[ind_T3, 1]
                x3, y3, z3 = self.spherical_to_cartesian(lon3, lat3, self.R)
                dx1, dy1, dz1 = x2 - x1, y2 - y1, z2 - z1
                dx2, dy2, dz2 = x3 - x1, y3 - y1, z3 - z1
                AA = dy1 * dz2 - dz1 * dy2
                BB = dz1 * dx2 - dx1 * dz2
                CC = dx1 * dy2 - dy1 * dx2
                d_grid_area[have_area_ind] = 0.5 * np.sqrt(AA**2 + BB**2 + CC**2)
            grid_area += d_grid_area

        self.grid_area = grid_area
                



    def implement_landocean(self, shapeFile):
        import shapefile
        from shapely.geometry import Point, Polygon

        
        if not(self.landocean is None):
            print("Landocean has already loaded.")
            return
        
        def is_point_in_poly(point, poly):
            shapely_point = Point(point)
            shapely_poly = Polygon(poly)
            return shapely_poly.contains(shapely_point)
        
        self.landocean = np.full( (self.nT), 1 ).astype("int")
        sf = shapefile.Reader(shapeFile)
        shape_record = sf.shapeRecords()[0]
        polygon = shape_record.shape.points
        
        for ind_T, (ilon, ilat) in enumerate(zip(self.lon_c, self.lat_c)):
            show_progress(ind_T, self.nT)
            point = (ilon, ilat)
            if is_point_in_poly(point, polygon):
                self.landocean[ind_T] = 0
        self.landocean_implemented = True

    def implement_landocean_fromnpy(self, npyFile):
        landocean = np.load(npyFile)
        assert len(landocean) == self.nT
        self.landocean = landocean
        self.landocean_implemented = True
        
    def implement_coastlines(self, jsonFile):
        with open(jsonFile) as fr:
            coastlines = json.load(fr)
        self.coastlines = coastlines
        self.coastlines_implemented = True
            
    def draw_Mesh(self, fig, row = None, col = None, **kwargs):
        
        m3d = go.Mesh3d(x = self.X, y = self.Y, z = self.Z, i = self.T[:, 0], j = self.T[:, 1], k = self.T[:, 2], **kwargs)
        if (row is None) or (col is None):
            fig.add_trace(m3d)
        else:
            fig.add_trace(m3d, row = row, col = col)
        return fig

    def draw_Vertex(self, fig, **kwargs):
        s3d = go.Scatter3d(x = self.X, y = self.Y, z = self.Z, **kwargs)
        fig.add_trace(s3d)
        return fig

    def draw_Edges(self, fig, **kwargs):

        for i,j in self.E:
            l3d = go.Scatter3d(x=[self.X[i], self.X[j]], y=[self.Y[i], self.Y[j]], z=[self.Z[i], self.Z[j]], **kwargs)
            fig.add_trace(l3d)
        return fig

    def plot(self, fig = None, outName = None, drawMesh = True, drawVertex = True, drawEdges = True, title = ""):
        
        lighting = self.default_lighting
        
        if fig is None:
            fig = go.Figure()

        if drawMesh:
            fig = self.draw_Mesh(fig, opacity = 0.8, color = "skyblue", lighting = lighting)
        if drawVertex:
            fig = self.draw_Vertex(fig, mode = "markers", marker = {"size": 1, "color": "yellow"})
        if drawEdges:
            fig = self.draw_Edges(fig, mode="lines", line = {"color": "red", "width": 2})

        fig.update_layout(
            title = title,
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "x (m)",
                "yaxis_title": "y (m)",
                "zaxis_title": "z (m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
            showlegend = False,
        )
        if not(outName is None):
            fig.write_html(outName)
        return fig

    def plot_data(self, fig = None, data = None, outName = None, title = "", output_html = False, **kwargs):
        
        if data is None:
            data = np.random.rand(self.nT)
        assert len(data) == self.nT
        if fig is None:
            fig = go.Figure()
        if not("opacity" in kwargs):
            kwargs["opacity"] = 0.3
        if not("colorscale" in kwargs):
            kwargs["colorscale"] = "Viridis"
        if not("showscale" in kwargs):
            kwargs["showscale"] = True
        if not("cmin" in kwargs):
            kwargs["cmin"] = np.nanmin(data)
            print("cmin = ", kwargs["cmin"])
        if not("cmax" in kwargs):
            kwargs["cmax"] = np.nanmax(data)
            print("cmax = ", kwargs["cmax"])
        if not("lighting" in kwargs):
            kwargs["lighting"] = self.default_lighting
        #print(kwargs)
        fig = self.draw_Mesh(fig, intensity = data, intensitymode = "cell", **kwargs)
        fig.update_layout(
            title = title,
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "x (m)",
                "yaxis_title": "y (m)",
                "zaxis_title": "z (m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
        )
        if not(outName is None):
            if output_html:
                fig.write_html(outName + ".html")
            else:
                fig.write_image(outName + ".png", width=960, height=540, scale=2)
        return fig
    
    
    def plot_coastlines(self, fig, threshold = 0, **line_kwargs):
        assert self.coastlines_implemented, "Please implement coastlines data first by running  self.implement_coastlines()"
        for (lon, lat) in self.coastlines:
            if len(lon) < threshold:
                continue
            lon_rad, lat_rad = np.deg2rad(np.array(lon)), np.deg2rad(np.array(lat))
            x, y, z = self.spherical_to_cartesian(lon_rad, lat_rad, self.R)
            line = go.Scatter3d(x = x, y = y, z = z, mode = "lines", line = line_kwargs)
            fig.add_trace(line)
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_line(self, fig, lonList, latList, **line_kwargs):
        lon_rad = np.deg2rad(lonList)
        lat_rad = np.deg2rad(latList)
        x, y, z = self.spherical_to_cartesian(lon_rad, lat_rad, self.R)
        line = go.Scatter3d(x = x, y = y, z = z, mode = "lines", line = line_kwargs)
        fig.add_trace(line)
        return fig
        
    def plot_gridlines(self, fig):
        nPoints = 100
        lonList = np.linspace(-180, 180, nPoints)
        
        for ilat in np.arange(-75, 90, 15):
            if ilat == 0:
                line_color = "red"
                line_dash = "solid"
            else:
                line_color = "black"
                line_dash = "dash"
            latList = np.full( (nPoints), ilat)
            fig = self.plot_line(fig, lonList, latList, color = line_color, width = 1, dash = line_dash)
        
    
        latList = np.linspace(-90, 90, nPoints)
        for ilon in np.arange(-180, 180, 15):
            lonList = np.full( (nPoints), ilon)
            if (ilon == -180) or (ilon == 0):
                line_color = "red"
                line_dash = "solid"
            else:
                line_color = "black"
                line_dash = "dash"
            
            fig = self.plot_line(fig, lonList, latList, color = line_color, width = 1, dash = line_dash)  
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_landocean(self, fig = None, outName = None, output_html = False, **kwargs):
        assert self.landocean_implemented, "Please implement the land ocean data first by running self.implement_landocean() or self.implement_landocean_fromnpy()"
        if not("colorscale" in kwargs):
            kwargs["colorscale"] = [[0, 'skyblue'],[1, 'grey']]
        if not("showscale" in kwargs):
            kwargs["showscale"] = False
        if not("opacity" in kwargs):
            kwargs["opacity"] = 1
        fig = self.plot_data(fig = None, data = self.landocean, outName = None, output_html = False, **kwargs)
        return fig
        
    def plot_grid(self, fig = None, outName = None, title = ""):
        if fig is None:
            fig = go.Figure()
        fig = self.draw_Mesh(fig, opacity = 0.95, color = "skyblue", lighting = self.default_lighting)
        x, y, z = self.spherical_to_cartesian(self.lon_c_rad, self.lat_c_rad, self.R)
        s3d = go.Scatter3d(x = x, y = y, z = z, mode = "markers", marker = {"size": 2, "color": "green"})
        fig.add_trace(s3d)

        x, y, z = self.spherical_to_cartesian(self.lon_s_rad, self.lat_s_rad, self.R)
        s3d = go.Scatter3d(x = x, y = y, z = z, mode = "markers", marker = {"size": 2, "color": "blue"})
        fig.add_trace(s3d)

        fig = self.draw_Edges(fig, mode="lines", line = {"color": "red", "width": 2})
        fig.update_layout(
            title = title,
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "x (m)",
                "yaxis_title": "y (m)",
                "zaxis_title": "z (m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
            showlegend = False,
        )
        if not(outName is None):
            fig.write_html(outName)
        return fig

    def plot_neighbor(self, ind_tri, outName = "neighbor.html"):
        fig = go.Figure()
        m3d = go.Mesh3d(x = self.X[self.T[ind_tri]], y = self.Y[self.T[ind_tri]], z = self.Z[self.T[ind_tri]], i = [0], j = [ 1], k = [2], opacity = 0.8, color = "red")
        fig.add_trace(m3d)
        for ind_neighbor in self.neighbor[ind_tri]:
            print(ind_neighbor)
            m3d = go.Mesh3d(x = self.X[self.T[ind_neighbor]], y = self.Y[self.T[ind_neighbor]], z = self.Z[self.T[ind_neighbor]], i = [0], j = [1], k = [2], opacity = 0.8, color = "skyblue")

            fig.add_trace(m3d)
        fig.update_layout(
            #title = "test",
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "(m)",
                "yaxis_title": "(m)",
                "zaxis_title": "(m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
        )
        for i,j in self.E:
            l3d = go.Scatter3d(x=[self.X[i], self.X[j]], y=[self.Y[i], self.Y[j]], z=[self.Z[i], self.Z[j]], mode="lines", line = {"color": "red", "width": 2})
            fig.add_trace(l3d)
        fig.write_html(outName)


class SphereTriMesh(SphereMesh):

    def __init__(self, from_file = False, first_generation = True, **kwargs):
        SphereMesh.__init__(self)
        if from_file:
            self.read_from_file_init(**kwargs)
        else:
            if first_generation:
                self.first_generation_init(**kwargs)
            else:
                self.future_generation_init(**kwargs)
        self.update()
        self.info_show()

        
    def first_generation_init(self, R):
        self.R = R
        self.init_icosahedron()
        
        
    def future_generation_init(self, R, V, T):
        self.R = R
        self.V = V
        self.T = T

    def read_from_file_init(self, fileName):
        ncf = nc.Dataset(fileName)
        self.undef_ind = int(ncf.variables["undef_ind"][0].filled(np.nan))
        self.R = float(ncf.variables["R"][0].filled(np.nan))
        self.V = ncf.variables["V"][:].filled(np.nan)
        self.T = ncf.variables["T"][:].filled(np.nan)
        ncf.close()

    def copy(self):
        obj_copy = SphereTriMesh(R = self.R, V = self.V, T = self.T, first_generation = False)
        return obj_copy

        
    def init_icosahedron(self):
        phi = (1 + np.sqrt(5)) / 2
        theta = np.arctan(1/2)
        self.V = [ [0, np.pi/2] ]
        lon0 = np.linspace(-np.pi, np.pi, 6)
        lon1 = lon0[:-1]
        lon2 = (lon0[1:] + lon0[:-1]) / 2
        
        # create vertices
        self.V += [ [lon_i, theta] for lon_i in lon2 ]
        self.V += [ [lon_i, -theta] for lon_i in lon1 ]
        self.V += [ [0, -np.pi/2] ]
        self.V = np.array(self.V)
        
        # create triangles
        self.T = []
        a_list = [1, 2, 3, 4, 5, 1]
        b_list = [7, 8, 9, 10, 6, 7]

        for i in range(5):
            self.T.append([0, a_list[i], a_list[i+1]])
        for i in range(5):
            self.T.append([a_list[i], b_list[i], a_list[i+1]])
            self.T.append([a_list[i+1], b_list[i], b_list[i+1]])
        for i in range(5):
            self.T.append([b_list[i], b_list[i+1], 11])
        self.T = np.array(self.T)
        
    def create_edges(self):
        
        edges = [((i,j) if i <= j else (j,i)) for i,j in zip(self.T[:,0], self.T[:,1])]
        edges += [((i,j) if i <= j else (j,i)) for i,j in zip(self.T[:,1], self.T[:,2])]
        edges += [((i,j) if i <= j else (j,i)) for i,j in zip(self.T[:,2], self.T[:,0])]
        self.E = np.array(list(set(edges)))

    
    def create_tri2edge_mapping(self):
        tri2edge = np.full((self.nT, 3), self.undef_ind)
        for ind_tri, tri in enumerate(self.T):
            for ind_E, (i,j) in enumerate([(0,1), (1,2), (2,0)]):
                tri2edge[ind_tri, ind_E] = self.edge2index[(tri[i],tri[j]) if tri[i] <= tri[j] else (tri[j],tri[i])]
        self.tri2edge = tri2edge
        
        for e_ind in np.arange(self.nE):
            assert np.sum(self.tri2edge == e_ind) == 2

    def create_neighbor_mapping(self):
        pairs = []
        for ind_E in np.arange(self.nE):
             pairs.append(np.where((self.tri2edge == ind_E))[0])
        self.pairs = np.array(pairs)
        neighbor = np.full((self.nT, 3), self.undef_ind)
        for ind_tri in np.arange(self.nT):
            loc_x, loc_y = np.where(self.pairs == ind_tri)
            neighbor[ind_tri, :] = self.pairs[loc_x, 1-loc_y]
        self.neighbor = neighbor


    def compute_lonlat_center_grid(self):
        index_1, index_2, index_3 = self.T[:, 0], self.T[:, 1], self.T[:, 2]
        x1, y1, z1 = self.X[index_1], self.Y[index_1], self.Z[index_1]
        x2, y2, z2 = self.X[index_2], self.Y[index_2], self.Z[index_2]
        x3, y3, z3 = self.X[index_3], self.Y[index_3], self.Z[index_3]
        x_center = (x1 + x2 + x3) / 3
        y_center = (y1 + y2 + y3) / 3
        z_center = (z1 + z2 + z3) / 3
        lon_rad, lat_rad, _ = self.cartesian_to_spherical(x_center, y_center, z_center)
        lon_deg = np.rad2deg(lon_rad)
        lat_deg = np.rad2deg(lat_rad)

        self.lon_c = lon_deg
        self.lat_c = lat_deg

        self.lon_c_rad = lon_rad
        self.lat_c_rad = lat_rad
        
    def compute_lonlat_stagger_grid(self):
        index_1, index_2 = self.E[:, 0], self.E[:, 1]
        x1, y1, z1 = self.X[index_1], self.Y[index_1], self.Z[index_1]
        x2, y2, z2 = self.X[index_2], self.Y[index_2], self.Z[index_2]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        z_center = (z1 + z2) / 2
        lon_rad, lat_rad, _ = self.cartesian_to_spherical(x_center, y_center, z_center)
        lon_deg = np.rad2deg(lon_rad)
        lat_deg = np.rad2deg(lat_rad)

        self.lon_s = lon_deg
        self.lat_s = lat_deg

        self.lon_s_rad = lon_rad
        self.lat_s_rad = lat_rad
   
    def update(self):
        self.create_edges()
        self.update_cartesian()
        self.h = self.compute_h()
        self.nV = self.V.shape[0]
        self.nT = self.T.shape[0]
        self.nE = self.E.shape[0]
        self.compute_edge2index()
        self.create_tri2edge_mapping()
        self.create_neighbor_mapping()
        self.compute_lonlat_center_grid()
        self.compute_lonlat_stagger_grid()
        self.compute_edge2neighbor()
        self.compute_neighbor_distance()
        self.compute_grid_area()
        
        assert self.nV + self.nT - 2 == self.nE, "nV = %d, nT = %d, nE = %d" % (self.nV, self.nT, self.nE)


    def info_show(self):
        SphereMesh.info_show(self, meshName = "triangular")

        
        
                     

            
    def bisection_old(self):
        
        nV_orig = self.nV
        index_orig = np.arange(nV_orig)
        
        index_1 = self.T[:,0]
        index_2 = self.T[:,1]
        index_3 = self.T[:,2]
        
        X1, Y1, Z1 = self.X[index_1], self.Y[index_1], self.Z[index_1]
        X2, Y2, Z2 = self.X[index_2], self.Y[index_2], self.Z[index_2]
        X3, Y3, Z3 = self.X[index_3], self.Y[index_3], self.Z[index_3]
        
        lon_m12, lat_m12, r_m12 = self.mid_point(X1, Y1, Z1, X2, Y2, Z2)
        lon_m23, lat_m23, r_m23 = self.mid_point(X2, Y2, Z2, X3, Y3, Z3)
        lon_m13, lat_m13, r_m13 = self.mid_point(X1, Y1, Z1, X3, Y3, Z3)
       

        V_m12 = np.column_stack((lon_m12, lat_m12))
        nV_m12 = V_m12.shape[0]
        index_m12 = np.arange(nV_orig, nV_orig + nV_m12)
        
        V_m23 = np.column_stack((lon_m23, lat_m23))
        nV_m23 = V_m23.shape[0]
        index_m23 = np.arange(nV_orig + nV_m12, nV_orig + nV_m12 + nV_m23)
        
        V_m13 = np.column_stack((lon_m13, lat_m13))
        nV_m13 = V_m13.shape[0]


        index_m13 = np.arange(nV_orig + nV_m12 + nV_m23, nV_orig + nV_m12 + nV_m23 + nV_m13)
        
        V_new = np.full((nV_orig + nV_m12 + nV_m23 + nV_m13, 2), np.nan)
        V_new[index_orig, :] = self.V.copy()
        V_new[index_m12, :] = V_m12
        V_new[index_m23, :] = V_m23
        V_new[index_m13, :] = V_m13
        
        tri1 = np.column_stack( (index_1, index_m13, index_m12) )
        tri2 = np.column_stack( (index_m12, index_m13, index_m23) )
        tri3 = np.column_stack( (index_m12, index_m23, index_2) )
        tri4 = np.column_stack( (index_m13, index_3, index_m23) )
        
        T_new = np.vstack((tri1, tri2, tri3, tri4))
        return V_new, T_new
    
    def bisection(self):
        nV_orig = self.nV
        
        index_E1 = self.E[:, 0]
        index_E2 = self.E[:, 1]
        edge2index = {((i,j) if i <= j else (j,i)): ind for i, j, ind in zip(index_E1, index_E2, np.arange(self.nE))}
        
        X1, Y1, Z1 = self.X[index_E1], self.Y[index_E1], self.Z[index_E1]
        X2, Y2, Z2 = self.X[index_E2], self.Y[index_E2], self.Z[index_E2]
        lon_m, lat_m, r_m = self.mid_point(X1, Y1, Z1, X2, Y2, Z2)

        V_m = np.column_stack((lon_m, lat_m))
        nV_m = V_m.shape[0]
        index_orig = np.arange(nV_orig)
        index_m = np.arange(nV_orig, nV_orig + nV_m)
        
        V_new = np.full((nV_orig + nV_m, 2), np.nan)
        V_new[index_orig, :] = self.V.copy()
        V_new[index_m, :] = V_m
        
        index_1 = self.T[:,0]
        index_2 = self.T[:,1]
        index_3 = self.T[:,2]
        
        edge_12 = [((i,j) if i <= j else (j,i)) for i, j in zip(self.T[:,0], self.T[:,1])]
        index_m12 = index_m[np.array([edge2index[edge] for edge in edge_12])]
        
        edge_23 = [((i,j) if i <= j else (j,i)) for i, j in zip(self.T[:,1], self.T[:,2])]
        index_m23 = index_m[np.array([edge2index[edge] for edge in edge_23])]
        
        edge_13 = [((i,j) if i <= j else (j,i)) for i, j in zip(self.T[:,0], self.T[:,2])]
        index_m13 = index_m[np.array([edge2index[edge] for edge in edge_13])]
        
        tri1 = np.column_stack( (index_1, index_m13, index_m12) )
        tri2 = np.column_stack( (index_m12, index_m13, index_m23) )
        tri3 = np.column_stack( (index_m12, index_m23, index_2) )
        tri4 = np.column_stack( (index_m13, index_3, index_m23) )
       
        T_new = np.vstack((tri1, tri2, tri3, tri4))
        
        return V_new, T_new
        
    def give_birth(self):
        V_new, T_new = self.bisection()
        obj_child = SphereTriMesh(R = self.R, V = V_new, T = T_new, first_generation = False)
        return obj_child

    def convert_to_hexagon(self):

        V_new = np.column_stack((self.lon_c_rad, self.lat_c_rad))
        lon_c_rad_new = self.V[:, 0]
        lat_c_rad_new = self.V[:, 1]

        T_new = np.full( (self.nV, 6), self.undef_ind )
        neighbor_new = np.full ( (self.nV, 6), self.undef_ind )

        is_pentagon_points = []
        is_hexagon_points = []
       
        print("Converting from triangle mesh to hexagon mesh ...")
        for ind_V in range(self.nV):
            show_progress(ind_V+1, self.nV)
            edges_number = np.sum(self.T == ind_V)
            assert edges_number == 5 or edges_number == 6
            if edges_number == 5:
                is_hexagon = False
                is_pentagon_points.append(ind_V)
            else:
                is_hexagon = True
                is_hexagon_points.append(ind_V)
            index_around = np.where(self.T == ind_V)[0]
            temp = self.T[index_around, :]
            temp1 = np.where(temp == ind_V, np.inf, temp)
            temp1.sort()
            temp2 = temp1[:, :2].astype("int")
            obj_indV = 0
            start_indT = 0
            current_tri = start_indT
            poly_hexagon = [current_tri]
            while True:
                neighbor_indV = temp2[current_tri, 1 - obj_indV]
                ii,jj = np.where(temp2 == neighbor_indV)
                kk = int(np.where(ii!=current_tri)[0][0])
                current_tri = ii[kk]
                obj_indV = jj[kk] 
                if current_tri == start_indT:
                    break
                poly_hexagon.append(current_tri)
            if is_hexagon:
                assert len(poly_hexagon) == 6
            else:
                assert len(poly_hexagon) == 5
            hexagon_points = index_around[poly_hexagon]
            
            T_new[ind_V, :len(poly_hexagon)] = hexagon_points
            ii, jj = np.where(self.E == ind_V)
            neighbor_points = self.E[ii,1-jj]
            neighbor_new[ind_V, :len(neighbor_points)] = neighbor_points

        is_pentagon_points = np.array(is_pentagon_points)
        is_hexagon_points = np.array(is_hexagon_points)
        
        hexMesh = SphereHexMesh(R = self.R, V = V_new, T = T_new, neighbor = neighbor_new, lon_c_rad = lon_c_rad_new, lat_c_rad = lat_c_rad_new, is_pentagon_points = is_pentagon_points, is_hexagon_points = is_hexagon_points)
        return hexMesh

    
    def save_to_file(self, fileName = None):
        if fileName is None:
            fileName = "meshgrid.nc"
        ncf = nc.Dataset(fileName, "w", format = "NETCDF4")
        ncf.createDimension("nV", self.nV)
        ncf.createDimension("nT", self.nT)
        ncf.createDimension("nE", self.nE)
        ncf.createDimension("1", 1)
        ncf.createDimension("2", 2)
        ncf.createDimension("3", 3)
        ncv_dict = {}

        ncv_dict["undef_ind"] = ncf.createVariable("undef_ind", "i8", ("1"))
        ncv_dict["undef_ind"][0] = self.undef_ind

        ncv_dict["R"] = ncf.createVariable("R", "f8", ("1"))
        ncv_dict["R"][0] = self.R

        ncv_dict["V"] = ncf.createVariable("V", "f8", ("nV", "2"))
        ncv_dict["V"][:] = self.V

        ncv_dict["T"] = ncf.createVariable("T", "i8", ("nT", "3"))
        ncv_dict["T"][:] = self.T

        ncf.close()

    def plot_edge2neighbor(self, E_list, outName = "edge2neighbor.html"):
        fig = go.Figure()
        colors = ["orange", "skyblue"]
        if isinstance(E_list, int):
            E_list = [E_list]
        for ind_E in E_list:
            neighbor = self.edge2neighbor[ind_E, :]
            for ind_tri, color in zip(neighbor, colors):
                m3d = go.Mesh3d(x = self.X[self.T[ind_tri]], y = self.Y[self.T[ind_tri]], z = self.Z[self.T[ind_tri]], i = [0], j = [1], k = [2], opacity = 0.4, color = color)
                fig.add_trace(m3d)

            i, j = self.E[ind_E, :]
            l3d = go.Scatter3d(x=[self.X[i], self.X[j]], y=[self.Y[i], self.Y[j]], z=[self.Z[i], self.Z[j]], mode="lines", line = {"color": "red", "width": 2})
            fig.add_trace(l3d)

        fig.update_layout(
            #title = "test",
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "(m)",
                "yaxis_title": "(m)",
                "zaxis_title": "(m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
        )
        fig.write_html(outName)


class SphereHexMesh(SphereMesh):

    def __init__(self, from_file = False, **kwargs):
        SphereMesh.__init__(self)
        if from_file:
            self.read_from_file_init(**kwargs)
        else:
            self.init_from_triMesh(**kwargs)

        assert self.nV + self.nT - 2 == self.nE, "nV = %d, nT = %d, nE = %d" % (self.nV, self.nT, self.nE)
        self.info_show()

    def info_show(self):
        SphereMesh.info_show(self, meshName = "hexagon")

    def init_from_triMesh(self, R, V, T, neighbor, lon_c_rad, lat_c_rad, is_pentagon_points, is_hexagon_points):
        self.R = R
        self.V = V
        self.T = T
        self.neighbor = neighbor
        self.lon_c_rad = lon_c_rad
        self.lat_c_rad = lat_c_rad
        self.pentagon_Tindex = is_pentagon_points
        self.hexagon_Tindex = is_hexagon_points
        self.update()

    def read_from_file_init(self, fileName):
        ncf = nc.Dataset(fileName)
        self.undef_ind = int(ncf.variables["undef_ind"][0].filled(np.nan))
        self.R = float(ncf.variables["R"][0].filled(np.nan))
        self.V = ncf.variables["V"][:].filled(np.nan)
        self.T = ncf.variables["T"][:].filled(np.nan)
        self.neighbor = ncf.variables["neighbor"][:].filled(np.nan)
        self.lon_c_rad = ncf.variables["lon_c_rad"][:].filled(np.nan)
        self.lat_c_rad = ncf.variables["lat_c_rad"][:].filled(np.nan)
        self.pentagon_Tindex = ncf.variables["pentagon_Tindex"][:].filled(np.nan)
        self.hexagon_Tindex = ncf.variables["hexagon_Tindex"][:].filled(np.nan)
        self.edge2neighbor = ncf.variables["edge2neighbor"][:].filled(np.nan)
        ncf.close()
        self.update(do_edge2neighbor = False)
   
    def update(self, do_edge2neighbor = True):
        self.lon_c = np.rad2deg(self.lon_c_rad)
        self.lat_c = np.rad2deg(self.lat_c_rad)
        self.create_edges()
        self.compute_edge2index()
        self.update_cartesian()
        self.compute_lonlat_stagger_grid()
        self.nT = self.T.shape[0]
        self.nV = self.V.shape[0]
        self.nE = self.E.shape[0]
        self.h = self.compute_h()
        if do_edge2neighbor:
            self.compute_edge2neighbor()
        self.compute_neighbor_distance()
        self.compute_grid_area()


    def create_edges(self):
        edges = []
        pentagon_index = [0, 1, 2, 3, 4, 0]
        for icol1, icol2 in zip(pentagon_index[:-1], pentagon_index[1:]):
            edges += [((i,j) if i <= j else (j,i)) for i,j in zip(self.T[self.pentagon_Tindex,icol1], self.T[self.pentagon_Tindex, icol2])]
        if len(self.hexagon_Tindex) > 0:
            hexagon_index = [0, 1, 2, 3, 4, 5, 0]
            for icol1, icol2 in zip(hexagon_index[:-1], hexagon_index[1:]):
                edges += [((i,j) if i <= j else (j,i)) for i,j in zip(self.T[self.hexagon_Tindex,icol1], self.T[self.hexagon_Tindex, icol2])]
        self.E = np.array(list(set(edges)))

    
    def plot_edge2neighbor(self, E_list, outName = "edge2neighbor.html"):
        fig = go.Figure()
        colors = ["orange", "skyblue"]
        if isinstance(E_list, int):
            E_list = [E_list]
        for ind_E in E_list:
            neighbor = self.edge2neighbor[ind_E, :]
            for ind_tri, color in zip(neighbor, colors):
                if ind_tri in self.pentagon_Tindex:
                    pentagon_drawList = [0, 1, 2, 3, 4]
                    for jj, kk in zip(pentagon_drawList[1:-1], pentagon_drawList[2:]):
                        m3d = go.Mesh3d(x = self.X[self.T[ind_tri, :-1]], y = self.Y[self.T[ind_tri, :-1]], z = self.Z[self.T[ind_tri, :-1]], i = [0], j = [jj], k = [kk], opacity = 0.4, color = color)
                        fig.add_trace(m3d)
                else:
                    hexagon_drawList = [0, 1, 2, 3, 4, 5]
                    for jj, kk in zip(hexagon_drawList[1:-1], hexagon_drawList[2:]):
                        m3d = go.Mesh3d(x = self.X[self.T[ind_tri]], y = self.Y[self.T[ind_tri]], z = self.Z[self.T[ind_tri]], i = [0], j = [jj], k = [kk], opacity = 0.4, color = color)
                        fig.add_trace(m3d)

            i, j = self.E[ind_E, :]
            l3d = go.Scatter3d(x=[self.X[i], self.X[j]], y=[self.Y[i], self.Y[j]], z=[self.Z[i], self.Z[j]], mode="lines", line = {"color": "red", "width": 2})
            fig.add_trace(l3d)

        fig.update_layout(
            #title = "test",
            margin = {"l": 0, "r": 0, "b": 0, "t": 30},
            scene = {
                "xaxis_title": "(m)",
                "yaxis_title": "(m)",
                "zaxis_title": "(m)",
                "xaxis": {"range": [-self.R, self.R]},
                "yaxis": {"range": [-self.R, self.R]},
                "zaxis": {"range": [-self.R, self.R]},
            },
        )
        fig.write_html(outName)





    def draw_Mesh(self, fig, row = None, col = None, z_offset = None, **kwargs):

        if "intensity" in kwargs:
            data = kwargs["intensity"].copy()
            kwargs1 = copy.deepcopy(kwargs)
            kwargs1["intensity"] = data[self.pentagon_Tindex]
            kwargs2 = copy.deepcopy(kwargs)
            kwargs2["intensity"] = data[self.hexagon_Tindex]
            
        else:
            kwargs1 = kwargs
            kwargs2 = kwargs
            
        if z_offset is None:
            do_offset = False
            plot_X = self.X
            plot_Y = self.Y
            plot_Z = self.Z
        else:
            do_offset = True
            print("Do vertical offset")
            plot_X, plot_Y, plot_Z = self.spherical_to_cartesian(self.V[:, 0], self.V[:, 1], self.R + z_offset)
        pentagon_drawList = [0, 1, 2, 3, 4]
        for jj, kk in zip(pentagon_drawList[1:-1], pentagon_drawList[2:]):
            m3d = go.Mesh3d(x = plot_X, y = plot_Y, z = plot_Z, i = self.T[self.pentagon_Tindex, 0], j = self.T[self.pentagon_Tindex, jj], k = self.T[self.pentagon_Tindex, kk], **kwargs1)
            if (row is None) or (col is None):
                fig.add_trace(m3d)
            else:
                fig.add_trace(m3d, row = row, col = col)

        if len(self.hexagon_Tindex) > 0:
            hexagon_drawList = [0, 1, 2, 3, 4, 5]
            for jj, kk in zip(hexagon_drawList[1:-1], hexagon_drawList[2:]):
                m3d = go.Mesh3d(x = plot_X, y = plot_Y, z = plot_Z, i = self.T[self.hexagon_Tindex, 0], j = self.T[self.hexagon_Tindex, jj], k = self.T[self.hexagon_Tindex, kk], **kwargs2)
                if (row is None) or (col is None):
                    fig.add_trace(m3d)
                else:
                    fig.add_trace(m3d, row = row, col = col)
        return fig

    def save_to_file(self, fileName = None):
        if fileName is None:
            fileName = "meshgrid.nc"
        ncf = nc.Dataset(fileName, "w", format = "NETCDF4")
        ncf.createDimension("nV", self.nV)
        ncf.createDimension("nT", self.nT)
        ncf.createDimension("nE", self.nE)
        ncf.createDimension("pentagon", 12)
        ncf.createDimension("hexagon", self.nT - 12)
        ncf.createDimension("1", 1)
        ncf.createDimension("2", 2)
        ncf.createDimension("3", 3)
        ncf.createDimension("6", 6)
        ncv_dict = {}

        ncv_dict["undef_ind"] = ncf.createVariable("undef_ind", "i8", ("1"))
        ncv_dict["undef_ind"][0] = self.undef_ind

        ncv_dict["R"] = ncf.createVariable("R", "f8", ("1"))
        ncv_dict["R"][0] = self.R

        ncv_dict["V"] = ncf.createVariable("V", "f8", ("nV", "2"))
        ncv_dict["V"][:] = self.V

        ncv_dict["T"] = ncf.createVariable("T", "i8", ("nT", "6"))
        ncv_dict["T"][:] = self.T

        ncv_dict["neighbor"] = ncf.createVariable("neighbor", "i8", ("nT", "6"))
        ncv_dict["neighbor"][:] = self.neighbor

        ncv_dict["lon_c_rad"] = ncf.createVariable("lon_c_rad", "f8", ("nT"))
        ncv_dict["lon_c_rad"][:] = self.lon_c_rad

        ncv_dict["lat_c_rad"] = ncf.createVariable("lat_c_rad", "f8", ("nT"))
        ncv_dict["lat_c_rad"][:] = self.lat_c_rad

        ncv_dict["pentagon_Tindex"] = ncf.createVariable("pentagon_Tindex", "i8", ("pentagon"))
        ncv_dict["pentagon_Tindex"][:] = self.pentagon_Tindex

        ncv_dict["hexagon_Tindex"] = ncf.createVariable("hexagon_Tindex", "i8", ("hexagon"))
        ncv_dict["hexagon_Tindex"][:] = self.hexagon_Tindex

        ncv_dict["edge2neighbor"] = ncf.createVariable("edge2neighbor", "i8", ("nE", "2"))
        ncv_dict["edge2neighbor"][:] = self.edge2neighbor

        ncf.close()



def test_transform():
    aa = SphereTriMesh(R=1)
    
    lon = np.random.rand(10000) * 2 * np.pi - np.pi
    lat = np.random.rand(10000) * np.pi - np.pi/2
    r = np.random.rand(10000)
    x, y, z = aa.spherical_to_cartesian(lon, lat, r)
    lon_, lat_, r_ = aa.cartesian_to_spherical(x, y, z)
    print(np.max(np.abs(lon - lon_)))
    print(np.max(np.abs(lat - lat_)))
    print(np.max(np.abs(r - r_)))
       
def create_grid(R, order, grid_type = "triangle", save_history_to_file = False, save_history_hexagon_to_file = False, fileDir = "."):
    assert grid_type in ["triangle", "hexagon"]
    mesh0 = SphereTriMesh(R = R)
    mesh_old = mesh0.copy()
    if save_history_to_file:
        mesh_old.save_to_file(fileDir + "/triangle_order0.nc")
    if save_history_hexagon_to_file and (grid_type == "hexagon"):
        mesh_hexagon = mesh_old.convert_to_hexagon()
        mesh_hexagon.save_to_file(fileDir + "/hexagon_order0.nc")
        del(mesh_hexagon)
    for i in range(order):
        mesh_new = mesh_old.give_birth()
        if save_history_to_file and ( (i != order - 1) or (grid_type != "triangle")) :
            mesh_new.save_to_file(fileDir + "/triangle_order" + str(i+1) + ".nc")
        if save_history_hexagon_to_file and (i != order - 1) and (grid_type == "hexagon"):
            mesh_hexagon = mesh_new.convert_to_hexagon()
            mesh_hexagon.save_to_file(fileDir + "/hexagon_order" + str(i+1) + ".nc")
            del(mesh_hexagon)
        del(mesh_old)
        mesh_old = mesh_new
    if grid_type == "hexagon":
        mesh_new = mesh_new.convert_to_hexagon()
    return mesh_new


if __name__ == "__main__":
    """
        !! For tests only !!
    """
    R = 6371000
    order = 4
    gridFileDir = "../meshgrid"
    save_history_to_file = True
    save_history_hexagon_to_file = True
    mesh = create_grid(
        R = R,
        order = order,
        save_history_to_file = save_history_to_file,
        save_history_hexagon_to_file = save_history_hexagon_to_file,
        fileDir = gridFileDir,
        grid_type = "hexagon"
        #grid_type = "triangle"
    )
    mesh.save_to_file(gridFileDir + "/hexagon_order" + str(order) + ".nc")

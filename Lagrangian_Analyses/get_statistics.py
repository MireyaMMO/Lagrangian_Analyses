import xarray as xr
import pandas as pd 
import numpy as np 
import matplotlib
import logging
from numpy import random, histogram2d
from scipy.interpolate import interp2d

logging.basicConfig(level=logging.INFO)

class get_statistics(object):
    def __init__(
        self,
        file_list,
        id=None,
        PDF=True,
        CM=True, 
        time_step=30,
        time_step_units='D'
        patch=False,
        ):
        self.file_list = file_list
        self.id = id
        self.PDF = PDF
        self.CM = CM
        self.patch = patch
        self.time_step = time_step
        self.time_step_units = time_step_units
        self.first_PDF = True
        self.first_CM = True
        self.logger = logging
        
    def get_requested_time_step_index(self, time):
        delta_time = time[1] - time[0]
        delta_time= int(delta_time.dt.total_seconds())
        index_interval = np.timedelta64(self.time_step,self.time_step_units) / np.timedelta64(delta_time, 's')
        return int(index_interval)
     
    def get_connectivity_matrix(self, CM, patches):      
        t=np.zeros((len(patches),275000))
        t[np.where(t==0)]=np.nan
        matrix=np.zeros((self.number_of_release_locations,len(patches)+1))
        matrix[np.where(matrix==0)]=np.nan
        for count, patch in enumerate(patches):
            pt1=matplotlib.path.Path(patch)
            grid=[]
            for row in CM:
                grid=np.append(grid,pt1.contains_point(row[0:2]))
            temp=CM[np.where(grid!=0),2][0]
            t[count,0:len(temp)]=temp
#                print(i)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j<matrix.shape[1]-1:
                    matrix[i,j]=len(np.where(t[j,:]==i)[0])
                else:
                    matrix[i,j]=self.number_of_particles-matrix[j,0:-1].sum()
        matrix[np.where(matrix==0)]=np.nan
        return matrix

    def centers(edges):
            return edges[:-1] + np.diff(edges[:2])/2
        
    def get_pdf(self, PDF):
        x = PDF[:,0]
        y = PDF[:,1]
        nogood = np.where(y>0)[0]
        x= np.delete(x,nogood)
        y= np.delete(y,nogood)     
        H, xedges, yedges = histogram2d(x, y, bins=[120,90])

        xcenters = self.centers(xedges)
        ycenters = self.centers(yedges)
        pdf = interp2d(xcenters, ycenters, H.T)
        n_pdf =  pdf(xedges, yedges)/pdf(xedges, yedges).sum()
        n_pdf[0:2,:]=0
        n_pdf[-2:-1,:]=0
        n_pdf[:,0:2]=0
        n_pdf[:,-2:-1]=0
        return n_pdf, xedges, yedges 

    def get_first_last_and_origin(self,ds):
        lon = ds.lon.where(ds.lon<9.0e+35, np.nan)
        lon = lon.where(lon>0 , lon+360)
        lat = ds.lat.where(ds.lat<9.0e+35, np.nan)
        requested_index = self.get_requested_time_step_index(self, ds.time )
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        number_of_release_locations= np.unique(origin_marker)[~np.isnan(np.unique(origin_marker))] #make sure there are no NaNs
        number_of_particles = len(ds['trajectory'])
        CM = np.empty((number_of_particles,3))
        for count in range(number_of_particles,1):
            active_particles = status.sel(trajectory=count)
            active_particles = np.where(active_particles==0)[0]
            if len(active_particles)>=requested_index:
                last_index = active_particles[requested_index]
            else:
                last_index = active_particles[-1]
            CM[count,0] = lon.sel(trajectory=count).isel(time=last_index).data
            CM[count,1] = lat.sel(trajectory=count).isel(time=last_index).data
            CM[count,2] = origin_marker.sel(trajectory=count).isel(time=last_index).data
        return CM
    
    def run(self):
        for file in self.filelist:
            ds = xr.open_dataset(file)
            lon = ds.lon.where(ds.lon<9.0e+35, np.nan)
            lon = lon.where(lon>0 , lon+360)
            lat = ds.lat.where(ds.lat<9.0e+35, np.nan)
            status = ds.status.where(~np.isnan(lon), np.nan)
            origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
            self.number_of_release_locations= np.unique(origin_marker)[~np.isnan(np.unique(origin_marker))] #make sure there are no NaNs
            self.trajectories = ds['trajectory']
            self.number_of_particles = len(self.trajectories)
            CM = self.get_first_last_and_origin(ds)
            if self.first_PDF:
                PDF=CM.copy()
                self.first_PDF = False
            else:
                PDF=np.vstack((PDF,CM))

            if self.CM:
                try:
                    patches  =np.load(self.patch, allow_pickle=True)
                except:
                    self.logger.warn('Need a path Collection')        
                matrix = self.get_connectivity_matrix(CM, patches)
                if self.first_CM:
                    Matrix=np.expand_dims(matrix,0)
                    self.first_CM = False
                else:
                    Matrix=np.vstack((Matrix,np.expand_dims(matrix,0)))
            
        if self.PDF:
            n_pdf, xedges, yedges = self.get_pdf(PDF)
            n_pdf.dump(f'{self.outdir}/normalised_PDF_{self.id}')
            xedges.dump(f'{self.outdir}/PDF_xedges_{self.id}')
            yedges.dump(f'{self.outdir}/PDF_yedges_{self.id}')
        if self.CM:
            Matrix.dump(f'{self.outdir}/CM_{self.id}')
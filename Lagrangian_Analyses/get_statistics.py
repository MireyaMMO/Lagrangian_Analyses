import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
import logging
from numpy import random, histogram2d
from scipy.interpolate import interp2d
from .utils import *
import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  

logging.basicConfig(level=logging.INFO)


class get_statistics(object):
    def __init__(
        self,
        file_list,
        id=None,
        PDF=[120, 90],
        CM=True,
        CV=True,
        stranded=True,
        time_of_interest=30,
        time_of_interest_units="D",
        patch=False,
    ):
        self.file_list = file_list
        self.id = id
        self.PDF = PDF
        self.CM = CM
        self.stranded = stranded
        self.patch = patch
        self.time_of_interest = time_of_interest
        self.time_interest_units = time_of_interest_units
        self.first_PDF = True
        self.first_CM = True
        self.first_stranded = True
        self.logger = logging

    def get_requested_time_step_index(self, time):
        delta_time = time[1] - time[0]
        delta_time = int(delta_time.dt.total_seconds())
        index_interval = np.timedelta64(
            self.time_of_interest, self.time_of_interest_units
        ) / np.timedelta64(delta_time, "s")
        return int(index_interval)

    def get_connectivity_matrix(self, CM, patches):
        temporal_arr = make_nan_array(len(patches), 275000)
        matrix = make_nan_array(self.number_of_release_locations, len(patches) + 1)
        for count, patch in enumerate(patches):
            pt1 = matplotlib.path.Path(patch)
            grid = []
            for row in CM:
                grid = np.append(grid, pt1.contains_point(row[0:2]))
            temp = CM[np.where(grid != 0), 2][0]
            temporal_arr[count, 0 : len(temp)] = temp
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j < matrix.shape[1] - 1:
                    matrix[i, j] = len(np.where(temporal_arr[j, :] == i)[0])
                else:
                    matrix[i, j] = np.nansum(self.number_of_particles - matrix[j, 0:-1])
        matrix[np.where(matrix == 0)] = np.nan
        return matrix

    def centers(edges):
        return edges[:-1] + np.diff(edges[:2]) / 2

    def get_pdf(self, PDF):
        x = PDF[:, 0]
        y = PDF[:, 1]
        nogood = np.where(y > 0)[0]
        x = np.delete(x, nogood)
        y = np.delete(y, nogood)
        H, xedges, yedges = histogram2d(x, y, bins=self.PDF)

        xcenters = self.centers(xedges)
        ycenters = self.centers(yedges)
        pdf = interp2d(xcenters, ycenters, H.T)
        n_pdf = pdf(xedges, yedges) / pdf(xedges, yedges).sum()
        n_pdf[0:2, :] = 0
        n_pdf[-2:-1, :] = 0
        n_pdf[:, 0:2] = 0
        n_pdf[:, -2:-1] = 0
        return n_pdf, xedges, yedges

    def get_data_from_file(self, ds):
        lon = ds.lon.where(ds.lon < 9.0e35, np.nan)
        lon = lon.where(lon > 0, lon + 360)
        lat = ds.lat.where(ds.lat < 9.0e35, np.nan)
        if isinstance(self.time_of_interest, int):
            requested_index = self.get_requested_time_step_index(self, ds.time)
        elif self.time_of_interest == "last":
            requested_index = -1
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        number_of_particles = len(ds["trajectory"])
        return lon, lat, requested_index, status, origin_marker, number_of_particles

    def get_lon_lat_and_origin(
        self, lon, lat, requested_index, status, origin_marker, number_of_particles
    ):
        CM = make_nan_array(number_of_particles, 3)
        for count in range(number_of_particles):
            active_particles = status.sel(trajectory=count + 1)
            active_particles = np.where(active_particles == 0)[0]
            if len(active_particles) >= requested_index:
                last_index = active_particles[requested_index]
            else:
                last_index = active_particles[-1]
            CM[count, 0] = lon.sel(trajectory=count + 1).isel(time=last_index).data
            CM[count, 1] = lat.sel(trajectory=count + 1).isel(time=last_index).data
            CM[count, 2] = (
                origin_marker.sel(trajectory=count + 1).isel(time=last_index).data
            )
        return CM

    def get_stranded_particles(
        self, lon, lat, requested_index, status, origin_marker, number_of_particles,
    ):
        for count in range(number_of_particles):
            particles = status.sel(trajectory=count + 1)
            index_of_release = np.where(particles == 0)[0][0]
            stranded_particles = np.where(particles == 1)[0]
            if stranded_particles.size>0 and stranded_particles < requested_index:
                time_of_stranding = lon.sel(trajectory=count + 1).isel(time=stranded_particles).time.data
                time_of_release = lon.sel(trajectory=count + 1).isel(time=index_of_release).time.data
                duration_s = (time_of_stranding - time_of_release)*1e-9 / 3600
                lon_s = lon.sel(trajectory=count + 1).isel(time=stranded_particles).data
                lat_s = lat.sel(trajectory=count + 1).isel(time=stranded_particles).data
                om_s = origin_marker.sel(trajectory=count + 1).isel(time=stranded_particles).data
                if self.first_stranded:
                    self.duration_before_stranding = duration_s
                    self.lon_stranding = lon_s
                    self.lat_stranding = lat_s
                    self.om_stranding = om_s 
                    self.first_stranded=False
                else:
                    self.duration_before_stranding = np.hstack((self.duration_before_stranding, duration_s))
                    self.lon_stranding = np.hstack((self.lon_stranding, lon_s))
                    self.lat_stranding = np.hstack((self.lat_stranding, lat_s))
                    self.om_stranding = np.hstack((self.om_stranding, om_s))
        
    def run(self):
        for file in self.filelist:
            ds = xr.open_dataset(file)
            (
                lon,
                lat,
                requested_index,
                status,
                origin_marker,
                self.number_of_particles,
            ) = self.get_data_from_file(ds)
            self.number_of_release_locations = np.unique(origin_marker)[
                ~np.isnan(np.unique(origin_marker))
            ]  # make sure there are no NaNs
            CM = self.get_lon_lat_and_origin(
                lon,
                lat,
                requested_index,
                status,
                origin_marker,
                self.number_of_particles,
            )

            if self.first_PDF:
                PDF = CM.copy()
                self.first_PDF = False
            else:
                PDF = np.vstack((PDF, CM))

            if self.CM:
                try:
                    patches = np.load(self.patch, allow_pickle=True)
                except:
                    self.logger.warn("Need a path Collection")
                matrix = self.get_connectivity_matrix(CM, patches)
                if self.first_CM:
                    Matrix = np.expand_dims(matrix, 0)
                    self.first_CM = False
                else:
                    Matrix = np.vstack((Matrix, np.expand_dims(matrix, 0)))
            if self.stranded:
                self.get_stranded_particles(lon, lat, requested_index, status, origin_marker, self.number_of_particles)
        
        if self.PDF:
            print(
                "Saving PDF associated parameters: normalised PDF (n_PDF), and bin edges (PDF_xedges,)"
            )
            self.norm_pdf, self.xedges, self.yedges = self.get_pdf(PDF)
            pickle.dump(
                [
                    self.norm_pdf,
                    self.xedges,
                    self.yedges,
                ],
                open(f"{self.outdir}/PDF_{self.id}.p", "wb"),
            )
        if self.CM:
            self.avg_CM = np.nanmean(Matrix, axis=0)[::-1]
            self.norm_CM = self.avg_CM / np.nansum(self.avg_CM, ax=1)
            self.norm_CM.dump(f"{self.outdir}/normalised_CM_{self.id}")
            pickle.dump(
                [
                    self.norm_CM,
                ],
                open(f"{self.outdir}/CM_{self.id}.p", "wb"),
            )
        if self.CV:
            std_CM = np.nanstd(Matrix, axis=0)[::-1]
            self.CV = std_CM / self.avg_CM
            pickle.dump(
                [
                    self.norm_CM,
                    self.CV,
                ],
                open(f"{self.outdir}/CM_{self.id}.p", "wb"),
            )
        if self.stranded:
            avg_duration = np.mean(self.duration_before_stranding)
            avg_particles_stranded = len(self.duration_before_stranding)/len(self.file_list)
            norm_avg_particles_stranded = np.round(avg_particles_stranded/self.number_of_particles,3)
            pickle.dump(
                [
                    avg_duration,
                    norm_avg_particles_stranded,
                    self.duration_before_stranding,
                    self.lon_stranding,
                    self.lat_stranding,
                    self.om_stranding,
                ],
                open(f"{self.outdir}/Stranded_particles_data_{self.id}.p", "wb"),
            )

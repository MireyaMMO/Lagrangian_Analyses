import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
import logging
from numpy import random, histogram2d
from scipy.interpolate import interp2d
from Lagrangian_Analyses.utils import *
import pickle
import os

logging.basicConfig(level=logging.INFO)

class get_statistics(object):
    """
    get_statistics
    Computes Lagrangian statistic from a given file or list of files

    Parameters:
    -----------
        file_list: str
            List of OpenDrift files to analyse
        outdir: str
            path of the directory where to save the output
        id: str
            Id to identify the experiment
        PDF: boolean or arr
            If True it calculates the normalised PDF. If true provide the number of bins required [120,90] for the BoP experiments
        CM: boolean
            If True it calculates the normalised Connectivity Matrix. (Average if multiple files are provided).
        CV: boolean
            If True it calculates the Coefficient of Variation of the Connectivity Matrix. This only works when there are multiple connectivity matrices available.
        stranded: boolean
            If True it estimates different values of interest of stranding/beaching of particles.
        time_of_interest: int
            Time at which the user wants to calculate all this parameters.
        time_of_interest_units: str
            Units of the time of interest defined previously. i.e. 'D' for days, 'H' for hours
        patch: str
            If CM is needed the user must provide a file containing the division of regions.
    """
    def __init__(
        self,
        file_list,
        outdir,
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
        self.outdir = outdir
        self.id = id
        self.PDF = PDF
        self.CM = CM
        self.CV = CV
        self.stranded = stranded
        self.patch = patch
        self.time_of_interest = time_of_interest
        self.time_of_interest_units = time_of_interest_units
        self.first_PDF = True
        self.first_CM = True
        self.first_stranded = True
        self.logger = logging

    def set_directories(self):
        """Create output directories."""
        self.logger.info("--- Creating output directory")
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def get_requested_time_step_index(self, time):
        """Obtains the requested time step index looking into the delta time of the data"""
        delta_time = time[1] - time[0]
        delta_time = int(delta_time.dt.total_seconds())
        index_interval = np.timedelta64(
            self.time_of_interest, self.time_of_interest_units
        ) / np.timedelta64(delta_time, "s")
        return int(index_interval)

    def get_connectivity_matrix(self, CM, patches):
        """With a provided file calculates the connectivity matrix for the domain"""
        temporal_arr = make_nan_array(len(patches), 275000)
        matrix = np.zeros((len(self.release_locations), len(patches) + 1))
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
                    self.number_of_particles_per_location = (
                        self.number_of_particles / len(self.release_locations)
                    )
                    matrix[i, j] = self.number_of_particles_per_location - np.nansum(
                        matrix[i, 0:-1]
                    )
        # matrix[np.where(matrix == 0)] = np.nan
        return matrix

    def centers(self, edges):
        return edges[:-1] + np.diff(edges[:2]) / 2

    def get_pdf(self, PDF):
        """Calculates de Probability Density Function in 2D"""
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
        """Obtains the needed data from the OpenDrift file"""
        lon = ds.lon.where(ds.lon < 9.0e35, np.nan)
        lon = lon.where(lon > 0, lon + 360)
        lat = ds.lat.where(ds.lat < 9.0e35, np.nan)
        if isinstance(self.time_of_interest, int):
            requested_index = self.get_requested_time_step_index(ds.time)
        elif self.time_of_interest == "last":
            requested_index = -1
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        number_of_particles = len(ds["trajectory"])
        return lon, lat, requested_index, status, origin_marker, number_of_particles

    def get_lon_lat_and_origin(
        self, lon, lat, requested_index, status, origin_marker, number_of_particles
    ):
        """Obtains the lon lat and origin marker of each trajectory at the desired time"""
        CM = make_nan_array(number_of_particles, 3)
        for count in range(number_of_particles):
            active_particle = status.sel(trajectory=count + 1)
            active_particles = np.where(active_particle == 0)[0]
            if len(active_particles) > requested_index:
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
        self,
        lon,
        lat,
        requested_index,
        status,
        origin_marker,
        number_of_particles,
    ):
        """Obtains data from the stranded particles"""
        for count in range(number_of_particles):
            particles = status.sel(trajectory=count + 1)
            index_of_release = np.where(particles == 0)[0][0]
            stranded_particles = np.where(particles == 1)[0]
            if stranded_particles.size > 0 and stranded_particles < requested_index:
                time_of_stranding = (
                    lon.sel(trajectory=count + 1)
                    .isel(time=stranded_particles)
                    .time.data
                )
                time_of_release = (
                    lon.sel(trajectory=count + 1).isel(time=index_of_release).time.data
                )
                duration_s = (time_of_stranding - time_of_release) * 1e-9 / 3600
                lon_s = lon.sel(trajectory=count + 1).isel(time=stranded_particles).data
                lat_s = lat.sel(trajectory=count + 1).isel(time=stranded_particles).data
                om_s = (
                    origin_marker.sel(trajectory=count + 1)
                    .isel(time=stranded_particles)
                    .data
                )
                if self.first_stranded:
                    self.duration_before_stranding = duration_s
                    self.lon_stranding = lon_s
                    self.lat_stranding = lat_s
                    self.om_stranding = om_s
                    self.first_stranded = False
                else:
                    self.duration_before_stranding = np.hstack(
                        (self.duration_before_stranding, duration_s)
                    )
                    self.lon_stranding = np.hstack((self.lon_stranding, lon_s))
                    self.lat_stranding = np.hstack((self.lat_stranding, lat_s))
                    self.om_stranding = np.hstack((self.om_stranding, om_s))

    def run(self):
        self.set_directories()
        for file in self.file_list:
            print(f"--- Analysing {file}")
            ds = xr.open_dataset(file)
            (
                lon,
                lat,
                requested_index,
                status,
                origin_marker,
                self.number_of_particles,
            ) = self.get_data_from_file(ds)
            self.release_locations = np.unique(origin_marker)[
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
                self.get_stranded_particles(
                    lon,
                    lat,
                    requested_index,
                    status,
                    origin_marker,
                    self.number_of_particles,
                )
        print(f"--- Calculating the average of desired variables")
        if self.PDF:
            print(
                f"--- Saving PDF associated parameters as {self.outdir}PDF_{self.id}.p \n - normalised PDF (n_PDF) \n - bin edges (xedges, yedges)"
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
            print(
                f"--- Saving Connectivity Matrix associated parameters as {self.outdir}/CM_{self.id}.p \n - normalised CM"
            )
            self.avg_CM = np.nanmean(Matrix, axis=0)
            self.norm_CM = np.divide(self.avg_CM.T, np.nansum(self.avg_CM, axis=1).T).T
            pickle.dump(
                [
                    self.norm_CM,
                ],
                open(f"{self.outdir}/CM_{self.id}.p", "wb"),
            )
        if self.CV:
            print(
                f"--- Adding Connectivity Matrix associated parameters to {self.outdir}/CM_{self.id}.p \n - CV"
            )
            std_CM = np.nanstd(Matrix, axis=0)
            std_CM[np.where(std_CM == 0)] = np.nan
            self.CV = std_CM / self.avg_CM
            pickle.dump(
                [
                    self.norm_CM,
                    self.CV,
                ],
                open(f"{self.outdir}/CM_{self.id}.p", "wb"),
            )
        if self.stranded:
            print(
                f"--- Saving stranded particles data associated parameters as {self.outdir}/Stranded_particles_data_{self.id}.p \n - avg_duration \n - norm_avg_particles_stranded \n - duration before stranding \n - stranding coordinates(lon, lat) \n - origin marker"
            )
            avg_duration = np.mean(self.duration_before_stranding)
            avg_particles_stranded = len(self.duration_before_stranding) / len(
                self.file_list
            )
            norm_avg_particles_stranded = np.round(
                avg_particles_stranded / self.number_of_particles, 3
            )
            pickle.dump(
                [
                    float(avg_duration),
                    norm_avg_particles_stranded,
                    self.duration_before_stranding,
                    self.lon_stranding,
                    self.lat_stranding,
                    self.om_stranding,
                ],
                open(f"{self.outdir}/Stranded_particles_data_{self.id}.p", "wb"),
            )

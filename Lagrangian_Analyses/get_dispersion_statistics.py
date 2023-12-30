import os
import xarray as xr
import numpy as np
import pickle
import pandas as pd
from Lagrangian_Analyses.utils import *
import logging


logging.basicConfig(level=logging.INFO)


class get_dispersion_statistics(object):
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
        predominant_direction: str
            Predominant direction to define the positive and negative values relative to the release location. Default: 'horizontal'
        time_of_interest: int
            Time at which the user wants to calculate all this parameters.
        time_of_interest_units: str
            Units of the time of interest defined previously. i.e. 'D' for days, 'H' for hours

    Returns
    -------
        GDK_{id}.p (pickle file)
            Pickle file containing a dictionary with an item per location that includes the x and y values for the Gaussian Dispersal Kernel
    """

    def __init__(
        self,
        file_list,
        outdir,
        id=None,
        RD=True,
        AD=True,
    ):
        self.file_list = file_list
        self.outdir = outdir
        self.id = id
        self.RD = RD
        self.first_RD = True
        self.AD = AD
        self.first_AD = True
        self.logger = logging

    def set_directories(self):
        """Create output directories."""
        self.logger.info("--- Creating output directory")
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def get_delta_time(self, time):
        delta_time = time[1] - time[0]
        self.delta_time = int(delta_time.dt.total_seconds())

    def get_data_from_file(self, ds):
        """Obtains the needed data from the OpenDrift file"""
        lon = ds.lon.where(ds.lon < 9.0e35, np.nan)
        lon = lon.where(lon > 0, lon + 360)
        lat = ds.lat.where(ds.lat < 9.0e35, np.nan)
        self.get_delta_time(ds.time)
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        age = ds.age_seconds.where(~np.isnan(lon), np.nan)
        number_of_particles = len(ds["trajectory"])
        return (
            lon,
            lat,
            status,
            origin_marker,
            age,
            number_of_particles,
        )

    def get_relative_dispersion(
        self,
        lon,
        lat,
        origin_marker,
    ):
        """Obtains the relative dispersion of the dataset"""
        RX = make_nan_array(len(self.release_locations), len(lon.time))
        RY = make_nan_array(len(self.release_locations), len(lon.time))
        RT = make_nan_array(len(self.release_locations), len(lon.time))
        for location in self.release_locations:
            print(
                f"Calculating relative dispersion for location {int(location+1)}/{len(self.release_locations)}"
            )
            lon_disp = lon.where(origin_marker == location, drop=True)
            lat_disp = lat.where(origin_marker == location, drop=True)
            identify_clusters = np.where(np.diff(lon_disp.trajectory) != 1)[0]
            cluster_index = 0
            RX_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            RY_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            RT_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            for cluster_count, cluster in enumerate(identify_clusters):
                print(
                    f"Location {int(location+1)}: Cluster {cluster_count+1}/{len(identify_clusters)}"
                )
                lon_cluster = lon_disp.isel(
                    trajectory=np.arange(cluster_index, cluster + 1, 1)
                ).dropna("time")
                lat_cluster = lat_disp.isel(
                    trajectory=np.arange(cluster_index, cluster + 1, 1)
                ).dropna("time")
                time_cluster = lon_cluster.time
                for time_count, time in enumerate(time_cluster):
                    lon_time_step = lon_cluster.sel(time=time).dropna("trajectory")
                    lat_time_step = lat_cluster.sel(time=time).dropna("trajectory")
                    if len(lon_time_step) > 1:
                        for first_count, (lon_trajectory, lat_trajectory) in enumerate(
                            zip(lon_time_step, lat_time_step), 1
                        ):
                            next_lats = lat_time_step.isel(
                                trajectory=np.arange(first_count, len(lat_time_step), 1)
                            )
                            next_lons = lon_time_step.isel(
                                trajectory=np.arange(first_count, len(lon_time_step), 1)
                            )
                            for second_count, (lon_, lat_) in enumerate(
                                zip(next_lons, next_lats)
                            ):
                                x = haversine(
                                    lon_trajectory, lat_trajectory, lon_, lat_trajectory
                                )
                                y = haversine(
                                    lon_trajectory, lat_trajectory, lon_trajectory, lat_
                                )
                                if time == time_cluster[0]:
                                    xo = np.copy(x)
                                    yo = np.copy(y)
                                    if second_count == 0:
                                        Xo = np.copy(xo)
                                        Yo = np.copy(yo)
                                    else:
                                        Xo = np.hstack((Xo, xo))
                                        Yo = np.hstack((Yo, yo))
                                if second_count == 0:
                                    X = np.copy(x)
                                    Y = np.copy(y)
                                else:
                                    X = np.hstack((X, x))
                                    Y = np.hstack((Y, y))
                            if first_count == 1:
                                Rx = X + Xo
                                Ry = Y + Yo
                            else:
                                Rx = np.hstack((Rx, X + Xo))
                                Ry = np.hstack((Ry, Y + Yo))

                    if time_count == 0:
                        RT_per_cluster = np.nanmean(Rx * Ry)
                        RX_per_cluster = np.nanmean(Rx**2)
                        RY_per_cluster = np.nanmean(Ry**2)
                    else:
                        RT_per_cluster = np.hstack(
                            (RT_per_cluster, np.nanmean(Rx * Ry))
                        )
                        RX_per_cluster = np.hstack(
                            (RX_per_cluster, np.nanmean(Rx**2))
                        )
                        RY_per_cluster = np.hstack(
                            (RY_per_cluster, np.nanmean(Ry**2))
                        )
                RT_per_location[cluster_count, 0 : time_count + 1] = RT_per_cluster
                RX_per_location[cluster_count, 0 : time_count + 1] = RX_per_cluster
                RY_per_location[cluster_count, 0 : time_count + 1] = RY_per_cluster
                cluster_index = cluster + 1
            rt = np.nanmean(RT_per_location, axis=0)
            rx = np.nanmean(RX_per_location, axis=0)
            ry = np.nanmean(RY_per_location, axis=0)
            RT[int(location), 0 : len(rt)] = rt
            RX[int(location), 0 : len(rx)] = rx
            RY[int(location), 0 : len(ry)] = ry
        return RX, RY, RT

    def get_absolute_dispersion(
        self,
        lon,
        lat,
        origin_marker,
    ):
        AX = make_nan_array(len(self.release_locations), len(lon.time))
        AY = make_nan_array(len(self.release_locations), len(lon.time))
        AT = make_nan_array(len(self.release_locations), len(lon.time))
        for location_count, location in enumerate(self.release_locations):
            print(
                f"Calculating absolute dispersion for location {int(location+1)}/{len(self.release_locations)}"
            )
            lon_disp = lon.where(origin_marker == location, drop=True)
            lat_disp = lat.where(origin_marker == location, drop=True)
            DX_per_location = make_nan_array(len(lon_disp), len(lon_disp.time))
            DY_per_location = make_nan_array(len(lon_disp), len(lon_disp.time))
            number_of_particles = len(lon_disp)
            for count in range(number_of_particles):
                lat_ = lat_disp.isel(trajectory=count).dropna("time")
                lon_ = lon_disp.isel(trajectory=count).dropna("time")
                times = lon_.time
                for time_count, time in enumerate(times):
                    if time_count == 0:
                        lon0 = lon_.isel(time=0)
                        lat0 = lat_.isel(time=0)
                        X = 0
                        Y = 0
                    else:
                        lon_trajectory = lon_.isel(time=time_count)
                        lat_trajectory = lat_.isel(time=time_count)
                        x = haversine(lon0, lat0, lon_trajectory, lat0)
                        y = haversine(lat0, lat0, lon0, lat_trajectory)
                        X = np.hstack((X, x))
                        Y = np.hstack((Y, y))
                DX_per_location[count, 0 : time_count + 1] = X
                DY_per_location[count, 0 : time_count + 1] = Y
            AX[location_count, :] = np.nanmean(DX_per_location**2, axis=0)
            AY[location_count, :] = np.nanmean(DY_per_location**2, axis=0)
            AT[location_count, :] = AX[location_count, :] + AY[location_count, :]
        return AX, AY, AT

    def run(self):
        self.set_directories()
        print(f"--- Generating output directory")
        for file in self.file_list:
            print(f"--- Analysing {file}")
            ds = xr.open_dataset(file)
            (
                lon,
                lat,
                _,
                origin_marker,
                _,
                self.number_of_particles,
            ) = self.get_data_from_file(ds)
            self.release_locations = np.unique(origin_marker)[
                ~np.isnan(np.unique(origin_marker))
            ]
            if self.AD:
                AX, AY, AT = self.get_absolute_dispersion(
                    lon,
                    lat,
                    origin_marker,
                )
                if self.first_AD:
                    self.AT = np.expand_dims(AT, 0)
                    self.AX = np.expand_dims(AX, 0)
                    self.AY = np.expand_dims(AY, 0)
                    self.first_AD = False
                else:
                    self.AT = np.vstack((self.AT, np.expand_dims(AT, 0)))
                    self.AX = np.vstack((self.AX, np.expand_dims(AX, 0)))
                    self.AY = np.vstack((self.AY, np.expand_dims(AY, 0)))

            if self.RD:
                RX, RY, RT = self.get_relative_dispersion(
                    lon,
                    lat,
                    origin_marker,
                )
                if self.first_RD:
                    self.RT = np.expand_dims(RT, 0)
                    self.RX = np.expand_dims(RX, 0)
                    self.RY = np.expand_dims(RY, 0)
                    self.first_RD = False
                else:
                    self.RT = np.vstack((self.RT, np.expand_dims(RT, 0)))
                    self.RX = np.vstack((self.RX, np.expand_dims(RX, 0)))
                    self.RY = np.vstack((self.RY, np.expand_dims(RY, 0)))
            if self.AD:
                print(f"--- Saving Absolute Dispersion to {self.outdir}/AD_{self.id}.p")
                pickle.dump(
                    np.nanmean(self.AT,0),
                    np.nanmean(self.AX,0),
                    np.nanmean(self.AY,0),
                    open(f"{self.outdir}/AD_{self.id}.p", "wb"),
                )

            if self.RD:
                print(f"--- Saving Relative Dispersion to {self.outdir}/RD_{self.id}.p")
                pickle.dump(
                    np.nanmean(self.RT,0),
                    np.nanmean(self.RX,0),
                    np.nanmean(self.RY,0),
                    open(f"{self.outdir}/RD_{self.id}.p", "wb"),
                )
            print("--- Done")

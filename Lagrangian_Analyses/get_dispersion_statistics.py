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
        AD: boolean
            If True calculates Absolute dispersion. Default True
        RD: boolean
            If True calculates relative dispersion. Default True

    Returns
    -------
        AD_{id}.p (pickle file)
            Pickle file containing a dictionary with an item per location that includes the time step, total absolute dispersion, absolute dispersion in x and y components and the number of particles that were available for the calculation
        RD_{id}.p (pickle file)
            Pickle file containing a dictionary with an item per location that includes the time step, total relative dispersion, relative dispersion in x and y components and the number of pair of partickes that were available for the calculation
                        
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
        lonmin = np.min(lon)
        latmin = np.min(lat)
        self.get_delta_time(ds.time)
        self.time_step = np.hstack((0, ds['time'].diff('time').dt.total_seconds().cumsum().data))
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        x,y = sph2xy(lonmin, lon, latmin, lat)
        age = ds.age_seconds.where(~np.isnan(lon), np.nan)
        number_of_particles = len(ds["trajectory"])
        return (
            lon,
            lat,
            np.abs(x)*1e-3,
            np.abs(y)*1e-3,
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
        PP = make_nan_array(len(self.release_locations), len(lon.time))
        for location in self.release_locations:
            print(
                f"--- Calculating relative dispersion for release location {int(location+1)}/{len(self.release_locations)}"
            )
            lon_disp = lon.where(origin_marker == location, drop=True)
            lat_disp = lat.where(origin_marker == location, drop=True)
            identify_clusters = np.where(np.diff(lon_disp.trajectory) != 1)[0]
            cluster_index = 0
            RX_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            RY_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            RT_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            PP_per_location = make_nan_array(len(identify_clusters), len(lon_disp.time))
            for cluster_count, cluster in enumerate(identify_clusters):
                print(
                    f"--- Location {int(location+1)}: Cluster {cluster_count+1}/{len(identify_clusters)}"
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
                    pair_of_particles = 1
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
                                #x = haversine(
                                #    lon_trajectory, lat_trajectory, lon_, lat_trajectory
                                #)
                                #y = haversine(
                                #    lon_trajectory, lat_trajectory, lon_trajectory, lat_
                                #)
                                x = np.abs(lon_trajectory - lon_)
                                y = np.abs(lat_trajectory - lat_)
                                pair_of_particles+=1
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
                        pp = pair_of_particles
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
                        pp = np.hstack((pp, pair_of_particles))
                
                RT_per_location[cluster_count, 0 : time_count + 1] = RT_per_cluster
                RX_per_location[cluster_count, 0 : time_count + 1] = RX_per_cluster
                RY_per_location[cluster_count, 0 : time_count + 1] = RY_per_cluster
                PP_per_location[cluster_count, 0 : time_count + 1] = pp
                cluster_index = cluster + 1
            rt = np.nanmean(RT_per_location, axis=0)
            rx = np.nanmean(RX_per_location, axis=0)
            ry = np.nanmean(RY_per_location, axis=0)
            pair_particles = np.nansum(PP_per_location, axis=0)
            RT[int(location), 0 : len(rt)] = rt
            RX[int(location), 0 : len(rx)] = rx
            RY[int(location), 0 : len(ry)] = ry
            PP[int(location), 0 : len(pair_particles)] = pair_particles
            
        return RX, RY, RT, PP

    def get_absolute_dispersion(
        self,
        lon,
        lat,
        origin_marker,
    ):
        """Obtains the absolute dispersion of the dataset"""
        AX = make_nan_array(len(self.release_locations), len(lon.time))
        AY = make_nan_array(len(self.release_locations), len(lon.time))
        AT = make_nan_array(len(self.release_locations), len(lon.time))
        NP = make_nan_array(len(self.release_locations), len(lon.time))
        for location_count, location in enumerate(self.release_locations):
            print(
                f"--- Calculating absolute dispersion for release location {int(location+1)}/{len(self.release_locations)}"
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
                        X = xr.DataArray(data=[0], dims='time', coords={'time':time_count})
                        Y = xr.DataArray(data=[0], dims='time', coords={'time':time_count})
                    else:
                        lon_trajectory = lon_.isel(time=time_count)
                        lat_trajectory = lat_.isel(time=time_count)
                        #x = haversine(lon0, lat0, lon_trajectory, lat0)
                        #y = haversine(lon0, lat0, lon0, lat_trajectory)
                        x = (lon_trajectory-lon0).assign_coords({'time':time_count}).drop('trajectory')
                        y = (lat_trajectory-lat0).assign_coords({'time':time_count}).drop('trajectory')
                        X = np.hstack((X, x))
                        Y = np.hstack((Y, y))
                DX_per_location[count, 0 : time_count + 1] = X
                DY_per_location[count, 0 : time_count + 1] = Y
            AX[location_count, :] = np.nanmean(DX_per_location**2, axis=0)
            AY[location_count, :] = np.nanmean(DY_per_location**2, axis=0)
            AT[location_count, :] = AX[location_count,:] + AY[location_count,:]
            NP[location_count, :] = self.get_number_of_particles_analysed(DX_per_location)
        
        return AX, AY, AT, NP

    def get_number_of_particles_analysed(self, var):
        total_time = var.shape[1]
        valid_particles = make_nan_array(total_time)
        for count in range(total_time):
            find_valid_particles = np.where(~np.isnan(var[:,count]))
            valid_particles[count] = len(find_valid_particles[0])
        return valid_particles
        
    def make_dict(self, vars, vars_names):
        """Makes dictionary of provided variables by location"""
        dict_disp={}
        for count, location in enumerate(self.release_locations):
            temp_dict= dict()
            for var, name in zip(vars,vars_names):
                if 'time' in name:
                    temp_dict[name] = var
                else:
                    temp_dict[name] = var[count,:]
            dict_disp[f'loc_{"%02d"%location}']=temp_dict
        return dict_disp

    def run(self):
        self.set_directories()
        print(f"--- Generating output directory")
        for count, file in enumerate(self.file_list,1):
            print(f"--- Analysing {file} file {count}/{len(self.file_list)}")
            ds = xr.open_dataset(file)
            (
                lon,
                lat,
                x,
                y,
                _,
                origin_marker,
                _,
                self.number_of_particles,
            ) = self.get_data_from_file(ds)
            self.release_locations = np.unique(origin_marker)[
                ~np.isnan(np.unique(origin_marker))
            ]
            if self.AD:
                AX, AY, AT, NP = self.get_absolute_dispersion(
                    x,
                    y,
                    origin_marker,
                )
                if self.first_AD:
                    self.AT = np.expand_dims(AT, 0)
                    self.AX = np.expand_dims(AX, 0)
                    self.AY = np.expand_dims(AY, 0)
                    self.NP = np.expand_dims(NP, 0)
                    self.first_AD = False
                else:
                    self.AT = np.vstack((self.AT, np.expand_dims(AT, 0)))
                    self.AX = np.vstack((self.AX, np.expand_dims(AX, 0)))
                    self.AY = np.vstack((self.AY, np.expand_dims(AY, 0)))
                    self.NP = np.vstack((self.NP, np.expand_dims(NP, 0)))

            if self.RD:
                RX, RY, RT, PP = self.get_relative_dispersion(
                    x,
                    y,
                    origin_marker,
                )
                if self.first_RD:
                    self.RT = np.expand_dims(RT, 0)
                    self.RX = np.expand_dims(RX, 0)
                    self.RY = np.expand_dims(RY, 0)
                    self.PP = np.expand_dims(PP, 0)
                    self.first_RD = False
                else:
                    self.RT = np.vstack((self.RT, np.expand_dims(RT, 0)))
                    self.RX = np.vstack((self.RX, np.expand_dims(RX, 0)))
                    self.RY = np.vstack((self.RY, np.expand_dims(RY, 0)))
                    self.PP = np.vstack((self.PP, np.expand_dims(PP, 0)))
        if self.AD:
            print(f"--- Saving Absolute Dispersion to {self.outdir}/AD_{self.id}.p")
            AT_avg = np.nanmean(self.AT,0)
            AX_avg = np.nanmean(self.AX,0)
            AY_avg = np.nanmean(self.AY,0)
            NP_accum = np.nansum(self.NP,0)
            AD_dict = self.make_dict([self.time_step, AT_avg, AX_avg, AY_avg, NP_accum], ['time_step','AD_t','AD_x','AD_y','Number_of_Particles'])
            pickle.dump(
                AD_dict,
                open(f"{self.outdir}/AD_{self.id}.p", "wb"),
            )

        if self.RD:
            RT_avg = np.nanmean(self.RT,0)
            RX_avg = np.nanmean(self.RX,0)
            RY_avg = np.nanmean(self.RY,0)
            PP_accum = np.nansum(self.PP,0)                
            print(f"--- Saving Relative Dispersion to {self.outdir}/RD_{self.id}.p")
            RD_dict = self.make_dict([self.time_step, RT_avg, RX_avg, RY_avg, PP_accum], ['time_step','RD_t','RD_x','RD_y','Number_of_Particle_Pairs'])
            pickle.dump(
                RD_dict,
                open(f"{self.outdir}/RD_{self.id}.p", "wb"),
            )
        print("--- Done")

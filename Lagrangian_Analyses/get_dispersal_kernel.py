import os
import xarray as xr
import numpy as np
import pickle
from Lagrangian_Analyses.utils import *
import logging

logging.basicConfig(level=logging.INFO)


class get_dispersal_kernel(object):
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
        predominant_direction="horizontal",
        xmin=None,
        xmax=None,
        bins=50,
        time_of_interest=None,
        time_of_interest_units=None,
    ):
        self.file_list = file_list
        self.outdir = outdir
        self.id = id
        self.predominant_direction = predominant_direction
        self.xmin = xmin
        self.xmax = xmax 
        self.bins = bins
        self.time_of_interest = time_of_interest
        self.time_of_interest_units = time_of_interest_units
        self.first_DK = True
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

    def get_data_from_file(self, ds):
        """Obtains the needed data from the OpenDrift file"""
        lon = ds.lon.where(ds.lon < 9.0e35, np.nan)
        lon = lon.where(lon > 0, lon + 360)
        lat = ds.lat.where(ds.lat < 9.0e35, np.nan)
        if isinstance(self.time_of_interest, int):
            requested_index = self.get_requested_time_step_index(ds.time)
        elif self.time_of_interest == None:
            requested_index = None
        status = ds.status.where(~np.isnan(lon), np.nan)
        origin_marker = ds.origin_marker.where(~np.isnan(lon), np.nan)
        number_of_particles = len(ds["trajectory"])
        return lon, lat, requested_index, status, origin_marker, number_of_particles

    def get_settlement_distance(
        self,
        lon,
        lat,
        requested_index,
        status,
        origin_marker,
        number_of_particles,
    ):
        for count in range(number_of_particles):
            particles = status.sel(trajectory=count + 1)
            index_of_release = np.where(particles == 0)[0][0]
            stranded_particles = np.where(particles == 1)[0]
            if stranded_particles.size > 0:  # and stranded_particles < requested_index:
                if not requested_index:
                    lon_ini = (
                        lon.sel(trajectory=count + 1).isel(time=index_of_release).data
                    )
                    lat_ini = (
                        lat.sel(trajectory=count + 1).isel(time=index_of_release).data
                    )
                    lon_s = (
                        lon.sel(trajectory=count + 1).isel(time=stranded_particles).data
                    )
                    lat_s = (
                        lat.sel(trajectory=count + 1).isel(time=stranded_particles).data
                    )
                    distance = haversine(lon_ini, lat_ini, lon_s, lat_s)
                else:
                    if stranded_particles < requested_index:
                        lon_ini = (
                            lon.sel(trajectory=count + 1)
                            .isel(time=index_of_release)
                            .data
                        )
                        lat_ini = (
                            lat.sel(trajectory=count + 1)
                            .isel(time=index_of_release)
                            .data
                        )
                        lon_s = (
                            lon.sel(trajectory=count + 1)
                            .isel(time=stranded_particles)
                            .data
                        )
                        lat_s = (
                            lat.sel(trajectory=count + 1)
                            .isel(time=stranded_particles)
                            .data
                        )
                        distance = haversine(lon_ini, lat_ini, lon_s, lat_s)
                if self.predominant_direction == "horizontal":
                    if lon_s < lon_ini:
                        distance *= -1
                elif self.predominant_direction == "vertical":
                    if lat_s < lat_ini:
                        distance *= -1
                om_s = (
                    origin_marker.sel(trajectory=count + 1)
                    .isel(time=stranded_particles)
                    .data
                )
                if self.first_DK:
                    self.distance = distance
                    self.origin_marker = om_s
                    self.first_DK = False
                else:
                    self.distance = np.hstack((self.distance, distance))
                    self.origin_marker = np.hstack((self.origin_marker, om_s))

    def calculate_gaussian_kernel(self):
        """Returns a 2D Gaussian kernel."""
        dict_kde = {}
        for location in self.release_locations:
            location_index = np.where(self.origin_marker == location)
            x = self.distance[location_index]
            if not self.xmin:
                self.xmin = np.min()
                self.xmax = np.max()
            xi = np.linspace(self.xmin, self.xmax, self.bins)
            std_dev = np.std(x)
            mean = np.mean(x)
            y = (
                1
                / (np.sqrt(2 * np.pi) * std_dev)
                * np.exp(-np.power((xi - mean) / std_dev, 2) / 2)
            )
            n_y = y / np.sum(y)
            dict_kde[f'loc_{"%02d"%location}'] = {'x':xi, 'y':n_y, 'mean':mean, 'std_dev':std_dev}
        return dict_kde

    def run(self):
        self.set_directories()
        print(f"--- Generating output directory")
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
            ]
            self.get_settlement_distance(
                lon,
                lat,
                requested_index,
                status,
                origin_marker,
                self.number_of_particles,
            )
        print(f"--- Calculating Gaussian Dispersal Kernel")
        self.dict_kde = self.calculate_gaussian_kernel()
        print(f"--- Saving to {self.outdir}/KDE_{self.id}.p")
        pickle.dump(
            self.dict_kde,
            open(f"{self.outdir}/GDK_{self.id}.p", "wb"),
        )
        print("--- Done")

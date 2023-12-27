#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

logging.basicConfig(level=logging.INFO)

class opendrift_run(object):
    """
    opendrift_run
    Runs a particle release experiment with the chosen parameters

    Environment Parameters
    ----------------------
        file_path: str
            path of the file where the velocities are located
        outdir: str
            path of the directory where to save the output
        id: str
            Id to identify the experiment
        vars_dict: dict
            Dictionary containing the names associated to the depth and time variables within the file. Default ROMS convention: {"depth": "h", "time": "ocean_time"}

    OpenDrift Parameters
    --------------------
        log_level: int
            Level of log wanted for OpenDrift processes, 0 to 50 range.
        opendrift_reader: str
            Reader used by opendrift. This depends on the hydrodynamic model being used. For this study the default is 'reader_ROMS_native_MOANA'
        opendrift_model: str
            Module used by opendrift. For the LCS we use OceanDrift (passive particles) however other modules are available (see OpenDrift for more information)
            Default 'OceanDrift'
        number_of_particles: int
            Number of particles to deploy per location
        number_of_release_locations: int
            Number of release locations
        release_locations: tuple
            If True, it must contain the coordinates of the release locations. Either this or release_on_isobath should be provided
        release_on_isobath: int
            If True, the script is going to find locations along the given isobath. Either this or release_locations should be provided
        spacing_locations: int
            If release on isobath is true, define the spacing between locations, this depends and type of the grid. Choose wisely.
        ignore_first: int
            Ignores the first (defined number) of locations given, useful when using release on isobath. Default 0
        ignore_last: int
            Ignores the last (defined number) of locations given, useful when using release on isobath. Default -1
        release_interval: int
            Particle release interval in hours. Default 3
        release_until: int
            Define how many days until particles are stop being released in seconds. Default 10 days 
        advection_duration: int
            Advection duration in days. Default 50
        cluster_std: int
            if True particles are going to be deployed in a cluster and a deviation of the cluster should be provided. Depends on the model grid. Used in BoP experiment 0.003 (~3 km)
        random_depth: boolean or int,
            if True is going to release the particles in each location at random depths. If a number is given this will be the limit i.e., if 20 is given the particles are going to be released between the surface and 20 meters
        max_speed: int
            Maximum speed that can be reached by the particles in m s^{-1}. Default 5.
        horizontal_diffusivity: int
            horizontal_diffusivity in m^{2} s^{-1}. Default 0.1
        advection_scheme: str
            Advection scheme for the particle releases. Default "runge-kutta4" (see OpenDrift for more options)
        coastline_action: str
            What happens to the particle when it reaches the coast. Default 'stranding'
        time_step_advection: int
            Advection time step in seconds.
        time_step_output: int
            Output time step in seconds

    Vertical Motion Parameters
    --------------------------
        vertical_motion: boolean
            If True all the vertical parameters below should be provided
        vertical_diffusivity: int
        vertical_mixing_timestep: int Default: 90 s

    Behaviour Parameters
    --------------------
        behaviour: boolean
            If True, parameters such as maximum age, minimum settlement age and maximum depth a particle can reach can be defined
        max_age_seconds: int
            In seconds Default: 30*24*3600, make sure is less or the same as advection_duration
        min_settlemente_age: int
            Minimum settlement age in seconds. Default 0
        maximum_depth: int
            Maximum depth a particle can reach. Default False

    Habitat Parameters
    ------------------
        habitat: boolean or str
            If True, a path of the shapefile with the habitats must be provided

    Other Parameters
    ----------------
        first_and_last_position: boolean
            If True a text file containing the first and last position of the particles is given.

    Returns
    -------
    Output files:
        %yyyy%mm_Particles_%isobathm.nc: netCDF
            File containing: lat, lon, z, trajectory, status, age_seconds and origin_marker

        %yyyy%mm_Particles_%isobathm.nc: textfile (Optional)
            File containing the initial and final coordinates of the released particles.
    """

    def __init__(
        self,
        file_path,
        outdir,
        id=None,
        vars_dict={'time':'ocean_time', 'depth':'h'},
        log_level = 50,
        opendrift_reader = "reader_ROMS_native",
        opendrift_model = "OceanDrift",
        number_of_particles = 20,
        cluster_std = 0.003,
        random_depth = 20,
        depth = 0,
        release_on_isobath = 200,
        spacing_locations = 10,
        ignore_first = 0,
        ignore_last = -1,
        release_locations=False,
        release_interval=3,
        release_until = 10*24*3600,
        advection_duration=5*24*3600,
        max_speed=5,
        advection_scheme="runge-kutta4",
        horizontal_diffusivity=0.1,
        time_step_advection=900,
        time_step_output=3600 * 3,
        coastline_action="stranding",
        vertical_motion=True,
        vertical_diffusivity=0.001,
        vertical_mixing_timestep=90,
        behaviour=False,
        max_age_seconds=None,
        min_settlemente_age=0,
        maximum_depth=False,
        habitat=False,
        first_and_last_position=False,
    ):
        # self.month = month
        # self.m = "%02d" % self.month
        # self.year = str(year)
        self.file_path = file_path
        self.outdir = outdir
        self.id = id
        self.vars_dict = vars_dict

        self.opendrift_reader = opendrift_reader
        self.opendrift_model = opendrift_model
        self.log_level = log_level

        self.number_of_particles = number_of_particles
        # self.number_of_release_locations = number_of_release_locations
        self.release_interval = release_interval
        self.release_until = release_until
        self.advection_duration = advection_duration
        self.release_on_isobath = release_on_isobath
        self.release_locations = release_locations
        self.spacing_locations = spacing_locations
        self.ignore_first = ignore_first
        self.ignore_last = -ignore_last
        self.cluster_std = cluster_std

        self.random_depth = random_depth
        self.depth = depth
        self.max_speed = max_speed
        self.advection_scheme = advection_scheme
        self.horizontal_diffusivity = horizontal_diffusivity
        self.vertical_mixing = vertical_motion
        self.vertical_advection = vertical_motion
        self.vertical_diffusivity = vertical_diffusivity
        self.vertical_mixing_timestep = vertical_mixing_timestep
        self.coastline_action = coastline_action

        self.behaviour = behaviour
        self.min_settlement_age = min_settlemente_age
        self.maximum_depth = maximum_depth
        self.max_age_seconds = max_age_seconds

        self.habitat = habitat

        self.time_step_advection = time_step_advection
        self.time_step_output = time_step_output

        self.first_and_last_position = first_and_last_position
        self.logger = logging

    def set_directories(self):
        """Create output directories."""
        self.logger.info("--- Creating output directory")
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def set_opendrift_configuration(self):
        # self.logger.info("--- Setting OpenDrift Configuration")
        exec(f"from opendrift.readers import {self.opendrift_reader}")
        exec(
            f"from opendrift.models.{self.opendrift_model.lower()} import {self.opendrift_model}"
        )
        o = eval(self.opendrift_model)(loglevel=self.log_level)
        reader = eval(self.opendrift_reader).Reader(self.file_path)
        # dynamical landmask if true
        o.set_config("general:use_auto_landmask", False)
        o.max_speed = self.max_speed
        o.add_reader(reader)
        # keep only particles from the "frame" that are on the ocean
        o.set_config("seed:ocean_only", True)
        ###############################
        # PHYSICS of Opendrift
        ###############################
        o.set_config("environment:fallback:x_wind", 0.0)
        o.set_config("environment:fallback:y_wind", 0.0)
        o.set_config("environment:fallback:x_sea_water_velocity", 0.0)
        o.set_config("environment:fallback:y_sea_water_velocity", 0.0)
        o.set_config("environment:fallback:sea_floor_depth_below_sea_level", 10000.0)

        # drift
        o.set_config("environment:fallback:land_binary_mask", 0)
        o.set_config("drift:advection_scheme", self.advection_scheme)
        # note current_uncertainty can be used to replicate an horizontal diffusion s
        o.set_config("drift:current_uncertainty", 0.0)
        Kxy = self.horizontal_diffusivity  # m2/s-1
        o.set_config(
            "drift:horizontal_diffusivity", Kxy
        )  # using new config rather than current uncertainty
        o.set_config("general:coastline_action", self.coastline_action)
        return o, reader

    def set_opendrift_vertical_motion_configuration(self):
        self.o.set_config("drift:vertical_mixing", self.vertical_mixing)
        self.o.set_config("drift:vertical_advection", self.vertical_advection)
        Kz = self.vertical_diffusivity  # m2/s-1
        self.o.set_config(
            "environment:fallback:ocean_vertical_diffusivity", Kz
        )  # specify constant ocean_vertical_diffusivity in m2.s-1
        self.o.set_config(
            "vertical_mixing:diffusivitymodel", "constant"
        )  # constant or environment
        self.o.set_config(
            "vertical_mixing:timestep", self.vertical_mixing_timestep
        )  # if some ocean_vertical_diffusivity!=0, turbulentmixing:timestep should be << 900 seconds
        # else:
        #    o.disable_vertical_motion()

    def set_opendrift_behaviour_configuration(self):
        self.o.set_config("drift:max_age_seconds", self.max_age_seconds)
        self.o.set_config("drift:min_settlement_age_seconds", self.min)
        self.o.set_config("drift:maximum_depth", -50)

    def set_opendrift_habitat_settlement(self):
        self.o.habitat(self.habitat)
        self.o.set_config("drift:settlement_in_habitat", True)

    def set_runtime(self, ds):
        start_time = pd.to_datetime(ds["ocean_time"][0].values)
        end_time = start_time + timedelta(seconds=self.release_until)
        self.run_until = start_time + timedelta(seconds=self.advection_duration)
        return [start_time, end_time]

    def create_seed_times(self,start, end, delta):
        """
        create times at given interval to seed particles
        """
        out = []
        start_t = start
        end_t = datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S")
        while start_t < end:
            out.append(start_t)
            start_t += delta
        return out

    def get_release_locations(self, ds, reader):
        h = ds[self.depth_var].values
        if len(h.shape)>2:
            h = h[0,:,:]
        lon = reader.lon
        lat = reader.lat
        latf = lon.flatten()
        lonf = lat.flatten()
        if self.release_on_isobath:
            self.isobath_levels = [self.release_on_isobath]
            print(f"--- Finding release locations over the {self.release_on_isobath}m isobath")
            cs = plt.contour(lon, lat, h, levels=self.isobath_levels)
            plt.close()
            p = cs.allsegs[0][:]
            plon = p[0][:, 0]
            locations_lon = plon[:: self.spacing_locations]
            locations_lon = locations_lon[self.ignore_first : self.ignore_last]
            plat = p[0][:, 1]
            locations_lat = plat[:: self.spacing_locations]
            locations_lat = locations_lat[self.ignore_first : self.ignore_last]
            print(f"--- {len(locations_lat)} release locations identified over the {self.release_on_isobath}m isobath")
        elif self.release_locations:
            locations_lon = self.release_locations[:, 0]
            locations_lat = self.release_locations[:, 1]
            print(f"--- {len(locations_lat)} release locations provided")
        self.number_of_release_locations = len(locations_lat)
        self.total_number_particles = (
            self.number_of_particles * self.number_of_release_locations
        )  # total number of particles released
        release_lon = np.zeros(self.total_number_particles)
        release_lat = np.zeros(self.total_number_particles)

        centers = np.array([locations_lon, locations_lat]).T
        cluster_std = np.tile(self.cluster_std, self.number_of_release_locations)
        X, Y = make_blobs(
            n_samples=self.total_number_particles,
            cluster_std=cluster_std,
            centers=centers,
            n_features=self.number_of_release_locations,
            random_state=1,
        )
        for i in range(self.number_of_release_locations):
            release_lon[i :: self.number_of_release_locations] = X[Y == i, 0]
            release_lat[i :: self.number_of_release_locations] = X[Y == i, 1]
        return release_lon, release_lat

    def get_first_and_last_position(self, o):
        lons_start = o.elements_scheduled.lon
        lats_start = o.elements_scheduled.lat
        name_con_file = os.path.join(
            self.outdir, f"{self.year}{self.month}_Particles_{self.id}.txt"
        )
        _, index_of_last = o.index_of_activation_and_deactivation()
        lons = o.get_property("lon")[0]
        lats = o.get_property("lat")[0]
        status = o.get_property("status")[0]
        lons_end = lons[index_of_last, range(lons.shape[1])]
        lats_end = lats[index_of_last, range(lons.shape[1])]
        status_end = status[index_of_last, range(lons.shape[1])]
        outFile = open(name_con_file, "w")
        for lon_start, lat_start, lon_end, lat_end, status in zip(
            lons_start, lats_start, lons_end, lats_end, status_end
        ):
            outFile.write(f"{lon_start},{lat_start},{lon_end},{lat_end},status\n")
        outFile.close()
        
    def run(self):
        self.set_directories()  # creates directory for output
        if len(self.file_path)>1: 
            ds = xr.open_mfdataset(self.file_path)
        else:
            ds = xr.open_dataset(self.file_path)
        self.time_var = self.vars_dict["time"]
        self.depth_var = self.vars_dict["depth"]
        ini, end =ds[self.time_var].isel({self.time_var:[0,-1]})
        file_period = (end-ini).dt.total_seconds()
        try:
            file_period>self.advection_duration
        except:
            print('Files provided do not cover the full advection period')
        self.month = ini.dt.month.data
        self.month = "%02d" % self.month
        self.year = ini.dt.year.data
        self.o, reader = self.set_opendrift_configuration()
        if self.vertical_mixing:
            self.set_opendrift_vertical_motion_configuration()
        if self.habitat:
            self.set_opendrift_habitat_settlement()
        if self.behaviour:
           self.set_opendrift_behaviour_configuration()
        print('--- OpenDrift Configuration set')

        runtime = self.set_runtime(ds)
        file_name = os.path.join(
            self.outdir, f"{self.year}{self.month}_Particles_{self.id}.nc"
        )
        times = self.create_seed_times(
            runtime[0], runtime[1], timedelta(hours=self.release_interval)
        )
        print(f'--- Particles seeded from {datetime.strftime(runtime[0], " %H:%M %m/%d/%Y")} to {datetime.strftime(runtime[1], " %H:%M %m/%d/%Y")} every {self.release_interval} hours')
        self.release_lon, self.release_lat = self.get_release_locations(ds, reader)
        print(f"--- Number of release locations: {self.number_of_release_locations}")
        for ii, time in enumerate(times):
            if self.random_depth:
                if isinstance(self.random_depth, bool):
                    z = np.random.uniform(
                        -self.release_on_isobath + 3,
                        0,
                        size=self.total_number_particles,
                    )
                    if ii==0:
                        print(f"--- Particles released at random depths throughout the water column")
                else:
                    random_depth = np.abs(self.random_depth)
                    z = np.random.uniform(
                        -random_depth, 0, size=self.total_number_particles
                    )
                    if ii==0:
                        print(f"--- Particles released at random depths throughout from {random_depth} to the surface")
            else:  # constant depth
                z = np.ones(self.total_number_particles) * self.depth
                if ii==0:
                    print(f"--- Particles released at a constant depth of {z} meters")
            for n in range(self.number_of_release_locations):
                self.o.seed_elements(
                    self.release_lon[n :: self.number_of_release_locations],
                    self.release_lat[n :: self.number_of_release_locations],
                    number=self.number_of_particles,
                    z=z[n :: self.number_of_release_locations],
                    time=time,
                    origin_marker=n,
                )  # , terminal_velocity=-0.001)
        self.o.plot()
        print('--- Particles seeded starting OpenDrift run')
        self.o.run(
            time_step=self.time_step_advection,
            end_time=self.run_until,
            outfile=file_name,
            time_step_output=self.time_step_output,
            export_variables=[
                "lat",
                "lon",
                "z",
                "trajectory",
                "status",
                "age_seconds",
                "origin_marker",
            ],
        )
        if self.first_and_last_position:
            self.get_first_and_last_position(o)
            print('--- Saving first and last position file')


#      o.animation(color='z',filename='/nesi/project/uoo02643/BoP_1km/Opendrift/2003'+m+'_Particles_'+str(iso[j])+'.mp4', corners=[175.7, 180, -38, -35.3], show_elements=True)

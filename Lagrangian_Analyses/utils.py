import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def sph2xy(lambda0, lambda1, theta0, theta1):
    ############# SPH2XY Spherical to curvilinear spherical. ############
    ##### where X,Y are in meters and LAMBDA0,THETA0 are in degrees#####
    R = 6371 * 1e3
    deg2rad = np.pi / 180
    x = R * (lambda0 - lambda1) * deg2rad * np.cos(theta1 * deg2rad)
    y = R * (theta0 - theta1) * deg2rad
    return x, y


def xy2sph(x, lambda1, y, theta1):
    ############# XY2SPH Curvilinear spherical to spherical. ############
    ##### where X,Y are in meters and LAMBDA1,THETA1 are in degrees#####
    R = 6371 * 1e3
    deg2rad = np.pi / 180
    lambda0 = lambda1 + x / (R * np.cos(theta1 * deg2rad)) / deg2rad
    theta0 = theta1 + y / R / deg2rad
    return lambda0, theta0

def make_nan_array(i,j=None,k=None,l=None):
    ##Makes arrays fill of NaN##
    if i and not j:
        arr = np.zeros((i))
        arr *= np.nan 
    elif j and not k:
        arr = np.zeros((i,j))
        arr *= np.nan  
    elif k and not l:
        arr = np.zeros((i,j,k))
        arr *= np.nan        
    elif l:
        arr = np.zeros((i,j,k,l))
        arr *= np.nan  
    return arr 
        
def get_colourmap(name):
    if name == "Zissou":
        colors = [
            (0.98, 0.98, 0.95),
            (0.23, 0.60, 0.69),
            (0.47, 0.71, 0.77),
            (0.92, 0.8, 0.16),
            (0.88, 0.68, 0),
            (0.95, 0.10, 0),
            (0.79, 0.08, 0),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueOrange":
        top = cm.get_cmap("Oranges", 128)  # r means reversed version
        bottom = cm.get_cmap("Blues_r", 128)  # combine it all
        colors = np.vstack(
            (bottom(np.linspace(0, 1, 128)), top(np.linspace(0, 1, 128)))
        )  # create a new colormaps with a name of OrangeBlue
        cmap = ListedColormap(colors, name)
    elif name == "Color_blind_1":
        colors = [
            (0.67, 0.34, 0.11),
            (0.89, 0.61, 0.34),
            (1, 0.87, 0.67),
            (0.67, 0.72, 0.86),
            (0.30, 0.45, 0.71),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "Duran_cLCS":
        colors = np.array(
            [
                [1.0000, 1.0000, 0.9987, 1],
                [0.9971, 1.0000, 0.9970, 1],
                [0.9896, 1.0000, 0.9931, 1],
                [0.9771, 1.0000, 0.9871, 1],
                [0.9593, 0.9900, 0.9789, 1],
                [0.9364, 0.9708, 0.9686, 1],
                [0.9084, 0.9484, 0.9564, 1],
                [0.8759, 0.9243, 0.9422, 1],
                [0.8395, 0.8994, 0.9264, 1],
                [0.8000, 0.8749, 0.9092, 1],
                [0.7585, 0.8516, 0.8906, 1],
                [0.7160, 0.8301, 0.8710, 1],
                [0.6738, 0.8110, 0.8506, 1],
                [0.6330, 0.7948, 0.8296, 1],
                [0.5949, 0.7817, 0.8081, 1],
                [0.5606, 0.7719, 0.7865, 1],
                [0.5310, 0.7657, 0.7649, 1],
                [0.5073, 0.7628, 0.7435, 1],
                [0.4900, 0.7633, 0.7225, 1],
                [0.4798, 0.7671, 0.7019, 1],
                [0.4771, 0.7737, 0.6821, 1],
                [0.4819, 0.7831, 0.6629, 1],
                [0.4943, 0.7946, 0.6446, 1],
                [0.5138, 0.8081, 0.6271, 1],
                [0.5399, 0.8230, 0.6105, 1],
                [0.5720, 0.8387, 0.5948, 1],
                [0.6090, 0.8548, 0.5800, 1],
                [0.6500, 0.8708, 0.5659, 1],
                [0.6938, 0.8860, 0.5525, 1],
                [0.7391, 0.9000, 0.5398, 1],
                [0.7847, 0.9120, 0.5275, 1],
                [0.8292, 0.9217, 0.5155, 1],
                [0.8716, 0.9284, 0.5037, 1],
                [0.9108, 0.9317, 0.4918, 1],
                [0.9457, 0.9310, 0.4797, 1],
                [0.9756, 0.9260, 0.4672, 1],
                [1.0000, 0.9162, 0.4541, 1],
                [1.0000, 0.9013, 0.4401, 1],
                [1.0000, 0.8810, 0.4251, 1],
                [1.0000, 0.8551, 0.4089, 1],
                [1.0000, 0.8235, 0.3912, 1],
                [1.0000, 0.7862, 0.3720, 1],
                [1.0000, 0.7432, 0.3511, 1],
                [1.0000, 0.6947, 0.3284, 1],
                [1.0000, 0.6408, 0.3039, 1],
                [1.0000, 0.5821, 0.2775, 1],
                [0.9900, 0.5190, 0.2494, 1],
                [0.9819, 0.4521, 0.2195, 1],
                [0.9765, 0.3822, 0.1882, 1],
                [0.9744, 0.3102, 0.1556, 1],
                [0.9756, 0.2372, 0.1222, 1],
                [0.9799, 0.1643, 0.0884, 1],
                [0.9864, 0.0931, 0.0547, 1],
                [0.9938, 0.0251, 0.0219, 1],
                [1.0000, 0, 0, 1],
                [1.0000, 0, 0, 1],
                [0.9989, 0, 0, 1],
                [0.9858, 0, 0, 1],
                [0.9601, 0, 0, 1],
                [0.9194, 0, 0, 1],
                [0.8618, 0, 0, 1],
                [0.7874, 0, 0, 1],
                [0.6982, 0, 0, 1],
                [0.6000, 0.0069, 0.0013, 1],
            ]
        )
        cmap = LinearSegmentedColormap.from_list(name, colors)
    elif name == "RedYellowBlue":
        colors = [
            (0.843, 0.188, 0.153),
            (0.988, 0.553, 0.349),
            (0.996, 0.878, 0.565),
            (0.569, 0.749, 0.859),
            (0.271, 0.459, 0.706),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueYellowRed":
        colors = [
            (0.843, 0.188, 0.153),
            (0.988, 0.553, 0.349),
            (0.996, 0.878, 0.565),
            (0.569, 0.749, 0.859),
            (0.271, 0.459, 0.706),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "AlgaeSalmon":
        colors = [
            (0.557, 0.792, 0.902),
            (0.165, 0.616, 0.561),
            (0.914, 0.769, 0.416),
            (0.957, 0.635, 0.38),
            (0.906, 0.435, 0.318),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "OceanSun":
        colors = [
            (0.0, 0.188, 0.286),
            (0.839, 0.157, 0.157),
            (0.969, 0.498, 0),
            (0.988, 0.749, 0.286),
            (0.918, 0.886, 0.718),
        ]
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "SunOcean":
        colors = [
            (0.0, 0.188, 0.286),
            (0.839, 0.157, 0.157),
            (0.969, 0.498, 0),
            (0.988, 0.749, 0.286),
            (0.918, 0.886, 0.718),
        ]
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "RedBlue":
        colors = [
            (0.792, 0.0, 0.125),
            (0.957, 0.647, 0.511),
            (0.969, 0.969, 0.969),
            (0.573, 0.773, 0.871),
            (0.024, 0.439, 0.690),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueRed":
        colors = [
            (0.792, 0.0, 0.125),
            (0.957, 0.647, 0.511),
            (0.969, 0.969, 0.969),
            (0.573, 0.773, 0.871),
            (0.024, 0.439, 0.690),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "PurpleOrange":
        colors = [
            (0.369, 0.235, 0.60),
            (0.698, 0.671, 0.824),
            (0.969, 0.969, 0.969),
            (0.992, 0.722, 0.388),
            (0.902, 0.380, 0.004),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "SeaLand":
        colors = [
            (0.004, 0.522, 0.443),
            (0.502, 0.804, 0.757),
            (0.969, 0.969, 0.969),
            (0.875, 0.761, 0.490),
            (0.651, 0.380, 0.102),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "Reds":
        colors = [
            (0.996, 0.941, 0.851),
            (0.992, 0.800, 0.541),
            (0.988, 0.553, 0.349),
            (0.843, 0.188, 0.122),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    return cmap
from laspy.file import File
import numpy as np
import scipy.io as sio

from pyproj import Proj, transform
from shapely.geometry import LineString, MultiLineString, Point

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import overpass


def copy_decimated_writable(las, decimate_step=10):
    n = "%s_decimated_%d.las" % (las.filename, decimate_step)
    las_out = File(n, mode='w', header=las.header)
    las_out.x = las.x[1::decimate_step]
    las_out.y = las.y[1::decimate_step]
    las_out.z = las.z[1::decimate_step]
    las_out.intensity = las.intensity[1::decimate_step]

    return las_out


# Converting to WGS84
def get_wgs84_sector(las, proj='epsg:32733'):
    in_proj = Proj(init=proj)
    out_proj = Proj(init='epsg:4326')
    lon, lat = transform(in_proj, out_proj, las.x, las.y)
    sector = (np.min(lat), np.min(lon), np.max(lat),np.max(lon))
    return sector


def convert_multilines(multilines, from_proj='epsg:4326', to_proj='epsg:32733'):
    to_proj = Proj(init=to_proj)
    from_proj = Proj(init=from_proj)

    linestrings = []
    for pl in multilines:
        lon = [c[0] for c in pl.coords]
        lat = [c[1] for c in pl.coords]
        x, y = transform(from_proj, to_proj, lon, lat)
        linestrings.append(LineString(zip(x, y)))

    return linestrings


def show_las(xs,ys,zs,c=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=c, marker='.')
    plt.show()


# def get_near_points(las, line, width):
#     x = las.x
#     y = las.y
#     r = width / 2
#
#     min_x = np.min(line.coords[0]) - r
#     min_y = np.min(line.coords[1]) - r
#     max_x = np.max(line.coords[0]) + r
#     max_y = np.max(line.coords[1]) + r
#
#     candidates = np.where((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y))
#
#     if len(candidates[0]) == 0:
#         return
#
#     x = x[candidates]
#     y = y[candidates]
#
#     points = [Point(px,py) for px,py in zip(x,y)]
#
#     print(len(points))


def get_near_points(las, segment_start, segment_end, segment_width):
    x = las.x
    y = las.y
    las_shape = x.shape
    r = segment_width / 2

    min_x = np.min([segment_start[0], segment_end[0]]) - r
    min_y = np.min([segment_start[1], segment_end[1]]) - r
    max_x = np.max([segment_start[0], segment_end[0]]) + r
    max_y = np.max([segment_start[1], segment_end[1]]) + r

    candidates = np.where((x >= min_x) & (x <= max_x) &
                          (y >= min_y) & (y <= max_y))
    candidates = candidates[0]

    mask = np.zeros(las_shape, dtype=bool)
    if len(candidates) == 0:
        return mask

    x = x[candidates]
    y = y[candidates]
    xy = np.transpose(np.vstack((x, y)))

    # Distance Line-Point
    d = segment_start-segment_end
    n = np.linalg.norm(d)
    xy0 = xy - segment_start
    c = np.cross(d, xy0)
    distances = np.abs(c / n)

    near = np.where(distances < r)[0]
    candidates = candidates[near]

    mask[candidates] = 1

    return mask


def get_road_mask(lines, widths):
    mask = np.zeros(las_data.x.shape, dtype=bool)
    n_segment = 0
    for line, width in zip(lines, widths):
        for i in range(len(line.coords) - 1):
            segment_start = s = np.array(line.coords[i])
            segment_end = np.array(line.coords[i + 1])
            mask_i = get_near_points(las_data, segment_start, segment_end, segment_width=width)
            mask = mask | mask_i
            print("Segment %d, Points detected %d" % (n_segment, np.sum(mask)))
            n_segment += 1

    return mask


def show_linestrings(linestrings):
    fig = plt.figure()
    for ls in linestrings:
        x = [c[0] for c in ls.coords]
        y = [c[1] for c in ls.coords]
        plt.plot(x, y)

    plt.show()


# Reading LAS
# filepath = "01_20190430_083945_1.las"
# filepath = "04_20190430_093008_0.las"
filepath = "293S_20190521_074137_4.las"
# las_proj = 'epsg:32733'  # Angola
las_proj = 'epsg:25830'  # Europe
las_data_read = File(filepath, mode='r')
las_data = copy_decimated_writable(las_data_read, decimate_step=10)

sector = get_wgs84_sector(las_data, proj=las_proj)
print("Sector: ", end="")
print(sector)
# sector = (-9.09333395170717, 13.28352294034062, -9.08682383512845, 13.306427631531223)
print(sector)

cache = overpass.ObjectCache()
query_roads = overpass.get_highway_query(sector)
linestrings, tags, widths = overpass.get_linestrings(query_roads, cache)


linestrings = convert_multilines(linestrings, from_proj='epsg:4326', to_proj=las_proj)
show_linestrings(linestrings)

road_mask = get_road_mask(linestrings, widths)
sio.savemat('roads_decimated.mat', {'mask': road_mask})

road_mask = road_mask.astype(np.uint8)

# mask = sio.loadmat('roads.mat')['mask']

# nth = 2000
# m = road_mask[1::nth][0]
# show_las(las_data.x[1::nth],las_data.y[1::nth],las_data.z[1::nth], c=m)

# las_out = File("01_20190430_083945_1_roads.las", mode='w', header=las_data.header)
# las_out.x = las_data.x
# las_out.y = las_data.y
# las_out.z = las_data.z
las_data.classification = road_mask

print("done")





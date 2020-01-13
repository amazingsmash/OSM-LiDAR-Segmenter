from laspy.file import File
import numpy as np
import scipy.io as sio
# import Encoding
import shutil
import os

from pyproj import Proj, transform
from waySet import WaySet
from shapely.geometry import LineString, MultiLineString, Point

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import pptk


# Reading LAS
las_data = File("01_20190430_083945_1.las", mode='r')


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


def get_near_points(las, line, width):
    x = las.x
    y = las.y
    r = width / 2

    min_x = np.min(line.coords[0]) - r
    min_y = np.min(line.coords[1]) - r
    max_x = np.max(line.coords[0]) + r
    max_y = np.max(line.coords[1]) + r

    candidates = np.where((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y))

    if len(candidates[0]) == 0:
        return

    x = x[candidates]
    y = y[candidates]

    points = [Point(px,py) for px,py in zip(x,y)]

    print(len(points))


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

    d = segment_start-segment_end
    n = np.linalg.norm(d)
    c = np.cross(d, xy - segment_start)
    distances = c / n

    near = np.where(distances < r)[0]
    candidates = candidates[near]

    mask[candidates] = 1

    return mask


def get_road_mask(lines):
    mask = np.zeros(las_data.x.shape, dtype=bool)
    n_segment = 0
    for line in lines:
        for i in range(len(line.coords) - 1):
            segment_start = s = np.array(line.coords[i])
            segment_end = np.array(line.coords[i + 1])
            mask_i = get_near_points(las_data, segment_start, segment_end, 10)
            mask = mask | mask_i
            print("Segment %d, Points detected %d" % (n_segment, np.sum(mask)))
            n_segment += 1

    return mask





# sector = get_sector(las_data, proj='epsg:32733')
sector = (-9.09333395170717, 13.28352294034062, -9.08682383512845, 13.306427631531223)
print(sector)

# ways = WaySet.download_all_ways(sector, tram_only=False, sector_padding=0)
# linestrings = convert_multilines(ways.multiline)
# road_mask = get_road_mask(linestrings)
# sio.savemat('roads.mat', {'mask': road_mask})

mask = sio.loadmat('roads.mat')['mask']

nth = 2000
m = mask[:, 1::nth][0]
show_las(las_data.x[1::nth],las_data.y[1::nth],las_data.z[1::nth], c=m)

print("done")





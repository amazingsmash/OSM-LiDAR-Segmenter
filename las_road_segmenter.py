import argparse
import sys

from laspy.file import File
from laspy.header import Header
import numpy as np
import scipy.io as sio

from pyproj import CRS, Transformer
from shapely.geometry import LineString

import matplotlib.pyplot as plt

import json_utils
import overpass_roads
from mpl_toolkits.mplot3d import Axes3D


# def copy_decimated_writable(las, decimate_step=10):
#     n = "%s_decimated_%d.las" % (las.filename, decimate_step)
#     las_out = File(n, mode='w', header=las.header)
#     las_out.x = las.x[1::decimate_step]
#     las_out.y = las.y[1::decimate_step]
#     las_out.z = las.z[1::decimate_step]
#     las_out.intensity = las.intensity[1::decimate_step]
#
#     return las_out


# def get_wgs84_sector(las, epsg_num):
#
#     crs_in = CRS.from_epsg(epsg_num)
#     crs_4326 = CRS.from_epsg(4326)
#     transformer = Transformer.from_crs(crs_from=crs_in, crs_to=crs_4326)
#
#     lat, lon = transformer.transform(las.x, las.y)
#
#     sector = (np.min(lat), np.min(lon), np.max(lat), np.max(lon))
#     return sector

def get_wgs84_sector(xyzc, epsg_num):
    crs_in = CRS.from_epsg(epsg_num)
    crs_4326 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_from=crs_in, crs_to=crs_4326)

    lat, lon = transformer.transform(xyzc[:, 0], xyzc[:, 1])

    sector = (np.min(lat), np.min(lon), np.max(lat), np.max(lon))
    return sector


# def convert_multilines(multilines, epsg_to=32733):
#     crs_from = CRS.from_epsg(4326)
#     crs_to = CRS.from_epsg(epsg_to)
#     transformer = Transformer.from_crs(crs_from=crs_from, crs_to=crs_to)
#
#     linestrings = []
#     for pl in multilines:
#         lon = [c[0] for c in pl.coords]
#         lat = [c[1] for c in pl.coords]
#         x, y = transformer.transform(lat, lon)
#         linestrings.append(LineString(zip(x, y)))
#
#     return linestrings


def show_las(xs, ys, zs, c=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=c, marker='.')
    plt.show()


# def get_near_points(las, segment_start, segment_end, segment_width):
#     x = las.x
#     y = las.y
#     las_shape = x.shape
#     r = segment_width / 2
#
#     min_x = np.min([segment_start[0], segment_end[0]]) - r
#     min_y = np.min([segment_start[1], segment_end[1]]) - r
#     max_x = np.max([segment_start[0], segment_end[0]]) + r
#     max_y = np.max([segment_start[1], segment_end[1]]) + r
#
#     candidates = np.where((x >= min_x) & (x <= max_x) &
#                           (y >= min_y) & (y <= max_y))
#     candidates = candidates[0]
#
#     mask = np.zeros(las_shape, dtype=bool)
#     if len(candidates) == 0:
#         return mask
#
#     x = x[candidates]
#     y = y[candidates]
#     xy = np.transpose(np.vstack((x, y)))
#
#     # Distance Line-Point
#     d = segment_start-segment_end
#     n = np.linalg.norm(d)
#     xy0 = xy - segment_start
#     c = np.cross(d, xy0)
#     distances = np.abs(c / n)
#
#     near = np.where(distances < r)[0]
#     candidates = candidates[near]
#
#     mask[candidates] = 1
#
#     return mask

def get_near_points(xyzc, segment_start, segment_end, segment_width):
    x = xyzc[:, 0]
    y = xyzc[:, 1]
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
    d = segment_start - segment_end
    n = np.linalg.norm(d)
    xy0 = xy - segment_start
    c = np.cross(d, xy0)
    distances = np.abs(c / n)

    near = np.where(distances < r)[0]
    candidates = candidates[near]

    mask[candidates] = 1

    return mask


# def get_road_mask(line_coords, widths):
#     mask = np.zeros(las_data.x.shape, dtype=bool)
#     n_segment = 0
#     for lc, width in zip(line_coords, widths):
#         for i in range(len(lc) - 1):
#             segment_start = np.array(lc[i])
#             segment_end = np.array(lc[i + 1])
#             mask_i = get_near_points(las_data, segment_start, segment_end, segment_width=width)
#             mask = mask | mask_i
#             print("Segment %d, Points detected %d" % (n_segment, np.sum(mask)))
#             n_segment += 1
#
#     mask = mask.astype(np.uint8)
#
#     return mask

def get_road_mask(xyzc, line_coords, widths):
    mask = np.zeros(xyzc.shape[0], dtype=bool)
    n_segment = 0
    for lc, width in zip(line_coords, widths):
        for i in range(len(lc) - 1):
            segment_start = np.array(lc[i])
            segment_end = np.array(lc[i + 1])
            mask_i = get_near_points(xyzc, segment_start, segment_end, segment_width=width)
            mask = mask | mask_i
            n_segment += 1
            num_points = np.sum(mask_i)
            if num_points > 0:
                print("Segment %d, Points detected %d" % (n_segment, num_points))
    return mask


def show_linestrings(linestrings):
    fig = plt.figure()
    for ls in linestrings:
        x = [c[0] for c in ls.coords]
        y = [c[1] for c in ls.coords]
        plt.plot(x, y)

    plt.show()


def get_map_url(sector):
    map_url = "http://layered.wms.geofabrik.de/std/demo_key?LAYERS=world%2Cbuildings%2Cpower%2Clanduse%2Cwater%2Cadmin%2Croads%2Cpoi%2Cplacenames&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image%2Fjpeg&SRS=EPSG%3A4326&BBOX={BBOX}&WIDTH=256&HEIGHT=256"
    return map_url.replace("{BBOX}", "%f,%f,%f,%f" % (sector[1], sector[0], sector[3], sector[2]))


# def get_road_points(las_path, epsg, decimation_step=10, cache=None, output_roads=None):
#     # Reading LAS
#     las_data_read = File(las_path, mode='r')
#     las_data = copy_decimated_writable(las_data_read, decimate_step=decimation_step)
#
#     # Getting Sector
#     sector = get_wgs84_sector(las_data, epsg_num=las_proj)
#     print("LAS WGS84 Sector [%f, %f - %f, %f] " % sector)
#     print("See the map: " + print(get_map_url(sector)))
#
#     #Getting OP data
#     roads = overpass_roads.get_roads(sector, cache, epsg_to=epsg)
#     if output_roads is not None:
#         json_utils.write_json(roads, output_roads)
#
#     #Computing mask
#     road_mask = get_road_mask(line_coords= [line["coords"] for line in roads],
#                               widths=[line["estimated_width_meters"] for line in roads])
#
#     return road_mask

def get_road_points(xyzc, epsg, cache=None):
    # Getting Sector
    sector = get_wgs84_sector(xyzc, epsg_num=las_proj)
    print("LAS WGS84 Sector [%f, %f - %f, %f] " % sector)
    print("See the map at: " + get_map_url(sector))

    # Getting OP data
    roads = overpass_roads.get_roads(sector, cache, epsg_to=epsg)

    # Computing mask
    road_mask = get_road_mask(xyzc,
                              line_coords=[line["coords"] for line in roads],
                              widths=[line["estimated_width_meters"] for line in roads])

    return road_mask, roads


def save_las(xyzc, header, path):
    # header = Header()
    outfile = File(path, mode="w", header=header)
    outfile.X = xyzc[:,0]
    outfile.Y = xyzc[:,1]
    outfile.Z = xyzc[:,2]
    outfile.Classification = xyzc[:, 3].astype(np.uint8)
    outfile.close()


if __name__ == "__main__":

    # filepath = "01_20190430_083945_1.las"
    # filepath = "04_20190430_093008_0.las"
    # filepath = "293S_20190521_074137_4.las"
    filepath = "294S_20190521_143953_6_EPSG_25830.las"
    # las_proj = 'epsg:32733'  # Angola
    las_proj = 25830  # Europe
    decimation = 10  # Take 1 every 10 points

    if len(sys.argv) < 2:
        print("Define LAS file path to process as first parameter")
        exit(0)

    filepath = sys.argv[1]

    parser = argparse.ArgumentParser(prog='overpass_roads')
    parser.add_argument("-e", help="Projection of LAS file (default 4326)", type=int, default=4326)
    parser.add_argument("-r", help="Output roads JSON file", type=str, default=None)
    parser.add_argument("-c", help="OSM cache file", type=str, default=None)
    parser.add_argument("-d", help="Take 1 every N points (default 1)", type=int, default=1)
    parser.add_argument("-o", help="Classified and decimated LAS output file (overwriting input as default)",
                        type=str,
                        default=filepath)

    parser.add_argument("-s", "--show_result",
                        help="Show resulting point cloud",
                        action='store_true')

    args = parser.parse_args(sys.argv[2:])

    # Reading LAS
    print("Processing %s" % filepath)
    las_data_read = File(filepath, mode='r')
    las_header = las_data_read.header.copy()
    xyzc_points = np.transpose(np.array([las_data_read.x,
                                         las_data_read.y,
                                         las_data_read.z,
                                         las_data_read.Classification.astype(float)]))
    las_data_read.close()

    if args.d != 1:
        xyzc_points = xyzc_points[::args.d, :]

    #las_data = copy_decimated_writable(las_data_read, decimate_step=decimation)

    road_mask, roads = get_road_points(xyzc=xyzc_points,
                                       epsg=args.e,
                                       cache=args.c)
    xyzc_points[:, 3] = road_mask

    if args.r is not None:
        json_utils.write_json(roads, args.r)

    save_las(xyzc_points, las_header, args.o)

    if args.show_result:
        show_las(xyzc_points[:,0],
                 xyzc_points[:,1],
                 xyzc_points[:,2],
                 xyzc_points[:,3])

    # Getting Sector
    # sector = get_wgs84_sector(las_data, epsg_num=las_proj)
    # print(sector)
    # print(get_map_url(sector))
    # sector = (-9.09333395170717, 13.28352294034062, -9.08682383512845, 13.306427631531223)

    # cache = overpass_roads.ObjectCache()
    # query_roads = overpass_roads.get_highway_query(sector)
    # road_linestrings, road_tags, road_widths = overpass_roads.get_linestrings(query_roads, cache)

    # road_linestrings = convert_multilines(road_linestrings, epsg_to=las_proj)
    # show_linestrings(road_linestrings)

    # road_mask = get_road_mask(road_linestrings, road_widths)

    # road_mask = road_mask.astype(np.uint8)

    # sio.savemat('roads_decimated.mat', {'mask': road_mask})
    # mask = sio.loadmat('roads.mat')['mask']


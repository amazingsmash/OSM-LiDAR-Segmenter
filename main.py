from laspy.file import File
import numpy as np
import scipy.io as sio

from pyproj import CRS, Transformer
from shapely.geometry import LineString

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
def get_wgs84_sector(las, epsg_num=32733):

    crs_in = CRS.from_epsg(epsg_num)
    crs_4326 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_from=crs_in, crs_to=crs_4326)

    lat, lon = transformer.transform(las.x, las.y)

    sector = (np.min(lat), np.min(lon), np.max(lat), np.max(lon))
    return sector


def convert_multilines(multilines, epsg_to=32733):

    crs_from = CRS.from_epsg(4326)
    crs_to = CRS.from_epsg(epsg_to)
    transformer = Transformer.from_crs(crs_from=crs_from, crs_to=crs_to)

    linestrings = []
    for pl in multilines:
        lon = [c[0] for c in pl.coords]
        lat = [c[1] for c in pl.coords]
        x, y = transformer.transform(lat, lon)
        linestrings.append(LineString(zip(x, y)))

    return linestrings


def show_las(xs,ys,zs,c=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=c, marker='.')
    plt.show()


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

def get_map_url(sector):
    map_url = "http://layered.wms.geofabrik.de/std/demo_key?LAYERS=world%2Cbuildings%2Cpower%2Clanduse%2Cwater%2Cadmin%2Croads%2Cpoi%2Cplacenames&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image%2Fjpeg&SRS=EPSG%3A4326&BBOX={BBOX}&WIDTH=256&HEIGHT=256"
    return map_url.replace("{BBOX}", "%f,%f,%f,%f" % (sector[1], sector[0], sector[3], sector[2]))


if __name__ == "__main__":

    # filepath = "01_20190430_083945_1.las"
    # filepath = "04_20190430_093008_0.las"
    # filepath = "293S_20190521_074137_4.las"
    filepath = "294S_20190521_143953_6_EPSG_25830.las"
    # las_proj = 'epsg:32733'  # Angola
    las_proj = 25830  # Europe
    decimation = 10  # Take 1 every 10 points

    # Reading LAS
    print("Processing %s" % filepath)
    las_data_read = File(filepath, mode='r')
    las_data = copy_decimated_writable(las_data_read, decimate_step=decimation)

    # Getting Sector
    sector = get_wgs84_sector(las_data, epsg_num=las_proj)
    print("Sector: ", end="")
    print(sector)
    print(get_map_url(sector))
    # sector = (-9.09333395170717, 13.28352294034062, -9.08682383512845, 13.306427631531223)

    cache = overpass.ObjectCache()
    query_roads = overpass.get_highway_query(sector)
    road_linestrings, road_tags, road_widths = overpass.get_linestrings(query_roads, cache)

    road_linestrings = convert_multilines(road_linestrings, epsg_to=las_proj)
    show_linestrings(road_linestrings)

    road_mask = get_road_mask(road_linestrings, road_widths)

    road_mask = road_mask.astype(np.uint8)

    # sio.savemat('roads_decimated.mat', {'mask': road_mask})
    # mask = sio.loadmat('roads.mat')['mask']

    las_data.classification = road_mask

    print("Point Classification Finished")





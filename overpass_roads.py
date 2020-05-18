import argparse
import json
import sys

import overpy
import time

from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point

import json_utils


class ObjectCache:

    def __init__(self, file="query_cache.json"):
        self.query_cache = None
        self.file = file

    def get_from_cache(self, query):
        try:
            if self.query_cache is None:
                with open(self.file, 'r') as f:
                    self.query_cache = json.load(f)

            return self.query_cache[query]
        except Exception:
            return None

    def store_in_cache(self, query, data):
        if self.query_cache is None:
            self.query_cache = {}
        self.query_cache[query] = data

        with open(self.file, 'w') as outfile:
            json.dump(self.query_cache, outfile, indent=4, sort_keys=True)


def get_highway_query(sector, timestamp=None, sector_padding=0.01):
    query = '[out:json]{{date}};(way["highway"]({{bbox}});node({{ext_bbox}}););out;'

    bbox = "%f,%f,%f,%f" % sector
    ext_bbox = "%f,%f,%f,%f" % (
        sector[0] - sector_padding, sector[1] - sector_padding,
        sector[2] + sector_padding, sector[3] + sector_padding)

    query = query.replace("{{bbox}}", bbox)
    query = query.replace("{{ext_bbox}}", ext_bbox)

    timestamp = "[date:\"%s\"]" % timestamp if timestamp is not None else ""
    query = query.replace("{{date}}", timestamp)

    print("Query: %s" % query)

    return query


def get_road_width(waytags):
    if "highway" in waytags:
        if "lanes" in waytags:
            return int(waytags["lanes"]) * 2.55

        if waytags["highway"] == "residential":
            return 5

        if waytags["highway"] == "tertiary":
            return 7

        if waytags["highway"] == "service":
            return 3

    return 10


def __op_query(q):
    api = overpy.Overpass()

    result = None
    r = 1
    while result is None:
        try:
            result = api.query(q)
        except (overpy.exception.OverpassGatewayTimeout, overpy.exception.OverpassTooManyRequests):
            time.sleep(2)
            print("Overpass query - too many requests. Retry %d." % r)
            result = api.query(q)

    return result


def __op_get_nodes_repeat(way):
    result = None
    resolve_missing = False
    r = 1
    while result is None:
        try:
            result = way.get_nodes(resolve_missing=resolve_missing)
        except (overpy.exception.OverpassGatewayTimeout, overpy.exception.OverpassTooManyRequests):
            time.sleep(2)
            print("Overpass get nodes - too many requests. Retry %d." % r)
            r = r + 1
            result = way.get_nodes(resolve_missing=resolve_missing)
        except overpy.exception.DataIncomplete:
            resolve_missing = True

    return result


def get_linestrings(query, cache=None):
    result = cache.get_from_cache(query) if cache is not None else None

    if result is None:
        result = __op_query(query)

        way_tags = []
        linestrings = []
        for w in result.ways:
            nodes = __op_get_nodes_repeat(w)

            if nodes is None:
                continue

            nodes = [[float(n.lon), float(n.lat)] for n in nodes]
            linestrings.append(nodes)
            way_tags.append(w.tags)

        result = {"linestrings": linestrings, "way_tags": way_tags}

        if cache is not None:
            cache.store_in_cache(query, result)

    linestrings = [LineString(c) for c in result["linestrings"]]

    widths = [get_road_width(wt) for wt in result["way_tags"]]

    return linestrings, result["way_tags"], widths


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


def get_roads(sector, cache=None, epsg_to=4326):
    query_roads = get_highway_query(sector)
    road_linestrings, road_tags, road_widths = get_linestrings(query_roads, cache)
    road_linestrings = convert_multilines(road_linestrings, epsg_to=epsg_to)

    data = [{"coords": list(l.coords),
             "tags": t,
             "estimated_width_meters": w} for l, t, w in zip(road_linestrings, road_tags, road_widths)]

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='overpass_roads')
    parser.add_argument("-minlat", help="Minimum Latitude", type=float)
    parser.add_argument("-minlon", help="Minimum Longitude", type=float)
    parser.add_argument("-maxlat", help="Maximum Latitude", type=float)
    parser.add_argument("-maxlon", help="Maximum Longitude", type=float)
    parser.add_argument("-e", help="Output EPSG Projection", type=int, default=4326)
    parser.add_argument("-o", help="Output JSON File", type=str, default=None)
    parser.add_argument("-c", help="Cache File", type=str, default="")

    arg = parser.parse_args(sys.argv[1:])  # getting args

    sector = (arg.minlat, arg.minlon, arg.maxlat, arg.maxlon)
    cache = None if arg.c == "" else ObjectCache(arg.c)
    proj = arg.e

    roads = get_roads(sector, cache, epsg_to=proj)

    if arg.o is None:
        print(roads)
    else:
        json_utils.write_json(roads, arg.o)

    print("done")




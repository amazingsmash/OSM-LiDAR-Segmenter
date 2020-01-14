import overpy
import time
from shapely.geometry import LineString, MultiLineString, Point
import json


class ObjectCache:

    def __init__(self):
        self.query_cache = None

    def get_from_cache(self, query):
        try:
            if self.query_cache is None:
                with open("query_cache.json", 'r') as f:
                    self.query_cache = json.load(f)

            return self.query_cache[query]
        except Exception:
            return None

    def store_in_cache(self, query, data):
        if self.query_cache is None:
            self.query_cache = {}
        self.query_cache[query] = data

        with open("query_cache.json", 'w') as outfile:
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
        if "lanes" in waytags: return int(waytags["lanes"]) * 2.55

        if waytags["highway"] == "residential": return 5

        if waytags["highway"] == "tertiary": return 7

        if waytags["highway"] == "service": return 3

    return 10


def get_linestrings(query, cache=None):
    result = cache.get_from_cache(query) if cache is not None else None

    if result is None:
        api = overpy.Overpass()
        try:
            result = api.query(query)
        except overpy.OverpassTooManyRequests:
            time.sleep(20)
            result = api.query(query)

        way_tags = []
        linestrings = []
        for w in result.ways:
            try:
                nodes = w.get_nodes(resolve_missing=False)
            except overpy.exception.DataIncomplete:
                try:
                    nodes = w.get_nodes(resolve_missing=True)
                except overpy.exception.DataIncomplete:
                    print("Overpass can't resolve nodes. Skipping way.")
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

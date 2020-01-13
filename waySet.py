from shapely.geometry import LineString, MultiLineString, Point
import overpy
import time

import matplotlib.pyplot as plt
import json

import route
from route import Route
import numpy as np


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


class WaySet:
    ways_cache = ObjectCache()

    def __init__(self, lines):
        self.multiline = MultiLineString(lines)
        self.headings = [Route.get_heading(l.coords[0][1], l.coords[0][0], l.coords[1][1], l.coords[1][0]) for l in
                         lines]

    @staticmethod
    def download_all_ways(sector, tram_only=False, timestamp=None, sector_padding=0.01):

        bbox = "%f,%f,%f,%f" % sector
        ext_bbox = "%f,%f,%f,%f" % (sector[0] - sector_padding, sector[1] - sector_padding, sector[2] + sector_padding, sector[3] + sector_padding)

        if tram_only:
            query = '[out:json]{{date}};(node({{ext_bbox}});way["railway"="tram"]({{bbox}}););out;'
        else:
            query = '[out:json]{{date}};(way["highway"]({{bbox}});node({{ext_bbox}}););out;'

        query = query.replace("{{bbox}}", bbox)
        query = query.replace("{{ext_bbox}}", ext_bbox)

        timestamp = "[date:\"%s\"]" % timestamp if timestamp is not None else ""
        query = query.replace("{{date}}", timestamp)

        ways = WaySet.ways_cache.get_from_cache(query)
        if ways is None:
            api = overpy.Overpass()
            try:
                result = api.query(query)
            except overpy.OverpassTooManyRequests:
                time.sleep(20)
                result = api.query(query)

            ways = []
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
                ways += [nodes]

            WaySet.ways_cache.store_in_cache(query, ways)

        lines = [LineString(c) for c in ways]

        return WaySet(lines)

    def snap_point_to_lines(self, point, next_point, priority_line_index):
        distances = [line.distance(point) for line in self.multiline]

        if priority_line_index > -1:
            distances[priority_line_index] /= 2

        min_distance = min(distances)
        i_dist = distances.index(min_distance)

        projection = self.multiline[i_dist].project(point)
        next_proj = 1 if next_point is None else self.multiline[i_dist].project(next_point)
        new_point = self.multiline[i_dist].interpolate(projection)
        new_next_point = self.multiline[i_dist].interpolate(next_proj)

        heading = Route.get_heading(new_point.y, new_point.x, new_next_point.y, new_next_point.x)

        return new_point, i_dist, heading

    def snap_point_to_multilines(self, point):
        return self.multiline.interpolate(self.multiline.project(point))

    def plot(self):
        for l in self.multiline:
            lon, lat = l.xy
            plt.plot(lon, lat)

    def get_closest_way(self, point):
        distances = [line.distance(point) for line in self.multiline]
        min_distance = min(distances)
        i_min_dist = distances.index(min_distance)
        return self.multiline[i_min_dist]

    def get_closest_way_with_heading(self, point, heading, heading_tolerance=45):
        heading = heading % 360
        lines = self.multiline
        projections = [l.project(point, normalized=True) for l in lines]
        lines = [l for l, p in zip(lines, projections) if 0 < p < 1]

        headings = np.array([WaySet.get_linestring_heading_at_projection(l, p)
                    for l, p in zip(lines, projections)])
        headings_diff = np.abs(headings % 360 - heading)
        # print(headings_diff)
        if (headings_diff < heading_tolerance).any():
            lines = [l for l, hd in zip(lines, headings_diff) if hd < heading_tolerance]

        distances = [line.distance(point) for line in lines]

        min_distance = min(distances)
        i_min_dist = distances.index(min_distance)

        return lines[i_min_dist]


    @staticmethod
    def get_linestring_heading_at_projection(linestring, projection):

        ps = [linestring.project(Point(c), normalized=True) for c in linestring.coords]

        for prev_c, c in zip(linestring.coords[:-1], linestring.coords[1:]):
            p = linestring.project(Point(c))
            if p > projection:
                return Route.get_heading(prev_c[1], prev_c[0], c[1], c[0])

        prev_c = linestring.coords[-2]
        c = linestring.coords[-1]
        return Route.get_heading(prev_c[1], prev_c[0], c[1], c[0])

    def snap_points(self, points):
        last_way = None
        snapped_points = []
        last_distance = None

        # headings before snapping
        headings = []
        for i, p in enumerate(points):
            if i < len(points) - 1:
                next_point = points[i + 1]
                headings.append(Route.get_heading(p.y, p.x, next_point.y, next_point.x))
        headings.append(headings[-1])

        h_diff = []

        for point_index in range(len(points)):

            point = points[point_index]
            heading = headings[point_index]

            # if last_way is not None:
            #     projection = last_way.project(point)
            #     distance = last_way.distance(point)
            #     if 0 <= projection <= 1 and distance < 1.1 * last_distance:
            #         snapped_point = last_way.interpolate(projection)
            #         last_distance = distance
            #         snapped_points += [snapped_point]
            #         continue

            last_way = self.get_closest_way_with_heading(point, heading)

            # last_way = self.get_closest_way(point)
            projection = last_way.project(point)
            # last_distance = last_way.distance(point)
            snapped_point = last_way.interpolate(projection)

            h = WaySet.get_linestring_heading_at_projection(last_way, projection)

            h_diff += [headings[point_index] - h]

            snapped_points += [snapped_point]

        # plt.plot(h_diff)
        # plt.show()

        # headings after snapping
        headings = []
        for i, p in enumerate(snapped_points):
            if i < len(snapped_points) - 1:
                next_point = snapped_points[i + 1]
                headings.append(Route.get_heading(p.y, p.x, next_point.y, next_point.x))
        headings.append(headings[-1])

        return Route.from_points(snapped_points, headings)

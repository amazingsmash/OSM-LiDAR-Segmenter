from shapely.geometry import Point
import math
import numpy as np

import matplotlib.pyplot as plt
from math import atan2, cos, sin, degrees

class MathUtils:

    @staticmethod
    def dot_product(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    @staticmethod
    def length(v):
        return math.sqrt(MathUtils.dot_product(v, v))

    @staticmethod
    def angle(v1, v2):
        try:
            return math.acos(MathUtils.dot_product(v1, v2) / (MathUtils.length(v1) * MathUtils.length(v2)))
        except:
            return np.nan

    @staticmethod
    def get_outliers(data, m=2):
        data = np.array(data)
        is_nan = np.isnan(data)
        data[is_nan] = np.mean(data[~is_nan])
        out = abs(data - np.mean(data)) > m * np.std(data)
        return is_nan | out

class Route:

    def __init__(self, latitudes, longitudes, headings=None):
        self.points = [Point(list(longitudes)[i], list(latitudes)[i]) for i in range(len(latitudes))]
        self.headings = headings

    @staticmethod
    def from_points(points, headings=None):
        latitudes, longitudes = Route.get_lat_lon_from_points(points)
        return Route(latitudes=latitudes, longitudes=longitudes, headings= headings)

    def not_too_close_points_indices(self, min_dist):

        indices = [0]
        for i, p in enumerate(self.points):
            l = indices[-1]
            if p.distance(self.points[l]) > min_dist:
                indices.append(i)

        return indices

    def get_next_point_at_min_distance(self, start_index, min_distance):
        pi = self.points[start_index]
        for j in range(1, len(self.points)-start_index):
            pj = self.points[j+start_index]
            if pi.distance(pj) > min_distance or j == len(self.points)-start_index-1:
                return pj

    def get_mean_of_points_in_range(self, start_index, end_index):

        end_index = np.clip(end_index, start_index, len(self.points))
        latitudes, longitudes = Route.get_lat_lon_from_points(self.points[start_index:end_index])
        return Point(np.mean(longitudes), np.mean(latitudes))

        # pi = self.points[start_index]
        # for j in range(1, len(self.points)-start_index):
        #     pj = self.points[j+start_index]
        #     if pi.distance(pj) > min_distance or j == len(self.points)-start_index-1:
        #         return pj

    @staticmethod
    def get_heading(lat1, lon1, lat2, lon2):
        angle = atan2(cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1), sin(lon2 - lon1) * cos(lat2))
        heading = (90 - (degrees(angle) + 360)) % 360
        return heading

    def get_sector(self):
        latitudes, longitudes = self.get_lat_lon_from_points(self.points)
        sector = (min(latitudes), min(longitudes), max(latitudes), max(longitudes))
        return sector

    @staticmethod
    def get_lat_lon_from_points(points):
        latitudes = [l.y for l in points]
        longitudes = [l.x for l in points]
        return latitudes, longitudes

    def snap_points(self, way_set):
        snapped_points = way_set.snap_points(self.points)
        return snapped_points

    def filter(self, selected_points):
        indices = list(np.where(selected_points)[0])
        ps = [self.points[i] for i in indices]
        latitudes, longitudes = self.get_lat_lon_from_points(ps)
        return Route(latitudes, longitudes)

    def plot(self, style='b*'):

        latitudes, longitudes = Route.get_lat_lon_from_points(self.points)
        # if hasattr(self, "headings"):
        #     pass
        # else:
        plt.plot(longitudes, latitudes, style)

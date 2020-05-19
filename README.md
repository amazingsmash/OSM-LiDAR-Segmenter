# OSM-LiDAR-Segmenter

Open point-clouds and scans are a common source of geographical information. However, their segmentation remains an open problem as, commonly, there is insufficient ground-truth data to train ML techniques. These tools present in this project intend to exploit openly available data from OSM, to provide some initial data to bootstrap these learning process.

In particular, the present version performs point cloud segmentation from Open Street Map road data for airborne LiDAR. In order to efficiently classify the dataset on a point-by-point basis, it estimates the road width based on the OSM metadata and computes a Digital Elevation Model based on the input geometry.

The repository contains two Python tools:

## overpass_roads

This tool retrieves OSM data on the roads present on a given WGS84 sector. The information is compiled from several queries against the Overpass API.
Based on the information metadata tags the application adds some estimation of the road widths.

        usage: overpass_roads: Get OSM road information from the Overpass API.
               [-h] [-minlat MIN_LATITUDE] [-minlon MIN_LONGITUDE]
               [-maxlat MAX_LATITUDE] [-maxlon MAX_LONGITUDE] [-e EPSG] [-o OUT]
               [-c CACHE] [-s]

        optional arguments:
          -h, --help            show this help message and exit
          -minlat MIN_LATITUDE, --min_latitude MIN_LATITUDE  Minimum Latitude
          -minlon MIN_LONGITUDE, --min_longitude MIN_LONGITUDE Minimum Longitude
          -maxlat MAX_LATITUDE, --max_latitude MAX_LATITUDE Maximum Latitude
          -maxlon MAX_LONGITUDE, --max_longitude MAX_LONGITUDE Maximum Longitude
          -e EPSG, --epsg EPSG  Output EPSG Projection
          -o OUT, --out OUT     Output JSON File
          -c CACHE, --cache CACHE  Cache File
          -s, --show_results    Show resulting point cloud

## las_road_segmenter

This tool segments LAS point clouds using the road data provided by overpass_roads. The LAS file can be passed on several EPSG reference systems.
The tool also accepts a height threshold that dismisses points over an estimated height above ground, as they presumably are non-road points. 

        usage: las_road_segmenter: Segment LAS point clouds using road data from OSM.
               [-h] [-e EPSG] [-f FLOOR_THRESHOLD] [-r ROADS_OUT] [-c CACHE]
               [-d DECIMATION_STEP] [-o OUT] [-s]

        optional arguments:
          -h, --help            show this help message and exit
          -e EPSG, --epsg EPSG  Projection of LAS file (default 4326)
          -f FLOOR_THRESHOLD, --floor_threshold FLOOR_THRESHOLD Filter points above height (default 0.5). < 0 means no filtering
          -r ROADS_OUT, --roads_out ROADS_OUT Output roads JSON file
          -c CACHE, --cache CACHE OSM cache file
          -d DECIMATION_STEP, --decimation_step DECIMATION_STEP Take 1 every N points (default 1)
          -o OUT, --out OUT     Classified and decimated LAS output file (overwriting input as default)
          -s, --show_result     Show resulting point cloud

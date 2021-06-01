import argparse
import re

from pyproj import Transformer


def latlon_to_metres(lat, lon):
    """
    Convert lat/lon coord into "geoscience australia lambert" metres.

    Working in a metres coordinate system makes it much easier to print out
    the distance of a route.
    """
    # epsg 4326 is lat/lon
    # epsg 3112 is "geoscience australia lambert" which uses metres as its unit
    transformer = Transformer.from_crs(4326, 3112)
    return transformer.transform(lat, lon)


def latlon(val):
    if not re.match(r"^[-\d\.]+,[-\d\.]+$", val):
        raise ValueError("r{val} should be 12.34,56.78 lat lon pair")
    lat, lon = map(float, val.split(","))

    # switch to epsg 3112 which is used everywhere else
    return latlon_to_metres(lat, lon)


parser = argparse.ArgumentParser()
parser.add_argument("postcode", type=int, help="3xxx postcode to ride in")
parser.add_argument("letter", type=str, help="ride streets beginning with this letter")
parser.add_argument(
    "home", type=latlon, help="lat,lon pair; where to start/finish rides"
)
parser.add_argument(
    "--outdir",
    default="tmp",
    help="write output imagery into subdirs at this path",
)
args = parser.parse_args()

import pyspark.sql.functions as F

from datetime import *
from pyspark.sql.types import *
import math
from geopy import distance

from constants import *
from helpers import *


@F.udf
def filter_coords(row):
    """filter out all station > 10 km to ZH HB"""
    lon = float(row[0])
    lat = float(row[1])
    distance_to_hb = distance.distance(
        (hb_coords.lon, hb_coords.lat), (lon, lat)).km <= 10

    return distance_to_hb


@F.udf
def filter_walking(row):
    """Find distance between two stations"""

    coords_1 = (row.lon, row.lat)
    coords_2 = (row.lon_2, row.lat_2)
    distance_walk = distance.distance(coords_1, coords_2).km

    return distance_walk





@F.udf(IntegerType())
def number_of_stops(x):
    """Compute the number of stop for every trip"""

    nb_stops = len(x)

    try:
        nb_stops = x[1:].index(x[0]) + 1

    except:
        pass

    return nb_stops


udf_shorten = F.udf(lambda x: x[0][:x.nb_stops], ArrayType(StringType()))


@F.udf(ArrayType(ArrayType(StringType())))
def edge_split(stops):
    """Create individual edges from stop names"""

    return [(stops[i], stops[i + 1]) for i in range(len(stops) - 1)]


compute_trip_time = F.udf(lambda struct: [
                          x - y for x, y in zip(struct[0], struct[1])], ArrayType(DoubleType()))


@F.udf(ArrayType(StringType()))
def extract_time(array):
    """Extract the time from a date"""

    ret = []

    for elem in array:
        date_obj = datetime.fromtimestamp(elem)
        ret.append('{:02d}:{:02d}:{:02d}'.format(
            date_obj.hour, date_obj.minute, date_obj.second))

    return ret


types = StructType([StructField('S_departure_time', ArrayType(StringType())),
                    StructField('S_arrival_time', ArrayType(StringType()))])


@F.udf(types)
def get_unique_time(struct_):
    """Build multiedges (one edge per departure time)"""

    # Sort unique departure times
    departure_times = sorted(list(set(struct_.S_departure_time)))

    # Sort unique arrival times
    arrival_times = sorted(list(set(struct_.S_arrival_time)))
    arrival_times_ordered = []

    # Match pairs of Departure/arrival time
    for idx, departure_time in enumerate(departure_times):
        stop = False
        idx_arrival = idx

        while(not stop and idx_arrival < len(arrival_times)):
            arrival_time = arrival_times[idx_arrival]

            if departure_time < arrival_time:
                arrival_times_ordered.append(arrival_time)
                stop = True

            idx_arrival += 1

    return departure_times, arrival_times_ordered


udf_sec = F.udf(get_sec, DoubleType())


@F.udf(DoubleType())
def get_lambda_mle(values_list):
    """compute statistics for exponential distribution"""

    list_true_samples = [x for x in values_list if x != -1]

    if len(list_true_samples) > 0:
        mean = sum(list_true_samples) / len(list_true_samples)

    else:

        return math.inf

    if mean == 0:

        return math.inf

    return 1. / mean


pxy_udf = F.udf(lambda x: pxy_2(x[0], x[1], x[2]), DoubleType())

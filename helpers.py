import routing_algo as route
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from datetime import *
from pyspark.sql.types import *
import math
import re
from pyspark.sql.types import Row
from pyproj import Proj, transform
import branca.colormap as cm
import folium
from constants import *
from udfs import *

@F.udf(DoubleType())
def to_timestamp(s):
    """Convert string date to timestamp"""

    try:
        ret = datetime.strptime(s, '%d.%m.%Y %H:%M:%S').timestamp()

    except:
        ret = datetime.strptime(s, '%d.%m.%Y %H:%M').timestamp()

    return ret



##################################################  Building nx graph ##############################################################################


def add_to_graph_edge(edge_data):
    """ For building multigraph edges """

    cols = edge_data.index
    edge_attribs = {x: edge_data[x]
                    for x in cols if type(edge_data[x]) != list}
    edge_attribs['weight'] = 0
    G.add_edge(edge_data.departure_station,
               edge_data.arrival_station, **edge_attribs)


def add_to_graph_node(node_data):
    """ For building multigraph nodes """
    cols = node_data.index
    node_attribs = {x: node_data[x] for x in cols}
    G.add_node(node_attribs['name'], **node_attribs)

##################################################  Map visualization ########################################################


def get_path_edges_ids(dst, path):
    """ Get edges from djikstra outputs """

    edges = []

    for i in range(1, len(path)):
        p_i = path[i]

        if i < len(path) - 1:
            edges.append((p_i[0], path[i + 1][0], p_i[1]))

        else:
            edges.append((p_i[0], dst, p_i[1]))

    return edges


def LongLat_to_EN(long, lat):
    """ Function to convert GPS coordinates """

    try:
        easting, northing = transform(
            Proj(init='epsg:4326'), Proj(init='epsg:3857'), long, lat)

        return easting, northing

    except:

        return None, None


def extract_data(G, edge, confidence):
    """ Extract data from one edge """

    record = {}
    edge = route.index_graph(G, *edge)

    station_x = edge['departure_station']
    station_y = edge['arrival_station']
    node_x = G.node[station_x]
    node_y = G.node[station_y]

    lon_x, lat_x = LongLat_to_EN(node_x['lon'], node_x['lat'])
    lon_y, lat_y = LongLat_to_EN(node_y['lon'], node_y['lat'])

    record['name_x'] = station_x
    record['name_y'] = station_y
    record['lon_x'] = lon_x
    record['lat_x'] = lat_x
    record['lon_y'] = lon_y
    record['lat_y'] = lat_y
    record['type'] = edge['type']
    record['confidence'] = confidence * 100
    record['time_x'] = edge['S_departure_time']
    record['time_x'] = edge['S_departure_time']
    record['time_y'] = edge['S_arrival_time']
    record['route'] = edge['trip_id']

    return record


def check_valid(selected_cutoff):

    try:
        h, m, s = selected_cutoff.split(':')

        return True

    except:

        return False


def interleave(serie_a, serie_b):
    """ Interleave two pd series """

    ret = np.empty((len(serie_a) + len(serie_b),), dtype=serie_a.dtype)
    ret[0::2] = serie_a.values
    ret[1::2] = serie_b.values

    return ret


def show_plan(json_data, durations, total_uncertainty, total_duration):
    """ Generate html to display optimal path """
    print(total_duration)
    html_returned = '<div id=path>'
    html_returned += '<div class=path_header><center>Total time: {:.01f} mn   &emsp;Confidence: {:.01f}%<center></div><br>'.format(total_duration / 60,
                                                                                                                                   total_uncertainty)
    for idx, hop in enumerate(json_data):
        tmp_string = '''
        <div class="table-header" >
        #{idx}&emsp;Type: {type_}&emsp;&emsp;Duration: {duration:.01f} mn&emsp;&emsp;Confidence: {confidence:.01f}%
        </div>
        '''
        tmp_string = tmp_string.format(type_=hop[0]['type'],
                                       duration=durations[idx], confidence=hop[0]['confidence'], idx=idx + 1)

        html_returned += tmp_string

        table_path = '<table width=90%>'

        for station in hop:
            table_path += '''
              <tr>
        <td>{station_name}</td>
        <td >{time_arrival}</td> </tr>'''.format(station_name=station['name'], time_arrival=station['time'])
        table_path += '</table>'
        html_returned += table_path
        html_returned += '<br><br>'

    return html_returned + '</div>'


def find_path_sub(G, src, dst, start_time, tresh, n_iter=1, cutoff=None):
    def get_duration(x):
        return get_sec(x.max()) - get_sec(x.min())

    if not check_valid(cutoff):
        cutoff = None

    # Find a path for a combination of parameters
    path, global_confidence, individual_confidence, _ = route.safest_path(G, src, dst, pxy,
                                                                          start_time=start_time,
                                                                          n_iters=n_iter, threshold=tresh, cutoff=cutoff)
    if path is None:

        return None, None, None, None, None

    df_data = []

    for idx, hop in enumerate(path):
        df_data.append(extract_data(G, hop, individual_confidence[idx]))

    columns_join = ["lat", "lon", "name", "time"]
    columns_other = ["confidence", "route", "type"]

    df_data = pd.DataFrame(df_data)
    path_display_data = pd.DataFrame(columns=columns_join + columns_other)

    for idx, row in df_data.iterrows():
        mini_df = [{}, {}]

        for column in columns_join:
            mini_df[0][column] = row[column + '_x']
            mini_df[1][column] = row[column + '_y']

        for column in columns_other:

            for i in [0, 1]:
                mini_df[i][column] = row[column]

        mini_df = pd.DataFrame(mini_df)
        path_display_data = path_display_data.append(
            mini_df, ignore_index=True)

    groups = [grp for idx, grp in df_data.groupby('route', sort=False)]
    grp_by = path_display_data.groupby('route', sort=False)

    groups_display = [grp.to_dict(orient='records') for idx, grp in grp_by]
    durations = grp_by.agg({'time': get_duration})['time'].values / 60

    total_time = get_sec(path_display_data['time'].max(
    )) - get_sec(path_display_data['time'].min())

    return groups, groups_display, durations, global_confidence * 100, total_time

##################################################  Isochrone map ########################################################
def add_circle(_map, coords, probability,
               time_left_in_seconds, popup_data, color_bar):
    # Plot a circle on the map corresponding to the walkable distance from
    walking_radius = time_left_in_seconds * AVERAGE_WALKING_SPEED_PER_SECOND

    folium.Circle(
        coords,
        walking_radius,
        fill=True,
        fill_color=color_bar(probability),
        fill_opacity=0.2,
        stroke=False,
        fill_rule='nonzero',
        popup="Time of arrival: {}<br\>Probability : {:.3f}<br\>Remaining Time: {} mins"
        .format(popup_data['arrival_time'], probability, (time_left_in_seconds // 60))
    ).add_to(_map)


def plot_isochrone(source, source_coords, stations_data, cutoff, threshold, G):
    # Plot the isochrone map
    color_bar = cm.StepColormap(
        ['blue','red','yellow','green'],
        index=[0, 0.25, 0.50, 0.75, 1]
    )

    color_bar.caption = 'Trip Uncertainty'

    m = folium.Map(source_coords, zoom_start=13, tiles='Stamen toner')
    m.add_child(color_bar)
    popup_data = {}
    for station_data in stations_data:
        station_name = station_data[0]
        quality = station_data[2]
        arrival_time = station_data[3]
        lat, long = G.node[station_name]['lat'], G.node[station_name]['lon']
        popup_data['station_name'] = station_name
        popup_data['arrival_time'] = arrival_time
        add_circle(m, (lat, long), quality, time_diff(
            cutoff, arrival_time), popup_data, color_bar)
    return m


def get_isochrone_map(source, source_coords, start_time, cutoff_time, threshold, G):
    # Computes the shortest path to all stations and return the isochrone map
    paths_iterator = route.get_isochrone(
        G, source, pxy, start_time,
        cutoff_time, n_iters=5, threshold=threshold)

    res = []
    for dst, edges, global_uncertainty, distance in paths_iterator:
        res.append((dst, edges, global_uncertainty, distance))

    return plot_isochrone(source, source_coords, res, cutoff_time, threshold, G)

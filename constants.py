from  dateutil import parser

distance_walk_thresh = 0.6
speed_walk = 1/780 #Km/s
null_timestamp = parser.parse('1970-01-01 01:00:00').timestamp()
null_date = '01.01.1970 01:00:00'
colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#ffffbf','#e6f598','#abdda4','#66c2a5','#3288bd']
default_text_div = '<div id=title_paths_div><h3>Path details</h3></id>'
AVERAGE_WALKING_SPEED_PER_SECOND = 1.28205 #  in meters per second
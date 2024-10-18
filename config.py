import matplotlib.pyplot as plt

groupby_key = [
    'ACADEMIC_PROGRAM_DESC',
    'ACADEMIC_PLAN',
    'ACADEMIC_LEVEL_TERM_START',
    'FIN_AID_FED_RES', 
    'UVA_ACCESS', 
    'REPORT_CODE',
    'Need based'
]

default_values = {
    'ACADEMIC_PROGRAM_DESC': 'Program',
    'ACADEMIC_PLAN': 'Academic Plan',
    'ACADEMIC_LEVEL_TERM_START': 'Level',
    'FIN_AID_FED_RES': 'Residency',
    'UVA_ACCESS': 'Access',
    'REPORT_CODE': 'Report Code',
    'Need based': 'Need Based'
}

# colors
prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
obs_color = next(prop_cycle)["color"]
pred_color = next(prop_cycle)["color"]

# should be 127.0.0.1 for debugging using local browser
# should be 0.0.0.0 for production
# host = '127.0.0.1' # '0.0.0.0'
# port = 8050
data_root = 'datawarehouse/'

time_column = 'AID_YEAR'
target = 'OFFER_BALANCE'
alpha = 0.05
confidence = int((1 - alpha) * 100)

# configurations
seq_len = 3
line_width = 5
pred_len = 6
marker_size = 10
fontsize = 12

# logging will be saved in data_root folder
log_file = 'myapp.log'

# colors
background_color = "lightblue"
uva_color = '#232d4b'
uva_header = '#dadada'

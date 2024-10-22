# https://dash.plotly.com/tutorial
# https://dash.plotly.com/basic-callbacks

# Import packages
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, callback, Output, Input, State
from util.utils import *
from util.plotter import *
from config import *
import argparse, json, dash_auth, warnings, logging, os
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.info('Started')

# data input
df = pd.read_csv(data_root + 'Merged.csv')
for col in ['FUNDED_PARTY', 'TOTAL_PARTY']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

# Initialize the app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__, external_stylesheets=external_stylesheets, 
    description='Financial Aid'
)
server = app.server

if os.path.exists(data_root + 'login.json'):
    login_info = json.load(open(data_root + 'login.json'))
    auth = dash_auth.BasicAuth(
        app, login_info['credentials'], 
        secret_key = login_info['secret_key']
    )
else:
    logging.info('Login file not found')

# remove ' Undergraduate' from ACADEMIC_PROGRAM_DESC
for value in [' Undergraduate', ' Undergrad']:
    df['ACADEMIC_PROGRAM_DESC'] = df['ACADEMIC_PROGRAM_DESC'].apply(lambda x: x.replace(value, ''))

program_ids = get_categories(df, 'ACADEMIC_PROGRAM_DESC', default_values['ACADEMIC_PROGRAM_DESC'])
academic_plans = get_categories(df, 'ACADEMIC_PLAN', default_values['ACADEMIC_PLAN'])
report_categories = get_categories(df, 'UVA_ACCESS', default_values['UVA_ACCESS'])
report_codes = get_categories(df, 'REPORT_CODE', default_values['REPORT_CODE'])
need_based = get_categories(df, 'Need based', default_values['Need based'])
residency = get_categories(df, 'FIN_AID_FED_RES', default_values['FIN_AID_FED_RES'])

# get_categories(df, 'ACADEMIC_LEVEL_TERM_START', default_values['ACADEMIC_LEVEL_TERM_START'])
academic_levels = [default_values['ACADEMIC_LEVEL_TERM_START']] + [
    'Level One', 'Level Two', 'Level Three', 'Level Four'
]

summed = pd.DataFrame(columns=[
   'AID_YEAR', 'OFFER_BALANCE','PREDICTED_MEAN',
    f'{confidence}% CI - LOWER', 
    f'{confidence}% CI - UPPER', 
    'FUNDED_PARTY', 'TOTAL_PARTY'
])

app.layout = get_layout(
    factors=[
        program_ids, academic_levels, academic_plans, 
        report_categories, report_codes, need_based, residency
    ], factor_labels=[
        'program-desc', 'academic-level', 'academic-plan', 
        'report-category', 'report-code', 'need-based', 'residency'
    ], summed=summed
)

# update list of academic plans based on selected program description
@callback(
    Output('dropdown-academic-plan', 'options'),
    Input('dropdown-program-desc', 'value')
)
def set_academic_plans(program):
    if program == default_values['ACADEMIC_PROGRAM_DESC']:
        return get_categories(df, 'ACADEMIC_PLAN', default_values['ACADEMIC_PLAN'])
    else:
        temp = df[df['ACADEMIC_PROGRAM_DESC']==program]
        new_academic_plans = get_categories(temp, 'ACADEMIC_PLAN', default_values['ACADEMIC_PLAN'])
        
        # if there are only two values, then remove the default value
        if len(new_academic_plans) == 2:
            new_academic_plans = new_academic_plans[1:]
    # print(program, new_academic_plans)
    return new_academic_plans

# update the prediction plot and party count based on the selection
callbacks = [
    Output("time-series-chart", "figure"), 
    Output("count-chart", "figure"), 
    Output('table', 'data')
] + [Input(f"dropdown-{value}", "value") for value in [
        'academic-level', 'program-desc', 'academic-plan', 
        'report-category', 'report-code', 'need-based', 'residency']
] + [Input('radio-time-series', 'value'), Input('radio-count-series', 'value')
] + [Input('constraint-table', 'data_timestamp'),
    State('constraint-table', 'data'),
    # State('constraint-table', 'columns')
]
@app.callback(callbacks)
def update_data(
    level, program_desc, academic_plan, 
    report_category, report_code, need_based, residency,
    radio_time, radio_count,
    timestamp, constraint_data
):
    dis = filter_disbursement(
        df, level, program_desc, academic_plan, 
        report_category, report_code, need_based,
        residency, logger
    )
    summed = dis.groupby(time_column)[target].sum().reset_index()
    years = summed[time_column].values
    
    count_df = dis.groupby(
        time_column
    )['FUNDED_PARTY'].sum().reset_index()
    
    count_df['TOTAL_PARTY'] = dis.groupby(
        time_column
    )['TOTAL_PARTY'].sum().values
    party_fig = draw_party_fig(years, count_df, radio_count)
    
    predictions = {
        'PREDICTED_MEAN': [], time_column:[],
        f'{confidence}% CI - LOWER':[], 
        f'{confidence}% CI - UPPER':[]
    }
    
    predictions = predict(summed, predictions)
    if len(constraint_data) > 0:
        predictions = limit_predictions(
            predictions, [program_desc, level, academic_plan, 
            report_category, report_code, need_based, residency],
            constraint_data
        )
        
    predictions = autoregressive(summed, predictions, pred_len)
    if len(constraint_data) > 0:
        predictions = limit_predictions(
            predictions, [program_desc, level, academic_plan, 
            report_category, report_code, need_based, residency],
            constraint_data
        ) 
    predictions = pd.DataFrame(predictions)
    
    fig = draw_main_fig(summed, predictions, radio_time)
    
    summed = summed.merge(count_df, on=time_column)
    summed = summed.merge(predictions, on=time_column, how='outer')
    summed.sort_values(by=time_column, ascending=False, inplace=True)
    
    for col in summed.columns:
        if col in ['AID_YEAR']: continue
        index = ~summed[col].isna()
        # summed.loc[index, col] = summed.loc[index, col].apply(numerize, args=(1,))
        if summed[col].dtype == 'float64':
            # summed.loc[index, col] = summed.loc[index, col].apply('{:,.2f}'.format)
            summed.loc[index, col] = summed.loc[index, col].apply(numerize, args=(1, ))
    
    return fig, party_fig, summed.round(0).to_dict('records')

# add rows to the constraint table
@callback(
    Output('constraint-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('constraint-table', 'data'),
    State('constraint-table', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks == 0: return rows
    
    # 'Program', 'Level', 'Plan', 'Access', 'Report',
    # 'Need Based', 'Residency','Start', 'End', 'Amount'
    column_mapping = {
        'Program': 'ACADEMIC_PROGRAM_DESC',
        'Level': 'ACADEMIC_LEVEL_TERM_START',
        'Plan': 'ACADEMIC_PLAN',
        'Access': 'UVA_ACCESS',
        'Report': 'REPORT_CODE',
        'Need Based': 'Need based',
        'Residency': 'FIN_AID_FED_RES'
    }
    
    new_row = {}
    for col in columns:
        column = col['id']
        if column in column_mapping:
            new_row[column] = default_values[column_mapping[column]]
        else: new_row[column] = ''
    
    if rows is None: rows = []
    rows.append(new_row)
    return rows

# update the table only if valid input and then cache it
# @callback(
#     # Output('constraint-table', 'data'),
#     Input('constraint-table', 'data_timestamp'),
#     State('constraint-table', 'data'),
#     State('constraint-table', 'columns'))
# def update_table(data_timestamp, rows, columns):
#     # convert to eastern timezone
#     date = pd.to_datetime(data_timestamp, unit='ms', utc=True, dayfirst=True).tz_convert('US/Eastern')
    # print(data_timestamp, date)
    # print(rows)
    # return rows

    
# Run the app
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Financial Aid Forecasting App', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--host', type=str, default='127.0.0.1', 
        help='Host address. Should be 0.0.0.0 if running on AWS, 127.0.0.1 if running locally.'
    )
    parser.add_argument('--port', type=int, default=8050) # default 80 for http, 443 for https
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()
    
    if args.log_file is not None:
        logging.basicConfig(
            filename=data_root + args.log_file, level=logging.INFO
        )

    app.run(host=args.host, port=args.port, debug=True)

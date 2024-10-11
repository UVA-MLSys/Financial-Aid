# https://dash.plotly.com/tutorial
# https://dash.plotly.com/basic-callbacks

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numerize.numerize import numerize
warnings.filterwarnings('ignore')

# logging helps find errors
import logging
logger = logging.getLogger(__name__)
data_root = 'datawarehouse/'
logging.basicConfig(
    filename=data_root + 'myapp.log', 
    level=logging.INFO
)
logger.info('Started')

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation 
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]

groupby_key = [
    'ACADEMIC_PROGRAM_DESC',
    'ACADEMIC_PLAN',
    'ACADEMIC_LEVEL_TERM_START',
    'FIN_AID_FED_RES', 
    'UVA_ACCESS', 
    'REPORT_CODE',
    'Need based'
]

# colors
prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
obs_color = next(prop_cycle)["color"]
pred_color = next(prop_cycle)["color"]

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

# data input
df = pd.read_csv(f'{data_root}/Merged.csv')

# Initialize the app
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# remove ' Undergraduate' from ACADEMIC_PROGRAM_DESC
for value in [' Undergraduate', ' Undergrad']:
    df['ACADEMIC_PROGRAM_DESC'] = df['ACADEMIC_PROGRAM_DESC'].apply(lambda x: x.replace(value, ''))
    
def get_categories(df, col, default_value):
    unique_values = list(sorted(df[col].unique()))
    # default value will work as Total
    unique_values = [x for x in unique_values if x != 'Total']
    
    return [default_value] + unique_values

default_values = {
    'ACADEMIC_PROGRAM_DESC': 'Program',
    'ACADEMIC_PLAN': 'Academic Plan',
    'ACADEMIC_LEVEL_TERM_START': 'Level',
    'FIN_AID_FED_RES': 'Residency',
    'UVA_ACCESS': 'Access',
    'REPORT_CODE': 'Report Code',
    'Need based': 'Need Based'
}

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
    
table_df = df.copy()

summed = pd.DataFrame(columns=[
   'AID_YEAR', 'OFFER_BALANCE','PREDICTED_MEAN',
    f'{confidence}% CI - LOWER', 
    f'{confidence}% CI - UPPER', 
    'FUNDED_PARTY', 'TOTAL_PARTY'
])

app.layout = html.Div([
    html.H3('Financial Aid Analysis (Undergrad)'),
    dbc.Row([
        dbc.Col([
            # html.H3("Program:"),
            dcc.Dropdown(
                id=f"dropdown-{column}",
                options = values,
                value=values[0],
                clearable=True,
                searchable=True,persistence=True,
                persistence_type='local'
            ),
        ])
        
        for column, values in zip(
            ('program-desc', 'academic-level', 'academic-plan', 'report-category', 'report-code', 'need-based', 'residency'),
            (program_ids, academic_levels, academic_plans, report_categories, report_codes, need_based, residency)
        )
    ], style={'margin':'auto', 'padding':'5px'}),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="time-series-chart"), 
            dcc.RadioItems(
                options=['Annotation On', 'Annotation Off'], 
                value='Annotation On', id='radio-time-series', 
                inline=True, 
                # https://stackoverflow.com/questions/75692815/increase-space-between-text-and-icon-in-dash-checklist
                labelStyle= {"margin":"1rem"} 
            ),
            ], width=7),
        dbc.Col([
            dcc.Graph(id="count-chart"),
            dcc.RadioItems(
                options=['Annotation On', 'Annotation Off'], 
                value='Annotation On', id='radio-count-series', 
                inline=True, labelStyle= {"margin":"1rem"} #,style = {'display': 'flex'}
            )
            ], width=5)
        ]
    ),
    dbc.Row(
        dbc.Col([
            html.H3("Financial Aid Table:"),
            # https://dash.plotly.com/datatable/style#styling-editable-columns
            dash_table.DataTable(
                id='table', columns=[{'id':c, 'name':c} for c in summed.columns],
                data=summed.to_dict('records'), page_size=10, 
                style_header={
                    'backgroundColor': 'rgb(210, 210, 210)',
                    'fontWeight': 'bold'
                },
            )
        ]), style={'width': '75vw', 'margin':'auto'}    
    )
], style={'width': '95vw', 'margin':'auto', 'text-align': 'center'})

def filter_disbursement(
    level, program_desc, academic_plan, 
    report_category, report_code, need_based, residency
):
    dis = df.copy()
    logger.info(f'Initial size: {len(dis)}')
        
    for (var, column) in [
        (program_desc,'ACADEMIC_PROGRAM_DESC'),
        (level, 'ACADEMIC_LEVEL_TERM_START'),
        (academic_plan, 'ACADEMIC_PLAN'),
        (report_category, 'UVA_ACCESS'),
        (report_code, 'REPORT_CODE'),
        (need_based, 'Need based'),
        (residency, 'FIN_AID_FED_RES')
    ]:
        
        logger.info(f'Filtering {column} == {var}...')
        # logger.info(f'Unique values: {list(dis[column].unique())}')
        default_value = default_values[column]
        
        if var == default_value or var == 'Total':
            if 'Total' in dis[column].values:  
                dis = dis[dis[column] == 'Total']  
            elif dis[column].nunique() != 1:
                logger.error(f'{column} has {dis[column].nunique()} unique values but does not contain Total.')   
                
        else: 
            if var not in dis[column].values: 
                logger.error(f'Error ! {column} does not contain {var}.')
                logger.info(f'Unique values: {list(dis[column].unique())}')
            dis = dis[dis[column] == var]
            
        if len(dis) == 0:
            if var == default_value: 
                logger.error(f'{column} does not contain Total. Please check data.')
            else:
                logger.error(f'{column} does not contain {var}. Please check data.')
            break
    
        logger.info(f'Reduced size: {len(dis)}.')
    
    logger.info('Done\n')
    return dis

def merge(left:pd.DataFrame, right:pd.DataFrame, key:list|str=None, how='inner'):
    common = set(left.columns) & set(right.columns)
    
    # if no key is provided, assume the common columns that 
    # ends with '_ID' or  '_YEAR' are the common keys
    if key is None:
        key = [
            column for column in common 
                if left[column].dtype in [object, str] or column.endswith('_YEAR') or column.endswith('_ID')
        ]
        if len(key) == 0: return None
    
    if type(key) != list: 
        common = [col for col in common if col != key]
    else:
        common = [col for col in common if col not in key]
    
    merged_data = left.merge(right.drop(columns=common), on = key, how=how)
    # logger.info(f'Shape: left {left.shape}, right {right.shape}, merged {merged_data.shape}.\n')
    
    return merged_data

def predict(summed, predictions):
    years = summed[time_column].values
    
    for time_step in range(seq_len, len(years)):
        Y = summed[summed[time_column]<years[time_step]][target].values
        
        model = ARIMA(Y, order=(seq_len, 1, 1))
        model.initialize_approximate_diffuse()
        model_fitted = model.fit() 
        
        result = model_fitted.get_forecast()
        forecast = result.predicted_mean
        forecast = np.round(np.where(forecast<=0, 0, forecast), 0)
        
        predictions['PREDICTED_MEAN'].append(forecast[0])
        predictions[time_column].append(years[time_step])
        
        ci = result.conf_int(alpha=alpha)
        ci = np.round(np.where(ci<=0, 0, ci), 0)
        predictions[f'{confidence}% CI - LOWER'].append(ci[0,0])
        predictions[f'{confidence}% CI - UPPER'].append(ci[0,1])
    
    return predictions

def autoregressive(summed, predictions):
    Y = list(summed[target].values)
    if len(Y) == 0:
        return predictions

    for i in range(0, pred_len):
        model = ARIMA(Y, order=(seq_len, 1, 1))
        model.initialize_approximate_diffuse()
        model_fitted = model.fit() 
        result = model_fitted.get_forecast()
        forecast = result.predicted_mean
        forecast = np.round(np.where(forecast<=0, 0, forecast), 0)
        
        predictions['PREDICTED_MEAN'].append(forecast[0])
        predictions[time_column].append(summed[time_column].values[-1]+i+1)
        
        ci = result.conf_int(alpha=alpha)
        ci = np.round(np.where(ci<=0, 0, ci), 0)
        predictions[f'{confidence}% CI - LOWER'].append(ci[0,0])
        predictions[f'{confidence}% CI - UPPER'].append(ci[0,1])
        Y.append(forecast[0])
        
    return predictions

def draw_party_fig(years, data, annotate):
    party_fig = go.Figure()
    
    if annotate == 'Annotation On':
        party_fig.add_trace(go.Scatter(
            name="Funded",
            mode="markers+lines+text", 
            x=years, y=data['FUNDED_PARTY'],
            text=data['FUNDED_PARTY'].apply(numerize, args=(1, )),
            # texttemplate = "%{y}", # "(%{x:.4g}, %{y:.4g})",
            textposition=improve_text_position(years), # 'bottom center',
            marker_symbol="circle", 
            line_color=obs_color,
            line=dict(width=line_width)
        ))
        
        party_fig.add_trace(go.Scatter(
            name="All",
            mode="markers+lines+text", 
            x=years, y=data['TOTAL_PARTY'],
            text=data['TOTAL_PARTY'].apply(numerize, args=(1, )),
            # texttemplate = "%{y}", # "(%{x:.4g}, %{y:.4g})",
            textposition= improve_text_position(years), # 'bottom center',
            marker_symbol="star", 
            line_color=obs_color,
            line=dict(width=line_width)
        ))
    else:
        party_fig.add_trace(go.Scatter(
            name="Funded",
            mode="markers+lines+text", 
            x=years, y=data['FUNDED_PARTY'],
            marker_symbol="circle", 
            line_color=obs_color,
            line=dict(width=line_width)
        ))
        
        party_fig.add_trace(go.Scatter(
            name="All",
            mode="markers+lines+text", 
            x=years, y=data['TOTAL_PARTY'],
            marker_symbol="star", 
            line_color=obs_color,
            line=dict(width=line_width)
        ))

    party_fig.update_layout(
        font=dict(
            # family="Courier New",
            size=fontsize,  # Set the font size here
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        title='Students per aid year',
        xaxis_title='Aid year',
        yaxis_title='Count',
        # hovermode='closest' # # Update the layout to show the coordinates
    )
    party_fig.update_xaxes(showgrid=True, ticklabelmode="period")
    party_fig.update_traces(marker_size=marker_size)
    # party_fig.update_yaxes(range=[0, None])
    
    return party_fig

def draw_main_fig(summed, predictions, annotate):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Ground Truth",
        mode="markers+text+lines", # markers+text+lines
        x=summed[time_column], y=summed[target],
        # text=summed[target].apply(numerize, args=(1, )),
        # texttemplate = "%{y}", # "(%{x:.4g}, %{y:.4g})",
        # textposition= improve_text_position(years),# 'bottom center',
        marker_symbol="circle", 
        line_color=obs_color,
        line=dict(width=line_width)
    ))

    fig.add_trace(go.Scatter(
        name="Prediction",
        mode="markers+lines+text", 
        x=predictions[time_column], 
        y=predictions['PREDICTED_MEAN'],
        # text=predictions['PREDICTED_MEAN'].apply(numerize),
        # texttemplate = "%{y:.3g}", 
        # textposition= 'bottom center',
        line=dict(width=line_width),
        line_color=pred_color
    ))
    
    if annotate == 'Annotation On':
        # https://plotly.com/python/text-and-annotations/
        for (x, y) in zip(summed[time_column], summed[target].values):
            fig.add_annotation(
                x=x, y=y, xref="x",
                yref="y", text=numerize(y, 1),
                showarrow=True, arrowhead=1, 
                arrowsize=2,arrowwidth=1,
            )
            
        for (x, y) in zip(
            predictions[time_column], 
            predictions['PREDICTED_MEAN'].values
        ):
            fig.add_annotation(
                x=x, y=y, xref="x",
                yref="y", text=numerize(y, 1),
                showarrow=True,yanchor='bottom',
                arrowhead=1, arrowsize=2,arrowwidth=1,
                ax=20,ay=40
            )
    
    # https://plotly.com/python/filled-area-plots/#filled-area-plot-in-dash
    fig.add_trace(
        go.Scatter(
            name=f'{confidence}% CI - LOWER',
            x=predictions[time_column], 
            y=predictions[f'{confidence}% CI - LOWER'], line=dict(width=1),
            line_color=pred_color, 
            opacity=0.1, mode="lines",
        )
    )
    
    fig.add_trace(
        go.Scatter(
            name=f'{confidence}% CI - UPPER',
            x=predictions[time_column], 
            y=predictions[f'{confidence}% CI - UPPER'],
            fill='tonexty', line=dict(width=1), # fillcolor='#ff7700',
            line_color=pred_color, mode="lines",
            opacity=0.1
        )
    )
    
    # fig.update_traces(textposition='top center')
    
    fig.update_layout(
        font=dict(
            # family="Courier New",
            size=fontsize,  # Set the font size here
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        title='Model: ARIMA. Input: Past offer balance as selected.',
        xaxis_title='Aid year',
        yaxis_title='Offer balance'
    )
    
    fig.update_traces(marker_size=marker_size)
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    # fig.update_yaxes(range=[0, None])
    # fig.update_layout(yaxis_range=[0, max(summed[target].max(), summed[f'Predicted_{target}'].max())*1.2])
    return fig

@app.callback(
    Output("time-series-chart", "figure"), 
    Output("count-chart", "figure"), 
    Output('table', 'data'),
    Input("dropdown-academic-level", "value"),
    Input("dropdown-program-desc", "value"),
    Input("dropdown-academic-plan", "value"),
    Input("dropdown-report-category", "value"),
    Input("dropdown-report-code", "value"),
    Input("dropdown-need-based", "value"),
    Input("dropdown-residency", "value"),
    Input('radio-time-series', 'value'),
    Input('radio-count-series', 'value')
)
def update_data(
    level, program_desc, academic_plan, 
    report_category, report_code, need_based, residency,
    radio_time, radio_count
):
    dis = filter_disbursement(
        level, program_desc, academic_plan, 
        report_category, report_code, need_based,
        residency
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
        'PREDICTED_MEAN': [],
        time_column:[],
        f'{confidence}% CI - LOWER':[], 
        f'{confidence}% CI - UPPER':[]
    }
    
    predictions = predict(summed, predictions)
    predictions = autoregressive(summed, predictions) 
    
    predictions = pd.DataFrame(predictions)   
    fig = draw_main_fig(summed, predictions, radio_time)
    
    summed = summed.merge(count_df, on=time_column)
    summed = summed.merge(predictions, on=time_column, how='outer')
    summed.sort_values(by=time_column, ascending=False, inplace=True)
    
    for col in summed.columns:
        if col in ['AID_YEAR']: continue
        index = ~summed[col].isna()
        # summed.loc[index, col] = summed.loc[index, col].apply(numerize, args=(1,))
        summed.loc[index, col] = summed.loc[index, col].apply('{:,.2f}'.format)
    
    return fig, party_fig, summed.round(0).to_dict('records')

    
# Run the app
if __name__ == '__main__':
    app.run(
        host="127.0.0.1", port=8050,
        debug=True
    )
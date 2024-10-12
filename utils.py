import pandas as pd
import numpy as np
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from numerize.numerize import numerize
import dash_bootstrap_components as dbc
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from config import *

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation 
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]


def merge(left:pd.DataFrame, right:pd.DataFrame, key=None, how='inner'):
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


def filter_disbursement(
    df, level, program_desc, academic_plan, 
    report_category, report_code, need_based, 
    residency, logger
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


def draw_party_fig(years, data, annotate):
    party_fig = go.Figure()
    
    for (name, column) in [('Funded', 'FUNDED_PARTY'), ('All', 'TOTAL_PARTY')]:
        if annotate == 'Annotation On':
            party_fig.add_trace(go.Scatter(
                name=name,
                mode="markers+lines+text", 
                x=years, y=data[column],
                text=data[column].apply(numerize, args=(1, )),
                # texttemplate = "%{y}", # "(%{x:.4g}, %{y:.4g})",
                textposition=improve_text_position(years), # 'bottom center',
                marker_symbol="circle", 
                line_color=obs_color,
                line=dict(width=line_width)
            ))
        else:
            party_fig.add_trace(go.Scatter(
                name=name,
                mode="markers+lines", 
                x=years, y=data[column],
                marker_symbol="circle", 
                line_color=obs_color,
                line=dict(width=line_width)
            ))

    party_fig.update_layout(
        font=dict(
            # family="Courier New",
            size=fontsize,  # Set the font size here
        ),
        legend=dict(
            orientation="h",yanchor="bottom",y=1.02,
            xanchor="center", x=0.5
        ),
        title='Students per aid year',
        xaxis_title='Aid year', yaxis_title='Count',
        # hovermode='closest' # # Update the layout to show the coordinates
    )
    party_fig.update_xaxes(showgrid=True, ticklabelmode="period")
    party_fig.update_traces(marker_size=marker_size)
    # party_fig.update_yaxes(range=[0, None])
    
    return party_fig

def get_layout(factors, factor_labels, summed):
    return html.Div([
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
        for column, values in zip(factor_labels, factors)
    ], style={'margin':'auto', 'padding':'5px'}),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id=graph_id), 
            dcc.RadioItems(
                options=['Annotation On', 'Annotation Off'], 
                value='Annotation On', id=radio_id, 
                inline=True, 
                # https://stackoverflow.com/questions/75692815/increase-space-between-text-and-icon-in-dash-checklist
                labelStyle= {"margin":"1rem"} 
            ),], width=width)
        
        for graph_id, radio_id, width in [
            ('time-series-chart', 'radio-time-series', 7), 
            ('count-chart', 'radio-count-series', 5)]
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
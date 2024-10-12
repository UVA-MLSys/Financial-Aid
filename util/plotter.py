from config import *
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from numerize.numerize import numerize
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation 
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]

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
        title='<b>Enrolled Students</b>',
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
                    options = values,value = values[0],
                    clearable=True, searchable=True,
                    # persistence=True, persistence_type='local'
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
            orientation="h", yanchor="bottom",
            y=1.02, xanchor="center", x=0.5
        ),
        title='<b>Model: ARIMA</b>',
        xaxis_title='Aid year',
        yaxis_title='Offer balance'
    )
    
    fig.update_traces(marker_size=marker_size)
    fig.update_xaxes(showgrid=True, ticklabelmode="period")
    # fig.update_yaxes(range=[0, None])
    # fig.update_layout(yaxis_range=[0, max(summed[target].max(), summed[f'Predicted_{target}'].max())*1.2])
    return fig
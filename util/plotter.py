from config import *
from dash import html, dash_table, dcc
from numerize.numerize import numerize
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import os, pandas as pd

def get_constraints(filename='constraints.csv'):
    if os.path.exists(data_root + filename):
        constraints = pd.read_csv(data_root + filename)
        # constraints['amount'] = constraints['amount'].apply(numerize, args=(1, ))
    else:
        # print('No constraints.csv found. Creating a new one...')
        constraints = pd.DataFrame(
            columns=[
                'Program', 'Level', 'Plan', 'Access', 'Report',
                'Need Based', 'Residency','Start', 'End', 'Amount']
        )
    return constraints

def get_contraints_table(constraints, factors):
    dash_columns = []
    for index, c in enumerate(constraints.columns):
        if c not in ['Start', 'End', 'Amount']: 
            dash_columns.append({'id':c, 'name':c, 'presentation': 'dropdown'})
        else: dash_columns.append({'id':c, 'name':c, 'type': 'numeric'})
    
    constraints_table = dbc.Col([
        html.H2("Policy Table", style={'fontWeight':'bold', 'textAlign':'center'}),
        dash_table.DataTable(
            id='constraint-table', columns=dash_columns,
            data=constraints.to_dict('records'), page_size=8, 
            style_header=style_header, 
            editable=True, row_deletable=True, 
            # row_selectable='multi',
            export_format='csv', is_focused=True,
            # style_table={'overflowX': 'scroll'},
            # https://dash.plotly.com/datatable/dropdowns
            dropdown={
                col: {
                    'options': [
                        {'label':i, 'value':i} for i in factors[factor_index]
                    ]
                }
                for factor_index, col in enumerate(constraints.columns[:len(factors)])
            }
        ),
        # html.Div(id='dropdown_per_row_container'),
        html.H2(),
        html.Button('Add Row', id='editing-rows-button', n_clicks=0),
    ])
    
    return constraints_table

def get_bottom_navbar():
    return [
        dbc.Row([
            dbc.Col(
                html.Img(src='./assets/uva-logo-footer-white.png', height="90px", style={'paddingLeft':padding})
            ),
            dbc.Col(
                [dbc.Row(row) for row in ["Student Financial Services",
                        "1001 North Emmet Street (map)",
                        "P. O. Box 400204",
                        "Charlottesville, VA 22904-420"]]
            ),
            dbc.Col(
                [dbc.Row(row) for row in [
                    'PHONE: 434-982-6000',
                    "EMAIL: sfs@virginia.edu",
                    html.Br(),
                    html.A(
                        "Contact Us", href='https://sfs.virginia.edu/contact-us', 
                        style={'color':'white', 'padding':'0px'}
                    ), html.Br(),
                    html.A(
                        "Staff and Faculty Resources", href='https://sfs.virginia.edu/sfresources', 
                        style={'color':'white', 'padding':'0px'}
                    )
                ]]
            ),
            dbc.Col([
                dbc.Row(row) for row in [
                    'Phone hours',
                    'Weekdays: 10-noon and 1-4',
                    dbc.Col([
                        html.Img(src='./assets/twitter-white.png', style={'block':'inline', 'height':'14px', 'width':'18px'}), 
                        html.A(
                            "X", href='https://twitter.com/UVASFS', 
                            style={'color':'white', 'paddingLeft':'6px'}
                        )
                    ], style={'padding':'0px', 'margin':'3px'}),
                    dbc.Col([
                        html.Img(src='./assets/facebook-white.png', style={'block':'inline', 'height':'16px', 'width':'20px'}), 
                        html.A(
                            "Facebook", href='https://facebook.com/UVASFS', 
                            style={'color':'white', 'paddingLeft':'6px'}
                        )
                    ], style={'padding':'0px', 'margin':'3px'}),
                    dbc.Col([
                        html.Img(src='./assets/instagram-white.png', style={'block':'inline', 'height':'20px', 'width':'20px'}), 
                        html.A(
                            "Instagram", href='https://instagram.com/UVASFS', 
                            style={'color':'white', 'paddingLeft':'6px'}
                        )
                    ], style={'padding':'0px', 'margin':'3px'})
                ]
            ])
            ],
            style={
                'color':'white', 'backgroundColor':uva_color, 
                'padding': '1em', 'fontSize':'0.86em',
                'paddingTop':'2em',
            },
        ), 
        dbc.Row(
            dbc.Col([
                html.A(
                    'Notice of Non-Discrimination and Equal Opportunity', 
                    href='http://eocr.virginia.edu/notice-non-discrimination-and-equal-opportunity',
                    style={'color':'white'}
                ), " | ",
                html.A(
                    "Report a Barrier", href='http://reportabarrier.virginia.edu/', 
                    style={'color':'white'} 
                )," | ",
                html.A(
                    "Privacy Policy", href='http://www.virginia.edu/siteinfo/privacy/', 
                    style={'color':'white'}
                ) ," | ",
                html.A(
                    "Emergency Relief Reporting", href='https://sfs.virginia.edu/emergency-federal-relief-funds', 
                    style={'color':'white'}
                )
            ], width=12)
            , style={
                'color':'white', 'backgroundColor':uva_color, 
                'padding': '15px', 'textAlign':'center', 'fontSize':'0.86em'
            }
        ),
        dbc.Row( 
            html.A('© 2024 By the Rector and Visitors of the University of Virginia'),
            style={
                'color':'white', 'backgroundColor':uva_color, 
                'textAlign':'center', 'fontSize':'0.86em', 'paddingBottom':'2em'
            })
    ]
    

def get_layout(factors, factor_labels, summed):    
    top_navbar = dbc.Navbar(
        color=uva_orange,
        style={'backgroundColor':uva_orange, 'margin':'0px', 'padding':'4px'},
        dark=True,
    )
    
    navbar = dbc.Row(
        dbc.Col(
            html.Img(
                src='./assets/uva-logo-inline.png', 
                style={'height':'35px'}
            )
        ),
        style={
            'color':'white', 'backgroundColor':uva_color, 
            'padding':'1.5em', 'paddingLeft': padding
        }
    )
    
    figures = [dbc.Col(
        dcc.Graph(id=graph_id)
        , width=width, style={
            'margin':'auto', 'padding':'0px', 'textAlign':'center'
        }) # , 'border':'1px solid black'
    
        for graph_id, width in [
            ('time-series-chart', 7), 
            ('count-chart', 5)]
    ]
    
    aid_table = dbc.Col([ 
        html.H2(
            "Financial Aid Table", 
            style={'fontWeight':'bold', 'textAlign':'center'}
        ),
        # https://dash.plotly.com/datatable/style#styling-editable-columns
        dash_table.DataTable(
            id='table', columns=[{'id':c, 'name':c} for c in summed.columns],
            data=summed.to_dict('records'), page_size=8, 
            style_header=style_header, is_focused=True,
            # editable=True, row_deletable=True, row_selectable=True, 
            export_format='csv', style_table={'overflowX': 'scroll'}
        )
    ])
    
    # get constraints table
    constraints = get_constraints(filename='constraints.csv')
    constraints_table = get_contraints_table(
        constraints, factors
    )
    
    return html.Div([
        top_navbar, navbar,
        dbc.Row([
            dbc.Col(
                html.H1('Student Financial Aid Prediction', 
                    style={
                        'textAlign':'left', 
                        'fontWeight':'bold','marginRight': '0px'
                    }
                )
            ), 
            dbc.Col([
                html.A('PAY ONLINE', href='https://virginia.myonplanu.com/login', style={'textDecoration': 'None'}),
                html.A(' / ', style={'color': uva_orange}), 
                html.A('SIS LOGIN', href='https://sisuva.admin.virginia.edu/ihprd/signon.html', style={'textDecoration': 'None'}), 
                html.A(' / ', style={'color': uva_orange}), 
                html.A('ESTIMATE COSTS', href='https://sfs.virginia.edu/estimate-your-costs-attend-uva', style={'textDecoration': 'None'})
                ], style={
                    'textAlign': 'right', 
                    'margin':'auto', 
                    'fontWeight':'bold',
                    'fontSize':'0.9em'
                }
            ),
        ], style={'margin': page_margin, 'marginTop':'1.5em', 'marginBottom':'1.5em'}),
        dbc.Row(
            [dbc.Col(dcc.Dropdown(
                id=f"dropdown-{column}",
                options = values, value = values[0],
                clearable=False, searchable=True, 
                style={'backgroundColor':uva_header, 'border':'0px'},
                # persistence=True, persistence_type='local'
            )) for column, values in zip(factor_labels, factors)]
            # + [
            #     dbc.Col(html.H5('Prediction Length')),
            #     dbc.Col(dcc.Input(id='pred-len', type='number', value=6, style={'width':'35px'})),
            # ]
            ,
            style={
                'padding':'3px', 'backgroundColor':uva_header, 
                'align':'center', 'fontWeight':'bold',
                'padding-left': padding, 'padding-right': padding
            }
        ),
        dbc.Row(figures, style={'margin':page_margin}), 
        dbc.Row(aid_table, style={'margin':page_margin, 'align':'center'}),
        dbc.Row(constraints_table, style={'margin':page_margin,'align':'center'})
    ] + get_bottom_navbar())

def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation 
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]

def draw_party_fig(years, data):
    party_fig = go.Figure()
    
    for (name, column) in [('Funded', 'FUNDED_PARTY'), ('All', 'TOTAL_PARTY')]:
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

    party_fig.update_layout(
        font=dict(
            # family="Courier New",
            size=fontsize,  # Set the font size here
        ),
        legend=dict(
            orientation="h",yanchor="bottom",y=1.02,
            xanchor="center", x=0.5
        ),
        title='<b>Enrolled Student Count</b>',
        xaxis_title='<b>Aid year</b>', 
        yaxis_title='<b>Count</b>',
        # hovermode='closest' # # Update the layout to show the coordinates
        # paper_bgcolor=background_color, # #dadada
        # plot_bgcolor='lightblue',
    )
    party_fig.update_xaxes(showgrid=True, ticklabelmode="period")
    party_fig.update_traces(marker_size=marker_size)
    # party_fig.update_yaxes(range=[0, None])
    
    return party_fig
    
def draw_main_fig(summed, predictions):
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
    
    # https://plotly.com/python/text-and-annotations/
    for (x, y) in zip(summed[time_column], summed[target].values):
        fig.add_annotation(
            x=x, y=y, xref="x",
            yref="y", text=numerize(y, 0),
            showarrow=True, arrowhead=1, 
            arrowsize=2,arrowwidth=1,
        )
        
    for (x, y) in zip(
        predictions[time_column], 
        predictions['PREDICTED_MEAN'].values
    ):
        fig.add_annotation(
            x=x, y=y, xref="x",
            yref="y", text=numerize(y, 0),
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
        title='<b>Aid Forecast. Model: ARIMA</b>',
        xaxis_title='<b>Aid year</b>',
        yaxis_title='<b>Offer balance</b>',
        # paper_bgcolor=background_color, # #dadada
        # plot_bgcolor='lightblue',
        xaxis=dict(linewidth=2), yaxis=dict(linewidth=2)
    )
    # fig.update_yaxes(title_font_color="red")
    fig.update_traces(marker_size=marker_size)
    # fig.update_yaxes(gridcolor='black')
    fig.update_xaxes(ticklabelmode="period")
    # fig.update_yaxes(range=[0, None])
    # fig.update_layout(yaxis_range=[0, max(summed[target].max(), summed[f'Predicted_{target}'].max())*1.2])
    return fig
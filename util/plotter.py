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

def get_contraints_table(constraints, style_header, factors):
    dash_columns = []
    for index, c in enumerate(constraints.columns):
        if c not in ['Start', 'End', 'Amount']: 
            dash_columns.append({'id':c, 'name':c, 'presentation': 'dropdown'})
        else: dash_columns.append({'id':c, 'name':c, 'type': 'numeric'})
    
    constraints_table = dbc.Col([
        html.H2("Policy Table", style={'textColor':uva_font, 'fontWeight':'bold', 'textAlign':'center'}),
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
                html.Img(src='./assets/uva-logo-footer-white.png', height="90px")
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
                        style={'color':'white'}
                    ), html.Br(),
                    "Staff and Faculty Resources"
                ]]
            ),
            dbc.Col([
                dbc.Row(row) for row in [
                    'Phone hours',
                    'Weekdays: 10-noon and 1-4',
                    'X', 'Facebook', 'Instagram'
                    # [html.Img(src='./assets/twitter-white.png', height=3, width=3), 'X'],
                    # [html.Img(src='./assets/facebook-white.png', style={'block':'inline', 'height':'10px'}), 'Facebook'],
                    # [html.Img(src='./assets/instagram-white.png', height="10px", width="10px"), 'Instagram']
                ]
            ])
            ],
            style={'color':'white', 'backgroundColor':uva_color, 'padding': '25px'},
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
            ,style={'color':'white', 'backgroundColor':uva_color, 'padding': '15px', 'textAlign':'center'}
        ),
        dbc.Row(
            html.H6('© 2024 By the Rector and Visitors of the University of Virginia'),
            style={'color':'white', 'backgroundColor':uva_color, 'padding': '0px', 'textAlign':'center'})
    ]
    

def get_layout(factors, factor_labels, summed):    
    # get constraints table
    constraints = get_constraints(filename='constraints.csv')
    constraints_table = get_contraints_table(
        constraints, style_header, factors
    )
    
    top_navbar = dbc.Navbar(
        color=uva_orange,
        style={'backgroundColor':uva_orange, 'margin':'0px', 'padding':'4px'},
        dark=True,
    )
    
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src='./assets/uva-logo-inline.png', height="40px")),
                            # dbc.Col(dbc.NavbarBrand("Navbar", className="ms-2")),
                        ],
                        # align="left"
                    ),
                    href="https://sfs.virginia.edu",
                    style={'margin':'8px'},
                )
            ]
        ),
        color=uva_color,
        style={'backgroundColor':uva_color, 'align': 'left'},
        dark=True,
    )
    
    bottom_navbars = get_bottom_navbar()
    
    aid_table = dbc.Col([
        html.H2(
            "Financial Aid Table", 
            style={'textColor':uva_font, 'fontWeight':'bold', 'textAlign':'center'}
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
    
    figures = [dbc.Col([
        dcc.Graph(id=graph_id), 
        dcc.RadioItems(
            options=['Annotation On', 'Annotation Off'], 
            value='Annotation On', id=radio_id, 
            inline=True, 
            # https://stackoverflow.com/questions/75692815/increase-space-between-text-and-icon-in-dash-checklist
            labelStyle= {"margin":"15px"},
            inputStyle={"margin": "5px"},
            style={'margin':'auto', 'padding':'2px'}
        )], width=width, style={'margin':'auto', 'padding':'0px', 'textAlign':'center'}) # , 'border':'1px solid black'
    
        for graph_id, radio_id, width in [
            ('time-series-chart', 'radio-time-series', 7), 
            ('count-chart', 'radio-count-series', 5)]
    ]
    
    return html.Div([
        top_navbar,navbar,
        dbc.Row(
            dbc.Col(
                html.H1('Financial Aid Predictive Analysis', 
                    style={'textAlign':'center', 'textColor':uva_font, 'fontWeight':'bold', 'padding':'8px', 'margin':'30px'}
                )
            )
        ),
        dbc.Row(
            [dbc.Col(dcc.Dropdown(
                        id=f"dropdown-{column}",
                        options = values,value = values[0],
                        # clearable=True, 
                        searchable=True, 
                        style={'backgroundColor':uva_header, 'border':'0px'},
                        # persistence=True, persistence_type='local'
                    )) for column, values in zip(factor_labels, factors)]
            # + [
            #     dbc.Col(html.H5('Prediction Length')),
            #     dbc.Col(dcc.Input(id='pred-len', type='number', value=6, style={'width':'35px'})),
            # ]
            ,
            style={
                'padding':'5px', 'backgroundColor':uva_header, 
                'margin':'5px', 'align':'center', 
                'textColor':uva_font, 'fontWeight':'bold'
            }
        ),
        dbc.Row(figures, style={'margin':'10px', 'marginBottom':'70px'}), 
        dbc.Row(aid_table, style={'margin':'10px', 'marginBottom':'70px', 'align':'center'}),
        dbc.Row(constraints_table, style={'margin':'10px', 'marginBottom':'80px', 'align':'center'})
    ]+bottom_navbars)

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
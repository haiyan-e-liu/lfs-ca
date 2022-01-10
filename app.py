#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import seaborn as sns
import datetime as dt

import plotly.graph_objects as go  # plot a grouped and stacked bar chart
from plotly.subplots import make_subplots
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# In[ ]:
lfs_tourism_employment = pd.read_csv('lfs_tourism_employment.csv')


# In[]:
## Calculate provincial total tourism related employment by month

# Calculate provincial total tourism employment by month
lfs_tourism_employment_tot = lfs_tourism_employment.groupby('Geography').sum().reset_index()

# Reshape data into long format with months in one column
lfs_tourism_employment_tot = pd.melt(lfs_tourism_employment_tot, id_vars = 'Geography', value_vars = lfs_tourism_employment_tot.columns[2:], var_name = 'Date', value_name = 'Total').sort_values(by = ['Geography', 'Date'])

# Calculate year over year changes by month
lfs_tourism_employment_tot = pd.concat([lfs_tourism_employment_tot,
                                        round(lfs_tourism_employment_tot.groupby('Geography')['Total'].pct_change(periods = 12)*100, 0)], axis = 1)
lfs_tourism_employment_tot.columns = ['Geography', 'Date', 'Total', 'YoY_Change']

# Only keep months with values of yoy change
lfs_tourism_employment_tot.dropna(subset = ['YoY_Change'], inplace = True)

# Order Geography by province from east to west
province = pd.DataFrame(lfs_tourism_employment['Geography'].unique(), columns = ['Geography'])
lfs_tourism_employment_tot = province.merge(lfs_tourism_employment_tot, on = 'Geography')

# In[]:

## Plot provincial tourism related employment by month
def plot_yoy_changes_by_month(province):

    dates = lfs_tourism_employment_tot.Date.unique()
    colors = sns.diverging_palette(150, 275, n=len(dates), center = 'dark').as_hex()

    df = lfs_tourism_employment_tot

    fig = go.Figure()

    for r, c in zip(dates, colors):
        plot_df = df[df['Date'] == r]
        fig.add_trace(go.Scatter(
            x = plot_df['Geography'],
            y = plot_df['YoY_Change'],
            name = r, marker_color= c,
            )
        )

    fig.update_layout(
            template = "simple_white",
            title = dict(text = '<b>' + 'Canadian Tourism Employment Y/Y Change by Month' + '</b>',
                         font_size = 16,
                         yanchor = 'top', y = .97, xanchor = 'center', x = .5),
            legend = dict(orientation = 'v',
                         traceorder = 'reversed', title = ""), # position of legend;
            autosize = False, width = 1000, height = 600,  # size of figure
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.update_xaxes(
        rangeslider_visible=False,    # Don't Add Range Slider
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])                      # Add Range Selector Buttons
        )
    )

    fig.update_yaxes(title = '%')
    fig.add_hline(y=0, line_dash="dot", line_width = 2, line_color = 'dark grey',
                 ),
    fig.add_vline(x=province, line_dash="dot", line_width = 2, line_color = 'dark grey',
         )

    return fig

## Tourism Related Employment in Saskatchewan

# In[ ]:

# Define function to format dataset and calculate total tourism employment by province
def calculate_tot_tourism_employment_by_province(province):
    # Select data for the selected province
    province_tourism_employment = lfs_tourism_employment[lfs_tourism_employment.Geography == province].reset_index(drop = True)

    # Calculate provincial total in all industries
    province_tourism_employment_tot = province_tourism_employment.groupby('Geography').sum().reset_index()
    province_tourism_employment_tot['Industry'] = 'Total'

    col = province_tourism_employment_tot.pop('Industry')
    province_tourism_employment_tot.insert(1, col.name, col)

    # Combine industry total with provincial total
    province_tourism_employment_tot = pd.concat([province_tourism_employment_tot, province_tourism_employment], ignore_index = True)

    province_tourism_employment_tot.drop('Geography', axis = 1, inplace = True)

    return province_tourism_employment_tot


# ## Provincial Y/Y Change in Monthly Total Tourism Employment

# ### Define Functions to Calculate YoY Changes

# In[ ]:

def yoy_change(province):

    # Generate total tourism employment for a specific province
    df = calculate_tot_tourism_employment_by_province(province)

    ## Organize data to calculate monthly year over year change
    # Only keep the total employment
    tot = df.loc[df['Industry'] == 'Total', df.columns[1:]]

    # Reshape data from wide to long
    tot_melted = pd.melt(tot, value_vars = tot.columns, var_name = 'Date', value_name = 'Total')

    # Calculate year over year changes by month
    tot_melted = pd.concat([tot_melted, round(tot_melted['Total'].pct_change(periods = 12)*100, 0)], axis = 1)

    tot_melted.columns = ['Date', 'Total', 'YoY_Change']

    # Total tourism employment in SK was estimated at 71,100 in August 2019, but the number got adjusted later on.
    # For now, manually change the number to 71,100
    if province == 'Saskatchewan':
        tot_melted.loc[tot_melted['Date'] == '2019-08-01', 'Total'] = 71100

    # Save the maximum monthly employment since April 2019
    tot_melted['Total_Max'] = tot_melted['Total'].max()

    tot_melted.dropna(subset = ['YoY_Change'], inplace = True) # only keep months with values of yoy change
    tot_melted['Date'] = pd.to_datetime(tot_melted['Date'])

    return tot_melted


# In[ ]:

def yoy_change_sectors(province):

    # Generate total tourism employment for a specific province
    df = calculate_tot_tourism_employment_by_province(province)

    # Keep only rows with industry total, not provincial total
    tot = df[df['Industry']!= 'Total']

    # Reshape data from wide to long
    tot_melted = pd.melt(tot, id_vars = ['Industry'], value_vars = tot.columns[1:],
                         var_name = 'Date', value_name = 'Total')

    # Calculate year over year changes by month
    tot_melted = pd.concat([tot_melted, round(tot_melted.groupby('Industry')['Total'].pct_change(periods = 12)*100, 0)], axis = 1)
    tot_melted.columns = ['Industry', 'Date', 'Total', 'YoY_Change']

    tot_melted.dropna(subset = ['YoY_Change'], inplace = True) # only keep months with values of yoy change
    tot_melted.sort_values(by = ['Industry', 'Date'], inplace = True)

    return tot_melted


# ### Define Functions to Plot YoY Changes**

# In[ ]:


# Plot monthy total tourism related employment and its yoy change
def plot_yoy_changes(province, fig_title):

    # Generate year over year change data for the plot
    df = yoy_change(province)

    # Plot the year over year change
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x = df['Date'], y = df['Total'], mode = 'lines + markers',
                   name = 'Total Employment', line_color = '#05712f'),
    )

    fig.add_trace(
        go.Scatter(x = df['Date'], y = df['YoY_Change'], mode = 'lines + markers',
                   name = 'Year Over Year Change', line_color = '#0000FF'),
        secondary_y = True,
    )

    fig.update_layout(
            template = "simple_white",
            title = dict(text = '<b>' + fig_title + ' in ' + province + '</b>',
                         font_size = 16,
                         yanchor = 'top', y = .96, xanchor = 'center', x = .5),
            showlegend=False,
            autosize = False, width = 1000, height = 600,  # size of figure

            xaxis = dict(title = 'Month-Year'),
            yaxis=dict(
                title="Total Employment",
                titlefont=dict(color="#05712f"),
                tickfont=dict(color="#05712f"),
                ),

            yaxis2=dict(
                title="Year over Year Change (%)",
                titlefont=dict(color='#0000FF'),
                tickfont=dict(color='#0000FF'),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.94,
                ),
    )

    fig.update_xaxes(
        dtick="M3",
        tickformat="%b\n%Y",)

    fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
                    ])                      # Add Range Selector Buttons
                  )
                )

    # Add a line to display the maximum total tourism employment in the province during the plotting period
    fig.add_hline(y = df['Total_Max'].max(), line_width = 3, line_dash = 'dash', line_color="black")

    # # Add an annotation to display the maximum total tourism employment in the province since April 2019
    fig.add_annotation(
        # Setting xref and/or yref to "paper" will cause the x and y attributes to be interpreted in paper coordinates.
        x = 0.02, y = 0.97,
        xref="paper",
        yref="paper",
        text="Max Employment Pre-COVID-19 = "+"{:,.0f}".format(df['Total_Max'].max()),
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff"
            ),
        align="center",
        ax=30,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor='#05712f',
        opacity=1
        )

    return fig

# In[ ]:


# Plot monthy total tourism related employment and its yoy change by sector
def plot_yoy_changes_sector(province, fig_title):
    df = yoy_change_sectors(province)

    fig = go.Figure()

    industry = sorted(df.Industry.unique().tolist())
    colors = sns.color_palette("Set1", len(industry)).as_hex()

    for r, c in zip(industry, colors):
        plot_df = df[df['Industry'] == r]
        fig.add_trace(go.Scatter(
            x = plot_df['Date'],
            y = plot_df['Total'],  # plot number of jobs in the sector rather than y/y change now
            name = r, marker_color= c,
            )
        )

    fig.update_layout(
            template = "simple_white",
            title = dict(text = '<b>' + fig_title + ' in ' + province + ' by Industry </b>',
                         font_size = 16,
                         yanchor = 'top', y = .97, xanchor = 'center', x = .5),
            legend = dict(orientation = 'v',
                         traceorder = 'grouped', title = ""), # position of legend;
            autosize = False, width = 1000, height = 600,  # size of figure
    )

    fig.update_xaxes(
        dtick="M3",
        tickformat="%b\n%Y")

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])                      # Add Range Selector Buttons
        )
    )

    fig.update_yaxes(title = '')
    fig.add_hline(y=0, line_dash="dot", line_width = 2, line_color = 'dark grey',
                 ),

    return fig

# In[ ]:

# Calculate YoY change of peak month (August) tourism employment by province
def plot_yoy_aug_tourism_employment(province):
    # Get tourism_employment for all months
    df = calculate_tot_tourism_employment_by_province(province)

    df = df[df['Industry'] != 'Total']

    # Calcualte YoY Change in Peak month by sector
    cy = dt.datetime.now().year   # current year
    if dt.datetime.now().month <8:
      cy = cy-1
    aug_cols = [str(cy-1) + '-08-01', str(cy) + '-08-01']

    yoy_all_sectors_aug = df[['Industry'] + aug_cols]
    yoy_change_aug = pd.DataFrame(round(yoy_all_sectors_aug[aug_cols].pct_change(axis = 'columns').iloc[:, 1]*100, 0)).fillna(0)
    yoy_change_aug.columns = ['YoY_Change']
    yoy_all_sectors_aug = yoy_all_sectors_aug.join(yoy_change_aug).sort_values(by = 'YoY_Change')


    # Plot YoY Change in Peak month by sector
    fig = px.bar(yoy_all_sectors_aug, x = 'Industry', y = 'YoY_Change', text = 'YoY_Change',
                color_discrete_sequence  = ['green']*len(yoy_all_sectors_aug))

    fig.update_layout(
        template = 'simple_white',
        title = dict(text = '<b> Tourism Employment in ' + province +
                     '<br> Y/Y Change, Aug ' + str(cy-1) + ' to Aug ' + str(cy) + '</b>',
                         font_size = 16, 
                         yanchor = 'top', y = .96, xanchor = 'center', x = .5),
        autosize = False, width = 1000, height = 600,  # size of figure
    )

    fig.update_xaxes(title = '')
    fig.update_yaxes(title = '%')

    return fig

# In[ ]:

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# app.config.suppress_callback_exceptions = True


province = lfs_tourism_employment['Geography'].unique()

app.layout = html.Div([
    html.Div([dcc.Dropdown(id='province-select',
                           options=[{'label': x, 'value': x} for x in province],
                           value='Saskatchewan', style={'width': '500px'}
                          )
             ],
             className = 'row',
             ),
    html.Div([
        dcc.Graph(id = 'yoy-by-month'),
        dcc.Graph(id = 'line-chart'),
        dcc.Graph(id = 'sector-line-chart'),
        dcc.Graph(id = 'aug-line-chart')
        ])
    ]
    )

@app.callback(
    [Output('yoy-by-month', 'figure'),
     Output('line-chart', 'figure'),
     Output('sector-line-chart', 'figure'),
     Output('aug-line-chart', 'figure')],
    [Input('province-select', 'value')]
)
def update_line_chart(province):
    return [plot_yoy_changes_by_month(province),
            plot_yoy_changes(province, 'Tourism Employment'),
            plot_yoy_changes_sector(province, 'Tourism Employment'),
            plot_yoy_aug_tourism_employment(province)]

if __name__ == '__main__':
    app.run_server(debug=False)

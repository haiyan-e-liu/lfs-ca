#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import re

import plotly.graph_objects as go  # plot a grouped and stacked bar chart
from plotly.subplots import make_subplots
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# In[ ]:

# cd 'C:\Users\hliu\Tourism Saskatchewan\GroupData - TourPlan\Statistics\Key Statistics\Employment Statistics\Labour Force Survey Raw Data'


# In[ ]:

lfs = pd.read_excel("0520_14 Table 1.xlsx", sheet_name = '0520_14 Table 1', skiprows = 3)


# In[ ]:


lfs.shape


# In[ ]:


lfs.tail()


# In[ ]:


# Convert column names to a list of strings
lfs.columns = ['Geography', 'NAICS', 'Variable'] + [x.strftime('%Y-%m-%d') for x in lfs.columns[3:]]


# In[ ]:


# lfs.drop('Code', axis = 1, inplace = True)


# In[ ]:


# Extract NAICS codes
def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return "".join(num)
lfs['Code'] = lfs['NAICS'].apply(lambda x: find_number(x))

# In[ ]:


# Extract NAICS descriptions
def find_description(text):
    txt = re.findall(r'[\D]+',text)
    return "".join(txt)
lfs['Description'] = lfs['NAICS'].apply(lambda x: find_description(x))

# In[ ]:


# Move column Code and Description to the front
col = lfs.pop('Code')
lfs.insert(2, col.name, col)

col = lfs.pop('Description')
lfs.insert(3, col.name, col)


# In[ ]:
    
# Trim leading and trailing white space from a column
for v in ['Geography', 'NAICS', 'Code', 'Description', 'Variable']:
    lfs[v] = lfs[v].apply(lambda x: x.strip())

# In[ ]:


lfs['Description'].unique()


# In[ ]:


# Check what non-digit characters are in column Code
a = [re.findall(r'\D+', x) for x in lfs['Code']]
b = list(np.concatenate(a).flat)
b


# In[ ]:


# Label the unclassified categories as '0', otherwise pd.to_numeric(, downcast = 'integer') only converts data to float64
lfs.loc[lfs['Code'] == '', 'Code'] = '0'


# In[ ]:


# Convert Code to numeric to filter out tourism related industries
lfs.Code = pd.to_numeric(lfs['Code'], downcast='integer')


# In[ ]:


lfs.Code.dtypes


# In[ ]:


# NAICS for tourism industry
naics_tourism_sub = [481, 485, 487, 5615, 711, 712, 713, 721, 722]
naics_tourism_total = naics_tourism_sub + [447, 453]


# In[ ]:


# Extract tourism related industries
lfs_tourism = lfs[lfs['Code'].isin(naics_tourism_total)]


# In[ ]:


lfs_tourism.Code.unique().tolist()


# In[ ]:


lfs_tourism.Geography.unique()


# In[ ]:


lfs_tourism.Variable.unique()


# In[ ]:


# Extract tourism related employment
lfs_tourism_employment = lfs_tourism[lfs_tourism.Variable == 'Employment (x 1,000)']


# In[ ]:


lfs_tourism_employment = lfs_tourism_employment.drop(['NAICS', 'Variable'], axis = 1)


# In[ ]:


lfs_tourism_employment.rename(columns = {'Code': 'NAICS'}, inplace = True)

# In[ ]:


# Convert employment into #jobs from thousands of jobs; if the value was suppressed to X, replace it with 0
def multiply_by_element(s):
    return s.apply(lambda x: x*1000 if isinstance(x, (int, float)) else 0)


# In[ ]:


lfs_tourism_employment.iloc[:, 3:] = lfs_tourism_employment.iloc[:, 3:].apply(lambda x: multiply_by_element(x))

# In[ ]:

lfs_tourism_employment

lfs_tourism_employment['Geography'].value_counts()


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
    
    # Plot the year over year change for key sectors
    # fig = px.line(yoy_change_sectors(province_name), x="Date", y="YoY_Change", color='Description', 
    #           color_discrete_sequence = ['#05712f', '#ffd92f', '#0000FF', '#e31a1c'])
    dates = lfs_tourism_employment_tot.Date.unique()
    colors = ['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', 
             '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff', '#fb9a99', '#05712f', '#0b559f', '#aa1016']
#     colors = sns.diverging_palette(130, 295, n=14, center = 'dark')
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
            title = dict(text = '<b>' + 'Tourism Related Employment Year over Year Change by Month' + '</b>', 
                         font_size = 16, 
                         yanchor = 'top', y = .97, xanchor = 'center', x = .5), 
            legend = dict(orientation = 'v', 
#                           yanchor = 'top', y = 1.05, xanchor = 'right', x = .98, 
                         traceorder = 'reversed', title = ""), # position of legend;    
            autosize = False, width = 1000, height = 600,  # size of figure  
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.update_xaxes(
        rangeslider_visible=False,    # Add Range Slider
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


# Define function to format dataset and calculate subtotal and total tourism employment by province
def calculate_tot_tourism_employment_by_province(province_name):
    province_tourism_employment = lfs_tourism_employment[(lfs_tourism_employment.Geography == province_name)].reset_index(drop = True)

    # Move non-essential tourism industries to the bottom
    province_tourism_employment = pd.concat([province_tourism_employment.shift(-2).iloc[:-2,:], province_tourism_employment.iloc[[0, 1]]], 
                                      ignore_index = True)

    # Calculate subtotal and totals
    province_tourism_employment_tot = province_tourism_employment.append(province_tourism_employment.iloc[0:-2, 3:].sum(axis = 0), ignore_index=True )
    province_tourism_employment_tot = province_tourism_employment_tot.append(province_tourism_employment.iloc[:, 3:].sum(axis = 0), ignore_index=True )

    province_tourism_employment_tot.loc[11, 'Description'] = 'Subtotal'
    province_tourism_employment_tot.loc[12, 'Description'] = 'Total'

    # Move subtotal row up
    province_tourism_employment_tot = province_tourism_employment_tot.reindex([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 9, 10, 12])

    province_tourism_employment_tot.drop('Geography', axis = 1, inplace = True)
    province_tourism_employment_tot.loc[[11, 12], 'NAICS'] = ''
    
    return province_tourism_employment_tot


# In[ ]:


calculate_tot_tourism_employment_by_province('Saskatchewan')


# ## Year Over Year Change in Monthly Total Tourism Employment

# ### Define Functions to Calculate YoY Changes

# In[ ]:

def yoy_change(province):
    
    # Generate total tourism employment for a specific province
    df = calculate_tot_tourism_employment_by_province(province)
    
    ## Organize data to calculate monthly year over year change
    # Only keep the total employment
    tot = df.loc[df['Description'] == 'Total', df.columns[2:]]

    # Reshape data from wide to long
    tot_melted = pd.melt(tot, value_vars = tot.columns, var_name = 'Date', value_name = 'Total')
    
    # Calculate year over year changes by month
    tot_melted = pd.concat([tot_melted, round(tot_melted['Total'].pct_change(periods = 12)*100, 0)], axis = 1)
    
    tot_melted.columns = ['Date', 'Total', 'YoY_Change']
    tot_melted.dropna(subset = ['YoY_Change'], inplace = True) # only keep months with values of yoy change    
    
    return tot_melted


# In[ ]:

def yoy_change_sectors(province):
    
    # Generate total tourism employment for a specific province
    df = calculate_tot_tourism_employment_by_province(province)
    
    ## Organize data to calculate monthly year over year change
    # Get employment data for the four sectors wanted
    tot = df[df['NAICS'].isin([711, 713, 721, 722])]

    # Reshape data from wide to long
    tot_melted = pd.melt(tot, id_vars = ['NAICS', 'Description'], value_vars = tot.columns[2:], 
                         var_name = 'Date', value_name = 'Total')
    
    # Calculate year over year changes by month
    tot_melted = pd.concat([tot_melted, round(tot_melted.groupby('NAICS')['Total'].pct_change(periods = 12)*100, 0)], axis = 1)
    tot_melted.columns = ['NAICS', 'Description', 'Date', 'Total', 'YoY_Change']

    tot_melted.dropna(subset = ['YoY_Change'], inplace = True) # only keep months with values of yoy change
    tot_melted.sort_values(by = ['NAICS', 'Date'], inplace = True)
    
    return tot_melted


# ### Define Functions to Plot YoY Changes**

# In[ ]:


# Plot monthy total tourism related employment and its yoy change
def plot_yoy_changes(province, fig_title):
    
    # Generate year over year change data for the plot
    data = yoy_change(province)
    
    # Plot the year over year change
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x = data['Date'], y = data['Total'], mode = 'lines + markers', 
                   name = 'Total Employment', line_color = '#05712f'),
    )

    fig.add_trace(
        go.Scatter(x = data['Date'], y = data['YoY_Change'], mode = 'lines + markers', 
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
        dtick="M1",
        tickformat="%b\n%Y",)
    
    fig.update_xaxes(
        rangeslider_visible=True,    # Add Range Slider
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
    
    return fig

# In[ ]:


# Plot monthy total tourism related employment and its yoy change for specific sectors
def plot_yoy_changes_sector(province, fig_title):
    
    # Plot the year over year change for key sectors
    # fig = px.line(yoy_change_sectors(province_name), x="Date", y="YoY_Change", color='Description', 
    #           color_discrete_sequence = ['#05712f', '#ffd92f', '#0000FF', '#e31a1c'])
    labels = ['Performing arts, spectator sports and related industries',
               'Amusement, gambling and recreation industries',
               'Accommodation services', 'Food services and drinking places']
    colors = ['#05712f', '#ffd92f', '#0000FF', '#e31a1c']
    
    df = yoy_change_sectors(province)
    
    fig = go.Figure()

    for r, c in zip(labels, colors):
        plot_df = df[df['Description'] == r]
        fig.add_trace(go.Scatter(
            x = plot_df['Date'], 
            y = plot_df['YoY_Change'],
            name = r, marker_color= c, 
            )
        ) 
        
    fig.update_layout(
            template = "simple_white", 
            title = dict(text = '<b>' + fig_title + ' in ' + province + 
                         '<br>' + 'Year over Year Change, Key Sectors' + '</b>', 
                         font_size = 16, 
                         yanchor = 'top', y = .97, xanchor = 'center', x = .5), 
            legend = dict(orientation = 'v', 
#                           yanchor = 'top', y = 1.05, xanchor = 'right', x = .98, 
                         traceorder = 'grouped', title = ""), # position of legend;    
            autosize = False, width = 1000, height = 600,  # size of figure  
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.update_xaxes(
        rangeslider_visible=False,    # Add Range Slider
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

    return fig

# In[ ]:

# Calculate YoY change of peak month (August) tourism employment by province
def plot_yoy_aug_tourism_employment(province):
    # Get tourism_employment for all months
    tot_employment = calculate_tot_tourism_employment_by_province(province)
    prov_tourism_employment = tot_employment[(tot_employment['Description'] != 'Total') & (tot_employment['Description'] != 'Subtotal')]
    
    # Calcualte YoY Change in Peak month by sector
    yoy_all_sectors_aug = prov_tourism_employment[['NAICS', 'Description', '2019-08-01', '2020-08-01']]
    yoy_change_aug = pd.DataFrame(round(yoy_all_sectors_aug[['2019-08-01', '2020-08-01']].pct_change(axis = 'columns')['2020-08-01']*100, 0)).fillna(0)
    yoy_change_aug.columns = ['YoY_Change']
    yoy_all_sectors_aug = yoy_all_sectors_aug.join(yoy_change_aug).sort_values(by = 'YoY_Change')
    
    
    # Plot YoY Change in Peak month by sector
    fig = px.bar(yoy_all_sectors_aug, x = 'Description', y = 'YoY_Change', text = 'YoY_Change', 
                color_discrete_sequence  = ['green']*len(yoy_all_sectors_aug))

    fig.update_layout(
        template = 'simple_white',
        title = dict(text = '<b> Tourism Related Employment in ' + province + '<br> Year over Year Change, August 2019 to August 2020 </b>', 
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
            plot_yoy_changes(province, 'Tourism Related Employment'), 
            plot_yoy_changes_sector(province, 'Tourism Related Employment'), 
            plot_yoy_aug_tourism_employment(province)]   

if __name__ == '__main__':
    app.run_server(debug=False)




import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import flask
import glob
import os

image_directory = '/Users/sue/Stockton MSDSSA/Data Practicum/images'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']        # There is an assets folder with a css file in it that prevents the undo from displaying on the graph; both this file and the assets folder contents should load                         

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

markdown_text = '''
## AWS DeepRacer - Reinforcement Learning Reward Function Analysis

AWS DeepRacer is a fun way to learn about Reinforcement Learning (RL).  Using the DeepRacer Console provided by AWS, you can create and
train model cars to race in a virtual racing league - and even in the real world!  

Once a reward function was working well on a trained model in the Empire City Circuit DeepRacer Virtual Circuit, its reward function, action
space and hyperparameters were used as the baseline to train six different models for three hours each - one on each of six tracks.  
Below is information on the rewards earned during my model training.  I competed in the July 2019 Empire City Circuit and ended in the
top 200 racers with my best model.

'''

markdown_text2 = '''

|Presented by| 
|:--- | ----:|
|Sue Catanoso|
|catanoss@go.stockton.edu|
 
  



'''



app.layout = html.Div([
        html.Div(
            dcc.Markdown(children = markdown_text)
        ),
      
        html.Div([
            html.H5('Select Track to View Training Data Rewards'),
            html.Div([   
                html.Label('Dropdown'),
                dcc.Dropdown(
                    id='track',
                    options=[
                        {'label': 'AWS', 'value': 'AWS'},
                        {'label': 'Bowtie', 'value': 'bowtie'},
                        {'label': 'London Loop', 'value': 'London'},
                        {'label': 'Oval', 'value': 'oval'},
                        {'label': 'reInvent', 'value': 'reinvent'},
                        {'label': 'Tokyo', 'value': 'Tokyo'}
                    ],
                    value = 'AWS',
                   )
                ], 
                style={'width': '27%', 'float': 'left', 'display': 'inline-block'}
                ),
            ]),   
  
        html.Div([
            html.Div([
                dcc.Graph(id='rewarditerations')
                ],
                style={'width': '97%', 'display': 'inline-block', 'position':'relative'}
             ), 
             html.Div([
                 dcc.Graph(id='rewardepisodes')
                 ],
                 style={'width': '97%', 'display': 'inline-block', 'position':'relative'}
             ),
    
            html.Div([
                html.Img(id='trackimage')
             ]),
          
#            html.Div([
#                html.Img(src=app.get_asset_url('bowtie.png'))
#             ]),
            
        html.Div(
                dcc.Markdown(children=markdown_text2)          
            ),
        ]),
])
    
 

@app.callback(                                                                     
    Output('rewarditerations', 'figure'), 
    [Input('track', 'value')])

def update_figure(trackname):

    fname = '%s.csv' %trackname
    df = pd.read_csv(fname)
    min_episodes = np.min(df['episode'])
    max_episodes = np.max(df['episode'])

    total_reward_per_episode = list()
    for epi in range(min_episodes, max_episodes):
        df_slice = df[df['episode'] == epi]
        total_reward_per_episode.append(np.sum(df_slice['reward']))

    total_reward_per_iteration = list()

    buffer_rew = list()
    for val in total_reward_per_episode:
        buffer_rew.append(val)

        if len(buffer_rew) == 20:                                             # This needs to correspond to episodes per iteration hyperparameter
            total_reward_per_iteration.append(np.sum(buffer_rew))
            # reset
            buffer_rew = list()
    df_iterations = pd.DataFrame(total_reward_per_iteration)
    df_iterations.columns = ['Total_Rewards']
    df_iterations['Iteration_Number'] = 1
    for i in range(len(df_iterations)):
        df_iterations['Iteration_Number']=df_iterations.index+1


    return {
        'data': [go.Scatter(
        x=df_iterations['Iteration_Number'],
        y=df_iterations['Total_Rewards'],
        mode='markers',
        opacity=0.7
        )],
        'layout': go.Layout(
            title = 'Total Rewards per Iteration - ' + trackname + ' Track',
            xaxis={'title': 'Iteration', 'range': [0, 106]},    
            yaxis={'title': 'Total Reward', 'range': [0, 1200]}    
        )
    }               


@app.callback(                                                                     
    Output('rewardepisodes', 'figure'), 
    [Input('track', 'value')])

def update_figure2(trackname):

    fname = '%s.csv' %trackname
    df = pd.read_csv(fname)
    
    min_episodes = np.min(df['episode'])
    max_episodes = np.max(df['episode'])

    total_reward_per_episode = list()
    for epi in range(min_episodes, max_episodes):
        df_slice = df[df['episode'] == epi]
        total_reward_per_episode.append(np.sum(df_slice['reward']))
 
    df_episodes = pd.DataFrame(total_reward_per_episode)
    df_episodes.columns = ['Total_Rewards']
    df_episodes['Episode_Number'] = 1
    for i in range(len(df_episodes)):
        df_episodes['Episode_Number']=df_episodes.index+1

    return {
        'data': [go.Scatter(
        x=df_episodes['Episode_Number'],
        y=df_episodes['Total_Rewards'],
        mode='markers',
        opacity=0.7
        )],
        'layout': go.Layout(
            title = 'Total Rewards per Episode - ' + trackname + ' Track',
            xaxis={'title': 'Episode', 'range': [0, 2120]},    
            yaxis={'title': 'Total Reward', 'range': [0, 225]}    
        )
    }               

@app.callback(                                                                     
    Output('trackimage', 'src'), 
    [Input('track', 'value')])


def update_image(trackname):
    
    imagename = '%s.png' %trackname
    src = app.get_asset_url(imagename)
    return src
 

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port='8051')
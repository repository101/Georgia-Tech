import logging
import modelDataRetrieval as predictor
from flask import Flask, jsonify, request

import random
import numpy as np
import pandas as pd
import random
import io
import requests

app = Flask(__name__, static_url_path='',static_folder='ui')

pitching_data = {}
players = []
ml_metadata = {}
stands = ["L","R"]

@app.before_first_request
def load_pitching_data():
    global pitching_data
    pitching_data = pd.read_csv('Modeling_data.csv')

@app.before_first_request
def load_player_data():
    global players
    players = pd.read_csv('ui/player_lookup.csv')

@app.before_first_request
def load_ml_metadata():
    global ml_metadata
    ml_metadata = pd.read_csv('results_40pitchers.csv')

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

@app.route("/predict", methods=["POST"])
def predictFrom():
    # data_url="https://s3.amazonaws.com/philly6242/Modeling_data.csv"
    # data_object=requests.get(url).content
    #
    # player_url="https://s3.amazonaws.com/philly6242/player_lookup.csv"
    # player_object=requests.get(player_url).content
    # players =pd.read_csv(io.StringIO(player_object.decode('utf-8')))
    balls = request.form['balls']
    pitcher = players[players["MLBID"] == int(request.form['pitcher'])].iloc[0]['MLBNAME']
    strikes = request.form['strikes']
    inning = request.form['inning']
    outs = request.form['outs']
    pitchnumpa = request.form['pitchnumpa']
    pitchnumgame = request.form['pitchnumgame']
    scorebatter = int(request.form['scorebatter'])
    scorepitcher = int(request.form['scorepitcher'])
    runneron1 = request.form['runneron1']
    runneron2 = request.form['runneron2']
    runneron3 = request.form['runneron3']
    batter = players[players["MLBNAME"] == request.form.get('batter')].iloc[0]["MLBID"]
    stand = random.choice(stands)
    outcomelags =  random.choices(predictor.outcomes,k=3)
    pitchlags = random.choices(predictor.pitch_types_4_pitcher(int(request.form['pitcher']), pitching_data),k=3)
    pitchResults = predictor.modelPrediction( pitching_data,players,pitcher=pitcher,
                    batter=batter,pitch_number=pitchnumpa,
                    game_pitch_number=pitchnumgame,outs_when_up=outs,
                    BR_1B=runneron1,BR_2B=runneron2,BR_3B=runneron3,
                    balls=balls,strikes=strikes,
                     pitcher_score=scorepitcher,batter_score=scorebatter,
                    stand=stand,outcomelag1=outcomelags[0],
                    outcomelag2=outcomelags[1],outcomelag3=outcomelags[2],
                     pitchlag1=pitchlags[0],pitchlag2=pitchlags[1],pitchlag3=pitchlags[2])
    return jsonify({"prediction": {k:float(v) for k,v in pitchResults.items()}, "accuracy": ml_metadata[ml_metadata["Pitcher"] == pitcher].iloc[0]['accuracy']})

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)

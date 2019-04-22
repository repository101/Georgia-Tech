import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from predictor import pitch_predictor


app = Flask(__name__)
CORS(app)
@app.route("/predict/",methods=['POST'])
def return_price():
id, balls, strikes, inning, outs, pitch_PA, pitch_game, score_batter, score_pitcher
  id = request.form.get('id')
  balls = request.form.get('balls')
  strikes = request.form.get('strikes')
  inning = request.form.get('inning')
  outs = request.form.get('outs')
  pitch_PA = request.form.get('pitch_PA')
  pitch_game = request.form.get('pitch_game')
  score_batter = request.form.get('score_batter')
  score_pitcher = request.form.get('score_pitcher')
  
  pitches = pitch_predictor.predict(date, month, year)
  price_dict = {
                {"pitch":"CH","usage": pitches["CH"]},
                {"pitch":"CU","usage": pitches["CU"]},
                {"pitch":"FC", "usage": pitches["FC"]},
                {"pitch":"FF","usage": pitches["FF"]},
                {"pitch":"FS","usage": pitches["FS"]},
                {"pitch":"FT","usage": pitches["FT"]},
                {"pitch":"KC","usage": pitches["KC"]},
                {"pitch":"KN","usage": pitches["KN"]},
                {"pitch":"SI","usage": pitches["SI"]},
                {"pitch":"SL","usage": pitches["SL"]}
                }
  return jsonify(price_dict)

@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to Pitch predictor <h1>"

if __name__ == "__main__":
    app.run()
steps to get this working:

install packages:
flask
flask-cors
jsonify
pandas
numpy
sklearn

to start flask on local host:

navigate to folder with the app.py and run `python app.py` and the console will tell you the local address to use.
You can now send post requests to the endpoint $(YOUR BASE)/predict

currently expects the following parameters:

id, balls, strikes, inning, outs, pitch_PA, pitch_game, score_batter, score_pitcher
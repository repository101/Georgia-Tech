# Baseball_Pitch_Prediction

Baseball Pitch Prediction is an app used to predict the type of pitch that will be thrown based on variables that the user can specify.

## Usage

```
Clone or download repo.

Specify the pitcher to predict with, then choose the various parameters, and finally click 'Predict!'
```

## Setup

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

```bash
pip install -r requirements.txt
```
### Download our Data

```
Download https://s3.amazonaws.com/philly6242/Modeling_data.zip
```
Extract Modeling_data.csv to the base directory for this application, the same directory where main.py is located


### (OPTIONAL) Get Data

Run GetBaseBallGameSchedule.py and specify the years you want game data for. This will generate .csv files corresponding to the years requested.
```python
> python GetBaseBallGameSchedule.py -start 2008 -end 2019
```
Move inside GetBaseballData.py and change the name of the csv to the filename the first script generated.
```python
# Load previous CSV data
#   2018BaseballGames.csv
game_csv = "2008-2018_Baseball_Games.csv"
```
**If needed you can use CombineCSV.py to combine csv files if needed.**

Change the filename for your model inside baseballNetworkCreation.py
```python
players = pd.read_csv("player_lookup.csv")
data = pd.read_csv("Modeling_data.csv")
```

**To start flask on local host:**

Open command prompt and cd to (BASE_DIRECTORY)/tools/flaskbackend/ which will contain app.py then run app.py like the example below

```bash
> python app.py
```
The console will tell you the local address to use.
You can now send post requests to the endpoint $(YOUR BASE)/predict

**Currently expects the following parameters:**

id, balls, strikes, inning, outs, pitch_PA, pitch_game, score_batter, score_pitcher


## License
[MIT](https://choosealicense.com/licenses/mit/)
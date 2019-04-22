# Baseball_Pitch_Prediction
[Baseball Pitch Prediction](https://github.com/Jadams29/Georgia-Tech/tree/master/CSE%206242%20-%20Data%20%26%20Visual%20Analytics/Group%20Project/ui_image.png)

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

**Currently expects the following parameters:**

id, balls, strikes, inning, outs, pitch_PA, pitch_game, score_batter, score_pitcher


## License
[![Licensed under the MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Microsoft/BosqueLanguage/blob/master/LICENSE.txt)
[![PR's Welcome](https://img.shields.io/badge/PRs%20-welcome-brightgreen.svg)](#contribute)

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score
from sklearn import preprocessing
import numpy as np
import itertools

import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import xgboost as xgb


players = pd.read_csv("./data/player_lookup.csv")
data = pd.read_csv("./data/Modeling_data.csv")

players['last'] = players['MLBNAME'].str.split(" ").str.get(-1)
# players[players['last']=='Happ']


def cleanData(df):
    """
    One-hot encode categorical attributes + drop not useful attributes
    input: df (pitch data)
    output: df (cleaned pitched data)
    """
    
    attributes = ['stand','outcomelag3','outcomelag2','outcomelag1',
                  'pitch_typelag1', 'pitch_typelag2', 'pitch_typelag3']
    
    for trait in attributes:
        try:
            dummies = pd.get_dummies(df[trait], prefix = trait)
            df = pd.concat([df, dummies], axis=1)
        except:
            pass
    
    attributes = attributes + ['inning1', 'inning2',
       'inning3', 'inning4', 'inning5', 'inning6', 'inning7', 'inning8',
       'inning9', 'inning10', 'BR_EMPTY', 'BR_1B_2B', 'BR_1B_3B',
        'BR_2B_3B','BR_FULL','batting_order1', 'batting_order2', 'batting_order3', 'batting_order4',
       'batting_order5', 'batting_order6', 'batting_order7', 'batting_order8',
       'batting_order9','pitcher','batter']
    df.drop(attributes,axis=1,inplace=True)
    
    return df


def simplifyOutcomes(df):
    """
    Standardize pitch outcomes.  Shrink dimension space from 23 results to 6 for
    all the "outcome lag" columns
    """
    
    
    subs = {'ball':'ball',
         'called_strike':'strike',
         'catcher_interf':'hit',
         'double':'hit',
         'double_play':'out',
         'field_error':'hit',
         'field_out':'out',
         'fielders_choice':'hit',
         'fielders_choice_out':'out',
         'force_out':'out',
         'foul':'foul',
         'grounded_into_double_play':'out',
         'hit_by_pitch':'hit',
         'home_run':'score',
         'offensive_substitution': 'other',
         'sac_bunt': 'score',
         'sac_bunt_double_play':'score',
         'sac_fly': 'score',
         'sac_fly_double_play': 'score',
         'single': 'hit',
         'swinging_strike':'strike',
         'triple': 'hit',
         'triple_play':'out'}
    
    attributes = ['outcomelag1', 'outcomelag2', 'outcomelag3']
    for trait in attributes:
        df[trait] = df[trait].map(subs)
    
    return df


def Determine_Pitch_Type_To_Keep_Pitcher_Specific(the_pitcher_id, the_data=None):
    print("Determining the PitchTypes to use with pitcher {}\n".format(the_pitcher_id))
    all_pitch_types = list(data.pitch_type.unique())
    # Getting count for each type of pitch
    pitch_type_count_dict = the_data.pitch_type.value_counts()
    num_of_pitch = len(the_data["pitch_type"])
    list_of_pitch_types_used = list(the_data.pitch_type.unique())
    threshold = 0.02
    knuckle_thresh = 0.5
    # Remove woba and swstrike columns relating to pitch types not thrown by the pitcher
    for pitchType in all_pitch_types:
        if pitchType not in list_of_pitch_types_used:
            woba_column_to_remove = "woba.{}".format(pitchType)
            swstrike_column_to_remove = "swstrike_pct.{}".format(pitchType)
            if woba_column_to_remove in the_data.columns:
                the_data.drop(woba_column_to_remove, axis=1, inplace=True)
            if swstrike_column_to_remove in the_data.columns:
                the_data.drop(swstrike_column_to_remove, axis=1, inplace=True)

    for key, value in pitch_type_count_dict.iteritems():
        current_pitch_type_percentage_of_total = value/(len(the_data["pitch_type"]))
        # If the pitcher throws more than knuckle_thresh, KN, then we remove all woba and swstrike columns
        if (key == "KN") and (current_pitch_type_percentage_of_total > knuckle_thresh):
            current_pitch_types = list(the_data.pitch_type.unique())
            for pitch in current_pitch_types:
                if pitch in the_data.columns:
                    the_data.drop(pitch, axis=1, inplace=True)
                if pitch in the_data.columns:
                    the_data.drop(pitch, axis=1, inplace=True)
        # Finds and removes pitch types if they have not been used enough by the pitcher
        #   specified by the threshold
        if current_pitch_type_percentage_of_total < threshold:
            print("Pitch total {}, current pitch type {} and total {}, percentage {}".format(
                (len(the_data["pitch_type"])), key, value, current_pitch_type_percentage_of_total))
            the_data = the_data[the_data.pitch_type != key]
            print("The number of pitches now {}".format(len(the_data["pitch_type"])))


def fillData(df):
    """
    1. Fill NaN values in WOBA fields with mean value of dataset
    2. Fill NaN values for all other fields with 0
    """
    
    attributes = ['woba.FF','woba.SL', 'woba.CH', 'woba.CU', 'woba.FT', 'woba.SI',
                  'woba.FC','woba.FS', 'woba.KC', 'woba.KN','swstrike_pct.FF', 'swstrike_pct.SL',
                  'swstrike_pct.CH', 'swstrike_pct.CU', 'swstrike_pct.FT',
                  'swstrike_pct.SI', 'swstrike_pct.FC', 'swstrike_pct.FS',
                  'swstrike_pct.KC', 'swstrike_pct.KN']#pitch_number','score_diff']
    
    for trait in attributes:
        try:
            df[trait] = df[trait].fillna((df[trait].mean()))
            #df[trait]=(df[trait]-df[trait].mean())/df[trait].std()
            df[trait]=(df[trait]-df[trait].min())/(df[trait].max()-df[trait].min())

        except:
            pass
        
    attributes = ['release_speedlag1', 'release_speedlag2',
       'release_speedlag3', 'avg2_release_speed', 'avg3_release_speed',
       'plate_xlag1', 'plate_xlag2', 'plate_xlag3','plate_xlag1', 
       'plate_xlag2', 'plate_xlag3', 'plate_zlag1',
       'plate_zlag2', 'plate_zlag3', 'avg2_plate_x', 'avg2_plate_z',
       'avg3_plate_x', 'avg3_plate_z', 'pfx_xlag1', 'pfx_xlag2', 'pfx_xlag3',
       'pfx_zlag1', 'pfx_zlag2', 'pfx_zlag3', 'avg2_pfx_x', 'avg2_pfx_z',
       'avg3_pfx_x', 'avg3_pfx_z']
    
    for trait in attributes:
        try:
            df[trait] = df[trait].fillna((0))
            #df[trait]=(df[trait]-df[trait].mean())/df[trait].std()
            df[trait]=(df[trait]-df[trait].min())/(df[trait].max()-df[trait].min())
        except:
            pass
        
    return df



def dataOrchestrator(mainData, reference, playerNames = ["Justin Verlander"]):
    """
    Receives mainData, MLBID reference data, and specific player name.
    
    Runs all the above cleaning code to produce a "cleaned" dataset that can be 
    used as an input for the "createTrainTest" function
    
    """
    
    
    mlbids = list(reference[reference['MLBNAME'].isin(playerNames)]['MLBID'])
    
    
    
    df = mainData[mainData['pitcher'].isin(mlbids)]
    
    Determine_Pitch_Type_To_Keep_Pitcher_Specific(mlbids,df)
    
    """df = df[['game_year','pitch_type', 'pitch_number','game_pitch_number',
             'outs_when_up', 'BR_1B',
       'BR_2B', 'BR_3B','zero_zero_count',
       'zero_one_count', 'zero_two_count', 'one_zero_count', 'one_one_count',
       'one_two_count', 'two_zero_count', 'two_one_count', 'two_two_count',
       'three_zero_count', 'three_one_count', 'three_two_count', 'score_diff',
              'stand','outcomelag1', 'outcomelag2', 'outcomelag3','woba.FF',
       'woba.SL', 'woba.CH', 'woba.CU', 'woba.FT', 'woba.SI', 'woba.FC',
       'woba.FS', 'woba.K C', 'woba.KN','release_speedlag1', 'release_speedlag2',
       'release_speedlag3', 'avg2_release_speed', 'avg3_release_speed','plate_xlag1', 'plate_xlag2', 'plate_xlag3'
             ]]"""
    
    df = simplifyOutcomes(df)
    df = cleanData(df)
    
    df = fillData(df)
    print("Number of rows:{0}".format(len(df)))
    
    return df


def createTrainTest(df,testYear=2018,trainYearStart=2016):
    """
    Create training and testing data outputs
    
    inputs:
        - testYear (int)
        - traingYearStart (int): all data starting from this value up to but not including
                                the testYear
    
    outputs:
        xTrain, yTrain, xTest, yTest numpy arrays
    """
    
    yData = pd.get_dummies(df['pitch_type'])
    
    yData['game_year'] = df['game_year']
    
    xTrain = df[(df['game_year'] < testYear) & (df['game_year'] >= trainYearStart)]
    xTest = df[df['game_year'] >= testYear]
    
    xTrain.drop(['game_year','pitch_type'],axis=1,inplace=True)
    xTest.drop(['game_year','pitch_type'],axis=1,inplace=True)
    
    yTrain = yData[(yData['game_year'] < testYear) & (yData['game_year'] >= trainYearStart)]
    yTest = yData[yData['game_year'] >= testYear]
    
    yTrain.drop(['game_year'],axis=1,inplace=True)
    yTest.drop(['game_year'],axis=1,inplace=True)
    
    xTrain = xTrain.values
    xTest = xTest.values
    yTrain = yTrain.values
    yTest = yTest.values
    
    return xTrain, xTest, yTrain, yTest


##-----------------------------
## Save results and models for each of 40 pitchers   
##-----------------------------
myPitchers = ['Justin Verlander','Jon Lester','Zack Greinke','Max Scherzer',
              'Felix Hernandez','Cole Hamels','CC Sabathia','Clayton Kershaw',
              'David Price','Rick Porcello','J.A. Happ','Madison Bumgarner',
              'Chris Sale','Jose Quintana','Jake Arrieta','Stephen Strasburg',
              'Corey Kluber','Charlie Morton','Chris Archer','Dallas Keuchel',
              'Carlos Carrasco','Gerrit Cole','Trevor Bauer','Wade Davis',
              'Patrick Corbin','Jacob deGrom','Yu Darvish','Carlos Martinez',
              'Masahiro Tanaka','Andrew Miller','James Paxton','Hyun-Jin-Ryu',
              'Craig Kimbrel','Aaron Nola','Luis Severino','Aroldis Chapman',
              'Noah Syndergaard','Blake Snell','Dellin Betances','Steven Wright']

num_pred_all = 0
num_correct_all = 0
out_preds = []

## parameter tuning
## 10 Fold Cross-Validation too expensive because we're
## fitting 40 separate models. Just going to directly judge
## on test set.  
n_estimators = [150,1000]
max_depth = [4,6]
learning_rate = [0.1,0.01]

for name in myPitchers:

    print("Working on:", name)

    cleaned = dataOrchestrator(data, players, playerNames = [name]);
    
    xTrain, xTest, yTrain, yTest = createTrainTest(cleaned,2018,2015)
    
    resultTrain = yTrain.argmax(axis=1)
    resultTest = yTest.argmax(axis=1)
    dtrain = xgb.DMatrix(xTrain, label=resultTrain)
    dtest = xgb.DMatrix(xTest, label=resultTest)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    precision_pitcher = 0
    itr = 0

    naive = cleaned[cleaned['game_year'] ==2018]
    naive = naive.groupby(['pitch_type'])[['pitch_type']].count()
    naive['rate'] = naive / naive.sum() * 100
    naive_guess = naive['rate'].max()

    for (a,b,c) in itertools.product(n_estimators, max_depth, learning_rate):

        itr += 1
        print('   ',name + ':', str(itr)) 

        param = {'max_depth': b,  # the maximum depth of each tree
                 'n_estimators' : a,
                 'learning_rate': c,
                 'eta': 0.1,  # the training step for each iteration
                 "min_child_weight" : 0.5,
                 "gamma" : 1, 
                 "subsample": 1,
                 "colsample_bytree" : 1, 
                 "scale_pos_weight": 1,
                 'objective': 'multi:softprob',  # error evaluation for multiclass training
                 'num_class': yTrain.shape[1],
                 "tree_method" :'gpu_hist',
                 "cv":kfold}  

        num_round = 100  # the number of training iterations

        bst = xgb.train(param, dtrain,num_round)
    
        preds = bst.predict(dtest)
        best_preds = np.asarray([np.argmax(line) for line in preds])
    
        precision = round(precision_score(resultTest, best_preds, average='micro'),4)
        print("Numpy array precision:", round(precision*100,2))

        # Write number correct
        num_pred = len(yTest)
        num_correct = precision_score(resultTest, best_preds, average='micro')*num_pred

        if precision > precision_pitcher: 
            precision_pitcher = precision
            num_pred_pitcher = num_pred
            num_correct_pitcher = num_correct
            n_estimators_pitcher = a
            max_depth_pitcher = b
            learning_rate_pitcher = c
            bst_pitcher = bst


    out_preds.append({'Pitcher': name, 
                      'num_pred': num_pred_pitcher,
                      'num_correct': num_correct_pitcher, 
                      'accuracy': precision_pitcher*100,
                      'naive': naive_guess,
                      'plus_minus': precision_pitcher*100 - naive_guess,
                      'n_estimators': n_estimators_pitcher,
                      'max_depth': max_depth_pitcher,
                      'learning_rate': learning_rate_pitcher})

    num_pred_all += num_pred_pitcher
    num_correct_all += num_correct_pitcher

    # save model to file
    pickle.dump(bst_pitcher, open("models/"+name+".dat", "wb"))


results_out = pd.DataFrame(out_preds)
results_out.to_csv("./results_40pitchers.csv")

total_accuracy = num_correct_all/num_pred_all*100
print(total_accuracy)
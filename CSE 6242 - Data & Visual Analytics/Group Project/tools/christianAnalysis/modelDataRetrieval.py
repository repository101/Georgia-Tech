#author: Christian Rivera and Josh Adams

import pickle
import pandas as pd
import numpy as np
import inflect
import xgboost as xgb


import warnings
warnings.filterwarnings('ignore')

# Reference lists to find index in numpy array model input
strikesBalls = ['zero_zero_count',
       'zero_one_count', 'zero_two_count', 'one_zero_count', 'one_one_count',
       'one_two_count', 'two_zero_count', 'two_one_count', 'two_two_count',
       'three_zero_count','three_one_count', 'three_two_count']

outcomes = ["ball","foul","hit","out","score","strike"]

all_pitches = ['FF', 'CH', 'FT', 'KC', 'CU', 'SL', 'SI', 'FC', 'FS', 'KN']



numConverter = inflect.engine()
##############################################################

# take in input data, convert it into numpy array, load model and return
# probability distribution

def modelPrediction( data,players,pitcher = "Justin Verlander",
                    batter="Mike Trout",pitch_number=0,
                    game_pitch_number=0,outs_when_up=0,
                    BR_1B=0,BR_2B=0,BR_3B=0,
                    balls=0,strikes=1,
                     pitcher_score=0,batter_score=0,
                    stand="L",outcomelag1="ball",
                    outcomelag2="ball",outcomelag3="ball",
                     pitchlag1=None,pitchlag2=None,pitchlag3=None):
    
    score_diff = pitcher_score - batter_score
    
    batter = players[players['MLBNAME'] == "Mike Trout"]['MLBID'].iloc[0]
    
    batterData = data[data['batter']==batter].iloc[-1]
    #bat['avg3_pfx_x']
    
    input_data = dataOrchestrator(data, players, playerNames = [pitcher]);
    
    pitches = list(pd.get_dummies(input_data['pitch_type']).columns)
    print(pitches)
    xTrain, xTest, yTrain, yTest = createTrainTest(input_data,2018,2015)
    
    input_data = input_data[input_data['game_year'] == 2018]
    record = np.zeros(shape=(xTrain.shape[1],))
    record[:6] = pitch_number,game_pitch_number,outs_when_up,BR_1B,BR_2B,BR_3B
    ballStrikeScenario = numConverter.number_to_words(balls)+"_"+numConverter.number_to_words(strikes)+"_count"
    
    strikeBallIndex = strikesBalls.index(ballStrikeScenario)
   
    record[6+strikeBallIndex] = 1
    
    record[18] = score_diff 
    
    record[19:22] = input_data["release_speedlag1"].mean(),input_data["release_speedlag2"].mean(),input_data["release_speedlag2"].mean()
    
    record[22:24] = input_data["avg2_release_speed"].mean(),input_data["avg3_release_speed"].mean()
    
    record[24:27] = input_data["plate_xlag1"].mean(),input_data["plate_xlag2"].mean(),input_data["plate_xlag3"].mean()
    
    record[27:30] = input_data["plate_zlag1"].mean(),input_data["plate_zlag2"].mean(),input_data["plate_zlag3"].mean()

    record[30:32] = input_data["avg2_plate_x"].mean(),input_data["avg2_plate_z"].mean()
    
    record[32:34] = input_data["avg3_plate_x"].mean(),input_data["avg3_plate_z"].mean()
    
    record[34:37] = input_data["pfx_xlag1"].mean(),input_data["pfx_xlag2"].mean(),input_data["pfx_xlag3"].mean()
    
    record[37:40] = input_data["pfx_zlag1"].mean(),input_data["pfx_zlag2"].mean(),input_data["pfx_zlag3"].mean()

    record[40:42] = input_data["avg2_pfx_x"].mean(),input_data["avg2_pfx_z"].mean()
    
    record[42:44] = input_data["avg3_pfx_x"].mean(),input_data["avg3_pfx_z"].mean()
    
    bookmark = 44
    for pitchtype in pitches:
        record[bookmark] = batterData["woba."+pitchtype]
        bookmark +=1
        
    for pitchtype in pitches:
        record[bookmark] = batterData["swstrike_pct."+pitchtype]
        bookmark +=1
    
    if stand =="L":
        record[bookmark]=1
    else:
        record[bookmark+1]=1
        
    bookmark +=2
    record[bookmark+ outcomes.index(outcomelag3)] = 1
    bookmark +=6
    record[bookmark+ outcomes.index(outcomelag2)] = 1
    bookmark +=6
    record[bookmark+ outcomes.index(outcomelag1)] =1 
    
    bookmark +=6
    record[bookmark+ pitches.index(pitchlag1)] = 1
    
    bookmark +=len(pitches)
    record[bookmark+ pitches.index(pitchlag2)] = 1
    
    bookmark +=len(pitches)
    record[bookmark+ pitches.index(pitchlag3)] = 1
     
    resultTest = yTest.argmax(axis=1)

    dtest = xgb.DMatrix(np.array([record]), label=resultTest)

    loaded_model = pickle.load(open("models/"+pitcher+".dat", "rb"))
    
    return dict(zip(pitches, np.round(loaded_model.predict(dtest)[0] * 100,2)))


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
    #print("Determining the PitchTypes to use with pitcher {}\n".format(the_pitcher_id))
    all_pitch_types = ['FF', 'CH', 'FT', 'KC', 'CU', 'SL', 'SI', 'FC', 'FS', 'KN']
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
            #print("Pitch total {}, current pitch type {} and total {}, percentage {}".format(
            #    (len(the_data["pitch_type"])), key, value, current_pitch_type_percentage_of_total))
            the_data = the_data[the_data.pitch_type != key]
            #print("The number of pitches now {}".format(len(the_data["pitch_type"])))


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
       'woba.FS', 'woba.KC', 'woba.KN','release_speedlag1', 'release_speedlag2',
       'release_speedlag3', 'avg2_release_speed', 'avg3_release_speed','plate_xlag1', 'plate_xlag2', 'plate_xlag3'
             ]]"""
    
    df = simplifyOutcomes(df)
    df = cleanData(df)
    
    df = fillData(df)
    #print("Number of rows:{0}".format(len(df)))
    
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
    



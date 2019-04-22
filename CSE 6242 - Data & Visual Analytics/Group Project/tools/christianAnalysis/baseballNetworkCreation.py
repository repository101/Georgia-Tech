#!/usr/bin/env python
# coding: utf-8

# # Baseball Pitch Data Analysis
# Class: cse6242
# Christian Rivera
# Team: Philly Philly

# In[1]:


import pandas as pd
from sklearn import tree 
from sklearn.metrics import accuracy_score
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Activation
from keras import optimizers


# ## Load the data sets

# In[2]:


players = pd.read_csv("player_lookup.csv")
data = pd.read_csv("Modeling_data.csv")

all_pitch_types = list(data.pitch_type.unique())


# In[3]:


data.columns


# ## Below are the functions to clean the data, select a player and create training and testing data.  All the cleaning functions are called by the dataOrchestrator function so that the user be only concerned with the dataOrchestrator and createTrainTest functions

# In[5]:


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


# In[6]:


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
    


# In[7]:


def Determine_Pitch_Type_To_Keep_Pitcher_Specific(the_pitcher_id, the_data=None):
    print("Determining the PitchTypes to use with pitcher {}\n".format(the_pitcher_id))
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


# In[8]:


def fillData(df):
    """
    1. Fill NaN values in WOBA fields with mean value of dataset
    2. Fill NaN values for all other fields with 0
    """
    
    attributes = ['woba.FF','woba.SL', 'woba.CH', 'woba.CU', 'woba.FT', 'woba.SI',
                  'woba.FC','woba.FS', 'woba.KC', 'woba.KN','swstrike_pct.FF', 'swstrike_pct.SL',
       'swstrike_pct.CH', 'swstrike_pct.CU', 'swstrike_pct.FT',
       'swstrike_pct.SI', 'swstrike_pct.FC', 'swstrike_pct.FS',
       'swstrike_pct.KC', 'swstrike_pct.KN']
    
    for trait in attributes:
        try:
            df[trait] = df[trait].fillna((df[trait].mean()))
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
        except:
            pass
        
    return df


# In[9]:


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
    


# In[10]:


def dataOrchestrator(mainData, reference, playerName = "Justin Verlander"):
    """
    Receives mainData, MLBID reference data, and specific player name.
    
    Runs all the above cleaning code to produce a "cleaned" dataset that can be 
    used as an input for the "createTrainTest" function
    
    """
    
    
    mlbid = int(reference[reference['MLBNAME'] ==playerName]['MLBID'].iloc[0])
    
    
    
    df = mainData[mainData['pitcher'] ==mlbid]
    
    Determine_Pitch_Type_To_Keep_Pitcher_Specific(mlbid,df)
    
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
    print("Number of rows:{0}".format(len(df)))
    
    return df
    
    


# In[11]:


cleaned = dataOrchestrator(data, players, playerName = 'Justin Verlander')


# In[12]:


xTrain, xTest, yTrain, yTest = createTrainTest(cleaned,2018,2015)


# ## Decision Tree.  Used as a baseline

# In[16]:



clf = tree.DecisionTreeClassifier()

clf.fit(xTrain,yTrain)
results = clf.predict(xTest)

print(accuracy_score(yTest,results))


# ## Keras neural network

# In[13]:


model = Sequential()
model.add(Dense(1024, input_shape=(xTrain.shape[1],), activation="sigmoid"))
model.add(Dropout(0.3))
model.add(Dense(512,activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(yTrain.shape[1], activation="relu"))


sgd = optimizers.SGD(lr=0.01, decay=1e-8, momentum=0.8, nesterov=False)
model.compile(loss="categorical_crossentropy",optimizer = sgd, metrics=['mae', 'acc'])

model.fit(xTrain,yTrain,epochs=5,batch_size=32)


# ### The result is a probability matrix showing the probability of each pitch.  Find the highest probability and round to 1 while rounding all other values to zero.  Then compare that with the yTest data to find accuracy.
# 

# In[14]:


kerasResults = model.predict(xTest)

row_maxes = kerasResults.max(axis=1).reshape(-1, 1)
kerasResults[:] = np.where(kerasResults == row_maxes, 1, 0)

accuracy_score(yTest,kerasResults)


# In[ ]:





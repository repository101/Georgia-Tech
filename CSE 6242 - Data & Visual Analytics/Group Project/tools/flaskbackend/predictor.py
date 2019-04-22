import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor


class pitch_predictor(object):
    def __init__(self):
        pass

    def deserialize(self,id):
        # de-serialize mlp_nn.pkl file into an object called model using pickle

        with open('Modela/{}.pkl'.format(id), 'rb') as handle:
            model = pickle.load(handle)
        return model

    #currently has these form elements:Pitcher
    #Balls
    #Strikes
    #Inning
    #Outs
    #Pitch # PA
    #Pitch # Game
    #Score Batter
    #Score Pitcher

#game_year,pitch_type,pitch_number,game_pitch_number,outs_when_up,BR_1B,BR_2B,BR_3B,zero_zero_count,zero_one_count,
    # zero_two_count,one_zero_count,one_one_count,one_two_count,two_zero_count,two_one_count,two_two_count,three_zero_count,
    # three_one_count,three_two_count,score_diff,release_speedlag1,release_speedlag2,release_speedlag3,avg2_release_speed,avg3_release_speed,
    # plate_xlag1,plate_xlag2,plate_xlag3,plate_zlag1,plate_zlag2,plate_zlag3,avg2_plate_x,avg2_plate_z,avg3_plate_x,avg3_plate_z,pfx_xlag1,
    # pfx_xlag2,pfx_xlag3,pfx_zlag1,pfx_zlag2,pfx_zlag3,avg2_pfx_x,avg2_pfx_z,avg3_pfx_x,avg3_pfx_z,woba.FF,woba.SL,woba.CH,
    # woba.CU,woba.FT,woba.FC,swstrike_pct.FF,swstrike_pct.SL,swstrike_pct.CH,swstrike_pct.CU,swstrike_pct.FT,swstrike_pct.FC,stand_L,
    # stand_R,outcomelag3_ball,outcomelag3_foul,outcomelag3_hit,outcomelag3_out,outcomelag3_score,outcomelag3_strike,outcomelag2_ball,
    # outcomelag2_foul,outcomelag2_hit,outcomelag2_out,outcomelag2_score,outcomelag2_strike,outcomelag1_ball,outcomelag1_foul,outcomelag1_hit,
    # outcomelag1_out,outcomelag1_score,outcomelag1_strike,pitch_typelag1_180523_181144,pitch_typelag1_CH,pitch_typelag1_CU,pitch_typelag1_FC,
    # pitch_typelag1_FF,pitch_typelag1_FT,pitch_typelag1_IN,pitch_typelag1_PO,pitch_typelag1_SL,pitch_typelag2_180523_181127,
    # pitch_typelag2_180523_181144,pitch_typelag2_CH,pitch_typelag2_CU,pitch_typelag2_FC,pitch_typelag2_FF,pitch_typelag2_FT,pitch_typelag2_IN,
    # pitch_typelag2_PO,pitch_typelag2_SL,pitch_typelag3_180523_181101,pitch_typelag3_180523_181127,pitch_typelag3_180523_181144,
    # pitch_typelag3_CH,pitch_typelag3_CU,pitch_typelag3_FC,pitch_typelag3_FF,pitch_typelag3_FT,pitch_typelag3_IN,pitch_typelag3_PO,
    # pitch_typelag3_SL



    def predict(self,id, balls, strikes, inning, outs, pitchPA, pitch_game, score_batter, score_pitcher):
        model = self.deserialize(id)
        parameter_array = self.build_parameter_values(id, balls, strikes, inning, outs, pitchPA, pitch_game, score_batter, score_pitcher)
        return model.predict(parameter_array)

    def build_parameter_values(self, id, balls, strikes, inning, outs, pitchPA, pitch_game, score_batter, score_pitcher):
        return []

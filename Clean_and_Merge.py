from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class Clean_and_Merge(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        categorical_attributes = ['preferred_foot', 'attacking_work_rate', 'defensive_work_rate']

        numercial_attributes = ['potential', 'crossing', 'finishing', 'heading_accuracy',
            'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
            'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
            'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
            'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
            'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
            'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
            'gk_reflexes']
        player_id = df['player_fifa_api_id'].unique()
        
        new_df = pd.DataFrame()

        for i in player_id:
            index = np.where(df['player_fifa_api_id'] == i)[0]

            temp_num = df.iloc[index][numercial_attributes].mean()
            temp_cat = df.iloc[index][categorical_attributes].mode().iloc[0]

            temp_df = pd.DataFrame(data=[temp_num.values], columns=temp_num.index)

            temp_df[temp_cat.index] = temp_cat.values

            new_df = new_df.append(temp_df)
            
        
        to_drop = ['norm', 'y', 'le', 'stoc']

        for i in to_drop:
            new_df['attacking_work_rate'].replace(i, np.nan, inplace=True)
            
        to_drop = ['ormal', 'ean', 'es', 'tocky', '_0', 'o']

        for i in to_drop:
            new_df['defensive_work_rate'].replace(i, np.nan, inplace=True)
            
            
        low_class = ['0', '1', '2']
        medium_class = ['3', '4', '5', '6']
        high_class = ['7', '8', '9']

        for i in low_class:
            new_df['defensive_work_rate'].replace(i, 'low', inplace=True)

        for i in medium_class:
            new_df['defensive_work_rate'].replace(i, 'medium', inplace=True)

        for i in high_class:
            new_df['defensive_work_rate'].replace(i, 'high', inplace=True)
            
            
        return new_df.reset_index(drop=True)
    
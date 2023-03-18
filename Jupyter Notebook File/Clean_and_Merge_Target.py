from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class Clean_and_Merge_Target(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        target_attribute = ['overall_rating']
        
        df.dropna(thresh=10, inplace=True)
        player_id = df['player_fifa_api_id'].unique()
        
        new_df = pd.DataFrame()

        for i in player_id:
            index = np.where(df['player_fifa_api_id'] == i)[0]

            temp_num = df.iloc[index][target_attribute].mean()

            temp_df = pd.DataFrame(data=[temp_num.values], columns=temp_num.index)

            new_df = new_df.append(temp_df)
            
        
        return new_df.reset_index(drop=True)
    
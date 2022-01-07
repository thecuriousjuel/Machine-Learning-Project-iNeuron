from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import joblib
import numpy as np
from functions import Clean_and_Merge

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def homepage():
    return render_template('index.html')


def transform_and_process_data(df):
    initial_transformer = joblib.load('pipeline/07_01_2022-12_56_07_custom_attribute_object.pkl')
    df = initial_transformer.transform(df)

    transformer = joblib.load('pipeline/07_01_2022-12_56_07_full_transformer.pkl')
    transformed_data = transformer.transform(df)

    model = joblib.load('model/07_01_2022-12_56_31_best_model.pkl')
    predicted_value = model.predict(transformed_data)

    print('Predicted Value : ', predicted_value)
    
    target_transformer = joblib.load('pipeline/07_01_2022-12_56_07_target_pipeline.pkl')
    actual_pred_value = np.round(target_transformer.inverse_transform([predicted_value]), 2)

    print('Actual predicted value : ', actual_pred_value)


@app.route('/file_predict', methods = ['POST'])
def file_predict():
    pass



@app.route('/predict', methods = ['POST'])
def predict():
    data = dict(potential = request.form['customRange2'],
    crossing = request.form['customRange3'],
    finishing = request.form['customRange4'],
    heading_accuracy = request.form['customRange5'],
    short_passing = request.form['customRange6'],
    volleys = request.form['customRange7'],
    dribbling = request.form['customRange8'],
    curve = request.form['customRange9'],
    free_kick_accuracy = request.form['customRange10'],
    long_passing = request.form['customRange11'],
    ball_control = request.form['customRange12'],
    acceleration = request.form['customRange13'],
    sprint_speed = request.form['customRange14'],
    agility = request.form['customRange15'],
    reactions = request.form['customRange16'],
    balance = request.form['customRange17'],
    shot_power = request.form['customRange18'],
    jumping = request.form['customRange19'],
    stamina = request.form['customRange20'],
    strength = request.form['customRange21'],
    long_shots = request.form['customRange22'],
    aggression = request.form['customRange23'],
    interceptions = request.form['customRange24'],
    positioning = request.form['customRange25'],
    vision = request.form['customRange26'],
    penalties = request.form['customRange27'],
    marking = request.form['customRange28'],
    standing_tackle = request.form['customRange29'],
    sliding_tackle = request.form['customRange30'],
    gk_diving = request.form['customRange31'],
    gk_handling = request.form['customRange32'],
    gk_kicking = request.form['customRange33'],
    gk_positioning = request.form['customRange34'],
    gk_reflexes = request.form['customRange35'],
    preferred_foot=request.form['preferred_foot'],
    attacking_work_rate = request.form['attacking_work_rate'],
    defensive_work_rate = request.form['defensive_work_rate'],
    player_fifa_api_id = 'Not Required')

    player_data = pd.DataFrame(data=[data.values()], columns=data.keys())
    
    player_data.dropna(thresh=5, inplace=True)

    transform_and_process_data(player_data)



    # new_dict = dict(pred_value = actual_pred_value[0,0],
    #                 Max_Rating = 94, Min_Rating = 33)

    # print(new_dict)

    # new_dict['Percentage'] = np.round(((new_dict['pred_value'] - new_dict['Min_Rating']) / (new_dict['Max_Rating'] - new_dict['Min_Rating'])) * 100, 2)

    # return render_template('output.html', new_dict=new_dict)

if __name__ == '__main__':
    app.run(debug=True)


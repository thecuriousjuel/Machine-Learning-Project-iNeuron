from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1000


@app.route('/', methods=['POST', 'GET'])
def homepage():
    return render_template('index.html')


@app.route('/form', methods=['POST', 'GET'])
def formpage():
    return render_template('form.html')


def transform_and_process_data(df):
    df.dropna(thresh=5, inplace=True)

    player_fifa_api_id = df['player_fifa_api_id'].unique() 

    initial_transformer = joblib.load('pipeline/07_01_2022-22_02_52_custom_attribute_object.pkl')
    df = initial_transformer.transform(df)    

    transformer = joblib.load('pipeline/07_01_2022-22_02_52_full_transformer.pkl')
    transformed_data = transformer.transform(df)

    model = joblib.load('model/07_01_2022-22_03_13_best_model.pkl')
    predicted_value = model.predict(transformed_data)

    # print('Predicted Value : ', predicted_value)
    
    target_transformer = joblib.load('pipeline/07_01_2022-22_02_52_target_pipeline.pkl')
    actual_pred_value = np.round(target_transformer.inverse_transform(predicted_value.reshape(-1, 1)), 2)

    # print('Actual predicted value : ', actual_pred_value)
    new_dict = {}

    for i in range(len(player_fifa_api_id)):
        temp_dict = {}
        temp_dict['id'] = i + 1
        temp_dict['pred_value'] = actual_pred_value[i, 0]
        # print(actual_pred_value[i, 0])

        # temp_dict['Max_Rating'] = 94
        # temp_dict['Min_Rating'] = 33

        # temp_dict['percentage'] = np.round(((temp_dict['pred_value'] - 33) / (94 - 33)) * 100, 2)

        new_dict[player_fifa_api_id[i]] = temp_dict

    return new_dict


@app.route('/file_predict', methods = ['POST'])
def file_predict():
    filename = request.files['filename']
    player_data = pd.read_csv(filename)
    
    output_dict = transform_and_process_data(player_data)
    # print('output_dict : ', output_dict)

    return render_template('output.html', output_dict=output_dict)


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

    output_dict = transform_and_process_data(player_data)
    # print('output_dict : ', output_dict)

    return render_template('output.html', output_dict=output_dict)

if __name__ == '__main__':
    app.run(debug=False)


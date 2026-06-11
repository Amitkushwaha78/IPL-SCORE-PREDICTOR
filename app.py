from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

model = joblib.load('models/model.pkl')

# =========================
# IPL 2026 TEAMS
# =========================

teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

# =========================
# HOME PAGE
# =========================

@app.route('/')
def index():

    return render_template(
        'index.html',
        teams=teams
    )

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    # =========================
    # GET FORM DATA
    # =========================

    batting_team = request.form['batting_team']

    bowling_team = request.form['bowling_team']

    current_score = int(
        request.form['current_score']
    )

    wickets_left = int(
        request.form['wickets_left']
    )

    overs = float(
        request.form['overs']
    )

    # =========================
    # VALIDATIONS
    # =========================

    if batting_team == bowling_team:

        return render_template(
            'index.html',
            prediction="Teams cannot be same!",
            teams=teams
        )

    if overs <= 0 or overs > 20:

        return render_template(
            'index.html',
            prediction="Overs must be between 0.1 and 20",
            teams=teams
        )

    if wickets_left < 0 or wickets_left > 10:

        return render_template(
            'index.html',
            prediction="Wickets must be between 0 and 10",
            teams=teams
        )

    # =========================
    # MATCH CALCULATIONS
    # =========================

    balls_bowled = int(overs * 6)

    balls_left = 120 - balls_bowled

    current_rr = current_score / overs

    overs_left = 20 - overs

    # =========================
    # MODEL INPUT
    # =========================

    input_df = pd.DataFrame({

        'batting_team': [batting_team],

        'bowling_team': [bowling_team],

        'current_score': [current_score],

        'balls_left': [balls_left],

        'wickets_left': [wickets_left],

        'crr': [current_rr]

    })

    # =========================
    # MACHINE LEARNING SCORE
    # =========================

    ml_prediction = model.predict(input_df)[0]

    # =========================
    # CONSERVATIVE IPL LOGIC
    # =========================

    if wickets_left >= 8:

        boost_rr = 1.8

    elif wickets_left >= 6:

        boost_rr = 1.2

    elif wickets_left >= 4:

        boost_rr = 0.6

    elif wickets_left >= 2:

        boost_rr = 0

    else:

        boost_rr = -1.5

    # Expected run rate
    expected_rr = current_rr + boost_rr

    # Realistic limits
    expected_rr = max(expected_rr, 5)

    expected_rr = min(expected_rr, 13)

    # Cricket logic prediction
    logic_score = current_score + (
        expected_rr * overs_left
    )

    # =========================
    # HYBRID PREDICTION
    # =========================

    prediction = (
        ml_prediction * 0.30 +
        logic_score * 0.70
    )

    # =========================
    # REALISTIC CAPS
    # =========================

    prediction = max(
        prediction,
        current_score
    )

    prediction = min(
        prediction,
        240
    )

    # Collapse handling
    if wickets_left <= 2:

        prediction = min(
            prediction,
            current_score + (overs_left * 7)
        )

    # Powerplay boost
    if overs <= 6 and wickets_left >= 8:

        prediction += 5

    # Death overs boost
    if overs >= 16 and wickets_left >= 5:

        prediction += 4

    # Final integer conversion
    prediction = int(prediction)

    # =========================
    # RETURN RESULT
    # =========================

    return render_template(

        'index.html',

        prediction=prediction,

        teams=teams

    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

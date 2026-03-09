from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(r"C:\Users\janaa\Python\travel_model_new.pkl")

def recommend_places(user_input, top_n=5):

    df = pd.DataFrame([user_input])

    probs = model.predict_proba(df)[0]

    places = model.named_steps["model"].classes_

    top_indices = np.argsort(probs)[::-1][:top_n]

    recommendations = [places[i] for i in top_indices]

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        user = {
    "Country": request.form["country"],
    "Season": request.form["season"],
    "Trip_Type": request.form["trip_type"],
    "Budget": float(request.form["budget"]),
    "Avg_Temp": float(request.form["temp"]),
    "Rain_Level": request.form["rain"],
    "Duration_Days": int(request.form["duration"]),
    "Crowd_Level": request.form["crowd"]
}

        places = recommend_places(user)

        return render_template("index.html", recommendations=places)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
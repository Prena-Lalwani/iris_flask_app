import csv
from datetime import datetime
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model, iris dataset, and accuracy
with open("iris_model.pkl", "rb") as f:
    model, iris, accuracy = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            ]
            pred = model.predict([features])[0]
            prediction = iris.target_names[pred]
        except:
            prediction = "Invalid input."

        # ✅ Save to CSV history
        with open("history.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(),
                *features,
                prediction
            ])

    return render_template("form.html", prediction=prediction, accuracy=accuracy)

# ✅ History route
@app.route("/history")
def history():
    rows = []
    try:
        with open("history.csv", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)
    except FileNotFoundError:
        pass  # No predictions yet

    return render_template("history.html", rows=rows)

# ✅ Start Flask app
if __name__ == "__main__":
    app.run(debug=True)

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

    return render_template("form.html", prediction=prediction, accuracy=accuracy)

# ✅ This line starts your Flask app
if __name__ == "__main__":
    app.run(debug=True)

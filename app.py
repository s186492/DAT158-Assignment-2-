from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("logistic_regression.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            prediction = model.predict([message])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import predictions, base64
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
@cross_origin()
def index_page():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_output():
    if request.method == "POST":
        image_name = request.form["img"]
        image_string = request.form["imagestr"]
        image_bytestrings = base64.b64decode(image_string)
        if not os.path.exists("./images"):
            os.makedirs("./images")
        with open("./images/{}".format(image_name), "wb") as f:
            f.write(image_bytestrings)
            a = predictions.A(f"./images/{image_name}")
            return jsonify(output = a.predic_out())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
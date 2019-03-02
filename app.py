from flask import Flask, jsonify
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

app = Flask(__name__)
def init():
    global model,graph
    # load the pre-trained Keras model
    with open('classifier.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('classifier.h5')
    graph = tf.get_default_graph()


@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>
    <img src="http://loremflickr.com/600/400" />
    """.format(time=the_time)


@app.route("/predict", methods=["GET","POST"])
def predict():
    inputFeature = np.asarray([99,80,21]).reshape(1, 3)
    raw_prediction = model.predict(inputFeature)[0][0]
    data = {"success": raw_prediction}
    return jsonify(data)  

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server...please wait until server has fully started"))
    init()
    app.run(debug=True, use_reloader=True)

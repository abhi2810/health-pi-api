import flask
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

app = flask.Flask(__name__)
with open('classifier.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('classifier.h5')
graph = tf.get_default_graph()
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
    parameters = []
    parameters.append(flask.request.args.get('temp'))
    parameters.append(flask.request.args.get('hr'))
    parameters.append(flask.request.args.get('bmi'))
    if len(parameters) == 3 :
        inputFeature = np.asarray(parameters).reshape(1, 3)
        with graph.as_default():
            raw_prediction = model.predict(inputFeature)[0][0]
        data = {"score": str(raw_prediction)}
    else:
        data = {"score": "0"}
    return flask.jsonify(data)  

if __name__ == '__main__':
    init()
    app.run(debug=True, use_reloader=True)

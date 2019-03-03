import flask
from datetime import datetime
import pytz
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

app = flask.Flask(__name__)
with open('classifier.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('classifier.h5')
graph = tf.get_default_graph()

cred = credentials.Certificate('cred/healthmeter-b5ac6-da53f53b29bb.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


timezone = pytz.timezone("Asia/Kolkata")

def init():
    global model,graph
    # load the pre-trained Keras model
    with open('classifier.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('classifier.h5')
    graph = tf.get_default_graph()


@app.route('/')
def homepage():
    the_time = datetime.now(timezone).strftime("%A, %d %b %Y %l:%M %p")
    return """
    <h1>Hello Bro!</h1>
    <p>It is currently {time}.</p>
    <p>It is really easy to use this API.<br>
    Just make a GET request to the url/predict with temp, hr and bmi.<br>
    You'll get a json with the healthscore.</p>
    <p>BTW, Enjoy this picture of a cat.</p>
    <img src="http://loremflickr.com/600/400" />
    """.format(time=the_time)

@app.route("/pistore", methods=["GET","POST"])
def pistore():
    parameters = []
    date = datetime.now(timezone).strftime("%d %b %Y")
    time = datetime.now(timezone).strftime("%l:%M %p")
    parameters.append(flask.request.args.get('temp'))
    parameters.append(flask.request.args.get('hr'))
    parameters.append(flask.request.args.get('bmi'))
    uuid = flask.request.args.get('uuid')
    if parameters[0] :
        inputFeature = np.asarray(parameters).reshape(1, 3)
        with graph.as_default():
            raw_prediction = model.predict(inputFeature)[0][0]
        score = str(raw_prediction)
    else:
        score = 0
    doc_ref = db.collection(u'user').document(uuid).collection(u'sensordata')
    doc_ref.add({
        u'temp': float(parameters[0]),
        u'hr': float(parameters[1]),
        u'date': date,
        u'time': time,
        u'score': float(score)
    })
    return 'valid'
    

@app.route("/predict", methods=["GET","POST"])
def predict():
    parameters = []
    parameters.append(flask.request.args.get('temp'))
    parameters.append(flask.request.args.get('hr'))
    parameters.append(flask.request.args.get('bmi'))
    if parameters[0] :
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

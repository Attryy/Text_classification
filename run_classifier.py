# -*- coding: utf-8 -*-


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import flask
import io
import os

# initialize our Flask application and the SVM model
app = flask.Flask(__name__)
model = None

def init_models():
    global model
    global vectorizer
    global label_map
    label_map = {0: 'Pinot Noir', 1: 'Chardonnay', 2: 'Cabernet Sauvignon', 3: 'Red Blend', 4: 'Bordeaux-style Red Blend',5: 'Riesling',6: 'Sauvignon Blanc',7: 'Syrah',8: 'Rosé',9: 'Merlot',10: 'Nebbiolo',11: 'Zinfandel',12: 'Sangiovese',13: 'Malbec',14: 'Portuguese Red',15: 'White Blend',16: 'Sparkling Blend',17: 'Tempranillo',18: 'Rhône-style Red Blend',19: 'Pinot Gris',20: 'Champagne Blend',21: 'Cabernet Franc',22: 'Grüner Veltliner',23: 'Portuguese White',24: 'Bordeaux-style White Blend'} 
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    model = pickle.load(open(os.path.join(PROJECT_ROOT,'svcModel.sav'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(PROJECT_ROOT, 'vectorizerDesc.pk'), 'rb'))
	


@app.route("/classifier", methods=["POST"])
def classification():
    result = {"prediction":None}
    req_desc = flask.request.get_json()
    desc = vectorizer.transform(req_desc)
    r=model.predict(desc)[0]
    result["prediction"]= label_map[r]
    return flask.jsonify(result)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading pre-trained model and Flask starting server..."
        "please wait until server has fully started"))
    init_models()
    app.run(host='localhost', port=5252)

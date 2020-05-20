import os
import json
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from profanity_check import predict, predict_prob

model = pickle.load(open('model.pkl', 'rb'))

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

objClass = {0:'discours haineux', 1 : 'discours offensant', 2:'discours neutre'}

def app():
    app = Flask(__name__)


    @app.route('/', methods=['GET'])
    def root():
        return render_template('index.html')

    @app.route('/tweet', methods=['POST'])
    def tweet():
        data = request.form
        if("neither" in data and "offensive" in data and "hate" in data):
            # output = predict([data['message']])
            values_form = [int(data['hate']),int(data['offensive']), int(data['neither'])]
            print(values_form)
            features = [np.array(values_form)]
            prediction = model.predict(features)
            resultList = "Pour un tweet comportant : " + str(data['hate']) + " mot(s) haineux, " + str(data['offensive']) + " mot(s) offensant(s), et " + str(data['neither']) + " mots neutres : "
            return render_template('index.html', result='Le texte est un {}'.format(objClass[prediction[0]]), listKeyWordVal = resultList )
        else:
            return render_template('index.html', error='Veuillez renseigner les champs.')
    return app

if __name__ == '__main__':
    app = app()
    app.run(debug=True, host='0.0.0.0')
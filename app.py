from flask import Flask, jsonify
from flask import render_template, request
import os
import json
import pandas as pd
import numpy

def default(o):
    if isinstance(o, numpy.int64): return int(o)  
    raise TypeError

LIMIT_URL = 1000

app = Flask(__name__)

models = ['RandomForest', 'AdaBoost', 'KNearestNeighbors', 'NaiveBayes']
df = {}
for model in models:
  df[model] = pd.read_csv('./web_data/{}.csv'.format(model))

urls = df[models[0]].head(LIMIT_URL)[['id', 'url']]
urls.columns = ['id', 'text']
urls = urls.to_dict('records')

@app.route('/api/predict/<model>/<id>')
def api_predict(model='', id=0):
  if model not in models: return {}
  record = df[model][df[model].id == int(id)].iloc[0]
  return jsonify(data=json.loads(json.dumps(record.to_dict(), default=default)))

@app.route('/api/url')
def api_url():
  return jsonify(urls=urls)

@app.route('/api/models')
def api_models():
  return jsonify(models=models)

@app.route('/')
def hello_world():
    return render_template('index.html')

if __name__ == '__main__':
    DEBUG = bool(os.getenv('DEBUG', 1))
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8088))
    app.run(host=HOST, port=PORT, debug=DEBUG)
    print('Server is running at http://{}:{}, DEBUG={}'
          .format(HOST, PORT, DEBUG))

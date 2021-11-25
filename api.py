import numpy as np
from helper import predict 
from flask import Flask, render_template,jsonify, request
#from flask import request, jsonify, make_response, send_file, render_template
app=Flask(__name__, template_folder='static/front')

@app.route('/')
def home():
   return render_template('index.html')

@app.route("/predict", methods=['POST'])
def prediction():
	data = request.json['uri']
	res = predict(data)
	return jsonify({"msg":res})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
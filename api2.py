from flask import Flask,jsonify
from i import get_pred
 app = Flask(__name__)
 @app.route('/pred-digit',methods = ['POST'])
def pred_data():
    img = request.files.get('digit')
    prediction = get_pred(img)
    return jsonify({"prediction":prediction}),200
if __name__ == '__main__':
    app.run(debug = True)


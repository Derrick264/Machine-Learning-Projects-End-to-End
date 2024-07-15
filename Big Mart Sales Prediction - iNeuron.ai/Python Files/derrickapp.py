from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    Item_Weight = float(request.form['Item_Weight'])
    Item_Fat_Content = float(request.form['Item_Fat_Content'])
    Item_Visibility = float(request.form['Item_Visibility'])
    Item_Type = float(request.form['Item_Type'])
    Item_Mrp = float(request.form['Item_Mrp'])
    Outlet_Establishment_Year = float(request.form['Outlet_Establishment_Year'])
    Outlet_Size = float(request.form['Outlet_Size'])
    Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
    Outlet_Type = float(request.form['Outlet_Type'])

    x = np.array([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_Mrp,
                   Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type]])

    scaler_path = r'sc.sav'
    sc = joblib.load(scaler_path)

    x_std = sc.transform(x)

    model_path = r'random_forest.sav'
    model = joblib.load(model_path)

    y_pred = model.predict(x_std)

    return jsonify({'Prediction': float(y_pred)})


if __name__ == "__main__":
    app.run(debug=True, port=9457)

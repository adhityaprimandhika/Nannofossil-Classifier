import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_restx import Resource, Api, reqparse

from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from config import config

app = Flask(__name__)

api = Api(app=app,
          version="1.0",
          title="Nannofossil Classifier API",
          description="Give prediction to classify nannofossil")

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@api.route("/")
class Index(Resource):
    @cross_origin()
    def get(self):
        return jsonify({"Message": "API Success"})


api_namespace = api.namespace(
    "api", description="To get prediction of the nannofossil")


@api_namespace.route("/prediction", methods=["GET"])
class PredictClass(Resource):
    @api_namespace.doc(responses={200: "OK", 400: "Invalid Argument", 500: "Mapping Key Error"}, params={"jumlah_lengan": {"type": "String", "required": False}, "cabang_lengan": {"type": "String", "required": False}, "bentuk_morfologi": {"type": "String", "required": False}, "knob": {"type": "String", "required": False}, "ukuran_lengan": {"type": "String", "required": False}, "bentuk_lengan": {"type": "String", "required": False}, "bentuk_ujung_lengan": {"type": "String", "required": False}, "bentuk_ujung_lengan_melengkung": {"type": "String", "required": False}})
    @cross_origin()
    def get(self):
        
        data = []
        result = {}
        
        jumlah_lengan = request.args.get("jumlah_lengan")
        cabang_lengan = request.args.get("cabang_lengan")
        bentuk_morfologi = request.args.get("bentuk_morfologi")
        knob = request.args.get("knob")
        ukuran_lengan = request.args.get("ukuran_lengan")
        bentuk_lengan = request.args.get("bentuk_lengan")
        bentuk_ujung_lengan = request.args.get("bentuk_ujung_lengan")
        bentuk_ujung_lengan_melengkung = request.args.get("bentuk_ujung_lengan_melengkung")
        
        data.append(int(jumlah_lengan))
        data.append(int(cabang_lengan))
        data.append(int(bentuk_morfologi))
        data.append(int(knob))
        data.append(int(ukuran_lengan))
        data.append(int(bentuk_lengan))
        data.append(int(bentuk_ujung_lengan))
        data.append(int(bentuk_ujung_lengan_melengkung))
        
        print("Data")
        print(data)
        print()

        df, species_dict = load_data()
        X = df[["Jumlah Lengan", "Cabang Lengan", "Bentuk Morfologi", "Knob", "Ukuran Lengan",
                "Bentuk Lengan", "Bentuk Ujung Lengan", "Bentuk Ujung Lengan Melengkung"]]
        y = df["Spesies"]

        X_train, X_test, y_train, y_test = train_test_datasplit(
            X, y)
        data = np.array(data).reshape((1, 8))
        prediction, accuracy = predictor(data, X_train, X_test, y_train, y_test)

        print("Data : {}".format(data[0]))
        print("Prediction : {}".format(species_dict[str(prediction[0])]))
        print()

        df = df.append({"Jenis": species_dict[str(prediction[0])], "Jumlah Lengan": data[0][0], 
                        "Cabang Lengan": data[0][1], "Bentuk Morfologi": data[0][2], 
                        "Knob": data[0][3], "Ukuran Lengan": data[0][4], 
                        "Bentuk Lengan": data[0][5], "Bentuk Ujung Lengan": data[0][6], 
                        "Bentuk Ujung Lengan Melengkung": data[0][7], "Spesies": prediction[0]}, ignore_index=True)
        print("New data added")
        print()
        df.to_csv("data/new_web_data.csv", index=False)
        print("Data saved")
        print()

        result["jenis"] = species_dict[str(prediction[0])]
        result["jumlah_lengan"] = str(data[0][0])
        result["cabang_lengan"] = str(data[0][1])
        result["bentuk_morfologi"] = str(data[0][2])
        result["knob"] = str(data[0][3])
        result["ukuran_lengan"] = str(data[0][4])
        result["bentuk_lengan"] = str(data[0][5])
        result["bentuk_ujung_lengan"] = str(data[0][6])
        result["bentuk_ujung_lengan_melengkung"] = str(data[0][7])
        result["prediction_species"] = str(prediction[0])
        result["accuracy"] = str(accuracy*100)
        print(result)

        return jsonify(result)


def load_data():
    xls = pd.ExcelFile("data/Data Discoaster.xlsx")
    temp = pd.read_excel(xls, 'Sheet2')
    
    ind = 1
    species_dict = {}
    for species in temp["Jenis"]:
        species_dict[str(ind)] = species
        ind+=1
        
    df = pd.read_csv("data/new_web_data.csv")
    return df, species_dict


def train_test_datasplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def predictor(data, X_train, X_test, y_train, y_test):
    X_train = np.array(X_train).reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = np.array(X_test).reshape((len(X_test), np.prod(X_test.shape[1:])))

    autoencoder = model_ml()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    history = autoencoder.fit(X_train, X_train,
                    epochs=500,
                    batch_size=1024,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    encoded_train = autoencoder.predict(X_train)
    encoded_test = autoencoder.predict(X_test)
    grid  = hyperparameter_tuning(encoded_train,encoded_test,y_train,y_test)
    classifier = ExtraTreesClassifier(max_features=grid.best_params_["max_features"], criterion=grid.best_params_["criterion"], n_estimators=grid.best_params_["n_estimators"])
    classifier.fit(encoded_train, y_train)
    y_pred = classifier.predict(encoded_test)
    accuracy = round(balanced_accuracy_score(y_test, y_pred),3)
    print("Accuracy  : {}".format(accuracy))
    print()

    prediction = classifier.predict(data)

    return prediction, accuracy


def model_ml():
    encoder_input = Input(shape=(8,))

    encoded_layer1 = Dense(8*4)(encoder_input)
    encoded_layer1 = LeakyReLU()(encoded_layer1)

    encoded_layer2 = Dense(8*2)(encoded_layer1)
    encoded_layer2 = LeakyReLU()(encoded_layer2)

    encoded_layer3 = Dense(3)(encoded_layer2)

    decoded_layer1 = Dense(8*2)(encoded_layer3)
    decoded_layer1 = LeakyReLU()(decoded_layer1)

    decoded_layer2 = Dense(8*4)(decoded_layer1)
    decoded_layer2 = LeakyReLU()(decoded_layer2)

    decoded_layer3 = Dense(8, activation='linear')(decoded_layer2)
    # This model maps an input to its reconstruction
    autoencoder = Model(encoder_input, decoded_layer3)

    # This model maps an input to its encoded representation
    encoder = Model(encoder_input, encoded_layer3)
    return autoencoder


def hyperparameter_tuning(encoded_train, encoded_test, y_train, y_test):
    # Defining parameter range 
    param_grid = {"n_estimators": [10,100,500,1000],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_features": ["sqrt", "log2", None]}  

    if len(encoded_train) < 60:
        grid = GridSearchCV(ExtraTreesClassifier(), param_grid, refit = True, verbose = 3, cv = 2) 
    elif len(encoded_train) > 60 & len(encoded_train) < 100:
        grid = GridSearchCV(ExtraTreesClassifier(), param_grid, refit = True, verbose = 3, cv = 3)
    else:
        grid = GridSearchCV(ExtraTreesClassifier(), param_grid, refit = True, verbose = 3, cv = 5) 
    
    # Fitting the model for grid search 
    grid.fit(encoded_train, y_train)
    
    # Print best parameter after tuning 
    print(grid.best_params_) 
    return grid


if __name__ == '__main__':
    app.run(debug=config.DEBUG)

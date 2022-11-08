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

from sklearn.model_selection import train_test_split

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

        df = load_data()
        X = df[["Jumlah Lengan", "Cabang Lengan", "Bentuk Morfologi", "Knob", "Ukuran Lengan",
                "Bentuk Lengan", "Bentuk Ujung Lengan", "Bentuk Ujung Lengan Melengkung"]]
        y = df["Class"]

        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        encoded_y = label_encoder.transform(y)
        categorized_y = np_utils.to_categorical(encoded_y)
        X_train, X_test, y_train, y_test = train_test_datasplit(
            X, categorized_y)
        data = np.array(data).reshape((1, 8))
        prediction, train_accuracy, test_accuracy = predictor(data, X_train, X_test, y_train, y_test)
        transformed_prediction = label_encoder.inverse_transform(
            prediction.argmax(1))

        print("Data : {}".format(data[0]))
        print("Prediction : {}".format(transformed_prediction[0]))
        print()

        df = df.append({"Jumlah Lengan": data[0][0], "Cabang Lengan": data[0][1],
                        "Bentuk Morfologi": data[0][2], "Knob": data[0][3],
                        "Ukuran Lengan": data[0][4], "Bentuk Lengan": data[0][5],
                        "Bentuk Ujung Lengan": data[0][6], "Bentuk Ujung Lengan Melengkung": data[0][7],
                        "Class": transformed_prediction[0]}, ignore_index=True)
        print("New data added")
        print()
        df.to_csv("data/data.csv", index=False)
        print("Data saved")
        print()

        result["jumlah_lengan"] = str(data[0][0])
        result["cabang_lengan"] = str(data[0][1])
        result["bentuk_morfologi"] = str(data[0][2])
        result["knob"] = str(data[0][3])
        result["ukuran_lengan"] = str(data[0][4])
        result["bentuk_lengan"] = str(data[0][5])
        result["bentuk_ujung_lengan"] = str(data[0][6])
        result["bentuk_ujung_lengan_melengkung"] = str(data[0][7])
        result["prediction_class"] = str(transformed_prediction[0])
        result["train_accuracy"] = str(train_accuracy)
        result["test_accuracy"] = str(test_accuracy)
        print(result)

        return jsonify(result)


def load_data():
    df = pd.read_csv("data/data.csv")
    return df


def train_test_datasplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def predictor(data, X_train, X_test, y_train, y_test):
    X_train = np.array(X_train).reshape(
        (len(X_train), np.prod(X_train.shape[1:])))
    X_test = np.array(X_test).reshape((len(X_test), np.prod(X_test.shape[1:])))

    deep_model = model_ml()
    deep_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 200
    history = deep_model.fit(X_train, y_train,
                             epochs=epochs,
                             batch_size=1024,
                             shuffle=True,
                             validation_data=(X_test, y_test))

    train_accuracy, test_accuracy = model_evaluate(deep_model, X_train, X_test, y_train, y_test)
    print()

    prediction = deep_model.predict(data)

    return prediction, train_accuracy, test_accuracy


def model_ml():
    window_length = 8
    input_layer = Input(shape=(window_length,))

    layer_1 = Dense(640)(input_layer)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = LeakyReLU()(layer_1)
    layer_1 = Dropout(0.25)(layer_1)

    layer_2 = Dense(320)(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = LeakyReLU()(layer_2)
    layer_2 = Dropout(0.25)(layer_2)

    layer_3 = Dense(160)(layer_2)
    layer_3 = BatchNormalization()(layer_3)
    layer_3 = LeakyReLU()(layer_3)
    layer_3 = Dropout(0.25)(layer_3)

    layer_4 = Dense(37, activation='softmax')(layer_3)

    deep_model = Model(input_layer, layer_4)

    return deep_model


def model_evaluate(model, X_train, X_test, y_train, y_test):
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(test_accuracy))
    return train_accuracy, test_accuracy


if __name__ == '__main__':
    app.run(debug=config.DEBUG)

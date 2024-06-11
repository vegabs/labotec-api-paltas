import sys
import os
import numpy as np
from PIL import Image
import cv2
from my_module.model_eval import *
from my_module.model_eval import predict_status
from flask import Flask, request, jsonify
import __main__

setattr(__main__, "CustomTrain", CustomTrain)

app = Flask(__name__)

@app.route("/")
def home():
    return "LABOTEC Servidor"

@app.route("/analisis", methods=['POST'])
def get_analisis():
    try:
        # Check if the request has a file part
        if 'file' not in request.files:
            return jsonify({"respuesta": "ERROR"}), 400

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({"respuesta": "ERROR"}), 400

        if file:
            # Save the file to a temporary location
            temp_filename = 'temp_image.jpg'
            file.save(temp_filename)

            # Process the image
            type_dict = {0: "Deficiencia", 1: "Control", 2: "Exceso"}
            image_bgr = cv2.imread(temp_filename)
            height, width = image_bgr.shape[:2]

            if width > height:
                new_size = (900, 600)
            else:
                new_size = (600, 900)

            image_bgr_resize = cv2.resize(
                image_bgr, new_size, interpolation=cv2.INTER_AREA)
            image_rgb = cv2.cvtColor(image_bgr_resize, cv2.COLOR_BGR2RGB)
            image_PIL = Image.fromarray(image_rgb)

            modelout = predict_status(temp_filename)

            nitrogeno_nivel = type_dict[int(list(modelout.keys())[0])]
            nitrogeno_conf = round(float(list(modelout.values())[0]), 2)

            potasio_nivel = type_dict[int(list(modelout.keys())[1])]
            potasio_conf = round(float(list(modelout.values())[1]), 2)

            fosforo_nivel = type_dict[int(list(modelout.keys())[2])]
            fosforo_conf = round(float(list(modelout.values())[2]), 2)

            haux = list(modelout.values())[-1]
            if haux > 0.5:
                hidrico_nivel = "Control"
                hidrico_conf = haux
            else:
                hidrico_nivel = "Deficiencia"
                hidrico_conf = round(1 - haux, 2)

            model_response = {
                "respuesta": "OK",
                "analisis": [
                    {"tipo": "nitrogeno", "confianza": nitrogeno_conf, "valor": nitrogeno_nivel},
                    {"tipo": "potasio", "confianza": potasio_conf, "valor": potasio_nivel},
                    {"tipo": "fosforo", "confianza": fosforo_conf, "valor": fosforo_nivel},
                    {"tipo": "hidrico", "confianza": hidrico_conf, "valor": hidrico_nivel}
                ]
            }

            # Remove the temporary file after processing
            os.remove(temp_filename)

            return jsonify(model_response)

    except Exception as e:
        return jsonify({"respuesta": "ERROR"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)

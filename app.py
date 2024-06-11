import sys
import os
import numpy as np
from PIL import Image, ImageTk
import cv2

from my_module.model_eval import *
from my_module.model_eval import predict_status
from flask import Flask
from flask import jsonify


import __main__
setattr(__main__, "CustomTrain", CustomTrain)

app = Flask(__name__)


# --------
type_dict = {
    0: "Deficiencia",
    1: "Control",
    2: "Exceso"
}

color_dict = {
    0: "red",
    1: "yellow",
    2: "green"
}

model_response = {
    'nitrogeno': {
        'nivel': 'none',
        'confianza': 0.0
    },
    'potasio': {
        'nivel': 'none',
        'confianza': 0.0
    },
    'fosforo': {
        'nivel': 'none',
        'confianza': 0.0
    },
    'hidrico': {
        'nivel': 'none',
        'confianza': 0.0
    }
}



def assertlen(st, n):
    while (len(st) < n):
        st += " "
    return st

@app.route("/")
def analyze_img():

    filename = 'pictures/A01.jpg'
    print(filename)
    image_bgr = cv2.imread(filename)
    height, width = image_bgr.shape[:2]
    # print(self.height, self.width)
    if width > height:
        new_size = (900, 600)
    else:
        new_size = (600, 900)

    image_bgr_resize = cv2.resize(
        image_bgr, new_size, interpolation=cv2.INTER_AREA)
    # Since imread is BGR, it is converted to RGB
    image_rgb = cv2.cvtColor(image_bgr_resize, cv2.COLOR_BGR2RGB)

    # self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB) #Since imread is BGR, it is converted to RGB
    image_PIL = Image.fromarray(image_rgb)  # Convert from RGB to PIL format

    print("Procesando")
    if filename:

        modelout = predict_status(filename)

        print("Procesando2")
        # print(self.modelout)
        n_tex = type_dict[int(list(modelout.keys())[0])] + \
            " en "+str(round(float(list(modelout.values())[0]), 2))
        p_tex = type_dict[int(list(modelout.keys())[1])] + \
            " en "+str(round(float(list(modelout.values())[1]), 2))
        k_tex = type_dict[int(list(modelout.keys())[2])] + \
            " en "+str(round(float(list(modelout.values())[2]), 2))
        p_tex = assertlen(p_tex, 24)
        n_tex = assertlen(n_tex, 24)
        k_tex = assertlen(k_tex, 24)

        model_response['nitrogeno']['nivel'] = type_dict[int(
            list(modelout.keys())[0])]
        model_response['nitrogeno']['confianza'] = round(
            float(list(modelout.values())[0]), 2)

        model_response['potasio']['nivel'] = type_dict[int(
            list(modelout.keys())[1])]
        model_response['potasio']['confianza'] = round(
            float(list(modelout.values())[1]), 2)

        model_response['fosforo']['nivel'] = type_dict[int(
            list(modelout.keys())[2])]
        model_response['fosforo']['confianza'] = round(
            float(list(modelout.values())[2]), 2)

        # self.button_n_.configure(text=n_tex, bg=self.color_dict[int(list(self.modelout.keys())[0])], relief='sunken')
        # self.button_p_.configure(text=p_tex, bg=self.color_dict[int(list(self.modelout.keys())[1])], relief='sunken')
        # self.button_k_.configure(text=k_tex, bg=self.color_dict[int(list(self.modelout.keys())[2])], relief='sunken')

        haux = list(modelout.values())[-1]
        if haux > 0.5:
            taux = "Control en " + str(haux)
            # baux = "green"

            model_response['hidrico']['nivel'] = 'Control'
            model_response['hidrico']['confianza'] = haux

        else:
            taux = "Deficiencia en " + str(round(1-haux, 2))
            # baux = "red"

            model_response['hidrico']['nivel'] = 'Deficiencia'
            model_response['hidrico']['confianza'] = round(1-haux, 2)
        taux = assertlen(taux, 24)

        # print("-----")
        # print(taux)
        # print("-----")
        # print(n_tex)
        # print(p_tex)
        # print(k_tex)
        # print("-----")
        # print(model_response)
        return jsonify(model_response)


if __name__ == '__main__':
    # from model_eval import predict_status
    app.run(host="0.0.0.0", port="5000", debug=True)
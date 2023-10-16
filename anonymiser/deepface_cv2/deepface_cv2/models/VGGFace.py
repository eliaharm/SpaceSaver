# -*- coding: utf-8 -*-

import cv2


# VGG_MODEL = cv2.dnn.readNetFromONNX(os.path.join(preprocess.get_deepface_home(),
#                                                  "vgg_face.onnx"))
VGG_MODEL = cv2.dnn.readNetFromONNX("C:/Users/elia/Downloads/vgg19-7.onnx")

input_shape = (224, 224)


def predict(img):
    VGG_MODEL.setInput(img)
    out = VGG_MODEL.forward()
    return out

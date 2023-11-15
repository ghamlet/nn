# -*- coding: utf-8 -*-
from typing import Tuple

import cv2
import numpy as np
# TODO: Допишите импорт библиотек, которые собираетесь использовать


def load_models():
    """ 
        Функция осуществляет загрузку модели(ей) нейросети(ей) из файла(ов).
        Выходные параметры: загруженный(е) модели(и)

        Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список []
        Если вы используете несколько моделей, возвращайте их список [model1, model2]

        То, что вы вернёте из этой функции, будет передано вторым аргументом в функцию сount_vehicles
    """

    # TODO: Отредактируйте функцию по своему усмотрению.
    # Модель нейронной сети, загрузите на онайн-платформу вместе с eval.py.

    # Пример загрузки моделей из файлов
    # Yolo-модели
    # net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo.weights')
    # yolo_model = cv2.dnn_DetectionModel(net)
    # yolo_model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
    # models = [yolo_model]

    # Пример загрузки модели TensorFlow (не забудьте импортировать библиотеку tensorflow)
    # tf_model = tf.keras.models.load_model('model.h5')
    # models.append(tf_model)
    # models = [yolo_model]

    models = []
    return models


def сount_vehicles(video, models) -> int:
    """
        Функция для детектирования сфетофоров.

        Входные данные: видео-объект (cv2.VideoCapture)
        Выходные данные: целое число транспортных средств на видео

        Примеры вывода:
            0

            2

            4
    """

    # TODO: Отредактируйте эту функцию по своему усмотрению.
    # Для удобства можно создать собственные функции в этом файле.
    # Алгоритм проверки один раз вызовет функцию load_models
    # и для каждого тестового изображения будет вызывать функцию сount_vehicles
    # Все остальные функции должны вызываться из вышеперечисленных.
    #
    # yolo_model = models[0]
    # while True:
    #   ret, frame = video.read()
    #   classes, scores, boxes = yolo_model.detect(frame, 0.45, 0.25)
    #   cars = len(boxes)
    # return cars
  
    cars = 0
    return cars
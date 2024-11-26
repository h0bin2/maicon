# -*- coding: utf-8 -*-
#!pip install nvidia-ml-py3
# 사용 주의 사항 custom model test 할 때 각 pt, engine 파일이름에 각 모델 명이 명시되어야 함, ex) yolo11.pt, rtdetr.pt, yolo~.engine, rtdetr.engine ~~
import numpy as np
import time
import sys
import os

from ultralytics import YOLO, RTDETR

IMG_SIZE = (640, 640, 3)
IMG_TEST = np.zeros(IMG_SIZE)


# models 폴더에 원하는 pt, engine 파일 넣으면 여기서도 뜰 수 있다.
MODEL = [
    "yolo11m.pt",
    "yolo11l.pt",
    "rtdetr-l.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]
ITERATION = 7
for model in os.listdir("models/"):
    MODEL.append(model)

MODEL = sorted(list(set(MODEL)))


def set_model():
    print("---------Model Menu ---------")
    for i in range(len(MODEL)):
        print(f"{i+1} : {MODEL[i]}")
    print("X : 종료하기")
    print("-----------------------------")

    model_num = input("입력 : ")

    if (model_num == "X") or (model_num == "x"):
        print("---------DONE---------")
        exit()

    return int(model_num)


def set_batch():
    print("---------Batch Menu---------")
    batch = input("Batch Num : ")
    try:
        batch = int(batch)
        print("---------OK---------")
        return batch
    except:
        print("숫자 입력")


def predict(model, batch):
    batch_input = [IMG_TEST] * batch
    results = model.predict(
        batch_input,
        show_labels=False,
        show_conf=False,
        show_boxes=False,
        verbose=False,
        device=0,
    )
    batch_infTime = []
    for result in results:
        infTime = sum(result.speed.values()) / 1000
        batch_infTime.append(infTime)

    return sum(batch_infTime) / len(batch_infTime), results


def module(handle):
    log = ""

    model_num = set_model()
    batch = set_batch()

    startTime = time.time()
    modelFile = MODEL[model_num - 1]

    file = open(f"logs/{modelFile}-batch({batch}).txt", "w")

    if "yolo" in modelFile:
        model = YOLO("models/" + modelFile)
    elif "rtdetr" in modelFile:
        model = RTDETR("models/" + modelFile)
    print(f"MODEL LOAD TIME : {time.time() - startTime:.03f}(s)")

    for iter in range(ITERATION):
        startTime = time.time()
        infTime, results = predict(model, batch)
        codeTime = time.time() - startTime

        inf_info = f"ITER({iter+1}) | {modelFile.split('/')[-1]} | Inference Time : {infTime:.03f}(s) | Code Time : {codeTime:.03f}(s)"
        print(inf_info)

        log += inf_info + "\n\n"

    file.write(log)

    file.close()
    print("")


def main(iters):
    handle = None

    while iters:
        module(handle)

    if not iters:
        module(handle)


if __name__ == "__main__":
    if sys.argv[1] in ("true", "True", "TRUE"):
        main(True)
    else:
        main(False)
    input()

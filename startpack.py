# -*- coding: utf-8 -*-
#!pip install nvidia-ml-py3
# 사용 주의 사항 custom model test 할 때 각 pt, engine 파일이름에 각 모델 명이 명시되어야 함, ex) yolo11.pt, rtdetr.pt, yolo~.engine, rtdetr.engine ~~
import numpy as np
import logging
import time

from ultralytics import YOLO, RTDETR
import nvidia_smi

# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

IMG_SIZE = (640, 640, 3)
IMG_TEST = np.zeros(IMG_SIZE)

# 파일 경로 잘 넣으면 테스트 가능
MODEL = ["yolo11m.pt", "yolo11l.pt", "rtdetr-l.pt", "yolo11l.engine"]
ITERATION = 7


def set_model():
    print("---------Model Menu ---------")
    for i in range(len(MODEL)):
        print(f"{i+1} : {MODEL[i]}")
    print("X : 종료하기")
    print("-----------------------------")

    model_num = input("입력 : ")

    if (model_num == "X") or (model_num == "x"):
        print("---------DONE---------")
        raise

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
        batch_input, show_labels=False, show_conf=False, show_boxes=False, verbose=False
    )
    batch_infTime = []
    for result in results:
        infTime = sum(result.speed.values()) / 1000
        batch_infTime.append(infTime)

    return sum(batch_infTime) / len(batch_infTime), results


def logWrite():
    return


def main():
    while True:

        model_num = set_model()
        batch = set_batch()

        startTime = time.time()
        modelFile = MODEL[model_num - 1]

        if "yolo" in modelFile:
            model = YOLO(modelFile)
        elif "rtdetr" in modelFile:
            model = RTDETR(modelFile)
        print(f"MODEL LOAD TIME : {time.time() - startTime:.03f}(s)")

        for iter in range(ITERATION):
            startTime = time.time()
            infTime, results = predict(model, batch)
            codeTime = time.time() - startTime

            print(
                f"ITER({iter+1}) | {modelFile.split('/')[-1]} | Inference Time : {infTime:.03f}(s) | Code Time : {codeTime:.03f}(s)"
            )

            # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # print(
            #     f"Total memory:{ (info.total/1073741824):03f} GB | Used memory:{ (info.used/1073741824):03f} GB | Free memory:{ (info.free/1073741824):03f} GB"
            # )

        # nvidia_smi.nvmlShutdown()

        print("")


if __name__ == "__main__":
    main()

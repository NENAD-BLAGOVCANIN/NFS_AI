import numpy as np
from screen_recorder import record_screen, close_recorder
from keyboard_tracker import get_arrow_keys
import cv2
import time
from alexnet import alexnet
import tensorflow as tf

MODEL_NAME = 'model/model_alexnet-47000'
WIDTH = 100
HEIGHT = 80
LR = 1e-4
EPOCHS = 10

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            screen = record_screen()

            prediction = model.predict([screen])[0]
            print(prediction)

            

        keys = get_arrow_keys()

main()       
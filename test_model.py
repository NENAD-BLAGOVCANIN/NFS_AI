import numpy as np
from screen_recorder import record_screen, close_recorder
from keyboard_tracker import get_arrow_keys
import cv2
import time
from alexnet import alexnet
import tensorflow as tf
from direct_keys import PressKey, ReleaseKey
import pydirectinput
pydirectinput.FAILSAFE = False

MODEL_NAME = 'model/model_alexnet-47000'
WIDTH = 100
HEIGHT = 80
LR = 1e-4
EPOCHS = 10

# Arrow key scan codes
UP_ARROW = 0x48
DOWN_ARROW = 0x50
LEFT_ARROW = 0x4B
RIGHT_ARROW = 0x4D

# Action map where each index corresponds to a combination of keys
action_map = {
    0: (0, 0, 0, 0),  # No keys pressed
    1: (0, 0, 0, 1),  # Down
    2: (0, 0, 1, 0),  # Up
    3: (0, 1, 0, 1),  # Down + Right
    4: (0, 1, 1, 1),  # Right + Up + Down
    5: (1, 0, 1, 1),  # Left + Up + Down
    6: (1, 1, 0, 0),  # Left + Right
    7: (1, 1, 1, 0)   # Left + Right + Up + Down
}

# Load model
model = alexnet(WIDTH, HEIGHT, LR, output=len(action_map))
model.load(MODEL_NAME)

def press_keys(action):
    """Press and hold/release keys based on the action using pydirectinput."""
    if action[0] == 1:
        pydirectinput.keyDown('left')
    else:
        pydirectinput.keyUp('left')

    if action[1] == 1:
        pydirectinput.keyDown('right')
    else:
        pydirectinput.keyUp('right')

    if action[2] == 1:
        pydirectinput.keyDown('up')
    else:
        pydirectinput.keyUp('up')

    if action[3] == 1:
        pydirectinput.keyDown('down')
    else:
        pydirectinput.keyUp('down')

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:
        if not paused:
            screen = record_screen()
            screen = screen.reshape(1, WIDTH, HEIGHT, 1)

            # Predict action
            prediction = model.predict(screen)[0]
            chosen_action_index = np.argmax(prediction)
            chosen_action = action_map[chosen_action_index]
            print(f"Chosen action: {chosen_action}")

            # Simulate key presses based on prediction
            press_keys(chosen_action)

        time.sleep(0.5)
main()
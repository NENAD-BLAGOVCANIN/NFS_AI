import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# Arrow key scan codes
UP_ARROW = 0x48
LEFT_ARROW = 0x4B
DOWN_ARROW = 0x50
RIGHT_ARROW = 0x4D

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actual Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == '__main__':
    # Test the up arrow key
    PressKey(UP_ARROW)
    time.sleep(1)
    ReleaseKey(UP_ARROW)
    time.sleep(1)

    # Test the down arrow key
    PressKey(DOWN_ARROW)
    time.sleep(1)
    ReleaseKey(DOWN_ARROW)
    time.sleep(1)

    # Test the left arrow key
    PressKey(LEFT_ARROW)
    time.sleep(1)
    ReleaseKey(LEFT_ARROW)
    time.sleep(1)

    # Test the right arrow key
    PressKey(RIGHT_ARROW)
    time.sleep(1)
    ReleaseKey(RIGHT_ARROW)
    time.sleep(1)

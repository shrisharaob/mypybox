import numpy as np

def AngleDiff(a, b):
    out = a - b
    while(out > 90):
       out -= 180
    while(out < -90):
        out  += 180
    return out

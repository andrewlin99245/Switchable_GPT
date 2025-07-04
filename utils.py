import time, json, math

def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg)

# Cyclic schedule util for CPT

def cyclic_precision(step, cycle_len, low_bit=3, high_bit=8):
    phase = (step % cycle_len) / cycle_len
    return high_bit if phase < 0.5 else low_bit
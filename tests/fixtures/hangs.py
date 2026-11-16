# Never returns: expect status=timeout after the task's test_timeout.
import time

def custom_kernel(data):
    while True:
        time.sleep(1)

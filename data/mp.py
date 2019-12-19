from multiprocessing import Process, Value, Array, Lock
import numpy as np
# not used in the final code, but some multiprocessing practice in python.
x = [1]
def f(a, lock):
    lock.acquire()
    for i in range(len(a)):
        a[i] = -a[i]
    lock.release()

if __name__ == '__main__':
    lock = Lock()
    jobs = []
    arr = Array('i', range(11))
    for i in range(11):
        p = Process(target=f, args=(arr, lock))
        jobs.append(p)
        p.start()
    for i in range(11):
        jobs[i].join()
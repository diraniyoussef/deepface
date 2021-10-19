import multiprocessing
from functools import partial
import os
import time
import random

def heavy_work():
    print("{} before sleep".format(os.getpid()))        
    time.sleep(1.5)
    print("{} after sleep".format(os.getpid()))

def _foo(a, b, c, d = "dd"):
    print("Worker process id for {0}: {1}\n".format(c, os.getpid()))
    heavy_work(c)
    return {"a": a, "b": b, "c": c}

if __name__ == "__main__":
    multiprocessing.freeze_support() #for windows machine
    mylist = [i for i in range(15)]
    with multiprocessing.Pool(4) as pool:
        func = partial(_foo, 66, 88)
		#p1 = pool.imap(func, range(10))
        tic = time.time()
        result = pool.map(func, mylist)
        print("spent time", str(time.time()-tic))
        print(result)
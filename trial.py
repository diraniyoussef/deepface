import multiprocessing
from functools import partial
import os
import time

def heavy_work(c):
    print("{} before sleep".format(os.getpid()))
    time.sleep(c)
    print("{} after sleep".format(os.getpid()))

def _foo(a, b, c, d = "dd"):
    print("Worker process id for {0}: {1}\n".format(c, os.getpid()))
    heavy_work(c)
    return {"a": a, "b": b, "c": c}

if __name__ == "__main__":
    multiprocessing.freeze_support() #for windows machine
    mylist = [1,2,3,4,5]
    with multiprocessing.Pool(2) as pool:
        func = partial(_foo, 66, 88)
		#p1 = pool.imap(func, range(10))
	
        result = pool.map(func, mylist)
	
        print(result)
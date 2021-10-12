import multiprocessing
from functools import partial
import os
import time


def square(n):
    print("Worker process id for {0}: {1}\n".format(n, os.getpid()))
    return (n*n)

def _foo(a, b, c, d = "dd"):
    #square = my_number * my_number
    time.sleep(1)
    print("b : {}\n".format(b))
    print("c : {}\n".format(c))
    print("d : {}\n".format(d))
    print("Worker process id for {0}: {1}\n".format(c, os.getpid()))
    return [a + c, c]  

if __name__ == "__main__":
    multiprocessing.freeze_support() #for windows machine
    mylist = [1,2,3,4,5]
    with multiprocessing.Pool(2) as pool:
        func = partial(_foo, 66, 88)
		#p1 = pool.imap(func, range(10))
	
        result = pool.map(func, mylist)
	
        print(result[0])


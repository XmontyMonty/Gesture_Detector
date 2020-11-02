import random

from threading import Thread
import time

Test = 1
stop = False


def para():
    while not stop:
        if Test > 10:
            print("Test is bigger than ten")
        time.sleep(1)


# I want this to start in parallel, so that the code below keeps executing without waiting for this function to finish

thread = Thread(target=para)
thread.start()

while True:
    Test = random.randint(1, 42)
    time.sleep(1)

    if Test == 42:
        break

# stop the parallel execution of the para() here (kill it)
stop = True
thread.join()

# ..some other code here
print('we have stopped')

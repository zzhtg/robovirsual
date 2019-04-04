import threading
def th1(i):
    while(1):
        print(i + 1)
def th2(i):
    while (1):
        print(i)
if __name__ == '__main__':
    t1 = threading.Thread(target=th1, args=[3])
    t2 = threading.Thread(target=th2, args=[3])
    t1.start()
    t2.start()
    t1.join()
    t2.join()
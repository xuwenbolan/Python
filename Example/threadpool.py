from concurrent.futures import ThreadPoolExecutor
import time

a=0
MB = 1024**2
speed = 0.0
starttime = time.time()
downsize = 0
progress = 0 
totalsize = 101*MB

def spider(page):
    time.sleep(page)
    print(f"crawl task{page} finished")
    return page

def add():
    global a
    a += 1
    time.sleep(0.1)

def get_speed(self):
    global downsize,speed,progress
    downsize += MB
    speed = downsize/(time.time()-starttime)
    progress = (downsize/totalsize)*100
    print("%.1fMB/s" % (speed/1000000))
    print("%d%%" % progress)

print(a)
t = ThreadPoolExecutor(max_workers=1)
for i in range(100):
    t.submit(add).add_done_callback(get_speed)
t.shutdown(wait=False,cancel_futures=True)
print("-"*10)
print(a)
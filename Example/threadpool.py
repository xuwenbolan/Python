from concurrent.futures import ThreadPoolExecutor
import time

a=0

def spider(page):
    time.sleep(page)
    print(f"crawl task{page} finished")
    return page

def add():
    global a
    a += 1

print(a)
with ThreadPoolExecutor(max_workers=200) as t:
    for i in range(200):
        t.submit(add)
print(a)
import time
import threadpool

#先用pip install threadpool 检查是否安装
#执行比较耗时的函数，需要开启多线程
def get_html(url):  
    time.sleep(3)
    print(url)

urls= [i for i in range(10)]#生成10个数 for的简洁写法 
pool = threadpool.ThreadPool(10)#建立线程池  开启10个线程

requests = threadpool.makeRequests(get_html,urls)#提交10个任务到线程池
print(requests)
for req in requests:#开始执行任务
    pool.putRequest(req)#提交  

pool.wait()#等待完成
print(123)   
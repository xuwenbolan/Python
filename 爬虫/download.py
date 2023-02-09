from concurrent.futures import ThreadPoolExecutor
import requests
import os
from urllib.parse import unquote
import time
import queue

#速度+进度
#异常处理（True，False）
#暂停、销毁
#文件路径传入

class muti_download:
    MB = 1024**2
    q=queue.Queue(10)
    URL = ""
    filename = ""

    def __init__(self,url,Path):
        self.URL = url
        self.headers = {"User-Agent":"pan.baidu.com"}
        self.Threads_num = 10
        self.Path = Path

    def range_download(self,filename , s_pos , e_pos):
        proxies = self.q.get()
        headers = {"Range": f"bytes={s_pos}-{e_pos}","User-Agent":"pan.baidu.com"}
        res = requests.get(self.URL, headers=headers,proxies=proxies, stream=True)
        with open(filename, "rb+") as f:
            f.seek(s_pos)
            for chunk in res.iter_content(chunk_size=64*1024):
                if chunk:
                    f.write(chunk)
        print(s_pos,"seccess")
        self.q.put(proxies)

    def split(self,start: int, end: int, step: int) -> list[tuple[int, int]]:
        parts = [(start, min(start+step, end))
                for start in range(0, end, step)]
        return parts

    def get_file_name(self, headers):
        filename = ''
        if 'Content-Disposition' in headers and headers['Content-Disposition']:
            disposition_split = headers['Content-Disposition'].split(';')
            if len(disposition_split) > 1:
                if disposition_split[1].strip().lower().startswith('filename='):
                    file_name = disposition_split[1].split('=')
                    if len(file_name) > 1:
                        filename = unquote(file_name[1])
        if not filename and os.path.basename(self.URL):
            filename = os.path.basename(self.URL).split("?")[0]
        if not filename:
            return time.time()
        filename = filename.replace('"',"")
        return filename

    def get_proixy(self):
        while(self.q.qsize()<9):
            proxies = {'http': '','https':''}
            proxies['http']=requests.get("http://43.138.161.163:5010/get/").json().get("proxy")
            proxies['https']=requests.get("http://43.138.161.163:5010/get?type=https").json().get("proxy")
            if(self.check_proixy(proxies)):
                self.q.put(proxies)
            else:
                continue   
        proxies = {'http': '','https':''}
        self.q.put(proxies) 

    def check_proixy(self,proxies):
        headers = {"Range": f"bytes={0}-{100}","User-Agent":"pan.baidu.com"}
        try:
            resp = requests.get(self.URL,headers=headers,proxies=proxies,timeout=3)
        except:
            return False
        print(proxies,resp.status_code)
        if(resp.status_code==206):
            return True
        else:
            return False
        
    def begin(self):
        self.get_proixy()
        start = time.time()
        res = requests.head(self.URL,headers=self.headers)
        res_headers = res.headers
        filesize = int(res_headers['Content-Length'])
        each_size = min(self.MB, filesize)
        parts = self.split(0, filesize, each_size)
        filename = self.get_file_name(res_headers)
        self.filename = filename
        with open(filename, "wb") as f:
            pass
        with ThreadPoolExecutor(max_workers=9) as t:
            for s_pos,e_pos in parts:
                t.submit(self.range_download,filename,s_pos,e_pos)
        end = time.time()
        print(end - start)
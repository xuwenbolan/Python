from concurrent.futures import ThreadPoolExecutor
import requests
import os
from urllib.parse import unquote
import time
import queue

MB = 1024**2
lastsize = 0
lasttime = time.time()
q=queue.Queue(10)

url4 = "https://114-233-56-49.d.cjjd20.com:30443/123-493/ca3884d7/1813944254-0/ca3884d7af72b861dd18df752539e1ed?v=2&t=1675949675&s=c060588cf78f29500fad12919dc4daab&filename=[Sakurato]+Mushoku+Tensei+Isekai+Ittara+Honki+Dasu+[15][AVC-8bit+1080p%4060FPS+AAC][CHS].zip&d=e27fc08"
url3 = "https://yunpan.aliyun.com/downloads/apps/desktop/aDrive.exe?spm=aliyundrive.sign.0.0.13871011kgQ9X6&file=aDrive.exe"
url = "http://175.6.53.36/b/dp0.baidupcs.com/file/fa160a103i7b9652c56eb1596ec9d846?bkt=en-2fb6763f1c8fb101509e72b061c2194a4ab5bffd784c7be12222642091e14f42c8ae2570a883886d&xcode=adfc8dc911f21c4f3e7157802848ff6d014b5d385737ada0eed857872f24f7b14c4a9dc080ab16680b2977702d3e6764&fid=1194533378-250528-38455223447052&time=1675863669&sign=FDTAXUbGERLQlBHSKfqi-DCb740ccc5511e5e8fedcff06b081203-vOz1fbtQrazWX8o%2FLDebwJ%2F0mzg%3D&to=p0&size=276549931&sta_dx=276549931&sta_cs=6&sta_ft=zip&sta_ct=7&sta_mt=6&fm2=MH%2CBaoding%2CAnywhere%2C%2C%E5%8C%97%E4%BA%AC%2Cany&ctime=1640702036&mtime=1655269375&resv0=-1&resv1=0&resv2=rlim&resv3=5&resv4=276549931&vuk=1194533378&iv=0&htype=&randtype=&tkbind_id=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-55d13dece44c761c3caa95205ea511c18f0d21b71c2726a85af945f0341c111004435c285f9a4bbc&sl=76480590&expires=8h&rt=pr&r=720277861&vbdid=1258173819&fin=b0008.zip&rtype=1&dp-logid=1038575693567739106&dp-callid=0.1&tsl=80&csl=80&fsl=-1&csign=INb2EjvWE14HvXAz4tjRDFx7TFE%3D&so=1&ut=6&uter=0&serv=0&uc=1559654663&ti=05df9239daa406479b542b8e32e166016abc0b18d6dc3097&hflag=21&from_type=3&adg=a_c7049768f8537840c86b5012b4833d0e&reqlabel=25571201_f_06768596e8270ae062c30f533d9ad28f_-1_57b367756dbd20ebd5abdd00fef07b72&by=themis"
url2 = "http://175.6.53.36/b/dp0.baidupcs.com/file/fa160a103i7b9652c56eb1596ec9d846?bkt=en-2fb6763f1c8fb101509e72b061c2194a4ab5bffd784c7be12222642091e14f42c8ae2570a883886d&xcode=d61804aaedb7b8a3232c15489dea1d19014b5d385737ada0842e23bf9e59178b7f844b9c121e97450b2977702d3e6764&fid=1194533378-250528-38455223447052&time=1675768382&sign=FDTAXUbGERLQlBHSKfqi-DCb740ccc5511e5e8fedcff06b081203-vJRkK58YhFjF%2FskwZSST8FZv6Nk%3D&to=p0&size=276549931&sta_dx=276549931&sta_cs=1&sta_ft=zip&sta_ct=7&sta_mt=6&fm2=MH%2CBaoding%2CAnywhere%2C%2C%E5%8C%97%E4%BA%AC%2Cany&ctime=1640702036&mtime=1655269375&resv0=-1&resv1=0&resv2=rlim&resv3=5&resv4=276549931&vuk=1194533378&iv=0&htype=&randtype=&tkbind_id=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-55d13dece44c761c3caa95205ea511c18f0d21b71c2726a85af945f0341c111004435c285f9a4bbc&sl=76480590&expires=8h&rt=pr&r=599770585&vbdid=1258173819&fin=b0008.zip&rtype=1&dp-logid=99748313556336228&dp-callid=0.1&tsl=80&csl=80&fsl=-1&csign=INb2EjvWE14HvXAz4tjRDFx7TFE%3D&so=1&ut=6&uter=0&serv=0&uc=1559654663&ti=51d76b5cdb7a87d2ef4a684abb72c70c556055ac1a20369a80d4af97bfb69cf0&hflag=21&from_type=3&adg=a_c7049768f8537840c86b5012b4833d0e&reqlabel=25571201_f_06768596e8270ae062c30f533d9ad28f_-1_57b367756dbd20ebd5abdd00fef07b72&by=themis"

def range_download(filename , s_pos , e_pos):
    # proxies = q.get()
    headers = {"Range": f"bytes={s_pos}-{e_pos}","User-Agent":"pan.baidu.com"}
    res = requests.get(url, headers=headers, stream=True)
    with open(filename, "rb+") as f:
        f.seek(s_pos)
        for chunk in res.iter_content(chunk_size=64*1024):
            if chunk:
                f.write(chunk)
    print(s_pos,"seccess")
    # q.put(proxies)

def get_file_name(url, headers):
    filename = ''
    if 'Content-Disposition' in headers and headers['Content-Disposition']:
        disposition_split = headers['Content-Disposition'].split(';')
        if len(disposition_split) > 1:
            if disposition_split[1].strip().lower().startswith('filename='):
                file_name = disposition_split[1].split('=')
                if len(file_name) > 1:
                    filename = unquote(file_name[1])
    if not filename and os.path.basename(url):
        filename = os.path.basename(url).split("?")[0]
    if not filename:
        return time.time()
    filename = filename.replace('"',"")
    return filename

def split(start: int, end: int, step: int) -> list[tuple[int, int]]:
    # 分多块
    parts = [(start, min(start+step, end))
             for start in range(0, end, step)]
    return parts

def main():
    # get_proixy()
    start = time.time()
    headers={"User-Agent":"pan.baidu.com"}
    res = requests.head(url,headers=headers)
    print(res.status_code)
    headers = res.headers
    filesize = int(headers['Content-Length'])
    each_size = min(MB, filesize)
    # print(headers)
    parts = split(0, filesize, each_size)
    filename = get_file_name(url,headers)
    print(filename)
    with open(filename, "wb") as f:
        pass
    with ThreadPoolExecutor(max_workers=200) as t:
        for s_pos,e_pos in parts:
            t.submit(range_download,filename,s_pos,e_pos)
    end = time.time()
    print(end - start)

def get_proixy():
    while(q.qsize()<5):
        proxies = {'http': '','https':''}
        proxies['http']=requests.get("http://43.138.161.163:5010/get/").json().get("proxy")
        proxies['https']=requests.get("http://43.138.161.163:5010/get?type=https").json().get("proxy")
        if(check_proixy(proxies)):
            q.put(proxies)          
        else:
            continue
    # for i in range(5):
    #     print(q.get())
    # proxies = {'http': '','https':''}
    # q.put(proxies)

def check_proixy(proxies):
    headers = {"Range": f"bytes={0}-{100}","User-Agent":"pan.baidu.com"}
    try:
        resp = requests.get(url,headers=headers,proxies=proxies,timeout=1)
    except:
        return False
    print(proxies,resp.status_code)
    if(resp.status_code==206):
        return True
    else:
        return False

def get_speed():
    pass

def get_buf():
    pass

main()
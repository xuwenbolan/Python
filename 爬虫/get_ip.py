import requests
import download

# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36 Edg/91.0.864.71',
#                'Connection': 'close'}

url = "https://114-233-56-49.d.cjjd20.com:30443/123-493/ca3884d7/1813944254-0/ca3884d7af72b861dd18df752539e1ed?v=2&t=1675945401&s=bc65c032847dc3bacfe5bd395cb881f2&filename=[Sakurato]+Mushoku+Tensei+Isekai+Ittara+Honki+Dasu+[15][AVC-8bit+1080p%4060FPS+AAC][CHS].zip&d=d965c598"

def get_ip():
    proxies = {'http': '','https':''}
    proxies['http']=requests.get("http://43.138.161.163:5010/get/").json().get("proxy")
    proxies['https']=requests.get("http://43.138.161.163:5010/get?type=https").json().get("proxy")
    # proxy['https']="127.0.0.1:7890"
    headers = {"Range": f"bytes={0}-{10000}","User-Agent":"pan.baidu.com"}
    resp = requests.get(url,headers=headers,proxies=proxies)
    print(resp.status_code)

def test():
    down = download.muti_download(url)
    down.begin()

test()

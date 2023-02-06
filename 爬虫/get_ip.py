import requests

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36 Edg/91.0.864.71',
               'Connection': 'close'}

url = 'https://www.baidu.com'

def get_ip():
    ip = requests.get("http://43.138.161.163:5010/get/").json().get("proxy")
    proxy = {"http":""}
    proxy['http']="http://"+ip
    resp = requests.get('http://wenshu.court.gov.cn/', proxies=proxy)
    print(resp)
get_ip()

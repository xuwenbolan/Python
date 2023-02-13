import requests
import download

# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36 Edg/91.0.864.71',
#                'Connection': 'close'}

url = "http://bjbgp01.baidupcs.com/file/fa160a103i7b9652c56eb1596ec9d846?bkt=en-2fb6763f1c8fb101509e72b061c2194a4ab5bffd784c7be12222642091e14f42c8ae2570a883886d&fid=1194533378-250528-38455223447052&time=1676262677&sign=FDTAXUbGERLQlBHSKfWaqi-DCb740ccc5511e5e8fedcff06b081203-MkZg7Msy25Dym67pgqljFFkP5%2Bk%3D&to=14&size=276549931&sta_dx=276549931&sta_cs=7&sta_ft=zip&sta_ct=7&sta_mt=6&fm2=MH%2CBaoding%2CAnywhere%2C%2C%E5%8C%97%E4%BA%AC%2Cany&ctime=1640702036&mtime=1655269375&resv0=-1&resv1=0&resv2=rlim&resv3=5&resv4=276549931&vuk=1194533378&iv=0&htype=&randtype=&tkbind_id=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-55d13dece44c761c3caa95205ea511c18f0d21b71c2726a85af945f0341c111004435c285f9a4bbc&sl=76480590&expires=8h&rt=pr&r=826909585&vbdid=1258173819&fin=b0008.zip&rtype=1&dp-logid=2284608458145399365&dp-callid=0.1&tsl=80&csl=80&fsl=-1&csign=INb2EjvWE14HvXAz4tjRDFx7TFE%3D&so=1&ut=6&uter=0&serv=0&uc=1559654663&ti=a50ef6a4e24b566d06c0faaaccb2ff09a535470ef95f6258&hflag=30&from_type=3&adg=a_fae0bef24e6070b28dc99ebca059a20a&reqlabel=25571201_f_06768596e8270ae062c30f533d9ad28f_-1_57b367756dbd20ebd5abdd00fef07b72&by=themis"

def get_ip():
    proxies = {'http': '','https':''}
    proxies['http']=requests.get("http://43.138.161.163:5010/get/").json().get("proxy")
    proxies['https']=requests.get("http://43.138.161.163:5010/get?type=https").json().get("proxy")
    # proxy['https']="127.0.0.1:7890"
    headers = {"Range": f"bytes={0}-{10000}","User-Agent":"pan.baidu.com"}
    resp = requests.get(url,headers=headers,proxies=proxies)
    print(resp.status_code)

def test():
    down = download.muti_download(url,"")
    down.begin()
    down.stop()

test()

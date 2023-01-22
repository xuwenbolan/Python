import requests
import json
import time
import os
from urllib import parse

url="http://192.168.1.1/stok={0}/ds"
def read():
    js = {"hosts_info":{"table":"online_host"},"method":"get"}
    js = json.dumps(js).encode('utf-8')
    for a in range(5):
        os.system("cls")
        resp = requests.post(url,js)
        data =json.loads(resp.text).get("hosts_info").get("online_host")
        i=0
        list1=[]
        for item in data:
            list1.append(item.get("host_info_{}".format(str(i))))
            i+=1
        for i in list1:
            print(parse.unquote(i.get("hostname")) + "：" +  i.get("down_speed"))
        time.sleep(1)
    
def write():
    js = {"hosts_info":{"set_flux_limit":{"mac":"8e-ef-67-d4-09-97","down_limit":"1","up_limit":"1","name":"恬恬","is_blocked":"0"}},"method":"do"}
    js = json.dumps(js).encode('utf-8')
    resp = requests.post(url,js)

write()
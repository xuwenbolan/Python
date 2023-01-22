import requests
import json
import time
import pandas as pd
import os
# from urllib import parse

url = "http://melogin.cn/stok=h!306nTm%08E!q56rPyXbV%7DJWnyyl%5Blf4w/ds"


def read():
    js = {"hosts_info": {"table": "online_host"}, "method": "get"}
    js = json.dumps(js).encode('utf-8')
    resp = requests.post(url, js)
    data = json.loads(resp.text).get("hosts_info").get("online_host")
    i = 0
    items = []
    for item in data:
        items.append(item.get("host_info_{}".format(str(i))))
        i += 1
    return items


def set(item):
    js = {"hosts_info": {"set_flux_limit": {"mac": "", "down_limit": "","up_limit": "", "name": "", "is_blocked": "0"}}, "method": "do"}
    js["hosts_info"]["set_flux_limit"]["mac"] = item.get("mac")
    js["hosts_info"]["set_flux_limit"]["down_limit"] = item.get("down_limit")
    js["hosts_info"]["set_flux_limit"]["up_limit"] = item.get("up_limit")
    js["hosts_info"]["set_flux_limit"]["name"] = item.get("hostname")
    js = json.dumps(js).encode('utf-8')
    requests.post(url, js)


def judgment():
    key=0
    while(1):
        for item in read():
            if(item.get("hostname")=="TV"):
                down_speed=item.get("down_speed")
                if(int(down_speed) >= 1024*30):
                    item["down_limit"]='1'
                    item["up_limit"]='1'
                    set(item)
                    key=1
                    break
        if(key==1):
            break
    return 0

def recode():
    data=[]
    for s in range(600):
        for item in read():
            if(item.get("hostname")=="TV"):
                list_1=[time.strftime('%H:%M:%S', time.localtime()),item.get("up_speed"),item.get("down_speed")]
                data.append(list_1)
                print(time.strftime('%H:%M:%S', time.localtime()),item.get("down_speed"))
                break
        time.sleep(1)
    df = pd.DataFrame(data,columns=['time','up_speed','down_speed'])
    df.to_csv("TV-data.csv",index=None)
    return 0

def get_time():
    print(time.strftime('%H:%M:%S', time.localtime()))
    return 0

judgment()

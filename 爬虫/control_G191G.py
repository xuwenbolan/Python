import requests
import json
import time
import os
import pandas as pd
import pymysql
import keyboard

URL = "http://melogin.cn/"

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

def get_url():
    js = {"method": "do", "login": {"password": "0KcgeXhc9TefbwK"}}
    js = json.dumps(js).encode('utf-8')
    resp = requests.post(URL, js)
    stok = json.loads(resp.text).get("stok")
    url = URL+"stok="+stok+"/ds"
    return url

def set(item):
    js = {"hosts_info": {"set_flux_limit": {"mac": "", "down_limit": "","up_limit": "", "name": "", "is_blocked": "0"}}, "method": "do"}
    js["hosts_info"]["set_flux_limit"]["mac"] = item.get("mac")
    js["hosts_info"]["set_flux_limit"]["down_limit"] = item.get("down_limit")
    js["hosts_info"]["set_flux_limit"]["up_limit"] = item.get("up_limit")
    js["hosts_info"]["set_flux_limit"]["name"] = item.get("hostname")
    js = json.dumps(js).encode('utf-8')
    requests.post(url, js)


def judgment(name):
    key=0
    while(1):
        for item in read():
            if(item.get("hostname")==name):
                down_speed=item.get("down_speed")
                print(get_time() +" "+ down_speed)
                if(int(down_speed) >= 1024*70):
                    slows_down(item)
                    # item["down_limit"]='1'
                    # item["up_limit"]='1'
                    # set(item)
                    key=1
                    break
        time.sleep(1)
        if(key==1):
            break
    return 0

def recode(name):
    data=[]
    for s in range(600):
        for item in read():
            if(item.get("hostname")==name):
                list_1=[time.strftime('%H:%M:%S', time.localtime()),item.get("up_speed"),item.get("down_speed")]
                data.append(list_1)
                print(time.strftime('%H:%M:%S', time.localtime()),item.get("down_speed"))
                break
        time.sleep(1)
    df = pd.DataFrame(data,columns=['time','up_speed','down_speed'])
    df.to_csv("{}-data.csv".format(name),index=None)
    return 0

def get_time():
    time1 = time.strftime('%H:%M:%S', time.localtime())
    return time1

def slows_down(item):
    for i in range(301,0,-5):
        item["up_limit"]=str(i)
        item["down_limit"]=str(i)
        print(get_time() +" "+ str(i))
        set(item)
        time.sleep(1)

def mysql_recode(name):
    mysql_conn = pymysql.connect(host='rm-7xvcob73w77pgn5v8to.mysql.rds.aliyuncs.com',
                             port=3306, user='app', password='Xuwenbo20040704', db='device_data')
    i=0
    while(1):
        if i==1000:
            i=0
            os.system("cls")
        if keyboard.is_pressed("enter"):
            break
        for item in read():
            if(item.get("hostname")==name):
                sql = "INSERT INTO game_speeds (time,up_speed,down_speed) VALUES ('{0}','{1}', '{2}');".format(time.strftime('%H:%M:%S', time.localtime()),item.get("up_speed"),item.get("down_speed"))
                try:
                    with mysql_conn.cursor() as cursor:
                        cursor.execute(sql)
                    mysql_conn.commit()
                except Exception as e:
                    mysql_conn.rollback()
                print(time.strftime('%H:%M:%S', time.localtime()),item.get("up_speed"),item.get("down_speed"))
                break
        time.sleep(1)
        i+=1
    mysql_conn.close()
    os.system("cls")

def test(name):
    js = {"hosts_info": {"set_flux_limit": { "down_limit": "9","mac": "00-24-68-CB-3E-12","up_limit": "1", "name": "TV", "is_blocked": "0"}}, "method": "do"}
    js = json.dumps(js).encode('utf-8')
    resp = requests.post(url, js)
    print(resp.text)
url = ""
if __name__ == '__main__':
    url = get_url()
    name="HUAWEI"
    recode(name)



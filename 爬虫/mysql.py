import pymysql
import keyboard
import time
# mysql_conn = pymysql.connect(host='rm-7xvcob73w77pgn5v8to.mysql.rds.aliyuncs.com',
#                              port=3306, user='app', password='Xuwenbo20040704', db='device_data')
# sql = "INSERT INTO tv_speeds (time,up_speed,down_speed) VALUES ('{0}','{1}', '{2}');".format("19:12:00",'12342', '13245235')
# try:
#     with mysql_conn.cursor() as cursor:
#         cursor.execute(sql)
#     mysql_conn.commit()
# except Exception as e:
#     mysql_conn.rollback()
# mysql_conn.close()
while(1):
    print(keyboard.is_pressed("enter"))
    time.sleep(1)

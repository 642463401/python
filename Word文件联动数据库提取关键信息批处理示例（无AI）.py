import pymysql
from openpyxl import load_workbook
import datetime
wb = load_workbook(r'工程量3.xlsx')
ws = wb.active
v = list(ws.values)
db=pymysql.connect(
    #本机使用localhost，服务器使用ip地址
    host='127.0.0.2',
    #用户名，如变动请按变动后修改
    user='root',
    #密码，如变动请按变动后修改
    password='123123qq.',
    #数据库名，如变动请按变动后修改
    database='工程量')
cursor = db.cursor()
for i in range(2,len(v)):
    区域 = v[i][0]
    工作内容 = v[i][1]
    工程量 = v[i][2]
    累计工程量 = v[i][3]
    日期 = v[i][4]
    录入时间 = datetime.datetime.now().strftime('%Y-%m-%d')
    sql = "insert into 工程量统计表(区域,工作内容,工程量,累计工程量,日期,录入时间) values('{}','{}','{}','{}','{}','{}')".format(区域,工作内容,工程量,累计工程量,日期,录入时间)
    cursor.execute(sql)
    #sql = "insert into study(区域,工作内容,工程量,累计工程量,日期) values(%s,%s,%s,$s,%s)"
    #cursor.execute(sql, [区域, 工作内容, 工程量, 累计工程量, 日期])
db.commit()
db.close()
# 拿整合数据以及细分数据


import time
import json
import pandas as pd
import requests
import websocket
from requests.adapters import HTTPAdapter
import time
from mybaseCode import *

try:
    import thread
except ImportError:
    import _thread as thread


class websocket_class:
    sheetIds = []  # 获取所有的标签ID

    targetID = {}

    finalDF = pd.DataFrame()

    # 这里之所以封装一个大的字典，是因为websocket代码启动后，不管是抛出请求还是接收数据都是走的 他自带的on_message的这个方法，所以说封装一个字典，自己可以通过我上面说的 randomID，找到发出的请求对应的接收数据。
    targetID['getDataOnWidgetV2'] = {}
    targetID['targetSecondID'] = {}
    targetID['tagTitle'] = {}  # 标签名称

    targetID['classifyId'] = {}

    newDF = pd.DataFrame()

    def __init__(self, cookie, appId, crowId, crowName, startDate, endDate, userChoice):

        # 配置requests
        self.s = requests.Session()
        self.NETWORK_STATUS = False
        self.REQUEST_TIMEOUT = False
        # 配置requests超时重试
        self.s.mount('http://', HTTPAdapter(max_retries=3))
        self.s.mount('https://', HTTPAdapter(max_retries=3))

        self.appId = appId
        self.startDate = startDate
        self.endDate = endDate
        self.userChoice = userChoice
        self.insertCookie(cookie)

        self.crowName = crowName
        self.getReportId(cookie, crowId)
        self.count = 0
        self.final = pd.DataFrame(columns=[''])

        self.argsId = []
        self.final_classify = pd.DataFrame(columns=['标签', '子标签', crowName + '内容浏览人数占比', crowName + '达人站内粉丝数量'])

        self.final_all = pd.DataFrame(columns=['主标签', crowName + '数据总和', crowName + '内容浏览人数占比', crowName + '达人站内粉丝数量'])

        self.n = 0
        self.l = 0
        self.idInfo = {}

    def insertCookie(self, cookie):
        # 读取cookie
        cookies = []
        try:
            for line in cookie.split(';'):
                name, value = line.strip().split('=', 1)
                cookies.append({'name': name, 'value': value})
        except ValueError:
            print('cookie格式错误!')
            return {'errCode': -1, 'errMsg': 'cookie格式错误!'}

        # cookie注入到requests.Session
        for cookie in cookies:
            if cookie['name'] == '_tb_token_':
                self.x_csrf_token = cookie['value']
            self.s.cookies.set(cookie['name'], cookie['value'])

    def getReportId(self, cookie, crowId):

        # 获取类目下所有的reportId
        first_url = 'https://strategy.tmall.com/api/scapi'
        data = {
            "contentType": "application/json",
            "param": {
                "crowdId": crowId
            },
            "reportType": "CELEBRITY_REPORT",
            "path": "/v2/reportcommon/getReportInfoByParam"
        }

        sendPackage = requests_method(cookie, 'post', first_url, json.dumps(data))
        info = json.loads(sendPackage)

        self.reportInfo = [[item['reportId'], item['reportName']] for item in info['data'] if
                           timeStamp(item['startDate']).split(' ')[0] == self.startDate and
                           timeStamp(item['endDate']).split(' ')[0] == self.endDate]

        print(self.reportInfo)
        if self.reportInfo == []:
            exit()

    def on_message(self, ws, message):

        print('进入message-----------------------%s' % (message))

        message = json.loads(message)

        if int(message['code']) == 0:

            subrid = message['headers']['rid']

            # 通过页面标签获取接口转接的标签ID
            if subrid in self.targetID['targetSecondID'].keys():
                print('--------进入第一个方法获取body----------------')
                print(message['body'])

                randomID = str(int(time.time() * 1000)) + str(self.count).zfill(2)
                sendMessage = {"method": "/iWidget/list", "headers": {"rid": randomID, "type": "PULL"},
                               "body": {"args": {"sheetId": message['body'], "appId": self.appId}}}

                ws.send(json.dumps(sendMessage))
                self.targetID['tagTitle'][randomID] = self.targetID['targetSecondID'][subrid]
                print(self.targetID)

                self.count += 1
                time.sleep(1)

            elif (subrid in self.targetID['tagTitle'].keys()):
                print('------------------进入第2个方法---拿id------------------')
                randomID = str(int(time.time() * 1000)) + str(self.count).zfill(2)

                self.idInfo = {str(item['id']): item['widgetKey'] for item in message['body']['list'] if
                               'widgetKey' in item.keys() and item['widgetKey'] == '内容浏览偏好'}

                print('self.idInfo -------:%s' % (self.idInfo))

                self.idInfo['173821'] = '达人领域内容浏览偏好'

                # 这里就不用默认是获取全部的数据，还是分类的数据，因为分类需要用到全部请求发送之后获得的数据
                for k, v in self.idInfo.items():
                    self.argsId.append(k)
                    print(k)
                    message = {
                        "method": "/queryDataService/queryDataOnWidget",
                        "headers": {
                            "rid": randomID,
                            "type": "PULL"
                        },
                        "body": {
                            "args": {
                                "id": str(k),
                                "isMock": 0,
                                "selections": [{
                                    "restrictList": [{
                                        "hide": 1,
                                        "oper": "eq",
                                        "value": self.reportInfo[0][0]
                                    }],
                                    "dimensionName": "reportId",
                                    "eq": [{
                                        "hide": 1,
                                        "oper": "eq",
                                        "value": self.reportInfo[0][0]
                                    }],
                                    "lt": None,
                                    "gt": None,
                                    "ge": None,
                                    "le": None,
                                    "ne": None,
                                    "showText": self.reportInfo[0][0]
                                }, {
                                    "restrictList": [{
                                        "hide": 1,
                                        "oper": "eq",
                                        "value": self.userChoice[0]
                                    }],
                                    "dimensionName": "type",
                                    "eq": [{
                                        "hide": 1,
                                        "oper": "eq",
                                        "value": self.userChoice[0]
                                    }],
                                    "lt": None,
                                    "gt": None,
                                    "ge": None,
                                    "le": None,
                                    "ne": None,
                                    "showText": self.userChoice[0]
                                }],
                                "rdPathInfoList": [],
                                "appId": self.appId
                            }
                        }
                    }
                    ws.send(json.dumps(message))

                    # 这里需要代码调整·
                    self.targetID['getDataOnWidgetV2'][randomID] = self.targetID['tagTitle'][subrid] + v
                    self.count += 1
                    # time.sleep(3)

                # message = '{"method":"/iCubeEngine/getDataOnWidgetV2","headers":{"rid":'+str(randomID)+',"type":"PULL"},"body":{"args":{"id":'+str(sendSheetId)+',"isMock":0,"appId":'+self.appId+'}}}'
                # print(type(message))
                # print(message)
                # ws.send(message)

            elif (subrid in self.targetID['getDataOnWidgetV2'].keys()):

                print('--------------------进入第三个方法-----------------------------------')
                # 这里就可以对全部数据进行保存，并且继续发送分类请求

                valueInfo = []
                for item in message['body']['axises'][0]['values']:

                    print(message)

                    if len(message['body']['datas']) < 4:
                        valueInfo.append(item['key'])
                    elif len(message['body']['datas']) == 4:
                        valueInfo.append(item['showName'])
                print('valueInfo------------------：%s' % (valueInfo))

                rate = message['body']['datas'][0]['values']
                crowd_cnt = message['body']['datas'][1]['values']

                if len(message['body']['datas']) < 4:
                    print('---获取分类数据')

                    for index, item in enumerate(valueInfo):
                        for k, v in self.idInfo.items():
                            randomID = str(int(time.time() * 1000)) + str(self.count).zfill(2)
                            message = {
                                "method": "/queryDataService/queryDataOnWidget",
                                "headers": {
                                    "rid": randomID,
                                    "type": "PULL"
                                },
                                "body": {
                                    "args": {
                                        "id": k,
                                        "isMock": 0,
                                        "selections": [{
                                            "restrictList": [{
                                                "hide": 1,
                                                "oper": "eq",
                                                "value": self.reportInfo[0][0]
                                            }],
                                            "dimensionName": "reportId",
                                            "eq": [{
                                                "hide": 1,
                                                "oper": "eq",
                                                "value": self.reportInfo[0][0]
                                            }],
                                            "lt": None,
                                            "gt": None,
                                            "ge": None,
                                            "le": None,
                                            "ne": None,
                                            "showText": self.reportInfo[0][0]
                                        }, {
                                            "restrictList": [{
                                                "hide": 1,
                                                "oper": "eq",
                                                "value": self.userChoice[0]
                                            }],
                                            "dimensionName": "type",
                                            "eq": [{
                                                "hide": 1,
                                                "oper": "eq",
                                                "value": self.userChoice[0]
                                            }],
                                            "lt": None,
                                            "gt": None,
                                            "ge": None,
                                            "le": None,
                                            "ne": None,
                                            "showText": self.userChoice[0]
                                        }, {
                                            "restrictList": [{
                                                "hide": 0,
                                                "oper": "eq",
                                                "value": item
                                            }],
                                            "dimensionName": "domain_name",
                                            "showName": "领域名称",
                                            "eq": [{
                                                "hide": 0,
                                                "oper": "eq",
                                                "value": item
                                            }],
                                            "lt": None,
                                            "gt": None,
                                            "ge": None,
                                            "le": None,
                                            "ne": None,
                                            "showText": item
                                        }],
                                        "rdPathInfoList": [],
                                        "appId": self.appId
                                    }
                                }
                            }
                            ws.send(json.dumps(message))
                            self.targetID['classifyId'][randomID] = v + '--' + item
                            self.count += 1
                            time.sleep(1)
                        self.final_all.loc[self.l] = [item, crowd_cnt[index], rate[index], '']
                        self.l += 1
                elif len(message['body']['datas']) == 4:
                    fans_cnt = message['body']['datas'][2]['values']
                    for index, item in enumerate(valueInfo):
                        self.final_all.loc[self.l] = [item, crowd_cnt[index], rate[index], fans_cnt[index]]
                        self.l += 1
                print(self.finalDF)
                self.final_all.to_excel(str(self.crowName) + self.userChoice[1] + '整合数据.xlsx')
            elif (subrid in self.targetID['classifyId'].keys()):

                print('-------------进入第四个方法--------------')
                if len(message['body']['datas']) >= 4:
                    print(message)
                    print('-------------- 取右边的数据-------------------')
                    rate = message['body']['datas'][0]['values']

                    showNames = [item['showName'] for item in message['body']['axises'][0]['values'] if
                                 'showName' in item.keys()]
                    crowd_cnt = message['body']['datas'][1]['values']

                    fans_cnt = message['body']['datas'][2]['values']
                    print('---------------')
                    print(self.targetID['classifyId'][subrid])
                    print('showNames---:%s' % (len(showNames)))
                    print('fans_cnt---:%s' % (len(fans_cnt)))
                    print('rate---:%s' % (len(rate)))

                    if showNames != [] and len(showNames) == len(crowd_cnt) == len(fans_cnt) == len(rate):
                        for index, item in enumerate(rate):
                            print('--------------')

                            self.final_classify.loc[self.n] = [self.targetID['classifyId'][subrid], showNames[index],
                                                               item, fans_cnt[index]]
                            self.n += 1
                        print(self.final_classify)
                        self.final_classify.to_excel(str(self.crowName) + self.userChoice[1] + '分类整合的数据.xlsx')
                        print(self.finalDF)



        else:
            print('')

        # else:
        #     print('')

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("关闭连接")

    def on_open(self, ws):
        print('进入方法')

        # 在一级标题的基础下发送请求，拿到所有的二级标题和对应的id
        def run(*args):
            print('进入run------------')
            id = '6987'
            randomID = str(int(time.time() * 1000)) + str(self.count).zfill(2)
            sendMessage = {"method": "/iSheet/getReportSheetId", "headers": {"rid": randomID, "type": "PULL"},
                           "body": {"args": {"appId": self.appId, "id": id}}}
            ws.send(json.dumps(sendMessage))

            self.count += 1
            self.targetID['targetSecondID'][randomID] = self.crowName

        run()
        # thread.start_new_thread(run, ())


# 更新cookie,更新Sec-WebSocket-Key
if __name__ == "__main__":
    cookie = 'cna=0Ad4FKsqxhcCAbfACu0Q09OV; hng=CN%7Czh-CN%7CCNY%7C156; _uab_collina=155200949730072619792649; _umdata=G9630C34C82DA27B4CEB56A5D266DF88F020D9F; _m_h5_tk=b6cfe39288605993b01e4577712e4ff4_1553689557032; _m_h5_tk_enc=ebda82b792d885619f3849b06580690f; uss=""; t=7f37bc10b1f658afa3153a168611aac7; lid=%E4%B8%8A%E6%B5%B7%E6%99%AE%E5%A4%AA%E4%BF%A1%E6%81%AF%E5%92%A8%E8%AF%A2%E5%B7%A5%E4%BD%9C%E5%AE%A4; _tb_token_=e8167e8f57a8b; cookie2=1e44039637fd44722e24501a9da85817; _l_g_=Ug%3D%3D; ck1=""; login=true; tracknick=%5Cu4E0A%5Cu6D77%5Cu70B9%5Cu6B63%5Cu4E92%5Cu8054%5Cu7F51%5Cu79D1%5Cu6280; unb=3246332809; lgc=%5Cu4E0A%5Cu6D77%5Cu70B9%5Cu6B63%5Cu4E92%5Cu8054%5Cu7F51%5Cu79D1%5Cu6280; cookie1=BYXMVOB%2F0gpSvIKO64slLGtNsh3h2mrU2YNsQG%2BLLag%3D; cookie17=UNJV26FcR9wX2A%3D%3D; _nk_=%5Cu4E0A%5Cu6D77%5Cu70B9%5Cu6B63%5Cu4E92%5Cu8054%5Cu7F51%5Cu79D1%5Cu6280; welcomeShownTime=1554086034090; uc1=cookie16=URm48syIJ1yk0MX2J7mAAEhTuw%3D%3D&cookie21=URm48syIZxx%2F&cookie15=W5iHLLyFOGW7aA%3D%3D&existShop=false&pas=0&cookie14=UoTZ4M%2BlTmQPFg%3D%3D&tag=8&lng=zh_CN; uc3=vt3=F8dByEnaRgaB6QOLstM%3D&id2=UNJV26FcR9wX2A%3D%3D&nk2=qiAr7C1v6U%2BpUQ0vr5j5%2BjWB&lg2=WqG3DMC9VAQiUQ%3D%3D; csg=3e6f4708; skt=66b922e84531875c; _mw_us_time_=1554103033902; __YSF_SESSION__={"baseId":"fed21222d74ba9fc","brandId":"f2afe39dfe825da7","departmentId":"a390981e5bb99005","smartId":"04df0fee0291dbc9","databankProjectId":"d0a753a2dc1af0d5"}; l=bBIM5N3gvm33S8iXBOfgdZi0a9QtqIRb8sPPh27gbICPO15W50DGBZsjQtTXC3GVZ6i6R3udQJkTBGmkVPRR.; isg=BKamBX-zxpBXJ5Li-qgzu3w59xoi3-ro_W0jUpBPWUmkE0ct-BduUFbla086u-JZ'

    appId = '7'  # 品牌ID
    crowInfo = [
        [2030149, "PT粉底液膏行业新客汇总"],
        [2030039, "PT粉底液膏品牌新客汇总"],

    ]

    crowId = crowInfo[0][0]
    crowName = crowInfo[0][1]

    print(crowName)
    startDate = '2018-12-23'
    endDate = '2019-03-31'
    # 默认是：图文   2：直播
    userChoices = [['1', '图文'], ['2', '直播']]
    userChoice = userChoices[1]
    # cateId=''
    ##############################上方的三个参数是需要修改的，AIPL对应4个不同的cookie来运行

    header = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,la;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'Upgrade',
        'Cookie': cookie,
        'Host': 'ws-nextbi.tmall.com',
        'Origin': 'https://insight-engine.tmall.com',
        'Pragma': 'no-cache',
        'Sec-WebSocket-Extensions': 'permessage-deflate; client_max_window_bits',
        'Sec-WebSocket-Key': '8Sj/vl2Hd+eoX3i9JKh/iA==',
        'Sec-WebSocket-Version': '13',
        'Upgrade': 'websocket',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }

    websocket.enableTrace(True)
    websocket_obj = websocket_class(cookie, appId, crowId, crowName, startDate, endDate, userChoice)
    ws = websocket.WebSocketApp("wss://ws-insight-engine.tmall.com/",
                                on_message=websocket_obj.on_message,
                                on_error=websocket_obj.on_error,
                                on_close=websocket_obj.on_close,
                                header=header)

    ws.on_open = websocket_obj.on_open
    ws.run_forever()
    ws.close()
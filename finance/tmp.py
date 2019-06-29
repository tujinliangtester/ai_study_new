import websocket
import requests

try:
    import thread
except ImportError:
    import _thread as thread
import time


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(self, ws):
    def run(*args):
        # 这里面就是写大家倒退出来页面请求第一步的代码
        pass

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    header = {'Host': ' gwt2.mql5.com',
              'User-Agent': ' Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',
              'Accept': ' */*',
              'Accept-Language': ' zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
              'Accept-Encoding': ' gzip, deflate, br',
              'Sec-WebSocket-Version': ' 13',
              'Origin': ' https//trade.mql5.com',
              'Sec-WebSocket-Extensions': ' permessage-deflate',
              'Sec-WebSocket-Key': ' dy2kgboZ7izia02KwX2wVQ',
              'Connection': ' keep-alive, Upgrade',
              'Cookie': ' uniq=C151C08D-31D2-S-180116; lang=zh',
              'Pragma': ' no-cache',
              'Cache-Control': ' no-cache',
              'Upgrade': ' websocket'}
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://gwt2.mql5.com/",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                header=header)
    ws.on_open = on_open
    # ws.run_forever()


    url='https://gwt2.mql5.com/'
    res=requests.get(url,header)
    print(res)

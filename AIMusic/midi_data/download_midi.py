'''
用selenium下载midi文件

'''

from selenium import webdriver
import time

browser = webdriver.Firefox()
browser.maximize_window()
url='http://midi.midicn.com/2000/06/06/%E5%8F%A4%E5%85%B8%E9%9F%B3%E4%B9%90MIDI'
browser.get(url)

'''
由于不好处理弹窗，所以第一次先手动选择保存，并勾选“以后均执行此操作”，后续就只需要点击各个链接即可进行下载
'''

el=browser.find_element_by_xpath('//section[@class="article-content"]/ul/li[1]/a')
el.click()
print('由于不好处理弹窗，所以第一次先手动选择保存，并勾选“以后均执行此操作”，后续就只需要点击各个链接即可进行下载')
time.sleep(15)
i=1
try:
    while(True):
        i+=1
        el_xpath='//section[@class="article-content"]/ul/li[{0}]/a'.format(i)
        el = browser.find_element_by_xpath(el_xpath)
        el.click()
        time.sleep(3)

except Exception:
    print(Exception)

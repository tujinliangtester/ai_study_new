from bs4 import BeautifulSoup
import urllib.request
import re,csv

def parse_url(url,filename):
    # url='http://bj.01fy.cn/sale/list_0_0_0_0-0_0_0-0_0_0_0-0_0_0-0_2_0_1_.html'
    html=urllib.request.urlopen(url).read().decode('utf-8')
    soup=BeautifulSoup(html,'html.parser')

    area=[]
    floor=[]
    site=[]
    price=[]

    list_house=soup.find_all('li',class_='clear-fix')

    pattern=re.compile('.+\n')
    for i in list_house:
        addr=pattern.findall(i.get_text())
        try:
            a=re.search('\d+㎡',addr[3]).group()[:-1]
            print(a)
            f=re.search('第\d层',addr[3]).group()[1:2]

            s=addr[4].split('-')[0]

            p=addr[9].split('元')[0]

            site.append(s)
            floor.append(f)
            area.append(a)
            price.append(p)
        except Exception:
            print(Exception)

    with open(filename,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(area)
        writer.writerow(floor)
        writer.writerow(site)
        writer.writerow(price)
        print(111)


if __name__=='__main__':
    i = 0
    while(True):
        filename='house_price1543.csv'
        i+=1
        url = 'http://bj.01fy.cn/sale/list_0_0_0_0-0_0_0-0_0_0_0-0_0_0-0_2_0_' + str(i) + '_.html'
        parse_url(url,filename=filename)
        if(i>=10):
            break
import urllib.request
import re

fy=[]
url='http://bj.01fy.cn/sale/house_8944648x.html'
res=urllib.request.urlopen(url).read().decode('utf-8')
print(res)

reg='\d+.\d+元/㎡'
price=re.search(reg,res).group()
price=price[:-3]
print(price)

reg='\d+㎡'
s=re.search(reg,res).group()
s=s[:-1]
print(s)
fy.append(s)


reg='<dd>北京 &nbsp;.*</dd>'
address=re.search(reg,res).group()
address=address[14:17]
print(address)
fy.append(address)

reg='第\d+层'
storey=re.search(reg,res).group()
storey=storey[1:-1]
print(storey)
fy.append(storey)

fy.append(price)

print(fy)

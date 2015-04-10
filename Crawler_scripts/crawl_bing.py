import json
import urllib
import urllib2
from bs4 import BeautifulSoup
import webbrowser
import re
import json
import time
import ast
import os

#logos = ['Addidas', 'Nike', 'DHL', 'Lufthansa','Starbucks','Fedex','McDonalds','Cocacola','Pepsi','Sprite']
logos = ['Addidas']#,'Nike']

for logo_name in logos:
   if not os.path.exists(logo_name):
     os.makedirs(logo_name)
   
   num=0
   corresponding_element_file = logo_name + '_elements.txt'
   infile=open(corresponding_element_file,'r')
   for line in infile:
   	elements= line
   	break
   
   
   #starturl ='http://www.bing.com/images/search?q=nike+art+work&go=Submit&qs=n&form=QBIR&pq=nike+art+work&sc=0-0&sp=-1&sk='
   
   
   #req = urllib2.Request(iriToUri(starturl), headers={ 'User-Agent': 'Mozilla/5.0' })
   #cont = urllib2.urlopen(req).read()
   #print cont
   soup = BeautifulSoup(elements)
   
   links= soup.find_all("div", { "class" : "dg_u" })
   
   for link in links:
   	num=num+1
   	#if num<78:
   	#	continue
   	print num
   
   	m= link.find('a')['m']
   	m=m.replace('{','').replace('}','').split(',')
   	#print m
   	for l in m:
   		if l.startswith('imgurl:"http'):
   			imgurl=l[8:-1]
   			print imgurl
   	#while True:
   	try:
   		urllib.urlretrieve(imgurl,logo_name+str(num)+imgurl[-4:])
                os.system('mv %s %s' %(logo_name+str(num)+imgurl[-4:],logo_name))
   		
   	except:
   		print 'Bad link !!'
   		#break
   	
   








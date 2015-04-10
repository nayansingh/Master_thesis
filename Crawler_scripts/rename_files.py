import os
import sys

Folder_path = '../Addidas/'
#List_of_folders = os.listdir(Folder_path)
counter = 1

#for j in range(0,len(List_of_folders)):
#    Path = List_of_folders[j]
#    if not ((Path.rfind('.py') == -1) and (Path.rfind('.txt') == -1) and (Path.rfind('.sh') == -1)):
#       continue

print 'path =',Folder_path
Name = 'Addidas'
counter += 1
start_num = 71

list_of_files = os.listdir(Folder_path)

for i in range(0,len(list_of_files)):
  if not ((list_of_files[i].rfind('.py') == -1) and (list_of_files[i].rfind('.sh') == -1) and (list_of_files[i].rfind('.txt') == -1)):
    continue
  os.rename(list_of_files[i],Name+str(start_num)+'.jpg')
  os.system('cp %s %s' % (Name+str(start_num)+'.jpg','/home/vion_labs/Vion_labs/crawler/Addidas_0/'))
  start_num += 1 
  #print 'List = ',list_of_files[i]



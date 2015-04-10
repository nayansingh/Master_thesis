import os
import sys

Folder_path = '../Addidas/'
#List_of_folders = os.listdir(Folder_path)
counter = 1

#for j in range(0,len(List_of_folders)):
#    Path = List_of_folders[j]
#    if not ((Path.rfind('.py') == -1) and (Path.rfind('.txt') == -1) and (Path.rfind('.sh') == -1)):
#       continue

print 'path =',Path
Name = Path
counter += 1
start_num = 71

list_of_files = os.listdir(Path)

for i in range(0,len(list_of_files)):
  if not ((list_of_files[i].rfind('.py') == -1) and (list_of_files[i].rfind('.sh') == -1) and (list_of_files[i].rfind('.txt') == -1)):
    continue
  os.rename(list_of_files[i],Name+str(i)+'.jpg')
  os.system('mv %s %s' % (Name+str(i)+'.jpg','/home/vion_labs/Vion_labs/crawler/Addidas_0/'))
  start_num += 1 
  #print 'List = ',list_of_files[i]



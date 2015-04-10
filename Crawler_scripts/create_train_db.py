import os
import sys
import PIL.Image


TRAIN_FILE_PATH = '../crawler/train.txt'
VAL_FILE_PATH = '../crawler/val.txt'
BASE_PATH = '/home/vion_labs/crawler/'
Folder_path = '../crawler/'

PATH_TO_TRAIN = '../crawler/logo_train/'

PATH_TO_VAL = '../crawler/logo_val/'

List_of_folders = os.listdir(Folder_path)

train_file_ptr = open(TRAIN_FILE_PATH,'w')
val_file_ptr = open(VAL_FILE_PATH,'w')

for j in range(0,len(List_of_folders)):
    Path = List_of_folders[j]
    if not ((Path.rfind('.py') == -1) and (Path.rfind('.txt') == -1) and (Path.rfind('.sh') == -1) and (Path.rfind('logo_train') == -1) and (Path.rfind('logo_val') == -1)):
        continue

    pos = List_of_folders[j].rfind('_')
    print 'path = ',List_of_folders[j].rfind('_')

    list_of_files = os.listdir(Path)

    for_training = len(list_of_files) * 80 / 100

    print 'Folder = ',List_of_folders[j]
    print 'for_training = ',for_training

    for i in range(0,len(list_of_files)):
        if not (list_of_files[i].rfind('.py') == -1) and (list_of_files[i].rfind('.sh') == -1) and (list_of_files[i].rfind('.txt') == -1):
            continue
 
        try:
#            print '111 Verify = '
            im = PIL.Image.open(List_of_folders[j]+'/'+list_of_files[i])
            im.verify()
            #print '111 Verify = ',im.verify()
        except Exception as e:
            print '\n 222 Verify = ',str(e)
            continue

        if i < int(for_training):
           FILE_NAME =  List_of_folders[j]
           string_in = FILE_NAME[pos+1:len(FILE_NAME)]
           #data_to_write = BASE_PATH+List_of_folders[j]+'/'+list_of_files[i]+'    '+ string_in +'\n'
           data_to_write = list_of_files[i]+'    '+ string_in +'\n'
           #print 'List - ',list_of_files[i]
           #print 'string_in', string_in
           train_file_ptr.write(data_to_write)
           os.system('cp %s %s' % (BASE_PATH+List_of_folders[j]+'/'+list_of_files[i],PATH_TO_TRAIN))
        else:
           FILE_NAME1 =  List_of_folders[j]
           string_in = FILE_NAME1[pos+1:len(FILE_NAME1)]
           #data_to_write1 = BASE_PATH+List_of_folders[j]+'/'+list_of_files[i]+'    '+ string_in +'\n'
           data_to_write1 = list_of_files[i]+'    '+ string_in +'\n'
           val_file_ptr.write(data_to_write1)
           os.system('cp %s %s' % (BASE_PATH+List_of_folders[j]+'/'+list_of_files[i],PATH_TO_VAL))


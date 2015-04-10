import numpy as np
import matplotlib.pyplot as plt
from html import HTML

# Make sure that caffe is on the python path:
caffe_root = '../../../../caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import draw_net
import matplotlib.pyplot as plt
import cv2
import os 
from PIL import Image, ImageDraw,ImageFont
from collections import namedtuple
import time

#function definitions 
def Predict_logo_for_image(input_image):
    try:
         #print 'Try prediction'
         caffe_input_image = caffe.io.load_image(Path+'/'+list_of_files[j])
    except IOError:
         #print 'IOError '
         exit(0)
    prediction = net.predict([caffe_input_image], oversample=False)
    #plt.plot(prediction[0])
    #print 'Image predicted as = ',LOGO_NAMES[prediction[0].argmax()]
    return(LOGO_NAMES[prediction[0].argmax()])

def Phase_1_Predict_logo_pos_x_for_image(input_image,hori_shift,phase):
     predicted_x = {} 
     Path_for_x = '/home/vion_labs/caffe-master/SPP_net-master/model-defs/deploy_spp_11_categories/cropped_x_' + str(phase) + '/'
     try:
       os.stat(Path_for_x)
     except:
       os.mkdir(Path_for_x,0755)
     
     im = Image.open(input_image)
     #if(phase == 2):
     #   im1 = outimg.transpose(Image.FLIP_LEFT_RIGHT)
     #   im = im1.transpose(Image.FLIP_TOP_BOTTOM)
     #   im.show()
     #   #cv2.imshow('Input Image',) 
     #   #cv2.waitKey(1000)
     #else:
     #  im = outimg
     
     index = 0
     x= 0
     while(x < (im.size[0]-hori_shift)):
        outfile = 'Cropped_'+ str(x)+'.jpg'
        #im = Image.open('fedex.jpg')
        #print 'point_x = {} point_y = {}'.format(point_x,point_y) 
        #print 'point_x shi = {} point_y shi = {}'.format(point_x + hori_shift,point_y + vert_shift) 
        if(phase == 1):
           region = im.crop((x,0,im.size[0],im.size[1]))
        else:
           region = im.crop((0,0,im.size[0]-x,im.size[1]))
        
        region.save(Path_for_x +  outfile,"JPEG")
        try:
           #print 'Try prediction at X direction '
           caffe_input_image = caffe.io.load_image(Path_for_x + outfile)
        except IOError:
           print 'IOError '
           x = x + hori_shift
           exit(0)

        prediction = net.predict([caffe_input_image], oversample=False)
        #plt.plot(prediction[0])
        #print 'prediction = ',prediction[0].argmax()
        #print '{} prediction = {}'.format(x,LOGO_NAMES[prediction[0].argmax()])
        #predicted_x[index] = LOGO_NAMES[prediction[0].argmax()]
        predicted_x[index] = prediction[0].argmax()
        index += 1 
        x = x + hori_shift
     return predicted_x

def Phase_1_Predict_logo_pos_y_for_image(input_image,vert_shift,phase):
     predicted_y = {} 
     Path_for_y = '/home/vion_labs/caffe-master/SPP_net-master/model-defs/deploy_spp_11_categories/cropped_y_' + str(phase) + '/'
     try:
       os.stat(Path_for_y)
     except:
       os.mkdir(Path_for_y,0755)
     
     im = Image.open(input_image)
     
     index = 0
     y = 0
     while(y < (im.size[1]-vert_shift)):
        outfile = 'Cropped_'+ str(y)+'.jpg'
        if(phase == 1):
           region = im.crop((0,y,im.size[0],im.size[1]))
        else:
           region = im.crop((0,0,im.size[0],im.size[1]-y))
        
        region.save(Path_for_y +  outfile,"JPEG")
        try:
           #print 'Try prediction at Y direction '
           caffe_input_image = caffe.io.load_image(Path_for_y + outfile)
        except IOError:
           #print 'IOError '
           y = y + vert_shift
           exit(0)
           continue

        prediction = net.predict([caffe_input_image], oversample=False)
        #plt.plot(prediction[0])
        #predicted_y[index] = LOGO_NAMES[prediction[0].argmax()] 
        predicted_y[index] = prediction[0].argmax() 
        index+=1
        y = y + vert_shift
     return predicted_y

def Phase_1_Predict_right_bottom_vert_x(predicted_x,hori_shift,phase,im_width):
      detections_x_direction = {}
      num_of_det_per_cat_x = np.zeros(len(LOGO_NAMES)) 
      for x in range(1,len(predicted_x)):
        if(predicted_x[x-1] == 7): # For No_Logo, Don't change
           continue
        prev_element = predicted_x[x-1] 
        next_element = predicted_x[x] 
        if(prev_element == next_element):
           continue   
        else:
           detections_x_direction[LOGO_NAMES[predicted_x[x-1]],num_of_det_per_cat_x[predicted_x[x-1]]] = (x * hori_shift) if (phase == 1) else ((x * hori_shift)) 
           num_of_det_per_cat_x[predicted_x[x-1]] += 1 
      return detections_x_direction

def Phase_1_Predict_right_bottom_vert_y(predicted_y,vert_shift,phase,im_height):
      detections_y_direction = {}
      num_of_det_per_cat_y = np.zeros(len(LOGO_NAMES)) 
      for y in range(1,len(predicted_y)):
        if(predicted_y[y-1] == 7): # For No_Logo, Don't change
           continue
        prev_element = predicted_y[y-1] 
        next_element = predicted_y[y] 
        if(prev_element == next_element):
           continue   
        else:
           detections_y_direction[LOGO_NAMES[predicted_y[y-1]],num_of_det_per_cat_y[predicted_y[y-1]]] = (y * vert_shift) if (phase == 1) else ((y * vert_shift))
           num_of_det_per_cat_y[predicted_y[y-1]] += 1 
      
      return detections_y_direction    

def search_for_logo_in_window(input_image,phase_1_detect,phase_2_detect,im_width,im_height):
    
    #max_num_keys = max(len(phase_1_detect.keys()),len(phase_2_detect.keys()))
    #print 'keys - 1 = ',phase_1_detect.keys()
    #print 'keys - 2 = ',phase_2_detect.keys()

    store_rect = [] 
    bound_rect = namedtuple('bound_rect',['Logo_id','Occurence_num','x1','y1','x2','y2'])

    #bound_rect = {}

    Path_for_y = '/home/vion_labs/caffe-master/SPP_net-master/model-defs/deploy_spp_11_categories/cropped_search_logo/'
    try:
       os.stat(Path_for_y)
    except:
       os.mkdir(Path_for_y,0755)
    
    y = 0
    for l_keys_ph_1 in phase_1_detect.keys():
        outfile = 'Cropped_ph1_'+ str(y)+'.jpg'

        phase1_out_val = phase_1_detect[l_keys_ph_1]
        phase_1_x = phase1_out_val % im_width
        phase_1_y = int(phase1_out_val / im_width)

        high_x_1 = 0 
        high_y_1 = 0 
        high_x_2 = 0 
        high_y_2 = 0 
 
        search_area = 100
        highest = 0
        str_keys = str(l_keys_ph_1)
        assumed_logo = str_keys[str_keys.find('\'')+1:str_keys.rfind('\'')]
        high_pred_logo = "NULL" 
        global_count_ph1 = 0 
        while(search_area < 300):
           im = Image.open(input_image)
           
           left   = max(0,phase_1_x - search_area)
           top    = max(0,phase_1_y - search_area)
           right  = min(im_width,phase_1_x + search_area)
           bottom = min(im_height,phase_1_y + search_area)
           #right  = min(im_width,phase_1_x + 40)
           #bottom = min(im_height,phase_1_y + 40)
           
           search_area += 10
  
           region = im.crop((left,top,right,bottom))
           region.save(Path_for_y +  outfile,"JPEG")
          
           try:
              #print 'Try prediction at Y direction '
              caffe_input_image = caffe.io.load_image(Path_for_y + outfile)
              y = y + 1
           except IOError:
              #print 'IOError '
              y = y + 1
              exit(0)
              continue

           prediction = net.predict([caffe_input_image], oversample=False)
           predicted_logo = LOGO_NAMES[prediction[0].argmax()]
           #print 'Predicted logo score = ',predicted_logo
           #print 'Predicted logo score  score = ',max(prediction[0])

           if(predicted_logo == assumed_logo):
               global_count_ph1 += 1
               if(max(prediction[0]) >= highest):
                    highest = max(prediction[0]) 
                    high_x_1 = left               
                    high_y_1 = top                
                    high_x_2 = right             
                    high_y_2 = bottom            
                    high_pred_logo = predicted_logo
               else:
                    high_pred_logo =  high_pred_logo
        
        print 'Assumd logo PH1 = ',assumed_logo 
        print 'High logo PH1 = ',high_pred_logo 
        if((high_pred_logo == assumed_logo) and global_count_ph1 > 4):
           temp = [high_pred_logo,int(float(str_keys[str_keys.find(',')+1:str_keys.rfind(')')])),high_x_1,high_y_1,high_x_2,high_y_2]
           element = bound_rect._make(temp)
           store_rect.append(element)
        else:
           kkk =0 
           print 'Detected logo is {} which is not same as predicted earlier {}'.format(high_pred_logo,assumed_logo)
      
# For Phase 2 detected points 

    y = 0
    for l_keys_ph_2 in phase_2_detect.keys():
        outfile = 'Cropped_ph2_'+ str(y)+'.jpg'

        phase2_out_val = phase_2_detect[l_keys_ph_2]
        phase_2_x = phase2_out_val % im_width
        phase_2_y = int(phase2_out_val / im_width)

        high_x_1 = 0 
        high_y_1 = 0 
        high_x_2 = 0 
        high_y_2 = 0 
 
        search_area = 100
        highest = 0
        str_keys = str(l_keys_ph_2)
        assumed_logo = str_keys[str_keys.find('\'')+1:str_keys.rfind('\'')]
        
        high_pred_logo = "NULL" 
        global_count_ph2 = 0 
        while(search_area < 300):
           im = Image.open(input_image)
           
           left   = max(0,phase_2_x - search_area)
           top    = max(0,phase_2_y - search_area)
           #left   = max(0,phase_2_x - 40)
           #top    = max(0,phase_2_y - 40)
           right  = min(im_width,phase_2_x + search_area)
           bottom = min(im_height,phase_2_y + search_area)
           
           search_area += 10
  
           region = im.crop((left,top,right,bottom))
           region.save(Path_for_y +  outfile,"JPEG")
          
           try:
              #print 'Try prediction at Y direction '
              caffe_input_image = caffe.io.load_image(Path_for_y + outfile)
              y = y + 1
           except IOError:
              #print 'IOError '
              y = y + 1
              exit(0)
              continue

           prediction = net.predict([caffe_input_image], oversample=False)
           predicted_logo = LOGO_NAMES[prediction[0].argmax()]
           #print 'PH2 Predicted logo score = ',predicted_logo
           #print 'PH2 Predicted logo score  score = ',max(prediction[0])

           if(predicted_logo == assumed_logo):
               global_count_ph2 += 1 
               if(max(prediction[0]) >= highest):
                    highest = max(prediction[0]) 
                    high_x_1 = left               
                    high_y_1 = top                
                    high_x_2 = right             
                    high_y_2 = bottom            
                    high_pred_logo = predicted_logo
               else:
                    high_pred_logo =  high_pred_logo
        
        print 'Assumd logo PH2 = ',assumed_logo 
        print 'High logo PH2 = ',high_pred_logo 
        if((high_pred_logo == assumed_logo) and global_count_ph2 > 4):
           temp = [high_pred_logo,int(float(str_keys[str_keys.find(',')+1:str_keys.rfind(')')])),high_x_1,high_y_1,high_x_2,high_y_2]
           element = bound_rect._make(temp)
           store_rect.append(element)
        else:
           kkk=0
           print 'PH2 Detected logo is {} which is not same as predicted earlier {}'.format(high_pred_logo,assumed_logo)
    
    print 'Bound = ',store_rect      
    return store_rect           


def perform_position_correction(input_image,phase_1_detect,phase_2_detect,im_width,im_height):
    
    #max_num_keys = max(len(phase_1_detect.keys()),len(phase_2_detect.keys()))
    #print 'keys - 1 = ',phase_1_detect.keys()
    #print 'keys - 2 = ',phase_2_detect.keys()
    Path_for_y = '/home/vion_labs/caffe-master/SPP_net-master/model-defs/deploy_spp_11_categories/cropped_pos_correct/'
    try:
       os.stat(Path_for_y)
    except:
       os.mkdir(Path_for_y,0755)
     
    y = 0 
    for l_keys in max(phase_1_detect.keys(),phase_2_detect.keys()):     
        outfile = 'Cropped_'+ str(y)+'.jpg'
        #im = Image.open(input_image)

        try:
            phase1_out_val = phase_1_detect[l_keys] 
            phase2_out_val = phase_2_detect[l_keys]
        except:
            continue

        logo_present = 0
        phase_1_x = phase1_out_val % im_width
        phase_1_y = int(phase1_out_val / im_width)
        phase_2_x = phase2_out_val % im_width
        phase_2_y = int(phase2_out_val / im_width)
        
        phase_1_x_ori = phase_1_x 
        phase_1_y_ori = phase_1_y 
        phase_2_x_ori = phase_2_x 
        phase_2_y_ori = phase_2_y 
       
        high_phase_1_x = phase_1_x        
        high_phase_1_y = phase_1_y
        high_phase_2_x = phase_2_x 
        high_phase_2_y = phase_2_y

        goto_except = 0
        #while(logo_present == 0):
        count = 0
        highest = 0
        while(count < 50):
           #print 'logo name 1 = ',phase_1_detect[l_keys]
           #print 'logo name 2 = ',phase_2_detect[l_keys]
           im = Image.open(input_image)
           count += 1
           try: 
              if(goto_except):
                ggg = 1/0

              #print 'Try '
              #print '1 X,Y = {} {}'.format(phase_2_x,phase_2_y)
              #print '2 X,Y = {} {}'.format(phase_1_x,phase_1_y)

              if((phase_2_x == phase_1_x) or (phase_1_y == phase_2_y)): 
                 break
               
              if((phase_2_x < phase_1_x) and (phase_2_y < phase_1_y)):
                 region = im.crop((phase_2_x,phase_2_y,phase_1_x,phase_1_y))
                 region.save(Path_for_y +  outfile,"JPEG")
                 phase_2_x = phase_2_x - 10
                 phase_2_y = phase_2_y - 10
              else:
                 region = im.crop((phase_1_x,phase_1_y,phase_2_x,phase_2_y))
                 region.save(Path_for_y +  outfile,"JPEG")
                 phase_1_x = phase_1_x - 10
                 phase_1_y = phase_1_y - 10
                 
           except :
              #print 'Except '
              #print '1 X,Y = {} {}'.format(phase_2_x,phase_1_y)
              #print '2 X,Y = {} {}'.format(phase_1_x,phase_2_y)
              goto_except = 1
              if((phase_2_x == phase_1_x) or (phase_1_y == phase_2_y)): 
                 break
              
              if((phase_2_x < phase_1_x) and (phase_1_y < phase_2_y)):
                  region = im.crop((phase_2_x,phase_1_y,phase_1_x,phase_2_y))
                  region.save(Path_for_y +  outfile,"JPEG")
                  phase_2_x = phase_2_x - 10
                  phase_1_y = phase_1_y - 10
              else:
                  region = im.crop((phase_1_x,phase_2_y,phase_2_x,phase_1_y))
                  region.save(Path_for_y +  outfile,"JPEG")
                  phase_1_x = phase_1_x - 10
                  phase_2_y = phase_2_y - 10
              
           try:
              #print 'Try prediction at Y direction '
              caffe_input_image = caffe.io.load_image(Path_for_y + outfile)
              y = y + 1
           except IOError:
              #print 'IOError '
              y = y + 1
              exit(0)
              continue

           prediction = net.predict([caffe_input_image], oversample=False)
           predicted_logo = LOGO_NAMES[prediction[0].argmax()]
           #print 'Predicted logo score = ',prediction[0].argmax()
           #print 'Predicted logo score  score = ',max(prediction[0])

           highest = (max(prediction[0])) if (max(prediction[0]) > highest) else highest
           high_phase_1_x = (phase_1_x) if (max(prediction[0]) > highest) else high_phase_1_x 
           high_phase_1_y = (phase_1_y) if (max(prediction[0]) > highest) else high_phase_1_y 
           high_phase_2_x = (phase_2_x) if (max(prediction[0]) > highest) else high_phase_2_x
           high_phase_2_y = (phase_2_y) if (max(prediction[0]) > highest) else high_phase_2_y

           logo_present = 0
           str_keys = str(l_keys)
           logo_present = (1) if (predicted_logo == str_keys[str_keys.find('\'')+1:str_keys.rfind('\'')]) else (0)
           if(phase_2_x < 0 or phase_2_y < 0 or phase_1_x < 0 or phase_1_y < 0):
             should_draw = 1
             # retain old values of BB's
             #print'RETAIN OLD BBs '
             phase_1_x = phase_1_x_ori 
             phase_1_y = phase_1_y_ori 
             phase_2_x = phase_2_x_ori 
             phase_2_y = phase_2_y_ori
             predicted_logo =  str_keys[str_keys.find('\'')+1:str_keys.rfind('\'')]
             break;
           else:
             should_draw = 1

        if(should_draw == 1): 
           draw = ImageDraw.Draw(im)
           if((abs(high_phase_1_x - high_phase_2_x)* abs(high_phase_1_y-high_phase_2_y)) > 10):
             if((high_phase_1_x < high_phase_2_x) and (high_phase_1_y < high_phase_2_y)):
                   draw.rectangle( [(high_phase_1_x,high_phase_1_y) ,(high_phase_2_x + 10,high_phase_2_y + 10)],fill=None,outline=20)
             else:
                   draw.rectangle( [(high_phase_2_x+10,high_phase_2_y+10) ,(high_phase_1_x,high_phase_1_y)],fill=None,outline=20)
             #draw.ellipse( (phase_2_x + 10,phase_2_y + 10,phase_1_x,phase_1_y),fill=None,outline='Green' )
             font = ImageFont.load_default()
             draw.text((high_phase_1_x,high_phase_1_y),predicted_logo,font=font, fill=(0,255,0,255))
    return im  



# arg 0 ==> exec
# arg 1 ==> caffe model
# arg 2 ==> npy file 
# arg 3 ==> gpu number (id) 
# Input image files are taken from the image folder 

if(len(sys.argv) != 4):
  #print 'Check the arguments please -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
  exit(0)
else:
  kkk=0
  #print 'Arguments are -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
LOGO_NAMES= ['Addidas', 'Fedex', 'Cocacola', 'DHL' ,'Lufthansa', 'McDonalds' ,'Nike', 'No_Logo', 'Pepsi' ,'Sprite' ,'Starbucks']
MODEL_FILE = 'deploy_11_classes.prototxt'
#PRETRAINED = 'train_val_spp_11_classes_iter17_iter_20001.caffemodel'
PRETRAINED = sys.argv[1]
#CLASS_FILE= '../data/ilsvrc12/synset_words.txt'
IMAGE_FILE = 'input_images.txt'
OUTPUT_FILE = 'out_caffe.txt'
#OUTPUT_HTML_FILE = 'output_caffe_imgnet/out_caffe.html'
#caffe.set_phase_test()
caffe.set_device(int(sys.argv[3]))
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(sys.argv[2]),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))

#class_file_ptr = open(CLASS_FILE)
output_file_ptr = open(OUTPUT_FILE,'w')
#output_html_file_ptr = open(OUTPUT_HTML_FILE,'w')
#print 'input file name = ',IMAGE_FILE
input_img_file_ptr = open(IMAGE_FILE)
Path = './images/'
Result_path = './Results/'

# Detection and localisation results in HTML file
HTMLFILE = Result_path + 'Logo_det_for_img'+time.strftime("%d_%m_%H:%M")+'.html'
try:
    os.stat(Result_path)
except:
    os.mkdir(Result_path,0755)

fp_html_file = open(HTMLFILE, 'w')

num_of_img = 1
list_of_files = os.listdir(Path)

for j in range(0,len(list_of_files)):
        if not ((list_of_files[j].rfind('.py') == -1) and (list_of_files[j].rfind('.sh') == -1) and (list_of_files[j].rfind('.txt') == -1)):
           continue
        #print 'List of files = ',list_of_files[j]
     
	im = Image.open(Path+'/'+list_of_files[j])
	#print 'Image width = ',im.size[0]
	#print 'Image height = ',im.size[1]
	hori_shift = 10
	vert_shift = 10   
       
        # Predict logo for entire image
        Logo_in_image = Predict_logo_for_image(Path+'/'+list_of_files[j])
  
        if(Logo_in_image != 'No_Logo'):
           
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& PASS 1 --> && | (image search from right side and from top to bottom)

           # Predict logo image location by performing regression in x direction
           phase_1_predicted_x = Phase_1_Predict_logo_pos_x_for_image(Path+'/'+list_of_files[j],hori_shift,1) 
           #print 'X = ',phase_1_predicted_x 
           # Predict logo image location by performing regression in x direction
           phase_1_predicted_y = Phase_1_Predict_logo_pos_y_for_image(Path+'/'+list_of_files[j],vert_shift,1) 
           #print 'Y = ',phase_1_predicted_y 

           # Predict the x co-ordi of right bottom vertice of the location of the logos in the image using phase_1_predicted_x  
           phase_1_detections_x_direction = Phase_1_Predict_right_bottom_vert_x(phase_1_predicted_x,hori_shift,1,im.size[0])
           #print 'PHase 1 Detect X = ',phase_1_detections_x_direction
   
           # Predict the y co-ordi of right bottom vertice of the location of the logos in the image using predicted_y  
           phase_1_detections_y_direction = Phase_1_Predict_right_bottom_vert_y(phase_1_predicted_y,vert_shift,1,im.size[1])
           #print 'Phase 1 Detect Y = ',phase_1_detections_y_direction
           iter_pt_x_1 = iter(phase_1_detections_x_direction)
           iter_pt_y_1 = iter(phase_1_detections_y_direction)
           
           min_len_ph1 = min(len(phase_1_detections_x_direction),len(phase_1_detections_y_direction)) 
           max_len_ph1 = max(len(phase_1_detections_x_direction),len(phase_1_detections_y_direction)) 
           
           phase_1_detect = {}  
           
           if(len(phase_1_detections_x_direction) < len(phase_1_detections_y_direction)):
              iter_pt_min_ph1 = iter_pt_x_1
              iter_pt_max_ph1 = iter_pt_y_1
              ph1_min_dict = phase_1_detections_x_direction
              ph1_max_dict =  phase_1_detections_y_direction
              ph1_num_dect = 0
              for min_index_ph1 in range(0,min_len_ph1):
                  min_iter_ph1 = iter_pt_min_ph1.next()
                  try:
                    phase_1_str1 = str(min_iter_ph1) 
                    phase_1_logo_id = phase_1_str1[phase_1_str1.find('\'')+1:phase_1_str1.rfind('\'')]
                    phase_1_num_of_occur = int(float(phase_1_str1[phase_1_str1.find(',')+1:phase_1_str1.rfind(')')]))
                    ph1_key = (phase_1_logo_id,phase_1_num_of_occur)
                    phase_1_detect[ph1_key] =  ph1_min_dict[min_iter_ph1] + ph1_max_dict[min_iter_ph1] * im.size[0] 
                  except: 
                    #print 'Phase 1 X continue ' 
                    continue
                        
           else:
              iter_pt_min_ph1 = iter_pt_y_1
              iter_pt_max_ph1 = iter_pt_x_1   
              ph1_min_dict = phase_1_detections_y_direction
              ph1_max_dict =  phase_1_detections_x_direction
              ph1_num_dect = 0
              for min_index_ph1 in range(0,min_len_ph1):
                  min_iter_ph1 = iter_pt_min_ph1.next()
                  try:
                    phase_1_str1 = str(min_iter_ph1) 
                    phase_1_logo_id = phase_1_str1[phase_1_str1.find('\'')+1:phase_1_str1.rfind('\'')]
                    phase_1_num_of_occur = int(float(phase_1_str1[phase_1_str1.find(',')+1:phase_1_str1.rfind(')')]))
                    ph1_key = (phase_1_logo_id,phase_1_num_of_occur)
                    phase_1_detect[ph1_key] =  ph1_max_dict[min_iter_ph1] + ph1_min_dict[min_iter_ph1] * im.size[0]
                  except: 
                    #print 'Phase 1 Y continue ' 
                    continue
 
                      
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& PASS 2 --> && | (image search from right side and from top to bottom)
           
           # Predict other diagonal vertice of bounding box of the rectangle
            
           # Predict logo image location by performing regression in x direction
           phase_2_predicted_x = Phase_1_Predict_logo_pos_x_for_image(Path+'/'+list_of_files[j],hori_shift,2) 
           #print 'X = ',phase_2_predicted_x 
           # Predict logo image location by performing regression in x direction
           phase_2_predicted_y = Phase_1_Predict_logo_pos_y_for_image(Path+'/'+list_of_files[j],vert_shift,2) 
           #print 'Y = ',phase_2_predicted_y 

           # Predict the x co-ordi of right bottom vertice of the location of the logos in the image using phase_2_predicted_x  
           phase_2_detections_x_direction = Phase_1_Predict_right_bottom_vert_x(phase_2_predicted_x,hori_shift,2,im.size[0])
           #print ' Phase 2 Detect X = ',phase_2_detections_x_direction
   
           # Predict the y co-ordi of right bottom vertice of the location of the logos in the image using predicted_y  
           phase_2_detections_y_direction = Phase_1_Predict_right_bottom_vert_y(phase_2_predicted_y,vert_shift,2,im.size[1])
           #print 'Phase 2 Detect Y = ',phase_2_detections_y_direction
           iter_pt_x_2 = iter(phase_2_detections_x_direction)
           iter_pt_y_2 = iter(phase_2_detections_y_direction)
          
          
           min_len_ph2 = min(len(phase_2_detections_x_direction),len(phase_2_detections_y_direction)) 
           max_len_ph2 = max(len(phase_2_detections_x_direction),len(phase_2_detections_y_direction)) 
           
           phase_2_detect = {}  

           if(len(phase_2_detections_x_direction) < len(phase_2_detections_y_direction)):
              iter_pt_min_ph2 = iter_pt_x_2
              iter_pt_max_ph2 = iter_pt_y_2
              ph2_min_dict = phase_2_detections_x_direction
              ph2_max_dict = phase_2_detections_y_direction
              ph2_num_dect = 0
              for min_index_ph2 in range(0,min_len_ph2):
                  min_iter_ph2 = iter_pt_min_ph2.next()
                  try:
                    phase_2_str1 = str(min_iter_ph2) 
                    phase_2_logo_id = phase_2_str1[phase_2_str1.find('\'')+1:phase_2_str1.rfind('\'')]
                    phase_2_num_of_occur = int(float(phase_2_str1[phase_2_str1.find(',')+1:phase_2_str1.rfind(')')]))
                    ph2_key = (phase_2_logo_id,phase_2_num_of_occur)
                    phase_2_detect[ph2_key] =  ph2_min_dict[min_iter_ph2] + ph2_max_dict[min_iter_ph2] * im.size[0]
                  except:
                    #print 'Phase 2 X continue ' 
                    continue
                        
           else:
              iter_pt_min_ph2 = iter_pt_y_2
              iter_pt_max_ph2 = iter_pt_x_2   
              ph2_min_dict = phase_2_detections_y_direction
              ph2_max_dict = phase_2_detections_x_direction
              ph2_num_dect = 0
              for min_index_ph2 in range(0,min_len_ph2):
                  min_iter_ph2 = iter_pt_min_ph2.next()
                  try:
                    phase_2_str1 = str(min_iter_ph2) 
                    phase_2_logo_id = phase_2_str1[phase_2_str1.find('\'')+1:phase_2_str1.rfind('\'')]
                    phase_2_num_of_occur = int(float(phase_2_str1[phase_2_str1.find(',')+1:phase_2_str1.rfind(')')]))
                    ph2_key = (phase_2_logo_id,phase_2_num_of_occur)
                    phase_2_detect[ph2_key] =  ph2_max_dict[min_iter_ph2] + ph2_min_dict[min_iter_ph2] * im.size[0]
                  except: 
                    #print 'Phase 2 Y continue ' 
                    continue

           #print 'last PH 1 ---> ',phase_1_detect 
           #print 'last PH 2 ---> ',phase_2_detect 
          # function to correctly apply position correction for detected logos in the image frame 
           #Final_image = perform_position_correction(Path+'/'+list_of_files[j],phase_1_detect,phase_2_detect,im.size[0],im.size[1])      
           Bounding_rects = search_for_logo_in_window(Path+'/'+list_of_files[j],phase_1_detect,phase_2_detect,im.size[0],im.size[1])      

           Final_image = im
           #print 'length = ',len(Bounding_rects)
           for num_of_rect in range(0,len(Bounding_rects)):
                temp = Bounding_rects[num_of_rect]
                draw = ImageDraw.Draw(Final_image)
                draw.rectangle( [(temp[2],temp[3]) ,(temp[4],temp[5])],fill=None,outline=20)
                font = ImageFont.load_default()
                draw.text((temp[2],temp[3]),str(temp[0])+'_'+str(temp[1]),font=font, fill=(0,255,0,255))

        else:
	   print 'Image {} INPUT IMAGE ==> {} and PREDICTED LOGO ==> {}'.format(num_of_img,list_of_files[j],LOGO_NAMES[7])
           out_image_no_logo = Path+'/'+list_of_files[j]
           Final_image = Image.open(out_image_no_logo)


        num_of_img += 1
	
# Display final detection and localisation image  
        Final_image.show()        
        input_img_file_ptr.close()

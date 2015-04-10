import numpy as np
import sys
import os
from PIL import Image

im = Image.open('fedex.jpg')
print 'Image width = ',im.size[0]
print 'Image height = ',im.size[1]
hori_factor = 4
vert_factor = 3
hori_shift = im.size[0] / hori_factor
vert_shift = im.size[1] / vert_factor
print 'Hori shift = ',hori_shift
print 'Vert shift = ',vert_shift

for y in range(0,vert_factor):
   for x in range (0,hori_factor):
     outfile = 'Cropped_' + str(y) + '_' + str(x)+'.jpg'
     im = Image.open('fedex.jpg')
     point_x = x * hori_shift
     point_y = y * vert_shift
     print 'point_x = {} point_y = {}'.format(point_x,point_y) 
     print 'point_x shi = {} point_y shi = {}'.format(point_x + hori_shift,point_y + vert_shift) 
     region = im.crop((point_x,point_y,point_x + hori_shift,point_y + vert_shift))
     region.save(outfile,"JPEG")

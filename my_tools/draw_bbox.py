src_image_dir="/Users/jt/Desktop/UISEE/image/Raw_Images/"
src_annotation_dir="/Users/jt/Desktop/UISEE/image/detection_labels/"
dst_images_dir="/Users/jt/Desktop/UISEE/image/draw_with_bbox/"


import os
if os.path.exists(dst_images_dir)==False:
	os.mkdir(dst_images_dir)
import matplotlib.pyplot as plt
import cv2
for name in os.listdir(src_annotation_dir):
	if name[-3:]=="txt" and os.path.isfile(os.path.join(src_image_dir,name[:-3]+"png")):
		im=cv2.imread(os.path.join(src_image_dir,name[:-3]+"png"))
		with open(os.path.join(src_annotation_dir,name),'r') as f:
			lines=f.read().splitlines()
			for line in lines:
				s = line.split(' ')
				if s[0] == 'sign':
					cv2.rectangle(im,(int(s[1]),int(s[2])),(int(s[3]),int(s[4])),(0,255,0),1)
		cv2.imwrite(os.path.join(dst_images_dir, name[:-3]+"png"), im)
		



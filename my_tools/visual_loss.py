import os
import re

every=100
log_file = '/Users/jt/Desktop/UISEE/code/py-faster-rcnn/nohup.out'
iters=[]
loss=[]
loss_bbox=[]
loss_cls=[]
loss_rpn_cls=[]
loss_rpn_bbox=[]

with open(log_file,'r') as f:
	while True:
		text=f.readline()
		if text=='':
			break
		match = re.findall("Iteration.+", text)
		if match:
			match = re.findall("\d+\.\d+|\d+", match[0])
			if len(match) == 5 and int(match[0])%every==0:
				iters.append(int(match[0])/every)
				loss.append(float(match[4]))
	
				match = re.findall("loss_bbox.+", f.readline())
				match = re.findall("\d+\.\d+e-\d+|\d+\.\d+", match[0])
				loss_bbox.append(float(match[1]))
				
				match = re.findall("loss_cls.+", f.readline())
				match = re.findall("\d+\.\d+", match[0])
				loss_cls.append(float(match[1]))
				
				match = re.findall("rpn_cls_loss.+", f.readline())
				match = re.findall("\d+\.\d+", match[0])
				loss_rpn_cls.append(float(match[1]))
				
				match = re.findall("rpn_loss_bbox.+", f.readline())
				match = re.findall("\d+\.\d+", match[0])
				loss_rpn_bbox.append(float(match[1]))

import numpy as np
import matplotlib.pyplot as plt


plt.subplot(5, 1, 1)
plt.plot(iters, loss, 'o-')
plt.xlabel('iter')
plt.ylabel('loss')

plt.subplot(5, 1, 2)
plt.plot(iters, loss_bbox, '.-')
plt.xlabel('iter')
plt.ylabel('loss_bbox')

plt.subplot(5, 1, 3)
plt.plot(iters, loss_cls, '.-')
plt.xlabel('iter')
plt.ylabel('loss_cls')

plt.subplot(5, 1, 4)
plt.plot(iters, loss_rpn_bbox, '.-')
plt.xlabel('iter')
plt.ylabel('loss_rpn_bbox')

plt.subplot(5, 1, 5)
plt.plot(iters, loss_rpn_cls, '.-')
plt.xlabel('iter')
plt.ylabel('loss_rpn_cls')

plt.show()




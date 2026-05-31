import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import time
import json

coorfile = open("coorfile.txt" , "rt")
coor_list = json.load(coorfile)
coorfile.close()

massfile = open("massfile.txt" , "rt")
mass = np.array(json.load(massfile),dtype=np.float64)
massfile.close()

coor = np.array(coor_list[0][:][:],dtype=np.float64)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set(xlim=(-1.5e13, 1.5e13), ylim=(-1.5e13, 1.5e13))
sc = ax.scatter(coor[:,0] ,coor[:,1] ,cmap='GnBu' ,c = mass ,s = 25+coor[:,2]/5e15 ,alpha=0.5 )
#sc = ax.scatter(coor[:,0] ,coor[:,1],  c = ['#618685','#ff8c66'] ,s = [30,70] )
plt.draw()


for x in coor_list:
	coor = np.array(x,dtype=np.float64)
	sc.set_sizes(25+coor[:,2]/5e15, dpi=72.0)
	sc.set_offsets(np.c_[coor[:,0] ,coor[:,1]])
	fig.canvas.draw_idle()
	plt.pause(0.03)

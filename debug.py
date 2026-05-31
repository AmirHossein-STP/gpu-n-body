import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from numba import cuda
import numba.types
import time

n = 1000
_G = 6.6743e-11
dt = 1e4
alpha = 6.6743e-1

mass =  ((np.random.random_sample(n)+1)*1e31).astype(np.float64)
coor =  ((np.random.random_sample((n,3))-0.5)*2e13).astype(np.float64)
dcoor = ((np.random.random_sample((n,3))-0.5)*2e6).astype(np.float64)
#mass = np.array([5.972e24,1.989e30],dtype=np.float64)
#coor = np.array([[8.637e10,8.637e10,8.637e10],[0,0,0]],dtype=np.float64)
#dcoor = np.array([[-1.119e4-4,1.119e4,0],[0,0,0]],dtype=np.float64)

@cuda.jit
def pull(mass ,coor ,dcoor ):
	global n ,_G ,dt ,alpha
	ds_coor = cuda.local.array((3),numba.types.float64)
	tid = cuda.grid(1)

	itt=n-1
	while tid>=itt :
		tid-=itt
		itt-=1
	i = itt
	j = tid

	for x in range (3):
		ds_coor[x] = coor[j,x] - coor[i,x]

	r2=0
	for x in range (3):
		r2 += ds_coor[x]*ds_coor[x]

	tmp =  (_G*(r2**0.5) - alpha)*(r2**-2)*dt


	'''
	if(tmp>1e20):
		tmp=1e20
	'''

	
	
	for x in range (3):
		cuda.atomic.add(dcoor ,(j,x) , -tmp*ds_coor[x]*mass[i] )
		cuda.atomic.add(dcoor ,(i,x) , +tmp*ds_coor[x]*mass[j] )

@cuda.jit
def move(coor ,dcoor):
	global dt
	tid = cuda.grid(1)
	for x in range (3):
		cuda.atomic.add(coor ,(tid,x) ,dcoor[tid,x]*dt)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.set(xlim=(-1.5e15, 1.5e15), ylim=(-1.5e15, 1.5e15))
sc = ax.scatter(coor[:,0] ,coor[:,1] ,cmap='GnBu' ,c = mass ,s = 25+coor[:,2]/1e16 ,alpha=0.5 )
#sc = ax.scatter(coor[:,0] ,coor[:,1],  c = ['#618685','#ff8c66'] ,s = [30,70] )
plt.draw()

mass_gpu = cuda.to_device(mass)
coor_gpu = cuda.to_device(coor)
dcoor_gpu = cuda.to_device(dcoor)

stream2 = cuda.stream()
stream3 = cuda.stream()

#time_ruler

tpb = 100
for j in range(1000):

	#start = time.time()
	for i in range(100):
		pull[int(n*(n-1)/2/tpb),tpb](mass_gpu,coor_gpu,dcoor_gpu)
		move[int(n/tpb),tpb, stream3](coor_gpu,dcoor_gpu)
		cuda.synchronize()

	coor = coor_gpu.copy_to_host(stream=stream2)
	
	cuda.synchronize()
	
	sc.set_sizes(25+coor[:,2]/1e16, dpi=72.0)
	sc.set_offsets(np.c_[coor[:,0] ,coor[:,1]])
	fig.canvas.draw_idle()
	plt.pause(0.01)
	#end = time.time()
	#print(debug)


#splt.waitforbuttonpress()

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from numba import cuda
import time


_G = 6.6743e-11
#_G = 6.6743e11
#_G = 6.6743e1

dt = 1e4
n = 100


mass =  ((np.random.random_sample(n)+1)*1e30).astype(np.float64)
coor =  ((np.random.random_sample((n,3))-0.5)*20e10).astype(np.float64)
dcoor = ((np.random.random_sample((n,3))-0.5)*20e4).astype(np.float64)
#mass = np.array([5.972e24,1.989e30],dtype=np.float64)
#coor = np.array([[8.637e10,8.637e10,8.637e10],[0,0,0]],dtype=np.float64)
#dcoor = np.array([[-2.119e4,2.119e4,0],[0,0,0]],dtype=np.float64)

@cuda.jit    
def gpu(mass ,coor ,dcoor ,n ,debug):
	global _G,dt
    
	tid = cuda.grid(1)
	move_treads_ter = int(n*(n-1)/2)
	
	if tid>=move_treads_ter :	
		i = tid - move_treads_ter
		for x in range (3):
			#print(dcoor[i,x],"  ",x)
			cuda.atomic.add(coor ,(i,x) ,dcoor[i,x]*dt)
		return
    
	itt=n-1
	while tid>=itt :
		tid-=itt
		itt-=1
	i = itt	
	j = tid
    
	
    
	r2=0
	for x in range (3):
		tmp = (coor[j,x]-coor[i,x])
		r2 += tmp*tmp
        
	tmp = _G*(r2**-1.5)*dt #1000/((1000000)*f(3))
	if(tmp>1e30):
                tmp=1e30
                
	for x in range (3):   

		if i==1 :
			debug[j,x] = tmp

		cuda.atomic.add(dcoor ,(j,x) , -tmp*(coor[j,x]-coor[i,x])*mass[i] )
		cuda.atomic.add(dcoor ,(i,x) , +tmp*(coor[j,x]-coor[i,x])*mass[j] )
		

	
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set(xlim=(-1.5e12, 1.5e12), ylim=(-1.5e12, 1.5e12))
sc = ax.scatter(coor[:,0] ,coor[:,1] ,cmap='GnBu' ,c = mass ,s = 25+coor[:,2]/5e9 ,alpha=0.5 )
#sc = ax.scatter(coor[:,0] ,coor[:,1],  c = ['#618685','#ff8c66'] ,s = [30,70] )
plt.draw()

mass_gpu = cuda.to_device(mass)
coor_gpu = cuda.to_device(coor)
#coor_gpu_output = cuda.device_array_like(coor_gpu)
dcoor_gpu = cuda.to_device(dcoor)
#dcoor_gpu_output = cuda.device_array_like(dcoor_gpu)

debug= np.zeros((1,3))

#time_ruler

tpb = 10
while True:
        
        #start = time.time()
        for i in range(10):
                gpu[int(n*(n+1)/2/tpb),tpb](mass_gpu,coor_gpu,dcoor_gpu,n,debug)
                cuda.synchronize()
                
        coor = coor_gpu.copy_to_host()
        sc.set_sizes(25+coor[:,2]/5e9, dpi=72.0)
        sc.set_offsets(np.c_[coor[:,0] ,coor[:,1]])
        fig.canvas.draw_idle()
        plt.pause(0.01)
        

        #end = time.time()
        #print(debug)


plt.waitforbuttonpress()

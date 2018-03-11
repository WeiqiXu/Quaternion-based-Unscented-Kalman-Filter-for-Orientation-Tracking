import pickle
import sys
import time 
import math
import numpy as np
import transforms3d
#import random

'''
def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

dataset="1"
cfile = "/Users/xuweiqi/Desktop/trainset/cam/cam" + dataset + ".p"
ifile = "/Users/xuweiqi/Desktop/trainset/imu/imuRaw" + dataset + ".p"
vfile = "/Users/xuweiqi/Desktop/trainset/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")


#--------------read matrix from Vicon 
M=vicd["rots"][:,:,0]
print(M)
'''

Ax=imud['vals'][0]
Ay=imud['vals'][1]
Az=imud['vals'][2]
Wz=imud['vals'][3]
Wx=imud['vals'][4]
Wy=imud['vals'][5]
Wz_total=0
Wx_total=0
Wy_total=0
biastotal_Wz=0
biastotal_Wx=0
biastotal_Wy=0
for i in range(100):
    Wz_total=Wz_total+Wz[i]
    Wx_total=Wx_total+Wx[i]
    Wy_total=Wy_total+Wy[i]
    
bias_Wz=Wz_total/100
bias_Wx=Wx_total/100
bias_Wy=Wy_total/100

N=len(imud['vals'][0])
pi=3.14
scale_factor_gyro = 3300/(1023*pi*3.3/180)
unit_Wz=[] # with physical units
unit_Wx=[]
unit_Wy=[]
rotation=np.zeros([N,3], dtype=np.float64) # wx,wy,wz
rotation_ts=np.zeros([N,3], dtype=np.float64) # w*delta_t
rotation_quaternion=np.zeros([N,4], dtype=np.float64 
norm=np.zeros([N,1], dtype=np.float64) #euclidian norm of w*delta_t
q=np.zeros([N,4], dtype=np.float64)
angles=np.zeros([N,3], dtype=np.float64)

for j in range(N):
    unit_Wz[j]=(Wz[j]-bias_Wz)*scale_factor_gyro
    unit_Wx[j]=(Wx[j]-bias_Wx)*scale_factor_gyro
    unit_Wy[j]=(Wy[j]-bias_Wy)*scale_factor_gyro
    rotation[j]=[unit_Wx[j],unit_Wy[j],unit_Wz[j]]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
 
q[0]=[1,0,0,0] #initial value
angles[0]=transforms3d.euler.quat2euler(q[0]) #angles transformed from rotation quaternions
for t in range(N-1):
    rotation_ts[t]=rotation[t]*0.5*(imu["ts"][t+1]-imu["ts"][t]) # 0.5*w*delta_t
    norm[t]=np.linalg.norm(rotation_ts[t])
    rotation_quaternion[t]=[math.cos(0.5*norm[t]),math.sin(0.5*norm[t])*rotation_ts[t]/norm[t]]
    q[t+1]=quaternion_multiply(q[t],rotation_quaternion[t])
    angles[t+1]=transforms3d.euler.quat2euler(q[t+1])







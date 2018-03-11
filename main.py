# -*- coding: utf-8 -*-
import pickle
import sys
import time 
import math
import numpy as np
import transforms3d
import matplotlib.pyplot as plt


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


# use quaternion to get rotation angles 
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

# turn Vicon rotation matrix to Euler angles
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
def inverse(q):
    Q=[q[0],-q[1],-q[2],-q[3]]/(np.linalg.norm(q)*np.linalg.norm(q))
    return Q

def exp(qs,q):
    qv1,qv2,qv3=q
    exp_q=np.zeros(4)
    exp_q[0]=math.exp(qs)*math.cos(np.linalg.norm(q))
    if np.linalg.norm([qv1,qv2,qv3])==0:
         exp_q[1]=0
         exp_q[2]=0
         exp_q[3]=0
    else:
        exp_q[1]=math.exp(qs)*qv1/np.linalg.norm(q)*math.sin(np.linalg.norm(q))
        exp_q[2]=math.exp(qs)*qv2/np.linalg.norm(q)*math.sin(np.linalg.norm(q))
        exp_q[3]=math.exp(qs)*qv3/np.linalg.norm(q)*math.sin(np.linalg.norm(q))
    return [exp_q[0],exp_q[1],exp_q[2],exp_q[3]]


def log(q):
    qs,qv1,qv2,qv3=q
    q_unit=np.zeros(4)
    w=np.zeros(3)
    q_norm=np.linalg.norm(q)
    q_unit=q/q_norm
    qv_norm=np.linalg.norm([q_unit[1],q_unit[2],q_unit[3]])
    if qv_norm==0:
        w=[0,0,0]
    else:
        w[0]=q_unit[1]/qv_norm*math.acos(q_unit[0])
        w[1]=q_unit[2]/qv_norm*math.acos(q_unit[0])
        w[2]=q_unit[3]/qv_norm*math.acos(q_unit[0])
    return w


####################################
Ax=(-1)*imud['vals'][0]
Ay=(-1)*imud['vals'][1]
Az=imud['vals'][2]
Wz=imud['vals'][3]
Wx=imud['vals'][4]
Wy=imud['vals'][5]
Ax_total=0
Ay_total=0
Az_total=0
Wz_total=0
Wx_total=0
Wy_total=0
for i in range(100):
    Wz_total=Wz_total+Wz[i]
    Wx_total=Wx_total+Wx[i]
    Wy_total=Wy_total+Wy[i]
    Ax_total=Ax_total+Ax[i]
    Ay_total=Ay_total+Ay[i]
    Az_total=Az_total+Az[i]

    
bias_Wz=Wz_total/100
bias_Wx=Wx_total/100
bias_Wy=Wy_total/100
bias_Ax=Ax_total/100
bias_Ay=Ay_total/100
bias_Az=Az_total/100

N=len(imud['vals'][0])
pi=3.14
scale_factor_gyro = 3300*pi/(1023*3.3*180)
scale_factor_acc=3300/1023/330
unit_Wz=np.zeros(N) # with physical units
unit_Wx=np.zeros(N)
unit_Wy=np.zeros(N)
unit_Ax=np.zeros(N)
unit_Ay=np.zeros(N)
unit_Az=np.zeros(N)
rotation=np.zeros([N,3], dtype=np.float64) # wx,wy,wz
rotation_ts=np.zeros([N,3], dtype=np.float64) # w*delta_t
rotation_quaternion=np.zeros([N,4], dtype=np.float64 )
norm=np.zeros([N,1], dtype=np.float64) #euclidian norm of w*delta_t
q=np.zeros([N,4], dtype=np.float64)
angles=np.zeros([N,3], dtype=np.float64)
a=np.zeros(3)

for j in range(N):
    unit_Wz[j]=(Wz[j]-bias_Wz)*scale_factor_gyro
    unit_Wx[j]=(Wx[j]-bias_Wx)*scale_factor_gyro
    unit_Wy[j]=(Wy[j]-bias_Wy)*scale_factor_gyro
    rotation[j]=[unit_Wx[j],unit_Wy[j],unit_Wz[j]]
    unit_Ax[j]=(Ax[j]-bias_Ax)*scale_factor_acc
    unit_Ay[j]=(Ay[j]-bias_Ay)*scale_factor_acc
    unit_Az[j]=(Az[j]-bias_Az)*scale_factor_acc

 
q[0]=[1,0,0,0] #initial value
angles[0]=transforms3d.euler.quat2euler(q[0]) #angles transformed from rotation quaternions
for t in range(N-1):
    rotation_ts[t]=0.5*rotation[t]*(imud["ts"][0,t+1]-imud["ts"][0,t]) #### w*delta_t
    norm[t]=np.linalg.norm(rotation_ts[t])
    a=math.sin(norm[t])*rotation_ts[t]/norm[t]
    rotation_quaternion[t]=[math.cos(norm[t]),a[0],a[1],a[2]]
    q[t+1]=quaternion_multiply(q[t],rotation_quaternion[t])
    angles[t+1]=transforms3d.euler.quat2euler(q[t+1])

#use angle integration to get rotation angles
Angles=np.zeros([N,3], dtype=np.float64)
Angles[0]=angles[0]
Angles_e2q=np.zeros([N,4], dtype=np.float64)
Angles_q2e=np.zeros([N,3], dtype=np.float64)
for i in range(N-1):
    Angles[i+1]=Angles[i]+rotation[i]*(imud["ts"][0,i+1]-imud["ts"][0,i])
    Angles_e2q[i+1]=transforms3d.euler.euler2quat(Angles[i+1][0],Angles[i+1][1],Angles[i+1][2])
    Angles_q2e[i+1]=transforms3d.euler.quat2euler(Angles_e2q[i+1])

Vicon_euler=np.zeros([5561,3])
for i in range(5561):
    Vicon_euler[i]=rotationMatrixToEulerAngles(vicd["rots"][:,:,i])

x=np.linspace(0,5561,5561)
plt.figure(1)
plt.plot(x,angles[0:5561,0],label='gyro x')
plt.plot(x,Vicon_euler[:,0],label='Vicon x')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')
plt.figure(2)
plt.plot(x,angles[0:5561,1],label='gyro y')
plt.plot(x,Vicon_euler[:,1],label='Vicon y')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')
plt.figure(3)
plt.plot(x,angles[0:5561,2],label='gyro z')
plt.plot(x,Vicon_euler[:,2],label='Vicon z')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')

Pk=np.zeros([N,6,6])
angles_UKF=np.zeros([N,3])
q2=q 

for b in range (N-1):
    
    #---------------UKF
    # process model Y=AX
    #q=np.zeros([N,4])
    #q[0]=[1,0,0,0]
    wq=np.zeros([13,3])
    ww=np.zeros([13,3])
    qw_cos=np.zeros([13,1])
    qw_sin=np.zeros([13,3])
    qw=np.zeros([13,4])
    X_q=np.zeros([13,4])
    X_w=np.zeros([13,3])
    X_w_total=np.zeros(3)
    Y_q=np.zeros([13,4])
    Y_w=np.zeros([13,3])
    Z_w=np.zeros([13,3])
    rw=np.zeros([13,3])
    x_mean_w=np.zeros([13,3])
    W1=np.zeros([13,6])
    
    
    Pk[0]=0.0001*np.eye(6)
    Q=0.0001*np.eye(6)
    W=np.linalg.cholesky(6*(Pk[b]+Q))
    W_=(-1)*W

    for i in range(13):
        if i<6:
            wq[i]=[W[0][i],W[1][i],W[2][i]] #turn the first 3 dim of Wi into quaternion qw
            ww[i]=[W[3][i],W[4][i],W[5][i]] #get angular velocity part of the sigma points
        elif i<12:
            wq[i]=[W_[0][i-6],W_[1][i-6],W_[2][i-6]]
            ww[i]=[W_[3][i-6],W_[4][i-6],W_[5][i-6]]
        else:
            wq[i]=[0,0,0]
            ww[i]=[0,0,0]
        qw_cos[i]=math.cos(np.linalg.norm(wq[i])/2)
        if np.linalg.norm(wq[i])==0:
            qw_sin[i]=[0,0,0]
        else:
            qw_sin[i]=wq[i]/np.linalg.norm(wq[i])*math.sin(np.linalg.norm(wq[i])/2)
        qw[i]=[qw_cos[i],qw_sin[i][0],qw_sin[i][1],qw_sin[i][2]] 
        X_q[i]=quaternion_multiply(q2[b],qw[i]) #get quaternion part of the sigma points
        X_w[i]=rotation[b]+ww[i]
        Y_q[i]=quaternion_multiply(X_q[i],rotation_quaternion[1])
        Y_w[i]=X_w[i]
        Z_w[i]=X_w[i]
        X_w_total=X_w_total+X_w[i]


    q1=np.zeros([100000,4])
    qe=np.zeros([13,4])
    ev=np.zeros(3)
    evi=np.zeros([13,3])
    evii=np.zeros([13,3])
    qi=Y_q
    q1[0]=[1,0,0,0]
    t=0
    
    
    while True:
        for i in range(13):
            qe[i]=quaternion_multiply(inverse(q1[t]),qi[i])
            evi[i]=[2*log(qe[i])[0],2*log(qe[i])[1],2*log(qe[i])[2]]
            if np.linalg.norm(evi[i])==0:
                evii[i]=[0,0,0]
            else:
                norm_evi = np.linalg.norm(evi[i])
                evii[i]=(-np.pi+math.fmod(norm_evi+np.pi,(2*np.pi)))*evi[i]/norm_evi
            ev=ev+1/13*evii[i]
        q1[t+1]=quaternion_multiply(q1[t],exp(0,[0.5*ev[0],0.5*ev[1],0.5*ev[2]]))
        if np.linalg.norm(ev)<0.0001:
        #print(ev) 
            break
        else:
            t=t+1
        






############################        
    x_mean_q=q1[t+1]    #quaternion_average(Y_q)
    x_mean_w=X_w_total/13
    Pk_=np.zeros([6,6])
    # getting Pk_
    for i in range(13):
        rw[i]=ev                #evi[i]
        W1[i]=[rw[i][0],rw[i][1],rw[i][2],Y_w[i][0]-x_mean_w[0],Y_w[i][1]-x_mean_w[1],Y_w[i][2]-x_mean_w[2]]
        if i<12:
            Pk_=Pk_+1/12*(W1[i]*np.transpose(W1[i]))
        else:
            Pk_=Pk_+2*(W1[i]*np.transpose(W1[i]))

    # measurement model Zi=HYi
    z_acc=np.zeros([13,3])
    g=np.zeros([13,4])
    g1=np.zeros([13,4])
    Z=np.zeros([13,6])
    Z_total=np.zeros(6)
    Z_mean=np.zeros(6)
    v_acc=[] #????????????
    z_acc_total=np.zeros(3)
    z_acc_mean=np.zeros(3)
    Pzz=np.zeros([3,3])
    Pzznew=np.zeros([6,6])
    Pr=0.001*np.eye(3)
    Prnew=0.001*np.eye(6)
    Pvv=np.zeros([3,3])
    Pvvnew=np.zeros([6,6])
    Pxz=np.zeros([3,3])
    Pxznew=np.zeros([6,6])
    Ktadd1=np.zeros([3,3])#kgain
    Ktadd1new=np.zeros([6,6])


    for i in range(13):
        g[0]=[0,unit_Ax[b],unit_Ay[b],unit_Az[b]]    
        g1[i]=quaternion_multiply(Y_q[i],quaternion_multiply(g[0],inverse(Y_q[i])))
        z_acc[i]=[g1[i][1],g1[i][2],g1[i][3]]
        z_acc_total=z_acc_total+z_acc[i]

    z_acc_mean=(z_acc_total-z_acc[12])/12
    Z=np.hstack((z_acc,Z_w))

    for i in range(13):
        Z_total=Z_total+Z[i]
    Z_mean=Z_total/13

    for i in range(13):
        if i<12:
            Pzznew=Pzznew+1/24*((Z[i]-Z_mean)*np.transpose(Z[i]-Z_mean))
        else:
            Pzznew=Pzznew+1/2*((Z[i]-Z_mean)*np.transpose(Z[i]-Z_mean))#(68)
    Pvvnew= Pzznew+Prnew
    for i in range(13):
        if i<12:
            Pxznew=Pxznew+1/24*(W1[i]*np.transpose(Z[i]-Z_mean))
        else:
            Pxznew=Pxznew+1/2*(W1[i]*np.transpose(Z[i]-Z_mean))#（70）
    Ktadd1new=Pxznew*np.linalg.inv(Pvvnew)
    for i in range(13):
        if i<12:
            Pzz=Pzz+1/12*((z_acc[i]-z_acc_mean)*np.transpose(z_acc[i]-z_acc_mean))
        else:
            Pzz=Pzz+2*((z_acc[i]-z_acc_mean)*np.transpose(z_acc[i]-z_acc_mean))
    Pvv=Pzz+Pr
    for i in range(13):
        if i<12:
            Pxz=Pxz+1/12*(wq[i]*(z_acc_mean-z_acc[i]))
        else:
           Pxz=Pxz+2*(wq[12]*(z_acc_mean-z_acc[i]))
    Ktadd1=Pxz*np.linalg.inv(Pvv)

    
    q2[b+1]=quaternion_multiply(q2[b],exp(0,1/2*Ktadd1*np.transpose(np.mat([unit_Ax[b+1],unit_Ay[b+1],unit_Az[b+1]]-z_acc_mean))))
    angles_UKF[b+1]=transforms3d.euler.quat2euler(q2[b+1])
    Pk[b+1,:,:]=Pk_-Ktadd1new*Pvvnew*np.transpose(Ktadd1new)
    
x=np.linspace(0,5561,5561)
plt.figure(4)
plt.plot(x,angles_UKF[0:5561,0],label='ukf x')
plt.plot(x,Vicon_euler[:,0],label='Vicon x')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')
plt.figure(5)
plt.plot(x,angles_UKF[0:5561,1],label='ukf y')
plt.plot(x,Vicon_euler[:,1],label='Vicon y')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')
plt.figure(6)
plt.plot(x,angles_UKF[0:5561,2],label='ukf z')
plt.plot(x,Vicon_euler[:,2],label='Vicon z')
plt.legend(loc="lower right")
plt.xlabel('time slot')
plt.ylabel('angles')



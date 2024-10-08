import numpy as np
import torch
import torch.nn as nn

def cal_acc(self,x_in,y_in):
    # turn one hot code into classes: class_change
    x = (x_in.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()

    x = np.concatenate([np.clip(np.argmax(x[..., :36], axis=-1, keepdims=True) / 36.,0.0,1.0) * 2.0 * np.pi,
                        np.clip(np.argmax(x[..., 36:54], axis=-1, keepdims=True) / 18.,0.0,1.0) * np.pi,
                        np.clip(np.argmax(x[..., 54:90], axis=-1, keepdims=True) / 36.,0.0,1.0)* 2.0 * np.pi,
                        x[..., 90:]],axis=-1)

    acc = []

    nx = x.shape[1]
    ny = x.shape[2]
    nz = x.shape[3]
    x = x * (x[..., 4:5] > 1e-10)

    x_true = (y_in.permute(0, 3, 4, 2, 1)).detach().cpu().numpy()
    x_true = np.concatenate([x_true[..., 0:1]/ 36.* 2.0 * np.pi,
                            x_true[..., 1:2]/ 18.* np.pi,
                            x_true[..., 2:3]/ 36.* 2.0 * np.pi,
                            x_true[..., 3:]], axis=-1)

    # calculate the difference between NCA and CA
    for i in range(len(x)):
        for j in range(nz//4, nz, nz//4):
            mis_ori = self.cal_misori(x[i, ..., j, :3],
                                x_true[i, ..., j, :3])  # misorientation angle
            filter = (mis_ori > 15.0) & (mis_ori < 75.0)  # diff>10.0
            # store the rsme and accuracy in the middle part of the domain
            acc.append(1.0 - np.sum(filter)/(nx * ny))
    acc = np.array(acc)*100
    acc_ave = np.average(acc)
    return acc_ave



def cal_misori(pred,true):
    p1 = pred[:,:,0]
    p = pred[:,:,1]
    p2 = pred[:,:,2]
    q1 = true[:,:,0]
    q = true[:,:,1]
    q2 = true[:,:,2]

    nx=p.shape[0]
    ny=p.shape[1]

    t1=np.zeros((nx,ny,24))
    t2=np.zeros((nx,ny,24))
    t3=np.zeros((nx,ny,24))
    theta=np.zeros((nx,ny,24))
    g1=np.zeros((nx,ny,3,3))
    g2=np.zeros((nx,ny,3,3))
    gp=np.zeros((nx,ny,3,3))
    gp1=np.zeros((nx,ny,3,3))
    gp2=np.zeros((nx,ny,3,3))
    gq=np.zeros((nx,ny,3,3))
    gq1=np.zeros((nx,ny,3,3))
    gq2=np.zeros((nx,ny,3,3))
    m=np.zeros((nx,ny,24,3,3))

    #converting in the form of matrices for both grains
    gp1[:,:,0,0]=np.cos(p1)
    gp1[:,:,1,0]=-np.sin(p1)
    gp1[:,:,0,1]=np.sin(p1)
    gp1[:,:,1,1]=np.cos(p1)
    gp1[:,:,2,2]=1
    gp2[:,:,0,0]=np.cos(p2)
    gp2[:,:,1,0]=-np.sin(p2)
    gp2[:,:,0,1]=np.sin(p2)
    gp2[:,:,1,1]=np.cos(p2)
    gp2[:,:,2,2]=1
    gp[:,:,0,0]=1
    gp[:,:,1,1]=np.cos(p)
    gp[:,:,1,2]=np.sin(p)
    gp[:,:,2,1]=-np.sin(p)
    gp[:,:,2,2]=np.cos(p)
    gq1[:,:,0,0]=np.cos(q1)
    gq1[:,:,1,0]=-np.sin(q1)
    gq1[:,:,0,1]=np.sin(q1)
    gq1[:,:,1,1]=np.cos(q1)
    gq1[:,:,2,2]=1
    gq2[:,:,0,0]=np.cos(q2)
    gq2[:,:,1,0]=-np.sin(q2)
    gq2[:,:,0,1]=np.sin(q2)
    gq2[:,:,1,1]=np.cos(q2)
    gq2[:,:,2,2]=1
    gq[:,:,0,0]=1
    gq[:,:,1,1]=np.cos(q)
    gq[:,:,1,2]=np.sin(q)
    gq[:,:,2,1]=-np.sin(q)
    gq[:,:,2,2]=np.cos(q)
    g1=np.matmul(np.matmul(gp2,gp),gp1)
    g2=np.matmul(np.matmul(gq2,gq),gq1)

    #symmetry matrices considering the 24 symmteries for cubic system
    T=np.zeros((24,3,3));
    T[0,:,:]=[[1,0,0],[0,1,0],[0, 0 ,1]]
    T[1,:,:]=[[0,0,-1],  [0 ,-1 ,0], [-1, 0 ,0]]
    T[2,:,:]=[[0, 0 ,-1],  [ 0 ,1, 0],  [ 1 ,0 ,0]]
    T[3,:,:]=[[-1 ,0 ,0],  [ 0 ,1, 0],  [ 0 ,0 ,-1]]
    T[4,:,:]=[[0, 0 ,1],  [ 0 ,1 ,0],  [ -1, 0 ,0]]
    T[5,:,:]=[[1, 0 ,0],  [ 0 ,0 ,-1],  [ 0 ,1 ,0]]
    T[6,:,:]=[[1 ,0 ,0],  [ 0 ,-1 ,0],  [ 0 ,0 ,-1]]
    T[7,:,:]=[[1, 0 ,0],  [ 0 ,0, 1],  [ 0 ,-1 ,0]]
    T[8,:,:]=[[0 ,-1, 0],  [ 1 ,0 ,0],  [ 0 ,0 ,1]]
    T[9,:,:]=[[-1, 0 ,0],  [ 0 ,-1, 0],  [ 0 ,0 ,1]]
    T[10,:,:]=[[0, 1 ,0],  [ -1 ,0, 0],  [ 0 ,0 ,1]]
    T[11,:,:]=[[0, 0 ,1],  [ 1 ,0 ,0],  [ 0 ,1 ,0]]
    T[12,:,:]=[[0, 1 ,0],  [ 0, 0 ,1],  [ 1 ,0 ,0]]
    T[13,:,:]=[[0 ,0 ,-1],  [ -1 ,0 ,0],  [ 0, 1 ,0]]
    T[14,:,:]=[[0 ,-1 ,0],  [ 0 ,0 ,1],  [ -1 ,0 ,0]]
    T[15,:,:]=[[0, 1 ,0],  [ 0, 0 ,-1],  [ -1,0 ,0]]
    T[16,:,:]=[[0 ,0 ,-1],  [ 1 ,0 ,0],  [ 0 ,-1 ,0]]
    T[17,:,:]=[[0 ,0 ,1],  [ -1, 0 ,0],  [ 0, -1, 0]]
    T[18,:,:]=[[0 ,-1 ,0],  [ 0 ,0 ,-1],  [ 1 ,0 ,0]]
    T[19,:,:]=[[0 ,1 ,0],  [ 1 ,0 ,0],  [ 0 ,0 ,-1]]
    T[20,:,:]=[[-1 ,0 ,0],  [ 0 ,0 ,1],  [ 0 ,1, 0]]
    T[21,:,:]=[[0, 0 ,1],  [ 0 ,-1 ,0],  [ 1 ,0 ,0]]
    T[22,:,:]=[[0 ,-1 ,0],  [ -1, 0, 0],  [ 0 ,0 ,-1]]
    T[23,:,:]=[[-1, 0 ,0],  [ 0, 0 ,-1],  [ 0 ,-1 ,0]]

    T = np.array(T[None,None,...])

    #finding the 24 misorientation matrices(also can be calculated for 576 matrices)
    for i in range(24):
        m[:,:,i,:,:]=np.matmul(np.linalg.inv(np.matmul(T[:,:,i,:,:],g1)),g2)
        t1[:,:,i]=m[:,:,i,0,0]
        t2[:,:,i]=m[:,:,i,1,1]
        t3[:,:,i]=m[:,:,i,2,2]
        theta[:,:,i]=np.arccos(0.5*(t1[:,:,i]+t2[:,:,i]+t3[:,:,i]-1))

    #minimum of 24 angles is taken as miorientation angle
    ansRad=np.nanmin(theta,axis=-1)
    ansTheta=ansRad*180.0/np.pi
    return ansTheta




#classification losses:  class_change
bc_loss = nn.BCEWithLogitsLoss(reduction='none')
cn_loss = nn.CrossEntropyLoss(reduction='none')
def cweigt_loss(self,p, t):
    l_weight = torch.sum(t[:, 3, ...] == 1.0, axis=[1, 2, 3])
    l_s = torch.mean(torch.mean(torch.square(p[:, 90:91, ...] - t[:, 3:4, ...]), axis=[1, 2, 3, 4]) / (
                l_weight + 1e-12) ) 

    l_ea1 = torch.mean(torch.mean(self.cn_loss(p[:, :36, ...], t[:, 0, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea1 = l_ea1/(l_ea1/l_s).item()
    l_ea2 = torch.mean(torch.mean(self.cn_loss(p[:, 36:54, ...], t[:, 1, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea2 = l_ea2/(l_ea2/l_s).item()
    l_ea3 = torch.mean(torch.mean(self.cn_loss(p[:, 54:90, ...], t[:, 2, ...].to(torch.long)), axis=[1, 2, 3]) / (l_weight+1e-12))
    l_ea3 = l_ea3/(l_ea3/l_s).item()

    l = l_s + l_ea1 + l_ea2+ l_ea3

    return l, l_ea1.item(), l_ea2.item(), l_ea3.item(), l_s.item()
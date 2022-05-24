import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

plt.interactive(True)


dx=0.1
dz=0.05

ni=34
nj=49
nk=34


viscos=1./5200.

# loop over nfiles
nfiles=16
#initialize fields
u3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
v3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
w3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
w3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
te3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
eps3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))

for n in range(1,nfiles+1):
   print('file no: n=',n)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_1 & transform v_1 to a 3D array (file 1)
   uvw = sio.loadmat('u'+str(n)+'_IDD_PANS.mat')
   ttu=uvw['u'+str(n)+'_IDD_PANS']
   u3d_nfiles[:,:,:,n]= np.reshape(ttu,(nk,nj,ni))  #v_1 velocity
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

   uvw = sio.loadmat('v'+str(n)+'_IDD_PANS.mat')
   tt=uvw['v'+str(n)+'_IDD_PANS']
   v3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #v_2 velocity

   uvw = sio.loadmat('w'+str(n)+'_IDD_PANS.mat')
   tt=uvw['w'+str(n)+'_IDD_PANS']
   w3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #v_3 velocity

   uvw = sio.loadmat('te'+str(n)+'_IDD_PANS.mat')
   tt=uvw['te'+str(n)+'_IDD_PANS']
   te3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #modeled turbulent kinetic energy

   uvw = sio.loadmat('eps'+str(n)+'_IDD_PANS.mat')
   tt=uvw['eps'+str(n)+'_IDD_PANS']
   eps3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #modeled turbulent dissipation


# merge nfiles. This means that new ni = nfiles*ni
u3d=u3d_nfiles[:,:,:,1]
v3d=v3d_nfiles[:,:,:,1]
w3d=w3d_nfiles[:,:,:,1]
te3d=te3d_nfiles[:,:,:,1]
eps3d=eps3d_nfiles[:,:,:,1]
for n in range(1,nfiles+1):
   u3d=np.concatenate((u3d, u3d_nfiles[:,:,:,n]), axis=0)
   v3d=np.concatenate((v3d, v3d_nfiles[:,:,:,n]), axis=0)
   w3d=np.concatenate((w3d, w3d_nfiles[:,:,:,n]), axis=0)
   te3d=np.concatenate((te3d, te3d_nfiles[:,:,:,n]), axis=0)
   eps3d=np.concatenate((eps3d, eps3d_nfiles[:,:,:,n]), axis=0)



# x coordinate direction = index 0, first index
# y coordinate direction = index 1, second index
# z coordinate direction = index 2, third index



ni=len(u3d)

x=dx*ni
z=dz*nk


umean=np.mean(u3d, axis=(0,2))
vmean=np.mean(v3d, axis=(0,2))
wmean=np.mean(w3d, axis=(0,2))
temean=np.mean(te3d, axis=(0,2))
epsmean=np.mean(eps3d, axis=(0,2))

# face coordinates
yc = np.loadtxt("yc.dat")

# cell cener coordinates
y= np.zeros(nj)
dy=np.zeros(nj)
for j in range (1,nj-1):
# dy = cell width
   dy[j]=yc[j]-yc[j-1]
   y[j]=0.5*(yc[j]+yc[j-1])

y[nj-1]=yc[nj-1]
tauw=viscos*umean[1]/y[1]
ustar=tauw**0.5
yplus=y*ustar/viscos

DNS=np.genfromtxt("LM_Channel_5200_mean_prof.dat", dtype=None,comments="%")
y_DNS=DNS[:,0]
yplus_DNS=DNS[:,1]
u_DNS=DNS[:,2]

DNS=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat", dtype=None,comments="%")

u2_DNS=DNS[:,2]
v2_DNS=DNS[:,3]
w2_DNS=DNS[:,4]
uv_DNS=DNS[:,5]

k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2

############################### U
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.semilogx(yplus,umean/ustar,'b-')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.axis([1, 8000, 0, 31])
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")

plt.savefig('u_log_zonal_python.png')

# T.3

uvmean1=np.mean((u3d-umean[None,:,None])*(v3d-vmean[None,:,None]), axis=(0,2))

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(y,uvmean1,'b')
plt.ylabel("$uv_{mean}$")
plt.xlabel("$y$")

plt.savefig('uvmean.png')

# T.4

uumean = np.mean((u3d-umean[None,:,None])*(u3d-umean[None,:,None]), axis=(0,2))
vvmean = np.mean((v3d-vmean[None,:,None])*(v3d-vmean[None,:,None]), axis=(0,2))
wwmean = np.mean((w3d-wmean[None,:,None])*(w3d-wmean[None,:,None]), axis=(0,2))

k_res = (uumean + vvmean + wwmean)/2

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(y[1:],k_res[1:],'b-')
plt.plot(y[1:],temean[1:],'g-')
plt.plot(y[1:],temean[1:]+k_res[1:],'r')
plt.legend(["modeled", "resolved", "sum"])
plt.grid()
plt.ylabel("$k}$")
plt.xlabel("$y$")
plt.savefig('kres.eps')

plt.show()

# T.5


dudx, dudy, dudz = np.gradient(u3d, dx, y, dz)
dvdx, dvdy, dvdz = np.gradient(v3d, dx, y, dz)

C_mu = 0.09
f_mu = 0.4
nu_t = C_mu*f_mu*(te3d[:,1:,:]**2)/eps3d[:,1:,:]

nu_t_mean = np.mean(nu_t, axis=(0,2))

nu_t_nu = nu_t_mean/viscos

tau_12 = -2*nu_t*(dudy[:,1:,:] + dvdx[:,1:,:])

tau_12_mean=np.mean(tau_12, axis=(0,2))

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(y[1:],tau_12_mean,'b-')
plt.plot(y[1:],uvmean1[1:],'g-')
plt.ylabel("stress")
plt.xlabel("y")
plt.grid()
plt.legend(["modeled", "resolved"])
plt.savefig('shear_stress.eps')

plt.show()

# T.6
C_des = 0.65
d_x = np.ones((1, nj))*dx
d_z = np.ones((1, nj))*dz
delta = np.maximum(d_x, dy)
delta = np.maximum(delta, d_z)
d = np.zeros((1, nj))
for j in range(nj):
    d_bot = abs(y[j]-y[0])
    d[0,j] = d_bot

d_bar = np.minimum(d, delta*C_des)

Lt = C_mu*(temean+k_res)**(3/2)/np.maximum(epsmean, np.ones((1, nj))**-10)
F_DES = np.maximum(Lt/(C_des*delta), np.ones((1,nj)))

beta_star = C_mu

omega = epsmean/(beta_star*temean)
xi = np.maximum(2*temean[1:]**(3/2)/(epsmean[1:]*y[1:]), 500*viscos/(omega[1:]*y[1:]**2))

F_2 = np.tanh(xi**2)

F_ddes = np.maximum(Lt[0,1:]/(C_des*delta[0, 1:])*(1-F_2), np.ones((1,nj-1)))

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(y[1:],d_bar[0,1:])
plt.plot(y[1:],d[0,1:])
plt.plot(y,np.transpose(F_DES))
plt.plot(y[1:],np.transpose(F_ddes))
plt.grid()
plt.ylabel("blending functions")
plt.xlabel("y")
plt.legend([r"\bar{d}","d","F_des","F_ddes"], prop={'size': 10})
plt.savefig('interface_loc.eps')

plt.show()
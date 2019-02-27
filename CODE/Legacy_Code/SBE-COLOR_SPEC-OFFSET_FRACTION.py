#Edited for VEGA Magnitudes, Absolute Magnitude uses kcorrectiong and apparent magnitudes with proper zero-point offset. LRH 5/6/17

import numpy as np
import kcorrect
import astropy as ap
import matplotlib.pyplot as plt
import kcorrect
import kcorrect.utils as ut
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import math as m
from scipy import stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model

filein='/home/lrhunt/CATALOGS/PHOT/LAMBDAR_MAG_R.txt'

plt.style.use('lcbg_paper')

IDtot,RAtot,DECtot,utot,uerrtot,btot,berrtot,vtot,verrtot,rtot,rerrtot,itot,ierrtot,ztot,zerrtot,ktot,kerrtot,NUVtot,NUVerrtot,rh,zbesttot,zusetot,zphottot,SGCLASStot=np.loadtxt(filein,unpack=True)

magrange=np.linspace(15,22.5,16)
zrange=np.linspace(0,1.2,7)
tf=open('weights.txt','w')
Weighttot=np.zeros_like(btot)
print('Calculating spectroscopic weights')
for k in range(0,len(magrange)-1):
	x=len(np.where((itot>=magrange[k]) & (itot<magrange[k+1]) & ((zusetot==1)|(zusetot==2)) & (SGCLASStot==0))[0])
	y=len(np.where((itot>=magrange[k]) & (itot<magrange[k+1]) & ((zusetot==3)|(zusetot==4)) & (SGCLASStot==0))[0])
        if (x>0):
            Weighttot[np.where((itot>=magrange[k]) & (itot<magrange[k+1]) & ((zusetot==1)|(zusetot==2)) & (SGCLASStot==0))[0]]=float(x+y)/float(x)   
            print('Good Spec={}, Bad Spec={}, Weight={}, magrange={}-{}'.format(x,y,float(x+y)/float(x),magrange[k],magrange[k+1]))
            tf.write('Good Spec={}, Bad Spec={}, Weight={}, magrange={}-{}\n'.format(x,y,float(x+y)/float(x),magrange[k],magrange[k+1]))

tf.close()


ID=IDtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
RA=RAtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
DECL=DECtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
SGCLASS=SGCLASStot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
umag=utot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
uerr=uerrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
bmag=btot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
berr=berrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
vmag=vtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
verr=verrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
imag=itot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
ierr=ierrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
zmag=ztot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
zerr=zerrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
kmag=ktot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
kerr=kerrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
zbest=zbesttot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
zuse=zusetot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
zphot=zphottot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
rmag=rtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
rerr=rerrtot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
rh=rh[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]
WeightArray=Weighttot[np.where((zbesttot>0)&(zbesttot<1.2)&(zusetot<=2)&(SGCLASStot==0))[0]]

umaggies=ut.mag2maggies(umag)
kmaggies=ut.mag2maggies(kmag)
bmaggies=ut.mag2maggies(bmag)
vmaggies=ut.mag2maggies(vmag)
rmaggies=ut.mag2maggies(rmag)
imaggies=ut.mag2maggies(imag)
zmaggies=ut.mag2maggies(zmag)

uinvervar=ut.invariance(umaggies,uerr)
kinvervar=ut.invariance(kmaggies,kerr)
binvervar=ut.invariance(bmaggies,berr)
vinvervar=ut.invariance(vmaggies,verr)
rinvervar=ut.invariance(rmaggies,rerr)
iinvervar=ut.invariance(imaggies,ierr)
zinvervar=ut.invariance(zmaggies,zerr)

allmaggies=np.stack((umaggies,bmaggies,vmaggies,rmaggies,imaggies,zmaggies),axis=-1)
allinvervar=np.stack((uinvervar,binvervar,vinvervar,rinvervar,iinvervar,zinvervar),axis=-1)

carr=np.ndarray((len(bmaggies),6))
rmarr=np.ndarray((len(bmaggies),7))
rmarr0=np.ndarray((len(bmaggies),7))
rmarr0B=np.ndarray((len(bmaggies),7))
rmarr0V=np.ndarray((len(bmaggies),7))
rmarr0U=np.ndarray((len(bmaggies),7))
rmarrB=np.ndarray((len(bmaggies),7))

print('Computing k-corrections')
kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/Lum_Func_Filters_US.dat')

for i in range(0,len(carr)):
	carr[i]=kcorrect.fit_nonneg(zbest[i],allmaggies[i],allinvervar[i])
for i in range(0,len(carr)):
	rmarr[i]=kcorrect.reconstruct_maggies(carr[i])
	rmarr0[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_B2.dat')

for i in range(0,len(carr)):
	rmarr0B[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)
	rmarrB[i]=kcorrect.reconstruct_maggies(carr[i])

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_V2.dat')

for i in range(0,len(carr)):
	rmarr0V[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_U2.dat')

for i in range(0,len(carr)):
	rmarr0U[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#print('Computing k-corrections')
#kcorrect.load_templates(v='vmatrix.defaultearly.dat',l='lambda.defaultearly.dat')
#kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/Lum_Func_Filters_US.dat')

#for i in range(0,len(carr)):
#	carrdefaultearly[i]=kcorrect.fit_nonneg(zbest[i],allmaggies[i],allinvervar[i])
#for i in range(0,len(carr)):
#	rmarrdefaultearly[i]=kcorrect.reconstruct_maggies(carr[i])
#	rmarr0defaultearly[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#kcorrect.load_templates(v='vmatrix.defaultearly.dat',l='lambda.defaultearly.dat')
#kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_B2.dat')

#for i in range(0,len(carr)):
#	rmarr0Bdefaultearly[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)
#	rmarrBdefaultearly[i]=kcorrect.reconstruct_maggies(carr[i])

#kcorrect.load_templates(v='vmatrix.defaultearly.dat',l='lambda.defaultearly.dat')
#kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_V2.dat')

#for i in range(0,len(carr)):
#	rmarr0Vdefaultearly[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#kcorrect.load_templates(v='vmatrix.defaultearly.dat',l='lambda.defaultearly.dat')
#kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_U2.dat')
#for i in range(0,len(carr)):
#	rmarr0Udefaultearly[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)



kcorr=-2.5*np.log10(rmarr/rmarr0)
kcorrM=-2.5*np.log10(rmarr/rmarr0B)
#kcorrV=-2.5*np.log10(rmarr/rmarr0V)
corrB=-2.5*np.log10(rmarr0B)
corrV=-2.5*np.log10(rmarr0V)
corrU=-2.5*np.log10(rmarr0U)
mags=-2.5*np.log10(rmarr)

vegaB=corrB[:,1]+0.09
vegaV=corrV[:,1]-0.02
vegaU=corrU[:,1]-0.79

#kcorrdefaultearly=-2.5*np.log10(rmarrdefaultearly/rmarr0defaultearly)
#kcorrMdefaultearly=-2.5*np.log10(rmarrdefaultearly/rmarr0Bdefaultearly)
#kcorrVdefaultearly=-2.5*np.log10(rmarrdefaultearly/rmarr0Vdefaultearly)
#corrBdefaultearly=-2.5*np.log10(rmarr0Bdefaultearly)
#corrVdefaultearly=-2.5*np.log10(rmarr0Vdefaultearly)
#corrUdefaultearly=-2.5*np.log10(rmarr0Udefaultearly)
#magsdefaultearly=-2.5*np.log10(rmarrdefaultearly)

print('Computing M')

M=np.zeros_like(zbest)
#M2=corrB[:,3]-cosmo.distmod(zbest).value
#MV=np.zeros_like(zbest)
#bv=corrB[:,3]-corrV[:,4]
#ub=corrU[:,1]-corrB[:,3]

Vegabv=vegaB-vegaV
Vegaub=vegaU-vegaB
VegaM=vegaB-cosmo.distmod(zbest).value

for i in range(0,len(zbest)):
	if zbest[i]<=0.1:
		M[i]=bmag[i]-0.05122-cosmo.distmod(zbest[i]).value-kcorrM[i][2]
	if zbest[i]<=0.35 and zbest[i]>0.1:
		M[i]=vmag[i]+0.069802-cosmo.distmod(zbest[i]).value-kcorrM[i][3]
	if zbest[i]<=0.55 and zbest[i]>0.35:
		M[i]=rmag[i]-0.01267-cosmo.distmod(zbest[i]).value-kcorrM[i][4]
	if zbest[i]<=0.75 and zbest[i]>0.55:
		M[i]=imag[i]-0.004512-cosmo.distmod(zbest[i]).value-kcorrM[i][5]
	if zbest[i]>0.75:
		M[i]=zmag[i]-0.00177-cosmo.distmod(zbest[i]).value-kcorrM[i][6]
diff=np.zeros_like(zbest)
for i in range(0,len(zbest)):
	if zbest[i]<=0.1:
		diff[i]=bmag[i]-0.05122-mags[i][2]
	if zbest[i]<=0.35 and zbest[i]>0.1:
		diff[i]=vmag[i]+0.069802-mags[i][3]
	if zbest[i]<=0.55 and zbest[i]>0.35:
		diff[i]=rmag[i]-0.01267-mags[i][4]
	if zbest[i]<=0.75 and zbest[i]>0.55:
		diff[i]=imag[i]-0.004512-mags[i][5]
	if zbest[i]>0.75:
		diff[i]=zmag[i]-0.00177-mags[i][6]
M=M+0.09

#SBe=np.zeros_like(zbest)
#SBe2=corrB[:,3]+2.5*np.log10(2*np.pi*np.power(rh*0.03,2))-10*np.log10(1+zbest)
#SBe3=M+cosmo.distmod(0.1).value+0.753+2.5*np.log10(np.pi*np.power(rh*0.03*cosmo.kpc_proper_per_arcmin(zbest).value/cosmo.kpc_proper_per_arcmin(0.1).value,2))-7.5*np.log10(1.1)
#SBe4=corrB[:,2]-cosmo.distmod(zbest).value+cosmo.distmod(0.1).value+0.753+2.5*np.log10(np.pi*np.power((rh*0.03*cosmo.kpc_proper_per_arcmin(zbest).value/cosmo.kpc_proper_per_arcmin(0.1).value),2))-7.5*np.log10(1.1)
#SBe5=M2+2.5*np.log10((2*np.pi*np.power(cosmo.angular_diameter_distance(zbest).value*np.tan(rh*0.03*4.84814e-6)*1e3,2)))+2.5*np.log10((360*60*60/(2*np.pi*0.01))**2)
#SBe6=SBe5-M2+M

VegaSBe=M+2.5*np.log10((2*np.pi*np.power(cosmo.angular_diameter_distance(zbest).value*np.tan(rh*0.03*4.84814e-6)*1.0659*1e3,2)))+2.5*np.log10((360*60*60/(2*np.pi*0.01))**2)

#for i in range(0,len(zbest)):
#     if zbest[i]<=0.1:
#          SBe[i]=bmag[i]-kcorrM[i][2]+0.753+2.5*np.log10(np.pi*np.power(rh[i]*0.03,2))-7.5*np.log10(1+zbest[i])
#     if zbest[i]<=0.35 and zbest[i]>0.1:
#          SBe[i]=vmag[i]-kcorrM[i][3]+0.753+2.5*np.log10(np.pi*np.power(rh[i]*0.03,2))-7.5*np.log10(1+zbest[i])
#     if zbest[i]<=0.55 and zbest[i]>0.35:
#          SBe[i]=rmag[i]-kcorrM[i][4]+0.753+2.5*np.log10(np.pi*np.power(rh[i]*0.03,2))-7.5*np.log10(1+zbest[i])
#     if zbest[i]<=0.75 and zbest[i]>0.55:
#          SBe[i]=imag[i]-kcorrM[i][5]+0.753+2.5*np.log10(np.pi*np.power(rh[i]*0.03,2))-7.5*np.log10(1+zbest[i])
#     if zbest[i]>0.75:
#          SBe[i]=zmag[i]-kcorrM[i][6]+0.753+2.5*np.log10(np.pi*np.power(rh[i]*0.03,2))-7.5*np.log10(1+zbest[i])

#print(bv)
LCBGS=np.where((M<=-18.5)&(VegaSBe<=21)&(Vegabv<0.6))[0]


#*******************************
#*******************************
#*******************************
#TO PLOT ALL GALAXIES IN REDSHIFT BINS (PAPER VERSION)
#*******************************
#*******************************
#*******************************

SBy=np.linspace(16,21,30)
SBx=np.full_like(SBy,0.6)
bvx=np.linspace(-.5,.6,30)
bvy=np.full_like(bvx,21)

f,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(16,16))
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][0].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
axes[0][0].set_xticks([-0.05,0.05,0.1,0.15,0.25,0.3,0.35,0.45,0.5,0.55,0.65,0.7,0.75,0.85,0.9,0.95,1.05],minor=True)
axes[0][0].set_xlim([-0.1,1.1])
axes[0][0].set_yticks([25,23,21,19,17])
axes[0][0].set_yticks([16.5,17.5,18,18.5,19.5,20,20.5,21.5,22,22.5,23.5,24,24.5,25.5],minor=True)
axes[0][0].set_ylim([26,16])
axes[0][0].plot(bvx,bvy,'b')
axes[0][0].plot(SBx,SBy,'b')
#axes[0][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][0].text(0,19,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][0].text(0,25,'z=0.01-0.25',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].plot(bvx,bvy,'b')
axes[0][1].plot(SBx,SBy,'b')
#axes[0][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][1].text(0,19,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][1].text(0,25,'z=0.25-0.50',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],'x',markersize=3.5,color='black')
axes[1][0].plot(bvx,bvy,'b')
axes[1][0].plot(SBx,SBy,'b')
#axes[1][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][0].text(0,19,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[1][0].text(0,25,'z=0.50-0.75',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<1')
axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<1')
axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<1')
axes[1][1].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],'x',markersize=3.5,color='black')
axes[1][1].plot(bvx,bvy,'b')
axes[1][1].plot(SBx,SBy,'b')
axes[1][1].text(0,19,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
#axes[1][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][1].text(0,25,'z=0.75-1.0',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})
f.text(0.02,0.52,'$\mu_{e}$(B)$_{0}$',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.52,0.02,'(B-V)$_{0}$',ha='center',va='center',fontsize=16)
plt.subplots_adjust(left=0.06,right=0.99,top=0.99,bottom=0.06)
for i, ax in enumerate(f.axes):
	ax.grid(False)
	ax.tick_params(axis='both',which='major',direction='in',length=6,width=1)
	ax.tick_params(axis='both',which='minor',direction='in',length=3,width=0.5)

plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/SurfaceBrightnessColor.pdf')



#*******************************
#*******************************
#*******************************
#TO PLOT ALL GALAXIES IN REDSHIFT BINS (B-V,M,PAPER VERSION)
#*******************************
#*******************************
#*******************************

SBy=np.linspace(16,21,30)
SBx=np.full_like(SBy,0.6)
bvx=np.linspace(-.5,.6,30)
bvy=np.full_like(bvx,-18.5)
My=np.linspace(-18.5,-25,30)
Mx=np.full_like(My,0.6)

f,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(16,16))
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.35')
axes[0][0].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][0].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
axes[0][0].set_xticks([-0.05,0.05,0.1,0.15,0.25,0.3,0.35,0.45,0.5,0.55,0.65,0.7,0.75,0.85,0.9,0.95,1.05],minor=True)
axes[0][0].set_xlim([-0.1,1.1])
axes[0][0].set_yticks([-24,-22,-20,-18,-16,-14])
axes[0][0].set_yticks([-13.5,-14.5,-15,-15.5,-16.5,-17,-17.5,-18.5,-19,-19.5,-20.5,-21,-21.5,-22.5,-23,-23.5,-24.5],minor=True)
axes[0][0].set_ylim([-13,-25])
axes[0][0].plot(bvx,bvy,'b')
axes[0][0].plot(Mx,My,'b')
#axes[0][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][0].text(0,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][0].text(0,-14,'z=0.01-0.25',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.55')
axes[0][1].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].plot(bvx,bvy,'b')
axes[0][1].plot(Mx,My,'b')
#axes[0][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][1].text(0,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][1].text(0,-14,'z=0.25-0.50',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})

axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.75')
axes[1][0].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],'x',markersize=3.5,color='black')
axes[1][0].plot(bvx,bvy,'b')
axes[1][0].plot(Mx,My,'b')
#axes[1][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][0].text(0,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[1][0].text(0,-14,'z=0.50-0.75',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<1')
axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<1')
axes[1][1].plot(Vegabv[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<1')
axes[1][1].plot(Vegabv[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],'x',markersize=3.5,color='black')
axes[1][1].plot(bvx,bvy,'b')
axes[1][1].plot(Mx,My,'b')
axes[1][1].text(0,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
#axes[1][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][1].text(0,-14,'z=0.75-1.0',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})
f.text(0.02,0.5,'M$_{B,0}$',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.52,0.02,'(B-V)$_{0}$',ha='center',va='center',fontsize=16)
plt.subplots_adjust(left=0.06,right=0.99,top=0.99,bottom=0.06)
for i, ax in enumerate(f.axes):
	ax.grid(False)
	ax.tick_params(axis='both',which='major',direction='in',length=6,width=1)
	ax.tick_params(axis='both',which='minor',direction='in',length=3,width=0.5)

plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/MagnitudeColor.pdf')

#*******************************
#*******************************
#*******************************
#TO PLOT ALL GALAXIES IN REDSHIFT BINS (SBe,M,PAPER VERSION)
#*******************************
#*******************************
#*******************************

SBx=np.linspace(16,21,30)
SBy=np.full_like(SBy,-18.5)
bvx=np.linspace(-.5,.6,30)
bvy=np.full_like(bvx,-18.5)
My=np.linspace(-18.5,-25,30)
Mx=np.full_like(My,21)

f,axes=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,figsize=(16,16))
axes[0][0].plot(VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.35')
axes[0][0].plot(VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.35')
axes[0][0].plot(VegaSBe[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.01)&(zbest<0.25)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.35')
axes[0][0].plot(VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.01)&(zbest[LCBGS]<0.25))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][0].set_xticks([25,23,21,19,17])
axes[0][0].set_xticks([16.5,17.5,18,18.5,19.5,20,20.5,21.5,22,22.5,23.5,24,24.5,25.5],minor=True)
axes[0][0].set_xlim([26,16])
axes[0][0].set_yticks([-24,-22,-20,-18,-16,-14])
axes[0][0].set_yticks([-13.5,-14.5,-15,-15.5,-16.5,-17,-17.5,-18.5,-19,-19.5,-20.5,-21,-21.5,-22.5,-23,-23.5,-24.5],minor=True)
axes[0][0].set_ylim([-13,-25])
axes[0][0].plot(SBx,SBy,'b')
axes[0][0].plot(Mx,My,'b')
#axes[0][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][0].text(19,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][0].text(19,-14,'z=0.01-0.25',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[0][1].plot(VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.55')
axes[0][1].plot(VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.55')
axes[0][1].plot(VegaSBe[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.25)&(zbest<0.5)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.55')
axes[0][1].plot(VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.25)&(zbest[LCBGS]<0.5))[0]]],'x',markersize=3.5,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].plot(SBx,SBy,'b')
axes[0][1].plot(Mx,My,'b')
#axes[0][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[0][1].text(19,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[0][1].text(19,-14,'z=0.25-0.50',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})

axes[1][0].plot(VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<0.75')
axes[1][0].plot(VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<0.75')
axes[1][0].plot(VegaSBe[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.5)&(zbest<0.75)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<0.75')
axes[1][0].plot(VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.5)&(zbest[LCBGS]<0.75))[0]]],'x',markersize=3.5,color='black')
axes[1][0].plot(SBx,SBy,'b')
axes[1][0].plot(Mx,My,'b')
#axes[1][0].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][0].text(19,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
axes[1][0].text(19,-14,'z=0.50-0.75',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})


axes[1][1].plot(VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='red',label='0.1<z<1')
axes[1][1].plot(VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub<-0.032*(M+21.52)+0.104))[0]],'.',markersize=3,color='blue',label='0.1<z<1')
axes[1][1].plot(VegaSBe[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],M[np.where((zbest>0.75)&(zbest<1)&(Vegaub>-0.032*(M+21.52)+0.104)&(Vegaub<-0.032*(M+21.52)+0.304))[0]],'.',markersize=3,color='green',label='0.1<z<1')
axes[1][1].plot(VegaSBe[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],M[LCBGS[np.where((zbest[LCBGS]>0.75)&(zbest[LCBGS]<1))[0]]],'x',markersize=3.5,color='black')
axes[1][1].plot(SBx,SBy,'b')
axes[1][1].plot(Mx,My,'b')
axes[1][1].text(19,-23,'LCBGs',fontsize=12,color='blue',bbox={'facecolor':'white'})
#axes[1][1].fill_between(bvx,21,16,facecolor='blue',alpha=0.1)
axes[1][1].text(19,-14,'z=0.75-1.0',verticalalignment='center',fontsize=12,bbox={'facecolor':'white'})
f.text(0.02,0.5,'M$_{B,0}$',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.52,0.02,'$\mu_{e}$(B)$_{0}$',ha='center',va='center',fontsize=16)
plt.subplots_adjust(left=0.06,right=0.99,top=0.99,bottom=0.06)
for i, ax in enumerate(f.axes):
	ax.grid(False)
	ax.tick_params(axis='both',which='major',direction='in',length=6,width=1)
	ax.tick_params(axis='both',which='minor',direction='in',length=3,width=0.5)

plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/MagnitudeSurfaceBrightness.pdf')



#*******************************
#*******************************
#*******************************
#Plotting Photometry Differences
#*******************************
#*******************************
#*******************************



@custom_model
def newline(x,B=0.5):
	return 0*x+B

fit=LevMarLSQFitter()
x=np.linspace(12,30,50)
y=np.full(50,0)

#PLOT UMAG

linefitinit=newline()
linefitu=fit(linefitinit,umag[np.where(~np.isnan(mags[:,2]))[0]],umag[np.where(~np.isnan(mags[:,2]))[0]]-mags[[np.where(~np.isnan(mags[:,2]))[0]],1])
plt.plot(umag[np.where(~np.isnan(mags[:,2]))[0]],umag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],1],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(umag[np.where(~np.isnan(mags[:,2]))[0]],umag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],1])
plt.plot(x,y)
plt.plot(x,linefitu(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{u*}$ LAMBDAR')
plt.ylabel('m$_{u* LAMBDAR}$-m$_{u* BLANTON}$')
plt.ylim(-1,1)
plt.xlim(17,30)

#PLOT BMAG

plt.figure(2)
linefitb=fit(linefitinit,bmag[np.where(~np.isnan(mags[:,2]))[0]],bmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],2])
plt.plot(bmag[np.where(~np.isnan(mags[:,2]))[0]],bmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],2],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(bmag[np.where(~np.isnan(mags[:,2]))[0]],bmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],2])
plt.plot(x,y)
plt.plot(x,linefitb(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{Bj}$ LAMBDAR')
plt.ylabel('m$_{Bj LAMBDAR}$-m$_{Bj BLANTON}$')
plt.ylim(-1,1)
plt.xlim(14,27)

#PLOT VMAG

plt.figure(3)
linefitv=fit(linefitinit,vmag[np.where(~np.isnan(mags[:,2]))[0]],vmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[[np.where(~np.isnan(mags[:,2]))[0]],3])
plt.plot(vmag[np.where(~np.isnan(mags[:,2]))[0]],vmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],3],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(vmag[np.where(~np.isnan(mags[:,2]))[0]],vmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],3])
plt.plot(x,y)
plt.plot(x,linefitv(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{V}$ LAMBDAR')
plt.ylabel('m$_{V LAMBDAR}$-m$_{V BLANTON}$')
plt.ylim(-1,1)
plt.xlim(15,26)

#PLOT RMAG

plt.figure(4)
linefitr=fit(linefitinit,rmag[np.where(~np.isnan(mags[:,2]))[0]],rmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[[np.where(~np.isnan(mags[:,2]))[0]],4])
plt.plot(rmag[np.where(~np.isnan(mags[:,2]))[0]],rmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],4],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(rmag[np.where(~np.isnan(mags[:,2]))[0]],rmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],4])
plt.plot(x,y)
plt.plot(x,linefitr(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{r}$ LAMBDAR')
plt.ylabel('m$_{r LAMBDAR}$-m$_{r BLANTON}$')
plt.ylim(-1,1)
plt.xlim(14,25)

#PLOT IMAG

plt.figure(5)
linefiti=fit(linefitinit,imag[np.where(~np.isnan(mags[:,2]))[0]],imag[np.where(~np.isnan(mags[:,2]))[0]]-mags[[np.where(~np.isnan(mags[:,2]))[0]],5])
plt.plot(imag[np.where(~np.isnan(mags[:,2]))[0]],imag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],5],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(imag[np.where(~np.isnan(mags[:,2]))[0]],imag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],5])
plt.plot(x,y)
plt.plot(x,linefiti(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{i}$ LAMBDAR')
plt.ylabel('m$_{i LAMBDAR}$-m$_{i BLANTON}$')
plt.ylim(-1,1)
plt.xlim(15,23)

#PLOT ZMAG

plt.figure(6)
linefitz=fit(linefitinit,zmag[np.where(~np.isnan(mags[:,2]))[0]],zmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[[np.where(~np.isnan(mags[:,2]))[0]],6])
plt.plot(zmag[np.where(~np.isnan(mags[:,2]))[0]],zmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],6],',')
slope,intercept,r_value,p_value,std_err=stats.linregress(zmag[np.where(~np.isnan(mags[:,2]))[0]],zmag[np.where(~np.isnan(mags[:,2]))[0]]-mags[np.where(~np.isnan(mags[:,2]))[0],6])
plt.plot(x,y)
plt.plot(x,linefitz(x),label='m={},b={},r^2={},$\sigma$={}'.format(slope,intercept,r_value**2,std_err))
plt.legend(prop={'size':9})
plt.xlabel('m$_{z}$ LAMBDAR')
plt.ylabel('m$_{z LAMBDAR}$-m$_{z BLANTON}$')
plt.ylim(-1,1)
plt.xlim(14,28)

plt.plot(M,corrU[:,2]-corrB[:,2]-0.81,',')
plt.plot(M,-0.032*(M+21.52)+0.204,label='Separation between Red Cloud and Blue Cloud (Willmer,2006)')
plt.legend(prop={'size':10})
plt.xlabel('M$_{B}')
plt.ylabel('(U-B)$_{0}$ Vega')



#*******************************
#*******************************
#*******************************
#PLOTTING FRACTION OF SOURCES
#*******************************
#*******************************
#*******************************

def autolabel(rects,thecolor,lcbgs,gals,ax):
     i=0
     for rect in rects:
          height=rect.get_height()
          print(height)
          if not m.isinf(height):
               axes[ax].text(rect.get_x() + rect.get_width()/2.,height+0.07,'{}'.format(lcbgs[i]) ,ha='center',va='bottom',fontsize='small',color=thecolor)
               axes[ax].text(rect.get_x() + rect.get_width()/2.,height+0.03,'{}'.format(gals[i]) ,ha='center',va='bottom',fontsize='small',color=thecolor)
               i=i+1

def autolabel2(rects,fraction,ax):
	i=0
	for rect in rects:
		height=rect.get_height()
		print(height)
		if not m.isinf(height):
			axes[ax].text(rect.get_x() + rect.get_width()/2.,height+0.01,'{}'.format(fraction[i]) ,ha='center',va='bottom',fontsize='small')
			i=i+1

ALLLCBGS=plt.hist(zbest[LCBGS],5,range=(0,1.0))
ALLGALS=plt.hist(zbest,5,range=(0,1.0))
fractionLCBGS=ALLLCBGS[0]/ALLGALS[0]
weightedfraction=np.zeros_like(fractionLCBGS)
for i in range(0,5):
	weightedfraction[i]=sum(WeightArray[LCBGS[np.where((zbest[LCBGS]>i/5.)&(zbest[LCBGS]<(i+1)/5.))[0]]])/sum(WeightArray[np.where((zbest>i/5.)&(zbest<(i+1)/5.))[0]])
redshift,galdensityeighteen,galdensityfifteen,lcbgdensity=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',skiprows=1,unpack=True)

fraceighteen=np.round(lcbgdensity/galdensityeighteen,4)
fracfifteen=np.round(lcbgdensity/galdensityfifteen,4)

x=ALLLCBGS[1][0:5]
x=x+0.1
FractionalError=fractionLCBGS[0:5]*np.sqrt((ALLLCBGS[0][0:5]/np.power(ALLLCBGS[0][0:5],2))+(ALLGALS[0][0:5]/np.power(ALLGALS[0][0:5],2)))
guzmanerror=12.0/21.0*102.0/301.0*np.sqrt(12.0/12.0**2+21.0/21.0**2+102.0/102.0**2+301.0/301.0**2)
guzman=12.0/21.0*102.0/301.0
tolerud=199/1744.
toleruderror=199./1744.*np.sqrt(199./199.**2+1744./1744.**2)

f,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1,1]},figsize=(8,10))

rectseighteen=axes[1].bar(x,lcbgdensity[0:5]/galdensityeighteen[0:5],0.2,align='center',color='white')

rects=axes[0].bar(x,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='white',ecolor='black')

rectsfifteen=axes[2].bar(x,lcbgdensity[0:5]/galdensityfifteen[0:5],0.2,align='center',color='white')

axes[0].text(0.05,0.6,'(a) N$_{LCBG}$/N$_{GAL}$ COSMOS',va='center')
axes[1].text(0.05,0.6,'(b) $\phi_{LCBG}$/$\phi_{GAL,M=-18.5}$',va='center')
axes[2].text(0.05,0.6,'(c) $\phi_{LCBG}$/$\phi_{GAL,M=-15}$',va='center')

axes[0].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
axes[1].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
axes[2].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')

asymmetric_error=[0.1,0.5]

axes[0].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')
axes[1].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud,2010',color='green')
axes[2].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')


axes[0].legend(loc=6,fontsize='small')
axes[1].legend(loc=6,fontsize='small')
axes[2].legend(loc=6,fontsize='small')

plt.subplots_adjust(hspace=0)
plt.xlabel('Redshift')
f.text(0.04,0.5,'Fraction of LCBGS',va='center',rotation='vertical')
plt.ylim(0,0.7)
autolabel(rects,'black',ALLLCBGS[0],ALLGALS[0],0)
autolabel2(rectseighteen,fraceighteen,1)
autolabel2(rectsfifteen,fracfifteen,2)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/FractionLCBGPLOT.pdf')
plt.show()

#*******************************
#Making Talk Plot
#*******************************

fractionLCBGS=ALLLCBGS[0]/ALLGALS[0]
weightedfraction=np.zeros_like(fractionLCBGS)
for i in range(0,5):
	weightedfraction[i]=sum(WeightArray[LCBGS[np.where((zbest[LCBGS]>i/5.)&(zbest[LCBGS]<(i+1)/5.))[0]]])/sum(WeightArray[np.where((zbest>i/5.)&(zbest<(i+1)/5.))[0]])
redshift,galdensityeighteen,galdensityfifteen,lcbgdensity=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',skiprows=1,unpack=True)

fraceighteen=np.round(lcbgdensity/galdensityeighteen,4)
fracfifteen=np.round(lcbgdensity/galdensityfifteen,4)

x=ALLLCBGS[1][0:5]
x=x+0.1
FractionalError=fractionLCBGS[0:5]*np.sqrt((ALLLCBGS[0][0:5]/np.power(ALLLCBGS[0][0:5],2))+(ALLGALS[0][0:5]/np.power(ALLGALS[0][0:5],2)))
guzmanerror=12.0/21.0*102.0/301.0*np.sqrt(12.0/12.0**2+21.0/21.0**2+102.0/102.0**2+301.0/301.0**2)
guzman=12.0/21.0*102.0/301.0
tolerud=199/1744.
toleruderror=199./1744.*np.sqrt(199./199.**2+1744./1744.**2)

f,axes=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1,1]},figsize=(14,6))

rectseighteen=axes[1].bar(x,lcbgdensity[0:5]/galdensityeighteen[0:5],0.2,align='center',color='white')

rects=axes[0].bar(x,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='white',ecolor='black')

rectsfifteen=axes[2].bar(x,lcbgdensity[0:5]/galdensityfifteen[0:5],0.2,align='center',color='white')

axes[0].text(0.05,0.6,'(a) N$_{LCBG}$/N$_{GAL}$ COSMOS',va='center')
axes[1].text(0.05,0.6,'(b) $\phi_{LCBG}$/$\phi_{GAL,M=-18.5}$',va='center')
axes[2].text(0.05,0.6,'(c) $\phi_{LCBG}$/$\phi_{GAL,M=-15}$',va='center')

axes[0].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
axes[1].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
axes[2].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')

asymmetric_error=[0.1,0.5]

axes[0].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')
axes[1].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud,2010',color='green')
axes[2].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')


axes[0].legend(loc=6,fontsize='small')
axes[1].legend(loc=6,fontsize='small')
axes[2].legend(loc=6,fontsize='small')

plt.subplots_adjust(wspace=0,left=0.08,right=0.95)
plt.xlabel('Redshift')
f.text(0.04,0.5,'Fraction of LCBGS',va='center',rotation='vertical')
plt.ylim(0,0.7)
autolabel2(rects,np.round(fractionLCBGS,4),0)
autolabel2(rectseighteen,fraceighteen,1)
autolabel2(rectsfifteen,fracfifteen,2)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/FractionLCBGPLOT.png')
plt.show()
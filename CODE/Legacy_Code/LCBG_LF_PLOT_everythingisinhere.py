import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
import uncertainties as unc
import uncertainties
from uncertainties import unumpy as unp


#Edited for VEGA Magnitudes, Absolute Magnitude uses kcorrectiong and apparent magnitudes with proper zero-point offset. LRH 5/6/17

import numpy as np
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
#PLOTTING LUMINOSITY FUNCTION FOR PAPER (VARYING ALPHA)
#*******************************
#*******************************
#*******************************



filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_0_20.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_0_20_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)

LFGAL020=sum(NGal)
LCBGGAL020=sum(LNGal)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

def autolabel(rects,thecolor,row,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[row][col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)

#def autolabel(rects,thecolor):
#         for rect in rects:
#                  height=rect.get_height()
#                  print(height)
#                  if not m.isinf(height):
#                           axes[1].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)

@custom_model
def schechter_func(x,phistar=0.0056,mstar=-21,alpha=-1.03):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


def schechter_func_scipy(x,phistar,mstar,alpha):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


LCBGFIT_init=schechter_func()
GALFIT_init=schechter_func()
LCBG_Range=np.linspace(-24,-15,30)

init_vals=[0.0056,-21,-1.03]	#Best guess at initial values, needed for scipy fitting

#PLOTTING 0<z<0.2

#Creating Mask Arrays (to easily remove points that are generally off

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros in LCBG Luminosity Function

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True

#Masking zeros in Luminosity Function

LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True

#Masking errant points in LCBG Luminosity Function

LLumFunc2.mask[6]=True
LMBINAVE2.mask[6]=True
LLumFuncErr2.mask[6]=True
LLumFunc2.mask[8]=True
LMBINAVE2.mask[8]=True
LLumFuncErr2.mask[8]=True
LumFunc2.mask[2:4]=True
MBINAVE2.mask[2:4]=True
LumFuncErr2.mask[2:4]=True

#Astropy Modelling

LCBG_FIT020=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT020=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT020u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT020l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_020_fit,scipy_LCBG_020_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG020ERRORS=np.array([np.sqrt(scipy_LCBG_020_cov[0][0]),np.sqrt(scipy_LCBG_020_cov[2][2]),np.sqrt(scipy_LCBG_020_cov[1][1])])
LUMFUNC020ERRORS=np.array([np.sqrt(scipy_LUMFUNC_020_cov[0][0]),np.sqrt(scipy_LUMFUNC_020_cov[2][2]),np.sqrt(scipy_LUMFUNC_020_cov[1][1])])

#Plotting
x=np.linspace(-23,-16,8)
xminor=np.linspace(-23.5,-15.5,9)
y=np.linspace(-7,-1,7)
yminor=np.linspace(-6.5,-1.5,6)

#f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
#LCBGcode=axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[0].plot(LCBG_Range,np.log10(LCBG_FIT020(LCBG_Range)),color='blue')
#axes[0].plot(LCBG_Range,np.log10(LUMFUNC_FIT020(LCBG_Range)),color='green')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[0].set_xticks(x)
#axes[0].set_xticks(xminor,minor=True)
#axes[0].set_yticks(y)
#axes[0].set_yticks(yminor,minor=True)
#axes[0].set_ylim([-7.5,-0.5])
#axes[1].set_yticks([3,2,1,0])
#axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[1].set_ylim([0,4])
#autolabel(lcbg,'black')
#autolabel(gals,'black')
#axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
#axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
#axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
#f.suptitle('z=0.01-0.2',fontsize=18)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC020.png')


f,axes=plt.subplots(nrows=4,ncols=3,sharex=True,gridspec_kw={'height_ratios':[3,1,3,1]},figsize=(24,13.5))
LCBGcode=axes[0][0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][0].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][0].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020(LCBG_Range)))
axes[0][0].plot(LCBG_Range,np.log10(LUMFUNC_FIT020(LCBG_Range)))
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][0].set_xticks(x)
axes[0][0].set_xticks(xminor,minor=True)
axes[0][0].set_yticks(y)
axes[0][0].set_yticks(yminor,minor=True)
axes[0][0].set_ylim([-7.5,-0.5])
axes[1][0].set_yticks([3,2,1,0])
axes[1][0].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][0].set_ylim([0,4])
autolabel(lcbg,'black',1,0)
autolabel(gals,'black',1,0)

#PLOTTING 0.2<z<0.4

#Defining Files to Read in

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_20_40.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_20_40_FULL.txt'

#Reading in Luminosity Function Files

LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)

LFGAL2040=sum(NGal)
LCBGGAL2040=sum(LNGal)


#Reading in Header parameters

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])

#Creating Mask Arrays (to easily remove points that are generally off

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros in LCBG Luminosity Function

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True

#Masking zeros in LCBG Luminosity Function

LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True

#Masking Errant Points

LLumFunc2.mask[11:12]=True
LMBINAVE2.mask[11:12]=True
LLumFuncErr2.mask[11:12]=True
LLumFunc2.mask[2]=True
LMBINAVE2.mask[2]=True
LLumFuncErr2.mask[2]=True
LumFunc2.mask[14:16]=True
MBINAVE2.mask[14:16]=True
LumFuncErr2.mask[14:16]=True
LumFunc2.mask[1:3]=True
MBINAVE2.mask[1:3]=True
LumFuncErr2.mask[1:3]=True

#Astropy fitting

LCBG_FIT2040=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT2040=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT2040u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT2040l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_2040_fit,scipy_LCBG_2040_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_2040_fit,scipy_LUMFUNC_2040_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG2040ERRORS=np.array([np.sqrt(scipy_LCBG_2040_cov[0][0]),np.sqrt(scipy_LCBG_2040_cov[1][1]),np.sqrt(scipy_LCBG_2040_cov[2][2])])
LUMFUNC2040ERRORS=np.array([np.sqrt(scipy_LUMFUNC_2040_cov[0][0]),np.sqrt(scipy_LUMFUNC_2040_cov[1][1]),np.sqrt(scipy_LUMFUNC_2040_cov[2][2])])

#Plotting


LCBGcode=axes[0][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040u(LCBG_Range)),color='red')
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].set_yticks(y)
axes[0][1].set_yticks(yminor,minor=True)
axes[0][1].set_ylim([-7.5,-0.5])
axes[1][1].set_yticks([3,2,1,0])
axes[1][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][1].set_ylim([0,4])
autolabel(lcbg,'black',1,1)
autolabel(gals,'black',1,1)


#f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
#LCBGcode=axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[0].plot(LCBG_Range,np.log10(LCBG_FIT2040(LCBG_Range)),color='blue')
#axes[0].plot(LCBG_Range,np.log10(LUMFUNC_FIT2040(LCBG_Range)),color='green')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[0].set_xticks(x)
#axes[0].set_xticks(xminor,minor=True)
#axes[0].set_yticks(y)
#axes[0].set_yticks(yminor,minor=True)
#axes[0].set_ylim([-7.5,-0.5])
#axes[1].set_yticks([3,2,1,0])
#axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[1].set_ylim([0,4])
#autolabel(lcbg,'black')
#autolabel(gals,'black')
#axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
#axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
#axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
#f.suptitle('z=0.2-0.4',fontsize=18)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC2040.png')

#PLOTTING 0.4<z<0.6

#New Files
filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_60.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_40_60_FULL.txt'

#Reading in to array

LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL4060=sum(NGal)
LCBGGAL4060=sum(LNGal)


#Redefining arrays to mask them

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True
LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True

#Masking errant points

LumFunc2.mask[11:15]=True
MBINAVE2.mask[11:15]=True
LumFuncErr2.mask[11:15]=True
#LumFunc2.mask[8]=True
#MBINAVE2.mask[8]=True
#LumFuncErr2.mask[8]=True
#LLumFunc2.mask[8]=True
#LMBINAVE2.mask[8]=True
#LLumFuncErr2.mask[8]=True

#Astropy Fitting

LCBG_FIT4060=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT4060=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT4060u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT4060l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_4060_fit,scipy_LCBG_4060_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_4060_fit,scipy_LUMFUNC_4060_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG4060ERRORS=np.array([np.sqrt(scipy_LCBG_4060_cov[0][0]),np.sqrt(scipy_LCBG_4060_cov[1][1]),np.sqrt(scipy_LCBG_4060_cov[2][2])])
LUMFUNC4060ERRORS=np.array([np.sqrt(scipy_LUMFUNC_4060_cov[0][0]),np.sqrt(scipy_LUMFUNC_4060_cov[1][1]),np.sqrt(scipy_LUMFUNC_4060_cov[2][2])])


LCBGcode=axes[0][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060(LCBG_Range)))
axes[0][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT4060(LCBG_Range)))
#axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060u(LCBG_Range)),color='red')
#axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][2].set_yticks(y)
axes[0][2].set_yticks(yminor,minor=True)
axes[0][2].set_ylim([-7.5,-0.5])
axes[1][2].set_yticks([3,2,1,0])
axes[1][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][2].set_ylim([0,4])
autolabel(lcbg,'black',1,2)
autolabel(gals,'black',1,2)

#f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
#LCBGcode=axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[0].plot(LCBG_Range,np.log10(LCBG_FIT4060(LCBG_Range)),color='blue')
#axes[0].plot(LCBG_Range,np.log10(LUMFUNC_FIT4060(LCBG_Range)),color='green')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[0].set_xticks(x)
#axes[0].set_xticks(xminor,minor=True)
#axes[0].set_yticks(y)
#axes[0].set_yticks(yminor,minor=True)
#axes[0].set_ylim([-7.5,-0.5])
#axes[1].set_yticks([3,2,1,0])
#axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[1].set_ylim([0,4])
#autolabel(lcbg,'black')
#autolabel(gals,'black')
#axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
#axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
#axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
#f.suptitle('z=0.4-0.6',fontsize=18)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC4060.png')

#PLOTTING 0.6<z<0.8

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_60_80.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_60_80_FULL.txt'


LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL6080=sum(NGal)
LCBGGAL6080=sum(LNGal)


LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)



LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True
LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True



LumFunc2.mask[9]=True
MBINAVE2.mask[9]=True
LumFuncErr2.mask[9]=True
LLumFunc2.mask[9]=True
LMBINAVE2.mask[9]=True
LLumFuncErr2.mask[9]=True
LLumFunc2.mask[0:3]=True
LMBINAVE2.mask[0:3]=True
LLumFuncErr2.mask[0:3]=True



LCBG_FIT6080=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT6080=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT6080u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT6080l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_6080_fit,scipy_LCBG_6080_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_6080_fit,scipy_LUMFUNC_6080_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG6080ERRORS=np.array([np.sqrt(scipy_LCBG_6080_cov[0][0]),np.sqrt(scipy_LCBG_6080_cov[1][1]),np.sqrt(scipy_LCBG_6080_cov[2][2])])
LUMFUNC6080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_6080_cov[0][0]),np.sqrt(scipy_LUMFUNC_6080_cov[1][1]),np.sqrt(scipy_LUMFUNC_6080_cov[2][2])])


LCBGcode=axes[2][0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][0].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][0].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080(LCBG_Range)))
axes[2][0].plot(LCBG_Range,np.log10(LUMFUNC_FIT6080(LCBG_Range)))
#axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080u(LCBG_Range)),color='red')
#axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][0].set_yticks(y)
axes[2][0].set_yticks(yminor,minor=True)
axes[2][0].set_ylim([-7.5,-0.5])
axes[3][0].set_yticks([3,2,1,0])
axes[3][0].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][0].set_ylim([0,4])
autolabel(lcbg,'black',3,0)
autolabel(gals,'black',3,0)

#f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
#LCBGcode=axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[0].plot(LCBG_Range,np.log10(LCBG_FIT6080(LCBG_Range)),color='blue')
#axes[0].plot(LCBG_Range,np.log10(LUMFUNC_FIT6080(LCBG_Range)),color='green')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[0].set_xticks(x)
#axes[0].set_xticks(xminor,minor=True)
#axes[0].set_yticks(y)
#axes[0].set_yticks(yminor,minor=True)
#axes[0].set_ylim([-7.5,-0.5])
#axes[1].set_yticks([3,2,1,0])
#axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[1].set_ylim([0,4])
#autolabel(lcbg,'black')
#autolabel(gals,'black')
#axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
#axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
#axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
#f.suptitle('z=0.6-0.8',fontsize=18)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC6080.png')

#PLOTTING 0.8<z<1

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_80_100.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_80_100_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL80100=sum(NGal)
LCBGGAL80100=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True
LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True

LumFunc2.mask[7]=True
MBINAVE2.mask[7]=True
LumFuncErr2.mask[7]=True
LLumFunc2.mask[7]=True
LMBINAVE2.mask[7]=True
LLumFuncErr2.mask[7]=True

LumFunc2.mask[0]=True
MBINAVE2.mask[0]=True
LumFuncErr2.mask[0]=True

LCBG_FIT80100=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT80100=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT80100u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT80100l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_80100_fit,scipy_LCBG_80100_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_80100_fit,scipy_LUMFUNC_80100_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG80100ERRORS=np.array([np.sqrt(scipy_LCBG_80100_cov[0][0]),np.sqrt(scipy_LCBG_80100_cov[1][1]),np.sqrt(scipy_LCBG_80100_cov[2][2])])
LUMFUNC80100ERRORS=np.array([np.sqrt(scipy_LUMFUNC_80100_cov[0][0]),np.sqrt(scipy_LUMFUNC_80100_cov[1][1]),np.sqrt(scipy_LUMFUNC_80100_cov[2][2])])



LCBGcode=axes[2][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100(LCBG_Range)))
axes[2][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT80100(LCBG_Range)))
#axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100u(LCBG_Range)),color='red')
#axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][1].set_yticks(y)
axes[2][1].set_yticks(yminor,minor=True)
axes[2][1].set_ylim([-7.5,-0.5])
axes[3][1].set_yticks([3,2,1,0])
axes[3][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][1].set_ylim([0,4])
autolabel(lcbg,'black',3,1)
autolabel(gals,'black',3,1)

#f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
#LCBGcode=axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[0].plot(LCBG_Range,np.log10(LCBG_FIT80100(LCBG_Range)),color='blue')
#axes[0].plot(LCBG_Range,np.log10(LUMFUNC_FIT80100(LCBG_Range)),color='green')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[0].set_xticks(x)
#axes[0].set_xticks(xminor,minor=True)
#axes[0].set_yticks(y)
#axes[0].set_yticks(yminor,minor=True)
#axes[0].set_ylim([-7.5,-0.5])
#axes[1].set_yticks([3,2,1,0])
#axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[1].set_ylim([0,4])
#autolabel(lcbg,'black')
#autolabel(gals,'black')
#axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
#axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
#axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
#f.suptitle('z=0.8-1.0',fontsize=18)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC80100.png')

#0.3<z<0.8

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_30_80.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_30_80_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL3080=sum(NGal)
LCBGGAL3080=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True
LumFunc2.mask[np.where(LumFunc==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc==0)[0]]=True

LumFunc2.mask[11:17]=True
MBINAVE2.mask[11:17]=True
LumFuncErr2.mask[11:17]=True
LLumFunc2.mask[11:17]=True
LMBINAVE2.mask[11:17]=True
LLumFuncErr2.mask[11:17]=True
LLumFunc2.mask[0:3]=True
LMBINAVE2.mask[0:3]=True
LLumFuncErr2.mask[0:3]=True

LCBG_FIT3080=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT3080=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT3080u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT3080l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_3080_fit,scipy_LCBG_3080_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_3080_fit,scipy_LUMFUNC_3080_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG3080ERRORS=np.array([np.sqrt(scipy_LCBG_3080_cov[0][0]),np.sqrt(scipy_LCBG_3080_cov[1][1]),np.sqrt(scipy_LCBG_3080_cov[2][2])])
LUMFUNC3080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_3080_cov[0][0]),np.sqrt(scipy_LUMFUNC_3080_cov[1][1]),np.sqrt(scipy_LUMFUNC_3080_cov[2][2])])


LCBGcode=axes[2][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080(LCBG_Range)))
axes[2][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT3080(LCBG_Range)))
#axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080u(LCBG_Range)),color='red')
#axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][2].set_yticks(y)
axes[2][2].set_yticks(yminor,minor=True)
axes[2][2].set_ylim([-7.5,-0.5])
axes[3][2].set_yticks([3,2,1,0])
axes[3][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][2].set_ylim([0,4.0])
autolabel(lcbg,'black',3,2)
autolabel(gals,'black',3,2)
axes[0][1].set_yticklabels([])
axes[0][2].set_yticklabels([])
axes[1][1].set_yticklabels([])
axes[1][2].set_yticklabels([])
axes[2][1].set_yticklabels([])
axes[2][2].set_yticklabels([])
axes[3][1].set_yticklabels([])
axes[3][2].set_yticklabels([])
axes[0][0].text(-23.5,-1,'z=0.01-0.2',verticalalignment='center',fontsize=12)
axes[0][1].text(-23.5,-1,'z=0.2-0.4',verticalalignment='center',fontsize=12)
axes[0][2].text(-23.5,-1,'z=0.4-0.6',verticalalignment='center',fontsize=12)
axes[2][0].text(-23.5,-1,'z=0.6-0.8',verticalalignment='center',fontsize=12)
axes[2][2].text(-23.5,-1,'z=0.3-0.8',verticalalignment='center',fontsize=12)
axes[2][1].text(-23.5,-1,'z=0.8-1.0',verticalalignment='center',fontsize=12)
f.text(0.52,0.05,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=12)
f.text(0.05,0.75,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.35,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.55,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.15,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=12)




#*******************************
#*******************************
#*******************************
#GENERATING OUTPUT 
#*******************************
#*******************************
#*******************************



redshiftrange=np.array([0.1,0.3,0.5,0.7,0.9,0.55])

allngal=np.array([LFGAL020,LFGAL2040,LFGAL4060,LFGAL6080,LFGAL80100,LFGAL3080])

allphistar=1000*np.array([LUMFUNC_FIT020.phistar.value,LUMFUNC_FIT2040.phistar.value,LUMFUNC_FIT4060.phistar.value,LUMFUNC_FIT6080.phistar.value,LUMFUNC_FIT80100.phistar.value,LUMFUNC_FIT3080.phistar.value])

allphistarerr=1000*np.array([LUMFUNC020ERRORS[0],LUMFUNC2040ERRORS[0],LUMFUNC4060ERRORS[0],LUMFUNC6080ERRORS[0],LUMFUNC80100ERRORS[0],LUMFUNC3080ERRORS[0]])

allmstar=np.array([LUMFUNC_FIT020.mstar.value,LUMFUNC_FIT2040.mstar.value,LUMFUNC_FIT4060.mstar.value,LUMFUNC_FIT6080.mstar.value,LUMFUNC_FIT80100.mstar.value,LUMFUNC_FIT3080.mstar.value])

allmstarerr=np.array([LUMFUNC020ERRORS[1],LUMFUNC2040ERRORS[1],LUMFUNC4060ERRORS[1],LUMFUNC6080ERRORS[1],LUMFUNC80100ERRORS[1],LUMFUNC3080ERRORS[1]])

allalpha=np.array([LUMFUNC_FIT020.alpha.value,LUMFUNC_FIT2040.alpha.value,LUMFUNC_FIT4060.alpha.value,LUMFUNC_FIT6080.alpha.value,LUMFUNC_FIT80100.alpha.value,LUMFUNC_FIT3080.alpha.value])

allalphaerr=np.array([LUMFUNC020ERRORS[2],LUMFUNC2040ERRORS[2],LUMFUNC4060ERRORS[2],LUMFUNC6080ERRORS[2],LUMFUNC80100ERRORS[2],LUMFUNC3080ERRORS[2]])

lcbgngal=np.array([LCBGGAL020,LCBGGAL2040,LCBGGAL4060,LCBGGAL6080,LCBGGAL80100,LCBGGAL3080])

lcbgphistar=1000*np.array([LCBG_FIT020.phistar.value,LCBG_FIT2040.phistar.value,LCBG_FIT4060.phistar.value,LCBG_FIT6080.phistar.value,LCBG_FIT80100.phistar.value,LCBG_FIT3080.phistar.value])

lcbgphistarerr=1000*np.array([LCBG020ERRORS[0],LCBG2040ERRORS[0],LCBG4060ERRORS[0],LCBG6080ERRORS[0],LCBG80100ERRORS[0],LCBG3080ERRORS[0]])

lcbgmstar=np.array([LCBG_FIT020.mstar.value,LCBG_FIT2040.mstar.value,LCBG_FIT4060.mstar.value,LCBG_FIT6080.mstar.value,LCBG_FIT80100.mstar.value,LCBG_FIT3080.mstar.value])

lcbgmstarerr=np.array([LCBG020ERRORS[1],LCBG2040ERRORS[1],LCBG4060ERRORS[1],LCBG6080ERRORS[1],LCBG80100ERRORS[1],LCBG3080ERRORS[1]])

lcbgalpha=np.array([LCBG_FIT020.alpha.value,LCBG_FIT2040.alpha.value,LCBG_FIT4060.alpha.value,LCBG_FIT6080.alpha.value,LCBG_FIT80100.alpha.value,LCBG_FIT3080.alpha.value])

lcbgalphaerr=np.array([LCBG020ERRORS[2],LCBG2040ERRORS[2],LCBG4060ERRORS[2],LCBG6080ERRORS[2],LCBG80100ERRORS[2],LCBG3080ERRORS[2]])

galdensityeighteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-18.5)[0]])

galdensityfifteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-15)[0]])

lcbgdensity=np.array([sp.integrate.quad(LCBG_FIT020,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT2040,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT4060,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT6080,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT80100,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT3080,-100,-18.5)[0]])


@uncertainties.wrap
def integratethings(ps,ms,al):
    def integrand(X):
        return (0.4*np.log(10)*ps)*(10**(0.4*(al+1)*(ms-X)))*(np.e**(-np.power(10,0.4*(ms-X))))
    integral,abserr=sp.integrate.quad(integrand,-100,-18.5)
    return integral

@uncertainties.wrap
def integratethings15(ps,ms,al):
    def integrand(X):
        return (0.4*np.log(10)*ps)*(10**(0.4*(al+1)*(ms-X)))*(np.e**(-np.power(10,0.4*(ms-X))))
    integral,abserr=sp.integrate.quad(integrand,-100,-15)
    return integral

phistarcorr020,mstarcorr020,alphacorr020=uncertainties.correlated_values(scipy_LCBG_020_fit,scipy_LCBG_020_cov)

phistarcorr2040,mstarcorr2040,alphacorr2040=uncertainties.correlated_values(scipy_LCBG_2040_fit,scipy_LCBG_2040_cov)

phistarcorr4060,mstarcorr4060,alphacorr4060=uncertainties.correlated_values(scipy_LCBG_4060_fit,scipy_LCBG_4060_cov)

phistarcorr6080,mstarcorr6080,alphacorr6080=uncertainties.correlated_values(scipy_LCBG_6080_fit,scipy_LCBG_6080_cov)

phistarcorr80100,mstarcorr80100,alphacorr80100=uncertainties.correlated_values(scipy_LCBG_80100_fit,scipy_LCBG_80100_cov)

galphistarcorr020,galmstarcorr020,galalphacorr020=uncertainties.correlated_values(scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov)

galphistarcorr2040,galmstarcorr2040,galalphacorr2040=uncertainties.correlated_values(scipy_LUMFUNC_2040_fit,scipy_LUMFUNC_2040_cov)

galphistarcorr4060,galmstarcorr4060,galalphacorr4060=uncertainties.correlated_values(scipy_LUMFUNC_4060_fit,scipy_LUMFUNC_4060_cov)

galphistarcorr6080,galmstarcorr6080,galalphacorr6080=uncertainties.correlated_values(scipy_LUMFUNC_6080_fit,scipy_LUMFUNC_6080_cov)

galphistarcorr80100,galmstarcorr80100,galalphacorr80100=uncertainties.correlated_values(scipy_LUMFUNC_80100_fit,scipy_LUMFUNC_80100_cov)


lcbg_density_err=np.array([integratethings(phistarcorr020,mstarcorr020,alphacorr020),integratethings(phistarcorr2040,mstarcorr2040,alphacorr2040),integratethings(phistarcorr4060,mstarcorr4060,alphacorr4060),integratethings(phistarcorr6080,mstarcorr6080,alphacorr6080),integratethings(phistarcorr80100,mstarcorr80100,alphacorr80100)])

gal_densityeighteen_err=np.array([integratethings(galphistarcorr020,galmstarcorr020,galalphacorr020),integratethings(galphistarcorr2040,galmstarcorr2040,galalphacorr2040),integratethings(galphistarcorr4060,galmstarcorr4060,galalphacorr4060),integratethings(galphistarcorr6080,galmstarcorr6080,galalphacorr6080),integratethings(galphistarcorr80100,galmstarcorr80100,galalphacorr80100)])

gal_densityfifteen_err=np.array([integratethings15(galphistarcorr020,galmstarcorr020,galalphacorr020),integratethings15(galphistarcorr2040,galmstarcorr2040,galalphacorr2040),integratethings15(galphistarcorr4060,galmstarcorr4060,galalphacorr4060),integratethings15(galphistarcorr6080,galmstarcorr6080,galalphacorr6080),integratethings15(galphistarcorr80100,galmstarcorr80100,galalphacorr80100)])

#FracGals=np.stack((redshiftrange,galdensityeighteen,galdensityfifteen,lcbgdensity),axis=-1)
FracGals=np.stack((redshiftrange[0:5],gal_densityeighteen_err,gal_densityfifteen_err,lcbg_density_err),axis=-1)

#np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',FracGals,header='#z	galden	lcbgden')

LFfittingparams=np.stack((redshiftrange,allngal,unp.nominal_values(galmstar,allmstarerr,allphistar,allphistarerr),axis=-1)
LCBGfittingparams=np.stack((redshiftrange,lcbgngal,lcbgmstar,lcbgmstarerr,lcbgphistar,lcbgphistarerr),axis=-1)
np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LFfitparams.txt',LFfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')
np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LCBGfitparams.txt',LCBGfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')

WILLMER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/WILLMER.txt')
FABER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FABER.txt')
COOL=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/COOL.txt')
ZUCCA=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA.txt')
FRITZ=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FRITZ.txt')
BEARE=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE.txt')
BEAREALPH=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE_CONSTANTALPHA.txt')
ZUCCAALPH=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA_CONSTANTALPHA.txt')


LCBGDENUP020ps=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00098,mstar=-20.33),-100,-18.5)
LCBGDENLOW020ps=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00044,mstar=-20.33),-100,-18.5)
LCBGDENUP2040ps=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00196,mstar=-20.08),-100,-18.5)
LCBGDENLOW2040ps=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00106,mstar=-20.08),-100,-18.5)
LCBGDENUP4060ps=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.0018,mstar=-20.55),-100,-18.5)
LCBGDENLOW4060ps=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00126,mstar=-20.55),-100,-18.5)
LCBGDENUP6080ps=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00405,mstar=-20.56),-100,-18.5)
LCBGDENLOW6080ps=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00243,mstar=-20.56),-100,-18.5)
LCBGDENUP80100ps=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.00479,mstar=-20.86),-100,-18.5)
LCBGDENLOW80100ps=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.00381,mstar=-20.86),-100,-18.5)
LCBGDENUP3080ps=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00296,mstar=-20.37),-100,-18.5)
LCBGDENLOW3080ps=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.0024,mstar=-20.37),-100,-18.5)

LCBGDENUP020ms=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00071,mstar=-19.92),-100,-18.5)
LCBGDENLOW020ms=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00071,mstar=-20.74),-100,-18.5)
LCBGDENUP2040ms=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00151,mstar=-19.83),-100,-18.5)
LCBGDENLOW2040ms=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00151,mstar=-20.33),-100,-18.5)
LCBGDENUP4060ms=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00153,mstar=-20.38),-100,-18.5)
LCBGDENLOW4060ms=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00153,mstar=-20.72),-100,-18.5)
LCBGDENUP6080ms=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00324,mstar=-20.3),-100,-18.5)
LCBGDENLOW6080ms=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00324,mstar=-20.82),-100,-18.5)
LCBGDENUP80100ms=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.0043,mstar=-20.98),-100,-18.5)
LCBGDENLOW80100ms=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.0043,mstar=-20.74),-100,-18.5)
LCBGDENUP3080ms=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00268,mstar=-20.24),-100,-18.5)
LCBGDENLOW3080ms=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00268,mstar=-20.5),-100,-18.5)

LCBGDENUP020al=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00071,mstar=-20.33),-100,-18.5)
LCBGDENLOW020al=sp.integrate.quad(schechter_func(alpha=-0.24,phistar=0.00071,mstar=-20.33),-100,-18.5)
LCBGDENUP2040al=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00151,mstar=-20.08),-100,-18.5)
LCBGDENLOW2040al=sp.integrate.quad(schechter_func(alpha=-0.82,phistar=0.00151,mstar=-20.08),-100,-18.5)
LCBGDENUP4060al=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00153,mstar=-20.55),-100,-18.5)
LCBGDENLOW4060al=sp.integrate.quad(schechter_func(alpha=-0.72,phistar=0.00153,mstar=-20.55),-100,-18.5)
LCBGDENUP6080al=sp.integrate.quad(schechter_func(alpha=-1.24,phistar=0.00324,mstar=-20.56),-100,-18.5)
LCBGDENLOW6080al=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00324,mstar=-20.56),-100,-18.5)
LCBGDENUP80100al=sp.integrate.quad(schechter_func(alpha=-1.33,phistar=0.0043,mstar=-20.86),-100,-18.5)
LCBGDENLOW80100al=sp.integrate.quad(schechter_func(alpha=-0.91,phistar=0.0043,mstar=-20.86),-100,-18.5)
LCBGDENUP3080al=sp.integrate.quad(schechter_func(alpha=-0.67,phistar=0.00268,mstar=-20.37),-100,-18.5)
LCBGDENLOW3080al=sp.integrate.quad(schechter_func(alpha=-0.37,phistar=0.00268,mstar=-20.37),-100,-18.5)

LCBGDENUPps=np.stack((LCBGDENUP020ps[0],LCBGDENUP2040ps[0],LCBGDENUP4060ps[0],LCBGDENUP6080ps[0],LCBGDENUP80100ps[0],LCBGDENUP3080ps[0]),axis=-1)

LCBGDENUPms=np.stack((LCBGDENUP020ms[0],LCBGDENUP2040ms[0],LCBGDENUP4060ms[0],LCBGDENUP6080ms[0],LCBGDENUP80100ms[0],LCBGDENUP3080ms[0]),axis=-1)

LCBGDENUPal=np.stack((LCBGDENUP020al[0],LCBGDENUP2040al[0],LCBGDENUP4060al[0],LCBGDENUP6080al[0],LCBGDENUP80100al[0],LCBGDENUP3080al[0]),axis=-1)

LCBGDENLOWps=np.stack((LCBGDENLOW020ps[0],LCBGDENLOW2040ps[0],LCBGDENLOW4060ps[0],LCBGDENLOW6080ps[0],LCBGDENLOW80100ps[0],LCBGDENLOW3080ps[0]),axis=-1)

LCBGDENLOWms=np.stack((LCBGDENLOW020ms[0],LCBGDENLOW2040ms[0],LCBGDENLOW4060ms[0],LCBGDENLOW6080ms[0],LCBGDENLOW80100ms[0],LCBGDENLOW3080ms[0]),axis=-1)

LCBGDENLOWal=np.stack((LCBGDENLOW020al[0],LCBGDENLOW2040al[0],LCBGDENLOW4060al[0],LCBGDENLOW6080al[0],LCBGDENLOW80100al[0],LCBGDENLOW3080al[0]),axis=-1)


GALDENUP020ps=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00416,mstar=-20.59),-100,-18.5)
GALDENLOW020ps=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00269,mstar=-20.59),-100,-18.5)
GALDENUP2040ps=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00943,mstar=-20.51),-100,-18.5)
GALDENLOW2040ps=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00831,mstar=-20.51),-100,-18.5)
GALDENUP4060ps=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00527,mstar=-20.67),-100,-18.5)
GALDENLOW4060ps=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00443,mstar=-20.67),-100,-18.5)
GALDENUP6080ps=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00839,mstar=-20.8),-100,-18.5)
GALDENLOW6080ps=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00639,mstar=-20.8),-100,-18.5)
GALDENUP80100ps=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00949,mstar=-20.91),-100,-18.5)
GALDENLOW80100ps=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00837,mstar=-20.91),-100,-18.5)

GALDENUP020ms=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00346,mstar=-20.83),-100,-18.5)
GALDENLOW020ms=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00346,mstar=-20.35),-100,-18.5)
GALDENUP2040ms=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00887,mstar=-20.57),-100,-18.5)
GALDENLOW2040ms=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00887,mstar=-20.45),-100,-18.5)
GALDENUP4060ms=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00485,mstar=-20.75),-100,-18.5)
GALDENLOW4060ms=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00485,mstar=-20.59),-100,-18.5)
GALDENUP6080ms=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00739,mstar=-20.94),-100,-18.5)
GALDENLOW6080ms=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00739,mstar=-20.66),-100,-18.5)
GALDENUP80100ms=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00893,mstar=-20.99),-100,-18.5)
GALDENLOW80100ms=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00893,mstar=-20.83),-100,-18.5)

GALDENUP020al=sp.integrate.quad(schechter_func(alpha=-1.35,phistar=0.00346,mstar=-20.59),-100,-18.5)
GALDENLOW020al=sp.integrate.quad(schechter_func(alpha=-1.25,phistar=0.00346,mstar=-20.59),-100,-18.5)
GALDENUP2040al=sp.integrate.quad(schechter_func(alpha=-1.03,phistar=0.00887,mstar=-20.51),-100,-18.5)
GALDENLOW2040al=sp.integrate.quad(schechter_func(alpha=-0.97,phistar=0.00887,mstar=-20.51),-100,-18.5)
GALDENUP4060al=sp.integrate.quad(schechter_func(alpha=-1.04,phistar=0.00485,mstar=-20.67),-100,-18.5)
GALDENLOW4060al=sp.integrate.quad(schechter_func(alpha=-0.92,phistar=0.00485,mstar=-20.67),-100,-18.5)
GALDENUP6080al=sp.integrate.quad(schechter_func(alpha=-1.11,phistar=0.00739,mstar=-20.8),-100,-18.5)
GALDENLOW6080al=sp.integrate.quad(schechter_func(alpha=-0.77,phistar=0.00739,mstar=-20.8),-100,-18.5)
GALDENUP80100al=sp.integrate.quad(schechter_func(alpha=-1.0,phistar=0.00893,mstar=-20.91),-100,-18.5)
GALDENLOW80100al=sp.integrate.quad(schechter_func(alpha=-0.7,phistar=0.00893,mstar=-20.91),-100,-18.5)

GALDENUP02015ps=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00416,mstar=-20.59),-100,-15)
GALDENLOW02015ps=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00269,mstar=-20.59),-100,-15)
GALDENUP204015ps=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00943,mstar=-20.51),-100,-15)
GALDENLOW204015ps=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00831,mstar=-20.51),-100,-15)
GALDENUP406015ps=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00527,mstar=-20.67),-100,-15)
GALDENLOW406015ps=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00443,mstar=-20.67),-100,-15)
GALDENUP608015ps=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00839,mstar=-20.8),-100,-15)
GALDENLOW608015ps=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00639,mstar=-20.8),-100,-15)
GALDENUP8010015ps=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00949,mstar=-20.91),-100,-15)
GALDENLOW8010015ps=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00837,mstar=-20.91),-100,-15)

GALDENUP02015ms=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00346,mstar=-20.83),-100,-15)
GALDENLOW02015ms=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00346,mstar=-20.35),-100,-15)
GALDENUP204015ms=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00887,mstar=-20.57),-100,-15)
GALDENLOW204015ms=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00887,mstar=-20.45),-100,-15)
GALDENUP406015ms=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00485,mstar=-20.75),-100,-15)
GALDENLOW406015ms=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00485,mstar=-20.59),-100,-15)
GALDENUP608015ms=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00739,mstar=-20.94),-100,-15)
GALDENLOW608015ms=sp.integrate.quad(schechter_func(alpha=-0.94,phistar=0.00739,mstar=-20.66),-100,-15)
GALDENUP8010015ms=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00893,mstar=-20.99),-100,-15)
GALDENLOW8010015ms=sp.integrate.quad(schechter_func(alpha=-0.85,phistar=0.00893,mstar=-20.83),-100,-15)

GALDENUP02015al=sp.integrate.quad(schechter_func(alpha=-1.35,phistar=0.00346,mstar=-20.59),-100,-15)
GALDENLOW02015al=sp.integrate.quad(schechter_func(alpha=-1.25,phistar=0.00346,mstar=-20.59),-100,-15)
GALDENUP204015al=sp.integrate.quad(schechter_func(alpha=-1.03,phistar=0.00887,mstar=-20.51),-100,-15)
GALDENLOW204015al=sp.integrate.quad(schechter_func(alpha=-0.97,phistar=0.00887,mstar=-20.51),-100,-15)
GALDENUP406015al=sp.integrate.quad(schechter_func(alpha=-1.04,phistar=0.00485,mstar=-20.67),-100,-15)
GALDENLOW406015al=sp.integrate.quad(schechter_func(alpha=-0.92,phistar=0.00485,mstar=-20.67),-100,-15)
GALDENUP608015al=sp.integrate.quad(schechter_func(alpha=-1.11,phistar=0.00739,mstar=-20.8),-100,-15)
GALDENLOW608015al=sp.integrate.quad(schechter_func(alpha=-0.77,phistar=0.00739,mstar=-20.8),-100,-15)
GALDENUP8010015al=sp.integrate.quad(schechter_func(alpha=-1.0,phistar=0.00893,mstar=-20.91),-100,-15)
GALDENLOW8010015al=sp.integrate.quad(schechter_func(alpha=-0.7,phistar=0.00893,mstar=-20.91),-100,-15)

GALDENUPps=np.stack((GALDENUP020ps[0],GALDENUP2040ps[0],GALDENUP4060ps[0],GALDENUP6080ps[0],GALDENUP80100ps[0]),axis=-1)

GALDENUPms=np.stack((GALDENUP020ms[0],GALDENUP2040ms[0],GALDENUP4060ms[0],GALDENUP6080ms[0],GALDENUP80100ms[0]),axis=-1)

GALDENUPal=np.stack((GALDENUP020al[0],GALDENUP2040al[0],GALDENUP4060al[0],GALDENUP6080al[0],GALDENUP80100al[0]),axis=-1)

GALDENLOWps=np.stack((GALDENLOW020ps[0],GALDENLOW2040ps[0],GALDENLOW4060ps[0],GALDENLOW6080ps[0],GALDENLOW80100ps[0]),axis=-1)

GALDENLOWms=np.stack((GALDENLOW020ms[0],GALDENLOW2040ms[0],GALDENLOW4060ms[0],GALDENLOW6080ms[0],GALDENLOW80100ms[0]),axis=-1)

GALDENLOWal=np.stack((GALDENLOW020al[0],GALDENLOW2040al[0],GALDENLOW4060al[0],GALDENLOW6080al[0],GALDENLOW80100al[0]),axis=-1)

gdupps=np.abs(galdensityeighteen[0:5]-GALDENUPps)
gdlowps=np.abs(galdensityeighteen[0:5]-GALDENLOWps)
gdupal=np.abs(galdensityeighteen[0:5]-GALDENUPal)
gdlowal=np.abs(galdensityeighteen[0:5]-GALDENLOWal)
gdupms=np.abs(galdensityeighteen[0:5]-GALDENUPms)
gdlowms=np.abs(galdensityeighteen[0:5]-GALDENLOWms)

GALDENUP15ps=np.stack((GALDENUP02015ps[0],GALDENUP204015ps[0],GALDENUP406015ps[0],GALDENUP608015ps[0],GALDENUP8010015ps[0]),axis=-1)

GALDENUP15ms=np.stack((GALDENUP02015ms[0],GALDENUP204015ms[0],GALDENUP406015ms[0],GALDENUP608015ms[0],GALDENUP8010015ms[0]),axis=-1)

GALDENUP15al=np.stack((GALDENUP02015al[0],GALDENUP204015al[0],GALDENUP406015al[0],GALDENUP608015al[0],GALDENUP8010015al[0]),axis=-1)

GALDENLOW15ps=np.stack((GALDENLOW02015ps[0],GALDENLOW204015ps[0],GALDENLOW406015ps[0],GALDENLOW608015ps[0],GALDENLOW8010015ps[0]),axis=-1)

GALDENLOW15ms=np.stack((GALDENLOW02015ms[0],GALDENLOW204015ms[0],GALDENLOW406015ms[0],GALDENLOW608015ms[0],GALDENLOW8010015ms[0]),axis=-1)

GALDENLOW15al=np.stack((GALDENLOW02015al[0],GALDENLOW204015al[0],GALDENLOW406015al[0],GALDENLOW608015al[0],GALDENLOW8010015al[0]),axis=-1)

gdupps15=np.abs(galdensityfifteen[0:5]-GALDENUP15ps)
gdlowps15=np.abs(galdensityfifteen[0:5]-GALDENLOW15ps)
gdupal15=np.abs(galdensityfifteen[0:5]-GALDENUP15al)
gdlowal15=np.abs(galdensityfifteen[0:5]-GALDENLOW15al)
gdupms15=np.abs(galdensityfifteen[0:5]-GALDENUP15ms)
gdlowms15=np.abs(galdensityfifteen[0:5]-GALDENLOW15ms)

#*******************************
#*******************************
#*******************************
# Plotting Evolution of Parameters
#*******************************
#*******************************
#*******************************


f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,2],yerr=LFfittingparams[0:5,3],fmt='-o',color='black',label='This Work')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,4],yerr=WILLMER[:,5],color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,4],yerr=FABER[:,5],color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,4],yerr=ZUCCA[:,5],color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,4],yerr=COOL[:,5],color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,4],yerr=FRITZ[:,5],color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,4],yerr=BEARE[:,5],color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,2],yerr=LCBGfittingparams[0:5,3],color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([-19.6,-21.8])
axes[1].set_yticks([-20,-20.5,-21,-21.5])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,-21.6,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,-21.6,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'M$^{*}$-5log(h$_{70}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVMSTAR.pdf')

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,4]/1000.,yerr=LFfittingparams[0:5,5]/1000.,color='black',label='This Work',fmt='-o')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,6]/1000.,yerr=WILLMER[:,7]/1000.,color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,6]/1000.,yerr=FABER[:,7]/1000.,color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,6]/1000.,yerr=ZUCCA[:,7]/1000.,color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,6]/1000.,yerr=COOL[:,7]/1000.,color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,6]/1000.,yerr=FRITZ[:,7]/1000.,color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,6]/1000.,yerr=BEARE[:,7]/1000.,color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,4]/1000.,yerr=LCBGfittingparams[0:5,5]/1000.,color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,0.009,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,0.009,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVPHISTAR.pdf')

lcbgmstaru=unp.uarray([lcbgmstar,lcbgmstarerr])
lcbgphistaru=unp.uarray([lcbgphistar,lcbgphistarerr])
allmstaru=unp.uarray([allmstar,allmstarerr])
allphistaru=unp.uarray([allphistar,allphistarerr])
BEAREmstaru=unp.uarray([BEARE[:,4],BEARE[:,5]])
BEAREphistaru=unp.uarray([BEARE[:,6],BEARE[:,7]])
WILLMERmstaru=unp.uarray([WILLMER[:,4],WILLMER[:,5]])
WILLMERphistaru=unp.uarray([WILLMER[:,6],WILLMER[:,7]])
FABERmstaru=unp.uarray([FABER[:,4],FABER[:,5]])
FABERphistaru=unp.uarray([FABER[:,6],FABER[:,7]])
COOLmstaru=unp.uarray([COOL[:,4],COOL[:,5]])
COOLphistaru=unp.uarray([COOL[:,6],COOL[:,7]])
ZUCCAmstaru=unp.uarray([ZUCCA[:,4],ZUCCA[:,5]])
ZUCCAphistaru=unp.uarray([ZUCCA[:,6],ZUCCA[:,7]])
FRITZmstaru=unp.uarray([FRITZ[:,4],FRITZ[:,5]])
FRITZphistaru=unp.uarray([FRITZ[:,6],FRITZ[:,7]])

#*******************************
# Calculating Luminosity Density (-24<M<-18.5) Varying Alpha
#*******************************

lcbgju=lcbgphistaru/1000*np.power(10,(lcbgmstaru-5.48)/-2.5)*(sp.special.gammainc(lcbgalpha+2,np.power(10,(lcbgmstar+23.5)/2.5))-sp.special.gammainc(lcbgalpha+2,np.power(10,(lcbgmstar+18.5)/2.5)))
galju=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*(sp.special.gammainc(allalpha+2,np.power(10,(allmstar+23.5)/2.5))-sp.special.gammainc(2+allalpha,np.power(10,(allmstar+18.5)/2.5)))
BEAREju=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*(sp.special.gammainc(2+BEARE[:,2],np.power(10,(BEARE[:,4]+23.5)/2.5))-sp.special.gammainc(2+BEARE[:,2],np.power(10,(BEARE[:,4]+18.5)/2.5)))
WILLMERju=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*(sp.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+23.5)/2.5))-sp.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+18.5)/2.5)))
FABERju=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*(sp.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+23.5)/2.5))-sp.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+18.5)/2.5)))
COOLju=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*(sp.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+23.5)/2.5))-sp.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+18.5)/2.5)))
ZUCCAju=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*(sp.special.gammainc(2+ZUCCA[:,2],np.power(10,(ZUCCA[:,4]+23.5)/2.5))-sp.special.gammainc(2+ZUCCA[:,2],np.power(10,(ZUCCA[:,4]+18.5)/2.5)))
FRITZju=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*(sp.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+23.5)/2.5))-sp.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+18.5)/2.5)))
#BEAREALPHju=BEAREALPHphistaru/1000*np.power(10,(BEAREALPHmstaru-5.48)/-2.5)*(sp.special.gammainc(2+BEAREALPH[:,2],np.power(10,(BEAREALPH[:,4]+23.5)/2.5))-sp.special.gammainc(2+BEAREALPH[:,2],np.power(10,(BEAREALPH[:,4]+18.5)/2.5)))
#ZUCCAALPHju=ZUCCAALPHphistaru/1000*np.power(10,(ZUCCAALPHmstaru-5.48)/-2.5)*(sp.special.gammainc(2+ZUCCAALPH[:,2],np.power(10,(ZUCCAALPH[:,4]+23.5)/2.5))-sp.special.gammainc(2+ZUCCAALPH[:,2],np.power(10,(ZUCCAALPH[:,4]+18.5)/2.5)))

#*******************************
# Evolution of j_u
#*******************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(redshiftrange,unp.nominal_values(galju)[0:5]/100000000,yerr=unp.std_devs(galju)[0:5]/100000000,color='black',label='This Work',fmt='-o')
axes[0].errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/100000000,yerr=unp.std_devs(BEAREju)/100000000,color='slateblue',label='Beare, 2015',fmt='-d')
axes[0].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/100000000,yerr=unp.std_devs(WILLMERju)/100000000,color='yellow',label='Willmer, 2006',fmt='->')
axes[0].errorbar(FABER[:,0],unp.nominal_values(FABERju)/100000000,yerr=unp.std_devs(FABERju)/100000000,color='green', label='Faber, 2007',fmt='-<')
axes[0].errorbar(COOL[:,0],unp.nominal_values(COOLju)/100000000,yerr=unp.std_devs(COOLju)/100000000,color='red',label='Cool, 2012',fmt='-s')
axes[0].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/100000000,yerr=unp.std_devs(ZUCCAju)/100000000,color='grey',label='Zucca, 2009',fmt='-8')
axes[0].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/100000000,yerr=unp.std_devs(FRITZju)/100000000,color='purple', label='Fritz, 2014',fmt='-*')
axes[1].errorbar(redshiftrange,unp.nominal_values(lcbgju)[0:5]/100000000,yerr=unp.std_devs(lcbgju)[0:5]/100000000,color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_ylim([0,3.5])
axes[1].set_yticks([1,2,3])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].grid()
axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,3.2,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,3.2,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'$j_{B}$ (10$^{8}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.pdf')

#*******************************
# Talk Version
#*******************************

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/100000000,yerr=unp.std_devs(galju)[0:5]/100000000,color='black',label='This Study',fmt='-o')
axes[0].errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/100000000,yerr=unp.std_devs(BEAREju)/100000000,color='slateblue',label='Beare, 2015',fmt='-d')
axes[0].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/100000000,yerr=unp.std_devs(WILLMERju)/100000000,color='yellow',label='Willmer, 2006',fmt='->')
axes[0].errorbar(FABER[:,0],unp.nominal_values(FABERju)/100000000,yerr=unp.std_devs(FABERju)/100000000,color='green', label='Faber, 2007',fmt='-<')
axes[0].errorbar(COOL[:,0],unp.nominal_values(COOLju)/100000000,yerr=unp.std_devs(COOLju)/100000000,color='red',label='Cool, 2012',fmt='-s')
axes[0].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/100000000,yerr=unp.std_devs(ZUCCAju)/100000000,color='grey',label='Zucca, 2009',fmt='-8')
axes[0].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/100000000,yerr=unp.std_devs(FRITZju)/100000000,color='purple', label='Fritz, 2014',fmt='-*')
axes[1].errorbar(LFfittingparams[0:5,0],unp.nominal_values(lcbgju)[0:5]/100000000,yerr=unp.std_devs(lcbgju)[0:5]/100000000,color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_ylim([0,3.5])
axes[1].set_yticks([1,2,3])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
plt.subplots_adjust(wspace=0,left=0.1,right=0.95)
axes[0].grid()
axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,4,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,4,'LCBG',fontsize=12,ha='center',va='center')
axes[0].set_ylabel('$j_{B}$ (10$^{8}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',fontsize=14)
axes[0].set_xlabel('z',fontsize=14)
axes[1].set_xlabel('z',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.png')


#*******************************
# calculating j_u (ALL M) Varying Alpha
#*******************************

galju2=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*sp.special.gamma(2+allalpha)
BEAREju2=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*sp.special.gamma(2+BEARE[:,2])
WILLMERju2=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*sp.special.gamma(2+WILLMER[:,2])
FABERju2=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*sp.special.gamma(2+FABER[:,2])
COOLju2=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*sp.special.gamma(2+COOL[:,2])
ZUCCAju2=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*sp.special.gamma(2+ZUCCA[:,2])
FRITZju2=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*sp.special.gamma(2+FRITZ[:,2])
lcbgju2=lcbgphistaru/1000*np.power(10,(lcbgmstaru-5.48)/-2.5)*sp.special.gamma(2+lcbgalpha)

f,axes=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
axes.errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju2)[0:5]/100000000,yerr=unp.std_devs(galju2)[0:5]/100000000,color='black')
axes.errorbar(BEARE[:,0],unp.nominal_values(BEAREju2)/100000000,yerr=unp.std_devs(BEAREju2)/100000000,color='yellow')
axes.errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju2)/100000000,yerr=unp.std_devs(WILLMERju2)/100000000,color='green')
axes.errorbar(FABER[:,0],unp.nominal_values(FABERju2)/100000000,yerr=unp.std_devs(FABERju2)/100000000,color='gray')
axes.errorbar(COOL[:,0],unp.nominal_values(COOLju2)/100000000,yerr=unp.std_devs(COOLju2)/100000000,color='red')
axes.errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju2)/100000000,yerr=unp.std_devs(ZUCCAju2)/100000000,color='purple')
axes.errorbar(FRITZ[:,0],unp.nominal_values(FRITZju2)/100000000,yerr=unp.std_devs(FRITZju2)/100000000,color='slateblue')
axes.errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/10000000,yerr=unp.std_devs(galju)[0:5]/10000000,color='black',ls='--')
axes.errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/10000000,yerr=unp.std_devs(BEAREju)/10000000,color='yellow',ls='--')
axes.errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/10000000,yerr=unp.std_devs(WILLMERju)/10000000,color='green',ls='--')
axes.errorbar(FABER[:,0],unp.nominal_values(FABERju)/10000000,yerr=unp.std_devs(FABERju)/10000000,color='gray',ls='--')
axes.errorbar(COOL[:,0],unp.nominal_values(COOLju)/10000000,yerr=unp.std_devs(COOLju)/10000000,color='red',ls='--')
axes.errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/10000000,yerr=unp.std_devs(ZUCCAju)/10000000,color='purple',ls='--')
axes.errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/10000000,yerr=unp.std_devs(FRITZju)/10000000,color='slateblue',ls='--')

galju2=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*sp.special.gamma(2+allalpha)
BEAREju2=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*sp.special.gamma(2+BEARE[:,2])
WILLMERju2=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*sp.special.gamma(2+WILLMER[:,2])
FABERju2=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*sp.special.gamma(2+FABER[:,2])
COOLju2=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*sp.special.gamma(2+COOL[:,2])
ZUCCAju2=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*sp.special.gamma(2+ZUCCA[:,2])
FRITZju2=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*sp.special.gamma(2+FRITZ[:,2])
lcbgju2=lcbgphistaru/1000*np.power(10,(lcbgmstaru-5.48)/-2.5)*sp.special.gamma(lcbgalpha+2)
#BEAREALPHju2=BEAREALPHphistaru/1000*np.power(10,(BEAREALPHmstaru-5.48)/-2.5)*sp.special.gamma(2+BEAREALPH[:,2])
#ZUCCAALPHju2=ZUCCAALPHphistaru/1000*np.power(10,(ZUCCAALPHmstaru-5.48)/-2.5)*sp.special.gamma(2+ZUCCAALPH[:,2])


#*******************************
# calculating j_u (ALL M) (COMPARE ALL M TO M LIMIT)
#*******************************

f,axes=plt.subplots(nrows=2,ncols=4,sharex=True,sharey=True)
axes[0][0].errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju2)[0:5]/100000000,yerr=unp.std_devs(galju2)[0:5]/100000000,color='black',label='This Study')
axes[0][1].errorbar(BEARE[:,0],unp.nominal_values(BEAREju2)/100000000,yerr=unp.std_devs(BEAREju2)/100000000,color='yellow',label='BEARE (ALL ALPHA)')
axes[0][2].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju2)/100000000,yerr=unp.std_devs(WILLMERju2)/100000000,color='green',label='WILLMER, 2006 (ALL ALPHA)')
axes[0][3].errorbar(FABER[:,0],unp.nominal_values(FABERju2)/100000000,yerr=unp.std_devs(FABERju2)/100000000,color='gray',label='FABER (ALL ALPHA)')
axes[1][0].errorbar(COOL[:,0],unp.nominal_values(COOLju2)/100000000,yerr=unp.std_devs(COOLju2)/100000000,color='red',label='COOL (ALL ALPHA)')
axes[1][1].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju2)/100000000,yerr=unp.std_devs(ZUCCAju2)/100000000,color='purple',label='ZUCCA (ALL ALPHA)')
axes[1][2].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju2)/100000000,yerr=unp.std_devs(FRITZju2)/100000000,color='slateblue',label='FRITZ (ALL ALPHA)')
#axes[1][3].errorbar(BEARE[:,0],unp.nominal_values(BEAREALPHju2)/100000000,yerr=unp.std_devs(BEAREALPHju2)/100000000,color='red',label='BEARE (CONSTALPHA) (ALL ALPHA)')
axes[0][0].errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/100000000,yerr=unp.std_devs(galju)[0:5]/100000000,color='black',ls='--',label='This Study (18.5)')
axes[0][1].errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/100000000,yerr=unp.std_devs(BEAREju)/100000000,color='yellow',ls='--',label='BEARE (18.5)')
axes[0][2].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/100000000,yerr=unp.std_devs(WILLMERju)/100000000,color='green',ls='--',label='WILLMER (18.5)')
axes[0][3].errorbar(FABER[:,0],unp.nominal_values(FABERju)/100000000,yerr=unp.std_devs(FABERju)/100000000,color='gray',ls='--',label='FABER (18.5)')
axes[1][0].errorbar(COOL[:,0],unp.nominal_values(COOLju)/100000000,yerr=unp.std_devs(COOLju)/100000000,color='red',ls='--',label='COOL (18.5)')
axes[1][1].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/100000000,yerr=unp.std_devs(ZUCCAju)/100000000,color='purple',ls='--',label='ZUCCA (18.5)')
axes[1][2].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/100000000,yerr=unp.std_devs(FRITZju)/100000000,color='slateblue',ls='--',label='FRITZ (18.5)')
#axes[1][3].errorbar(BEARE[:,0],unp.nominal_values(BEAREALPHju)/100000000,yerr=unp.std_devs(BEAREALPHju)/100000000,color='yellow',ls='--',label='BEARE (18.5)')

axes[0][0].legend()
axes[0][1].legend()
axes[0][2].legend()
axes[1][0].legend()
axes[1][1].legend()
axes[1][2].legend()
axes[0][3].legend()
axes[1][3].legend()

#*******************************
#*******************************
#*******************************
#CALCULATING LCBG PHI IN SAME BINS AS WERK PAPER
#*******************************
#*******************************
#*******************************

#0.4<z<0.7

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_70.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()

lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])

fit=LevMarLSQFitter()

LCBGGAL4070=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True


LLumFunc2.mask[7:17]=True
LMBINAVE2.mask[7:17]=True
LLumFuncErr2.mask[7:17]=True

LCBG_FIT4070=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())


#Scipy Modelling

scipy_LCBG_4070_fit,scipy_LCBG_4070_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG4070ERRORS=np.array([np.sqrt(scipy_LCBG_4070_cov[0][0]),np.sqrt(scipy_LCBG_4070_cov[1][1]),np.sqrt(scipy_LCBG_4070_cov[2][2])])

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT4070(LCBG_Range)))

#0.7<z<1.0


filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_70_100.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()

lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])

fit=LevMarLSQFitter()

LCBGGAL70100=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc==0)[0]]=True


LLumFunc2.mask[6:17]=True
LMBINAVE2.mask[6:17]=True
LLumFuncErr2.mask[6:17]=True

LCBG_FIT70100=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT70100(LCBG_Range)))


#Scipy Modelling

scipy_LCBG_70100_fit,scipy_LCBG_70100_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG70100ERRORS=np.array([np.sqrt(scipy_LCBG_70100_cov[0][0]),np.sqrt(scipy_LCBG_70100_cov[1][1]),np.sqrt(scipy_LCBG_70100_cov[2][2])])

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT70100(LCBG_Range)))


GaldensityPhillips=np.array([sp.integrate.quad(LCBG_FIT4070,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT70100,-100,-18.5)[0]])

LCBGDENUP4070ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value+LCBG4070ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN4070ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value-LCBG4070ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP4070ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value+LCBG4070ERRORS[1]),-100,-18.5)[0]
LCBGDENDOWN4070ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value-LCBG4070ERRORS[1]),-100,-18.5)[0]
LCBGDENUP4070al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value+LCBG4070ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN4070al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value-LCBG4070ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP70100ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value+LCBG70100ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN70100ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value-LCBG70100ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP70100ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value+LCBG70100ERRORS[1]),-100,-18.5)[0]
LCBGDENDOWN70100ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value-LCBG70100ERRORS[1]),-100,-18.5)[0]
LCBGDENUP70100al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value+LCBG70100ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN70100al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value-LCBG70100ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]

LCBGDENUPphps=np.stack((LCBGDENUP4070ps,LCBGDENUP70100ps),axis=-1)
LCBGDENLOWphps=np.stack((LCBGDENDOWN4070ps,LCBGDENDOWN70100ps),axis=-1)
LCBGDENUPphms=np.stack((LCBGDENUP4070ms,LCBGDENUP70100ms),axis=-1)
LCBGDENLOWphms=np.stack((LCBGDENDOWN4070ms,LCBGDENDOWN70100ms),axis=-1)
LCBGDENUPphal=np.stack((LCBGDENUP4070al,LCBGDENUP70100al),axis=-1)
LCBGDENLOWphal=np.stack((LCBGDENDOWN4070al,LCBGDENDOWN70100al),axis=-1)

yupphps=abs(GaldensityPhillips-LCBGDENUPphps)
ylowphps=abs(GaldensityPhillips-LCBGDENLOWphps)
yupphms=abs(GaldensityPhillips-LCBGDENUPphms)
ylowphms=abs(GaldensityPhillips-LCBGDENLOWphms)
yupphal=abs(GaldensityPhillips-LCBGDENUPphal)
ylowphal=abs(GaldensityPhillips-LCBGDENLOWphal)

yupps=abs(lcbgdensity[0:5]-LCBGDENUPps[0:5])
ylowps=abs(lcbgdensity[0:5]-LCBGDENLOWps[0:5])
yupms=abs(lcbgdensity[0:5]-LCBGDENUPms[0:5])
ylowms=abs(lcbgdensity[0:5]-LCBGDENLOWms[0:5])
yupal=abs(lcbgdensity[0:5]-LCBGDENUPal[0:5])
ylowal=abs(lcbgdensity[0:5]-LCBGDENLOWal[0:5])

yup=np.array([yupps[0],yupps[1],yupps[2],yupal[3],yupal[4]])
ylow=np.array([ylowps[0],ylowps[1],ylowps[2],ylowal[3],ylowal[4]])

phz=np.array([0.55,0.85])

f,ax=plt.subplots(figsize=(8.5,8))
#ax.errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yup,ylow],xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='black',marker='s',ls='none',label='This Study')
ax.errorbar(LCBGfittingparams[0:5,0],unp.nominal_values(lcbg_density_err),yerr=unp.std_devs(lcbg_density_err),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='black',marker='s',ls='none',label='This Study')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
#ax.errorbar(phz,GaldensityPhillips,color='green',marker='o',xerr=0.15,yerr=[yupphps,ylowphps],ls='none',label='COSMOS in Phillips bin')
ax.set_xlim([0,1])
plt.xlabel('Redshift',fontsize=14)
plt.ylabel('N (Mpc$^{-3}$)',fontsize=14)
plt.legend(fontsize='small',loc='2')
plt.subplots_adjust(right=0.92)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityEvolution.png')


f,ax=plt.subplots(3,1,sharex=True,sharey=True)
ax[0].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupps,ylowps],color='red',marker='s',ls='none',label='$\phi^{*}$')
ax[0].errorbar(phz,GaldensityPhillips,yerr=[yupphps,ylowphps],color='red',marker='s',ls='none',label='$\phi^{*}$')
ax[1].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupms,ylowms],color='blue',marker='s',ls='none',label='$M^{*}$')
ax[1].errorbar(phz,GaldensityPhillips,yerr=[yupphms,ylowphms],color='blue',marker='s',ls='none',label='$M^{*}$')
ax[2].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupal,ylowal],color='green',marker='s',ls='none',label='$\alpha$')
ax[2].errorbar(phz,GaldensityPhillips,yerr=[yupphal,ylowphal],color='green',marker='s',ls='none',label='$\alpha$')
ax[0].set_xlim(0,1)

galdensity195=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-19.5)[0]])

lcbgdensity195=np.array([sp.integrate.quad(LCBG_FIT020,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT2040,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT4060,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT6080,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT80100,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT3080,-100,-19.5)[0]])

import uncertainties.umath as um
def line_scipy(x,slope,intercept):
    return (slope*x+intercept)
logoneplusz=np.linspace(np.log10(1),np.log10(3),30)
line_fit,line_cov=sp.optimize.curve_fit(line_scipy,np.log10(1+LCBGfittingparams[0:5,0]),unp.nominal_values(unp.log10(lcbg_density_err)),p0=[0.5,1],sigma=unp.std_devs(unp.log10(lcbg_density_err)))
plt.errorbar(np.log10(1+LCBGfittingparams[0:5,0]),unp.nominal_values(unp.log10(lcbg_density_err)),fmt='-o',yerr=unp.std_devs(unp.log10(lcbg_density_err)),label='points')
plt.plot(logoneplusz,line_fit[0]*logoneplusz+line_fit[1],label='Fit')
plt.legend(loc=2)
plt.xlim(0,0.3)
plt.ylim(-3.2,-1.95)
plt.xlabel('log$_{10}$(1+z)')
plt.ylabel('log$_{10}$(N) (mpc$^{-3}$)')
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/FitTrendNumberDensity.pdf')
plt.show()


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
	weightedfraction[i]=sum(WeightArray[LCBGS[np.where((zbest[LCBGS]>i/10.)&(zbest[LCBGS]<(i+1)/10.))[0]]])/sum(WeightArray[np.where((zbest>i/10.)&(zbest<(i+1)/10.))[0]])
redshift,galdensityeighteen,galdensityfifteen,lcbgdensity=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',skiprows=1,unpack=True)

lcbgdenu=unp.uarray([lcbgdensity[0:5],yup])



galup=np.array([gdupps[0],gdupps[1],gdupps[2],gdupal[3],gdupal[4]])
galup15=np.array([gdupps15[0],gdupal15[1],gdupal15[2],gdupal15[3],gdupal15[4]])

galdeneighteenu=unp.uarray([galdensityeighteen[0:5],galup])
galdenfifteenu=unp.uarray([galdensityfifteen[0:5],galup15])

fraceighteen=np.round(lcbgdensity/galdensityeighteen,4)
fracfifteen=np.round(lcbgdensity/galdensityfifteen,4)

#fraceighteenu=lcbgdenu/galdeneighteenu
#fracfifteenu=lcbgdenu/galdenfifteenu

fraceighteenu=lcbg_density_err/gal_densityeighteen_err
fracfifteenu=lcbg_density_err/gal_densityfifteen_err

x=ALLLCBGS[1][0:5]
x=x+0.1
FractionalError=fractionLCBGS[0:5]*np.sqrt((ALLLCBGS[0][0:5]/np.power(ALLLCBGS[0][0:5],2))+(ALLGALS[0][0:5]/np.power(ALLGALS[0][0:5],2)))
guzman=12.0/21.0*102.0/301.0
guzmanerror=12.0/21.0*102.0/301.0*np.sqrt(12.0/12.0**2+21.0/21.0**2+102.0/102.0**2+301.0/301.0**2)
tolerud=199/1744.
toleruderror=199./1744.*np.sqrt(199./199.**2+1744./1744.**2)

f,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1,1]},figsize=(8,10))

#f,axes=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1,1]})

rectseighteen=axes[1].bar(x,lcbgdensity[0:5]/galdensityeighteen[0:5],0.2,align='center',color='white',yerr=unp.std_devs(fraceighteenu),ecolor='black')

rects=axes[0].bar(x,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='white',ecolor='black')

rectsfifteen=axes[2].bar(x,lcbgdensity[0:5]/galdensityfifteen[0:5],0.2,align='center',color='white',yerr=unp.std_devs(fracfifteenu),ecolor='black')

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



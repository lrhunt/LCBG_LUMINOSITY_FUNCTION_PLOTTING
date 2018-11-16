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
import kcorrect
import kcorrect.utils as ut
from astropy.cosmology import WMAP9 as cosmo

def autolabel(rects,thecolor,row,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[row][col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)

@custom_model
def schechter_func(x,phistar=0.0056,mstar=-21,alpha=-1.03):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy(x,phistar,mstar,alpha):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def read_lf_file(lcbgfile,galfile):
	LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(lcbgfile,unpack=True,skiprows=1)
	MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(galfile,unpack=True,skiprows=1)
	LCBGPARAMS=np.stack((LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight))
	GALPARAMS=np.stack((MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight))
	with open(lcbgfile,'r') as lf:
		LSPECVALS=lf.readline().strip().split()
	with open(galfile,'r') as gf:
		SPECVALS=gf.readline().strip().split()
	for i in range(1,len(SPECVALS)):
		LSPECVALS[i]=float(LSPECVALS[i])
		SPECVALS[i]=float(SPECVALS[i])
	return LCBGPARAMS,GALPARAMS,LSPECVALS,SPECVALS

def ap_fitting(PARAMARRAY,MASKARRAY=None):
	if MASKARRAY==None:
		MASKARRAY=[]
	#Creating Mask Array
	LumFunc=np.ma.array(PARAMARRAY[2],mask=False)
	MBINAVE=np.ma.array(PARAMARRAY[1],mask=False)
	LumFuncErr=np.ma.array(PARAMARRAY[3],mask=False)
	#Masking zeros in Luminosity Function
	LumFunc.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	MBINAVE.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	LumFuncErr.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	#Masking errant points
	LumFunc.mask[MASKARRAY]=True
	MBINAVE.mask[MASKARRAY]=True
	LumFuncErr.mask[MASKARRAY]=True
	#Astropy Modelling
	LFFIT_init=schechter_func()
	LFFIT=fit(LFFIT_init,MBINAVE.compressed(),LumFunc.compressed(),weights=1/LumFuncErr.compressed())
	return LFFIT,LumFunc.mask

def sp_fitting(PARAMARRAY,MASKARRAY=None,INITARRAY=None):
	if INITARRAY==None:
		INITARRAY=[0.0056,-21,-1.03]
	if MASKARRAY==None:
		MASKARRAY=[]
	#Creating Mask Array
	LumFunc=np.ma.array(PARAMARRAY[2],mask=False)
	MBINAVE=np.ma.array(PARAMARRAY[1],mask=False)
	LumFuncErr=np.ma.array(PARAMARRAY[3],mask=False)
	#Masking zeros in Luminosity Function
	LumFunc.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	MBINAVE.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	LumFuncErr.mask[np.where(PARAMARRAY[2]==0)[0]]=True
	#Masking errant points
	LumFunc.mask[MASKARRAY]=True
	MBINAVE.mask[MASKARRAY]=True
	LumFuncErr.mask[MASKARRAY]=True
	#Astropy Modelling
	LFFIT,LFCOV=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE.compressed(),LumFunc.compressed(),p0=INITARRAY,sigma=LumFuncErr.compressed())
	return LFFIT,LFCOV


#def LF_FITTING(lcbgfile,galfile,LCBG_Range=np.linspace(-24,-15,30),init_vals=None,LCBGMASK=None,GALMASK=None):
#	LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(lcbgfile,unpack=True,skiprows=1)
#	MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(galfile,unpack=True,skiprows=1)
#	LCBGPARAMS=np.stack((LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight))
#	GALPARAMS=np.stack((MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight))
#	if init_vals==None:
#		init_vals=[0.0056,-21,-1.03]
#	if LCBGMASK==None:
#		LCBGMASK=[]
#	if GALMASK==None:
#		GALMASK=[]
#	print(init_vals)
#	with open(lcbgfile,'r') as lf:
#		LSPECVALS=lf.readline().strip().split()
#	with open(galfile,'r') as gf:
#		SPECVALS=gf.readline().strip().split()
#	for i in range(1,len(SPECVALS)):
#		LSPECVALS[i]=float(LSPECVALS[i])
#		SPECVALS[i]=float(SPECVALS[i])
#	LCBGFIT_init=schechter_func()
#	GALFIT_init=schechter_func()
#	#Creating Mask Arrays (to easily remove points that are generally off
#	LLumFunc2=np.ma.array(LCBGPARAMS[2],mask=False)
#	LMBINAVE2=np.ma.array(LCBGPARAMS[1],mask=False)
#	LLumFuncErr2=np.ma.array(LCBGPARAMS[3],mask=False)
#	LumFunc2=np.ma.array(GALPARAMS[2],mask=False)
#	MBINAVE2=np.ma.array(GALPARAMS[1],mask=False)
#	LumFuncErr2=np.ma.array(GALPARAMS[3],mask=False)
#	#Masking zeros in LCBG Luminosity Function
#	LLumFunc2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
#	LMBINAVE2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
#	LLumFuncErr2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
#	#Masking zeros in Luminosity Function
#	LumFunc2.mask[np.where(GALPARAMS[2]==0)[0]]=True
#	MBINAVE2.mask[np.where(GALPARAMS[2]==0)[0]]=True
#	LumFuncErr2.mask[np.where(GALPARAMS[2]==0)[0]]=True
#	#Masking errant points in LCBG Luminosity Function
#	LLumFunc2.mask[LCBGMASK]=True
#	LMBINAVE2.mask[LCBGMASK]=True
#	LLumFuncErr2.mask[LCBGMASK]=True
#	LumFunc2.mask[GALMASK]=True
#	MBINAVE2.mask[GALMASK]=True
#	LumFuncErr2.mask[GALMASK]=True
#	#Astropy Modelling
#	LCBG_FIT020=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
#	LUMFUNC_FIT020=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
#	#Scipy Modelling
#	scipy_LCBG_020_fit,scipy_LCBG_020_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
#	scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())
#	return LCBG_FIT020,LUMFUNC_FIT020,scipy_LCBG_020_fit,scipy_LCBG_020_cov,scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov,LCBGPARAMS,GALPARAMS,LSPECVALS,SPECVALS,LLumFunc2.mask,LumFunc2.mask



fit=LevMarLSQFitter()

#*************************************************
#*************************************************
#*******        z=0.01-0.2 Fitting        ********
#*************************************************
#*************************************************

LCBGPARAMS020,GALPARAMS020,LCBGSPECVALS020,GALSPECVALS020=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_0_20.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_0_20_FULL.txt')
LCBGFIT020,LCBGMASK020=ap_fitting(LCBGPARAMS020,MASKARRAY=[6,8])
GALFIT020,GALMASK020=ap_fitting(GALPARAMS020,MASKARRAY=[2,3])
spLCBGFIT020,spLCBGCOV020=sp_fitting(LCBGPARAMS020,MASKARRAY=[6,8])
spGALFIT020,spGALCOV020=sp_fitting(GALPARAMS020,MASKARRAY=[2,3])

#************************************************
#************************************************
#*******        z=0.2-0.4 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS2040,GALPARAMS2040,LCBGSPECVALS2040,GALSPECVALS2040=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_20_40.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_20_40_FULL.txt')
LCBGFIT2040,LCBGMASK2040=ap_fitting(LCBGPARAMS2040,MASKARRAY=[2,11])
GALFIT2040,GALMASK2040=ap_fitting(GALPARAMS2040,MASKARRAY=[1,2,14,15])
spLCBGFIT2040,spLCBGCOV2040=sp_fitting(LCBGPARAMS2040,MASKARRAY=[2,11])
spGALFIT2040,spGALCOV2040=sp_fitting(GALPARAMS2040,MASKARRAY=[1,2,14,15])

#************************************************
#************************************************
#*******        z=0.4-0.6 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS4060,GALPARAMS4060,LCBGSPECVALS4060,GALSPECVALS4060=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_60.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_40_60_FULL.txt')
LCBGFIT4060,LCBGMASK4060=ap_fitting(LCBGPARAMS4060)
GALFIT4060,GALMASK4060=ap_fitting(GALPARAMS4060,MASKARRAY=[11,12,13,14])
spLCBGFIT4060,spLCBGCOV4060=sp_fitting(LCBGPARAMS4060)
spGALFIT4060,spGALCOV4060=sp_fitting(GALPARAMS4060,MASKARRAY=[11,12,13,14])


#************************************************
#************************************************
#*******        z=0.6-0.8 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS6080,GALPARAMS6080,LCBGSPECVALS6080,GALSPECVALS6080=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_60_80.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_60_80_FULL.txt')
LCBGFIT6080,LCBGMASK6080=ap_fitting(LCBGPARAMS6080,MASKARRAY=[0,1,2,9])
GALFIT6080,GALMASK6080=ap_fitting(GALPARAMS6080,MASKARRAY=[9])
spLCBGFIT6080,spLCBGCOV6080=sp_fitting(LCBGPARAMS6080,MASKARRAY=[0,1,2,9])
spGALFIT6080,spGALCOV6080=sp_fitting(GALPARAMS6080,MASKARRAY=[9])

#************************************************
#************************************************
#*******        z=0.8-1.0 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS80100,GALPARAMS80100,LCBGSPECVALS80100,GALSPECVALS80100=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_80_100.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_80_100_FULL.txt')
LCBGFIT80100,LCBGMASK80100=ap_fitting(LCBGPARAMS80100,MASKARRAY=[7])
GALFIT80100,GALMASK80100=ap_fitting(GALPARAMS80100,MASKARRAY=[7])
spLCBGFIT80100,spLCBGCOV80100=sp_fitting(LCBGPARAMS80100,MASKARRAY=[7])
spGALFIT80100,spGALCOV80100=sp_fitting(GALPARAMS80100,MASKARRAY=[7])

LCBG_Range=np.linspace(-24,-15,30)
x=np.linspace(-23,-16,8)
xminor=np.linspace(-23.5,-15.5,9)
y=np.linspace(-7,-1,7)
yminor=np.linspace(-6.5,-1.5,6)

#**************************************************
#**************************************************
#*******        z=0.01-0.2 Plotting        ********
#**************************************************
#**************************************************

f,axes=plt.subplots(nrows=4,ncols=3,sharex=True,gridspec_kw={'height_ratios':[3,1,3,1]},figsize=(24,13.5))
LCBG020code=axes[0][0].errorbar(LCBGPARAMS020[1],np.log10(LCBGPARAMS020[2]),yerr=LCBGPARAMS020[4],xerr=[LCBGPARAMS020[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LCBGPARAMS020[1]],fmt=',',label='1/V$_{MAX}$ code')
GAL020code=axes[0][0].errorbar(GALPARAMS020[1],np.log10(GALPARAMS020[2]),yerr=GALPARAMS020[4],xerr=[GALPARAMS020[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS020[1]],fmt=',',label='1/V$_{MAX}$ code')
axes[0][0].errorbar(GALPARAMS020[1][GALMASK020],np.log10(GALPARAMS020[2][GALMASK020]),yerr=GALPARAMS020[4][GALMASK020],fmt='x')
axes[0][0].errorbar(LCBGPARAMS020[1][LCBGMASK020],np.log10(LCBGPARAMS020[2][LCBGMASK020]),yerr=LCBGPARAMS020[4][LCBGMASK020],fmt='x',label='1/V$_{MAX}$ code')
GALS020hist=axes[1][0].bar(GALPARAMS020[0],np.log10(GALPARAMS020[5]),float(GALSPECVALS020[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG020hist=axes[1][0].bar(LCBGPARAMS020[0],np.log10(LCBGPARAMS020[5]),float(LCBGSPECVALS020[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][0].plot(LCBG_Range,np.log10(LCBGFIT020(LCBG_Range))) 
axes[0][0].plot(LCBG_Range,np.log10(GALFIT020(LCBG_Range)))
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
autolabel(LCBG020hist,'black',1,0)
autolabel(GALS020hist,'black',1,0)

#*************************************************
#*************************************************
#*******        z=0.2-0.4 Plotting        ********
#*************************************************
#*************************************************

LCBG2040code=axes[0][1].errorbar(LCBGPARAMS2040[1],np.log10(LCBGPARAMS2040[2]),yerr=LCBGPARAMS2040[4],xerr=[LCBGPARAMS2040[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LCBGPARAMS2040[1]],fmt=',',label='1/V$_{MAX}$ code')
GAL2040code=axes[0][1].errorbar(GALPARAMS2040[1],np.log10(GALPARAMS2040[2]),yerr=GALPARAMS2040[4],xerr=[GALPARAMS2040[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS2040[1]],fmt=',',label='1/V$_{MAX}$ code')
axes[0][1].errorbar(GALPARAMS2040[1][GALMASK2040],np.log10(GALPARAMS2040[2][GALMASK2040]),yerr=GALPARAMS2040[4][GALMASK2040],fmt='x')
axes[0][1].errorbar(LCBGPARAMS2040[1][LCBGMASK2040],np.log10(LCBGPARAMS2040[2][LCBGMASK2040]),yerr=LCBGPARAMS2040[4][LCBGMASK2040],fmt='x',label='1/V$_{MAX}$ code')
GALS2040hist=axes[1][1].bar(GALPARAMS2040[0],np.log10(GALPARAMS2040[5]),float(GALSPECVALS2040[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG2040hist=axes[1][1].bar(LCBGPARAMS2040[0],np.log10(LCBGPARAMS2040[5]),float(LCBGSPECVALS2040[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][1].plot(LCBG_Range,np.log10(LCBGFIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(GALFIT2040(LCBG_Range)))
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].set_xticks(x)
axes[0][1].set_xticks(xminor,minor=True)
axes[0][1].set_yticks(y)
axes[0][1].set_yticks(yminor,minor=True)
axes[0][1].set_ylim([-7.5,-0.5])
axes[1][1].set_yticks([3,2,1,0])
axes[1][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][1].set_ylim([0,4])
autolabel(LCBG2040hist,'black',1,1)
autolabel(GALS2040hist,'black',1,1)

#*************************************************
#*************************************************
#*******        z=0.4-0.6 Plotting        ********
#*************************************************
#*************************************************

LCBG4060code=axes[0][2].errorbar(LCBGPARAMS4060[1],np.log10(LCBGPARAMS4060[2]),yerr=LCBGPARAMS4060[4],xerr=[LCBGPARAMS4060[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LCBGPARAMS4060[1]],fmt=',',label='1/V$_{MAX}$ code')
GAL4060code=axes[0][2].errorbar(GALPARAMS4060[1],np.log10(GALPARAMS4060[2]),yerr=GALPARAMS4060[4],xerr=[GALPARAMS4060[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS4060[1]],fmt=',',label='1/V$_{MAX}$ code')
axes[0][2].errorbar(GALPARAMS4060[1][GALMASK4060],np.log10(GALPARAMS4060[2][GALMASK4060]),yerr=GALPARAMS4060[4][GALMASK4060],fmt='x')
axes[0][2].errorbar(LCBGPARAMS4060[1][LCBGMASK4060],np.log10(LCBGPARAMS4060[2][LCBGMASK4060]),yerr=LCBGPARAMS4060[4][LCBGMASK4060],fmt='x',label='1/V$_{MAX}$ code')
GALS4060hist=axes[1][2].bar(GALPARAMS4060[0],np.log10(GALPARAMS4060[5]),float(GALSPECVALS4060[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG4060hist=axes[1][2].bar(LCBGPARAMS4060[0],np.log10(LCBGPARAMS4060[5]),float(LCBGSPECVALS4060[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][2].plot(LCBG_Range,np.log10(LCBGFIT4060(LCBG_Range)))
axes[0][2].plot(LCBG_Range,np.log10(GALFIT4060(LCBG_Range)))
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][2].set_xticks(x)
axes[0][2].set_xticks(xminor,minor=True)
axes[0][2].set_yticks(y)
axes[0][2].set_yticks(yminor,minor=True)
axes[0][2].set_ylim([-7.5,-0.5])
axes[1][2].set_yticks([3,2,1,0])
axes[1][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][2].set_ylim([0,4])
autolabel(LCBG4060hist,'black',1,2)
autolabel(GALS4060hist,'black',1,2)

#*************************************************
#*************************************************
#*******	z=0.6-0.8 Plotting	********
#*************************************************
#*************************************************

LCBG6080code=axes[2][0].errorbar(LCBGPARAMS6080[1],np.log10(LCBGPARAMS6080[2]),yerr=LCBGPARAMS6080[4],xerr=[LCBGPARAMS6080[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LCBGPARAMS6080[1]],fmt=',',label='1/V$_{MAX}$ code')
GAL6080code=axes[2][0].errorbar(GALPARAMS6080[1],np.log10(GALPARAMS6080[2]),yerr=GALPARAMS6080[4],xerr=[GALPARAMS6080[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS6080[1]],fmt=',',label='1/V$_{MAX}$ code')
axes[2][0].errorbar(GALPARAMS6080[1][GALMASK6080],np.log10(GALPARAMS6080[2][GALMASK6080]),yerr=GALPARAMS6080[4][GALMASK6080],fmt='x')
axes[2][0].errorbar(LCBGPARAMS6080[1][LCBGMASK6080],np.log10(LCBGPARAMS6080[2][LCBGMASK6080]),yerr=LCBGPARAMS6080[4][LCBGMASK6080],fmt='x',label='1/V$_{MAX}$ code')
GALS6080hist=axes[3][0].bar(GALPARAMS6080[0],np.log10(GALPARAMS6080[5]),float(GALSPECVALS6080[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG6080hist=axes[3][0].bar(LCBGPARAMS6080[0],np.log10(LCBGPARAMS6080[5]),float(LCBGSPECVALS6080[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][0].plot(LCBG_Range,np.log10(LCBGFIT6080(LCBG_Range)))
axes[2][0].plot(LCBG_Range,np.log10(GALFIT6080(LCBG_Range)))
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][0].set_xticks(x)
axes[2][0].set_xticks(xminor,minor=True)
axes[2][0].set_yticks(y)
axes[2][0].set_yticks(yminor,minor=True)
axes[2][0].set_ylim([-7.5,-0.5])
axes[3][0].set_yticks([3,2,1,0])
axes[3][0].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][0].set_ylim([0,4])
autolabel(LCBG6080hist,'black',3,0)
autolabel(GALS6080hist,'black',3,0)

#*************************************************
#*************************************************
#*******        z=0.8-1.0 Plotting        ********
#*************************************************
#*************************************************

LCBG80100code=axes[2][1].errorbar(LCBGPARAMS80100[1],np.log10(LCBGPARAMS80100[2]),yerr=LCBGPARAMS80100[4],xerr=[LCBGPARAMS80100[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LCBGPARAMS80100[1]],fmt=',',label='1/V$_{MAX}$ code')
GAL80100code=axes[2][1].errorbar(GALPARAMS80100[1],np.log10(GALPARAMS80100[2]),yerr=GALPARAMS80100[4],xerr=[GALPARAMS80100[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS80100[1]],fmt=',',label='1/V$_{MAX}$ code')
axes[2][1].errorbar(GALPARAMS80100[1][GALMASK80100],np.log10(GALPARAMS80100[2][GALMASK80100]),yerr=GALPARAMS80100[4][GALMASK80100],fmt='x')
axes[2][1].errorbar(LCBGPARAMS80100[1][LCBGMASK80100],np.log10(LCBGPARAMS80100[2][LCBGMASK80100]),yerr=LCBGPARAMS80100[4][LCBGMASK80100],fmt='x',label='1/V$_{MAX}$ code')
GALS80100hist=axes[3][1].bar(GALPARAMS80100[0],np.log10(GALPARAMS80100[5]),float(GALSPECVALS80100[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG80100hist=axes[3][1].bar(LCBGPARAMS80100[0],np.log10(LCBGPARAMS80100[5]),float(LCBGSPECVALS80100[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][1].plot(LCBG_Range,np.log10(LCBGFIT80100(LCBG_Range)))
axes[2][1].plot(LCBG_Range,np.log10(GALFIT80100(LCBG_Range)))
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][1].set_xticks(x)
axes[2][1].set_xticks(xminor,minor=True)
axes[2][1].set_yticks(y)
axes[2][1].set_yticks(yminor,minor=True)
axes[2][1].set_ylim([-7.5,-0.5])
axes[3][1].set_yticks([3,2,1,0])
axes[3][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][1].set_ylim([0,4])
autolabel(LCBG80100hist,'black',3,1)
autolabel(GALS80100hist,'black',3,1)

#*************************************************
#*************************************************
#*******        Overall formatting        ********
#*************************************************
#*************************************************

axes[0][1].set_yticklabels([])
axes[0][2].set_yticklabels([])
axes[1][1].set_yticklabels([])
axes[1][2].set_yticklabels([])
axes[2][1].set_yticklabels([])
axes[2][2].set_yticklabels([])
axes[3][1].set_yticklabels([])
axes[3][2].set_yticklabels([])
axes[0][0].text(-23.5,-1,'z=0.01-0.2',verticalalignment='center')
axes[0][1].text(-23.5,-1,'z=0.2-0.4',verticalalignment='center')
axes[0][2].text(-23.5,-1,'z=0.4-0.6',verticalalignment='center')
axes[2][0].text(-23.5,-1,'z=0.6-0.8',verticalalignment='center')
axes[2][2].text(-23.5,-1,'z=0.3-0.8',verticalalignment='center')
axes[2][1].text(-23.5,-1,'z=0.8-1.0',verticalalignment='center')
f.text(0.52,0.05,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center')
f.text(0.05,0.75,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical')
f.text(0.05,0.35,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical')
f.text(0.05,0.55,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical')
f.text(0.05,0.15,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical')

#*******************************************************************
#*******************************************************************
#*******	Old way of plotting (colors/fontsize        ********
#*******************************************************************
#*******************************************************************

#LCBGcode=axes[2][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,xerr=[LMBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-LMBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='blue')
#code=axes[2][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,xerr=[MBINAVE-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-MBINAVE],fmt=',',label='1/V$_{MAX}$ code',color='green')
#axes[2][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
#axes[2][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
#gals=axes[3][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
#lcbg=axes[3][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
#axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080(LCBG_Range)))
#axes[2][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT3080(LCBG_Range)))
#axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080u(LCBG_Range)),color='red')
#axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080l(LCBG_Range)),color='yellow')
#plt.subplots_adjust(hspace=0,wspace=0)
#axes[2][2].set_yticks(y)
#axes[2][2].set_yticks(yminor,minor=True)
#axes[2][2].set_ylim([-7.5,-0.5])
#axes[3][2].set_yticks([3,2,1,0])
#axes[3][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
#axes[3][2].set_ylim([0,4.0])


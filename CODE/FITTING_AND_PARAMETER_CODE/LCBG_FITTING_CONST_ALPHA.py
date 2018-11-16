import numpy as np
import astropy as ap
from astropy.visualization import astropy_mpl_style
import matplotlib as mpl
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

plt.style.use(astropy_mpl_style)

def autolabel(rects,thecolor,row,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[row][col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)

@custom_model
def schechter_func(x,phistar=0.0056,mstar=-21,alpha=-1.03):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

@custom_model
def schechter_func_onethree(x,phistar=0.0056,mstar=-21):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.3+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

@custom_model
def schechter_func_onetwo(x,phistar=0.0056,mstar=-21):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.2+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

@custom_model
def schechter_func_oneone(x,phistar=0.0056,mstar=-23):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.1+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


def schechter_func_scipy(x,phistar,mstar,alpha):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy_onethree(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.3+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy_onetwo(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.2+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy_oneone(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.1+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


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

def ap_fitting_c(PARAMARRAY,MASKARRAY=None,alph=None):
	if MASKARRAY==None:
		MASKARRAY=[]
	if alph==None:
		alph=1
	if alph==1:
		LFFIT_init=schechter_func_oneone()
	if alph==2:
		LFFIT_init=schechter_func_onetwo()
	if alph==3:
		LFFIT_init=schechter_func_onethree()
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

def sp_fitting_c(PARAMARRAY,MASKARRAY=None,INITARRAY=None,alph=None):
	if INITARRAY==None:
		INITARRAY=[0.0056,-21]
	if MASKARRAY==None:
		MASKARRAY=[]
	if alph==None:
		alph=1
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
	if alph==1:
		LFFIT,LFCOV=sp.optimize.curve_fit(schechter_func_scipy_oneone,MBINAVE.compressed(),LumFunc.compressed(),p0=INITARRAY,sigma=LumFuncErr.compressed())
	if alph==2:
		LFFIT,LFCOV=sp.optimize.curve_fit(schechter_func_scipy_onetwo,MBINAVE.compressed(),LumFunc.compressed(),p0=INITARRAY,sigma=LumFuncErr.compressed())
	if alph==3:
		LFFIT,LFCOV=sp.optimize.curve_fit(schechter_func_scipy_onethree,MBINAVE.compressed(),LumFunc.compressed(),p0=INITARRAY,sigma=LumFuncErr.compressed())
	return LFFIT,LFCOV

fit=LevMarLSQFitter()
LCBG_Range=np.linspace(-24,-15,30)
x=np.linspace(-23,-16,8)
xminor=np.linspace(-23.5,-15.5,9)
y=np.linspace(-7,-1,7)
yminor=np.linspace(-6.5,-1.5,6)

#*************************************************
#*************************************************
#*******        z=0.01-0.2 Fitting        ********
#*************************************************
#*************************************************

LCBGPARAMS020,GALPARAMS020,LCBGSPECVALS020,GALSPECVALS020=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_0_20.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_0_20_FULL.txt')
cLCBGFIT020,cLCBGMASK020=ap_fitting_c(LCBGPARAMS020,MASKARRAY=[6,8],alph=3)
cGALFIT020,cGALMASK020=ap_fitting_c(GALPARAMS020,MASKARRAY=[2,3],alph=1)
cspLCBGFIT020,cspLCBGCOV020=sp_fitting_c(LCBGPARAMS020,MASKARRAY=[6,8],alph=3)
cspGALFIT020,cspGALCOV020=sp_fitting_c(GALPARAMS020,MASKARRAY=[2,3],alph=1)
phistarcorr020,mstarcorr020=uncertainties.correlated_values(cspLCBGFIT020,cspLCBGCOV020)
galphistarcorr020,galmstarcorr020=uncertainties.correlated_values(cspGALFIT020,cspGALCOV020)

#****************************************************
#*******        z=0.01-0.2 LCBG Error        ********
#****************************************************

LCBGFIT020UP=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr020)+unp.std_devs(phistarcorr020),unp.nominal_values(mstarcorr020)+unp.std_devs(mstarcorr020))
LCBGFIT020DOWN=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr020)-unp.std_devs(phistarcorr020),unp.nominal_values(mstarcorr020)-unp.std_devs(mstarcorr020))

#***************************************************
#*******        z=0.01-0.2 GAL Error        ********
#***************************************************

GALFIT020UP=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr020)+unp.std_devs(galphistarcorr020),unp.nominal_values(galmstarcorr020)+unp.std_devs(galmstarcorr020))
GALFIT020DOWN=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr020)-unp.std_devs(galphistarcorr020),unp.nominal_values(galmstarcorr020)-unp.std_devs(galmstarcorr020))

#c1LCBGFIT020,cLCBGMASK020=ap_fitting_oneone(LCBGPARAMS020,MASKARRAY=[6,8])
#c2LCBGFIT020,cLCBGMASK020=ap_fitting_onetwo(LCBGPARAMS020,MASKARRAY=[6,8])
#c3LCBGFIT020,cLCBGMASK020=ap_fitting_onethree(LCBGPARAMS020,MASKARRAY=[6,8])

#************************************************
#************************************************
#*******        z=0.2-0.4 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS2040,GALPARAMS2040,LCBGSPECVALS2040,GALSPECVALS2040=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_20_40.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_20_40_FULL.txt')
cLCBGFIT2040,cLCBGMASK2040=ap_fitting_c(LCBGPARAMS2040,MASKARRAY=[2,11],alph=3)
cGALFIT2040,cGALMASK2040=ap_fitting_c(GALPARAMS2040,MASKARRAY=[1,2,14,15],alph=1)
cspLCBGFIT2040,cspLCBGCOV2040=sp_fitting_c(LCBGPARAMS2040,MASKARRAY=[2,11],alph=3)
cspGALFIT2040,cspGALCOV2040=sp_fitting_c(GALPARAMS2040,MASKARRAY=[1,2,14,15],alph=1)
phistarcorr2040,mstarcorr2040=uncertainties.correlated_values(cspLCBGFIT2040,cspLCBGCOV2040)
galphistarcorr2040,galmstarcorr2040=uncertainties.correlated_values(cspGALFIT2040,cspGALCOV2040)

#****************************************************
#*******        z=0.02-0.4 LCBG Error        ********
#****************************************************

LCBGFIT2040UP=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr2040)+unp.std_devs(phistarcorr2040),unp.nominal_values(mstarcorr2040)+unp.std_devs(mstarcorr2040))
LCBGFIT2040DOWN=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr2040)-unp.std_devs(phistarcorr2040),unp.nominal_values(mstarcorr2040)-unp.std_devs(mstarcorr2040))

#***************************************************
#*******        z=0.02-0.4 GAL Error        ********
#***************************************************

GALFIT2040UP=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr2040)+unp.std_devs(galphistarcorr2040),unp.nominal_values(galmstarcorr2040)+unp.std_devs(galmstarcorr2040))
GALFIT2040DOWN=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr2040)-unp.std_devs(galphistarcorr2040),unp.nominal_values(galmstarcorr2040)-unp.std_devs(galmstarcorr2040))


#c3LCBGFIT2040,cLCBGMASK2040=ap_fitting_onethree(LCBGPARAMS2040,MASKARRAY=[2,11])
#c2LCBGFIT2040,cLCBGMASK2040=ap_fitting_onetwo(LCBGPARAMS2040,MASKARRAY=[2,11])
#c1LCBGFIT2040,cLCBGMASK2040=ap_fitting_oneone(LCBGPARAMS2040,MASKARRAY=[2,11])

#************************************************
#************************************************
#*******        z=0.4-0.6 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS4060,GALPARAMS4060,LCBGSPECVALS4060,GALSPECVALS4060=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_60.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_40_60_FULL.txt')
cLCBGFIT4060,cLCBGMASK4060=ap_fitting_c(LCBGPARAMS4060,alph=3)
cGALFIT4060,cGALMASK4060=ap_fitting_c(GALPARAMS4060,MASKARRAY=[11,12,13,14],alph=1)
cspLCBGFIT4060,cspLCBGCOV4060=sp_fitting_c(LCBGPARAMS4060,alph=3)
cspGALFIT4060,cspGALCOV4060=sp_fitting_c(GALPARAMS4060,MASKARRAY=[11,12,13,14],alph=1)
phistarcorr4060,mstarcorr4060=uncertainties.correlated_values(cspLCBGFIT4060,cspLCBGCOV4060)
galphistarcorr4060,galmstarcorr4060=uncertainties.correlated_values(cspGALFIT4060,cspGALCOV4060)

#****************************************************
#*******        z=0.4-0.6 LCBG Error        ********
#****************************************************

LCBGFIT4060UP=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr4060)+unp.std_devs(phistarcorr4060),unp.nominal_values(mstarcorr4060)+unp.std_devs(mstarcorr4060))
LCBGFIT4060DOWN=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr4060)-unp.std_devs(phistarcorr4060),unp.nominal_values(mstarcorr4060)-unp.std_devs(mstarcorr4060))

#***************************************************
#*******        z=0.4-0.6 GAL Error        ********
#***************************************************

GALFIT4060UP=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr4060)+unp.std_devs(galphistarcorr4060),unp.nominal_values(galmstarcorr4060)+unp.std_devs(galmstarcorr4060))
GALFIT4060DOWN=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr4060)-unp.std_devs(galphistarcorr4060),unp.nominal_values(galmstarcorr4060)-unp.std_devs(galmstarcorr4060))

#c3LCBGFIT4060,cLCBGMASK4060=ap_fitting_onethree(LCBGPARAMS4060)
#c2LCBGFIT4060,cLCBGMASK4060=ap_fitting_onetwo(LCBGPARAMS4060)
#c1LCBGFIT4060,cLCBGMASK4060=ap_fitting_oneone(LCBGPARAMS4060)

#************************************************
#************************************************
#*******        z=0.6-0.8 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS6080,GALPARAMS6080,LCBGSPECVALS6080,GALSPECVALS6080=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_60_80.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_60_80_FULL.txt')
cLCBGFIT6080,cLCBGMASK6080=ap_fitting_c(LCBGPARAMS6080,MASKARRAY=[0,1,2,9],alph=3)
cGALFIT6080,cGALMASK6080=ap_fitting_c(GALPARAMS6080,MASKARRAY=[9],alph=1)
cspLCBGFIT6080,cspLCBGCOV6080=sp_fitting_c(LCBGPARAMS6080,MASKARRAY=[0,1,2,9],alph=3)
cspGALFIT6080,cspGALCOV6080=sp_fitting_c(GALPARAMS6080,MASKARRAY=[9],alph=1)
phistarcorr6080,mstarcorr6080=uncertainties.correlated_values(cspLCBGFIT6080,cspLCBGCOV6080)
galphistarcorr6080,galmstarcorr6080=uncertainties.correlated_values(cspGALFIT6080,cspGALCOV6080)

#****************************************************
#*******        z=0.6-0.8 LCBG Error        ********
#****************************************************

LCBGFIT6080UP=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr6080)+unp.std_devs(phistarcorr6080),unp.nominal_values(mstarcorr6080)+unp.std_devs(mstarcorr6080))
LCBGFIT6080DOWN=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr6080)-unp.std_devs(phistarcorr6080),unp.nominal_values(mstarcorr6080)-unp.std_devs(mstarcorr6080))

#***************************************************
#*******        z=0.6-0.8 GAL Error        ********
#***************************************************

GALFIT6080UP=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr6080)+unp.std_devs(galphistarcorr6080),unp.nominal_values(galmstarcorr6080)+unp.std_devs(galmstarcorr6080))
GALFIT6080DOWN=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr6080)-unp.std_devs(galphistarcorr6080),unp.nominal_values(galmstarcorr6080)-unp.std_devs(galmstarcorr6080))

#c3LCBGFIT6080,cLCBGMASK6080=ap_fitting_onethree(LCBGPARAMS6080,MASKARRAY=[0,1,2,9])
#c2LCBGFIT6080,cLCBGMASK6080=ap_fitting_onetwo(LCBGPARAMS6080,MASKARRAY=[0,1,2,9])
#c1LCBGFIT6080,cLCBGMASK6080=ap_fitting_oneone(LCBGPARAMS6080,MASKARRAY=[0,1,2,9])

#************************************************
#************************************************
#*******        z=0.8-1.0 Fitting        ********
#************************************************
#************************************************

LCBGPARAMS80100,GALPARAMS80100,LCBGSPECVALS80100,GALSPECVALS80100=read_lf_file('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_80_100.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_80_100_FULL.txt')
cLCBGFIT80100,cLCBGMASK80100=ap_fitting_c(LCBGPARAMS80100,MASKARRAY=[7],alph=3)
cGALFIT80100,cGALMASK80100=ap_fitting_c(GALPARAMS80100,MASKARRAY=[7],alph=1)
cspLCBGFIT80100,cspLCBGCOV80100=sp_fitting_c(LCBGPARAMS80100,MASKARRAY=[7],alph=3)
cspGALFIT80100,cspGALCOV80100=sp_fitting_c(GALPARAMS80100,MASKARRAY=[7],alph=1)
phistarcorr80100,mstarcorr80100=uncertainties.correlated_values(cspLCBGFIT80100,cspLCBGCOV80100)
galphistarcorr80100,galmstarcorr80100=uncertainties.correlated_values(cspGALFIT80100,cspGALCOV80100)

#****************************************************
#*******        z=0.8-1.0 LCBG Error        ********
#****************************************************

LCBGFIT80100UP=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr80100)+unp.std_devs(phistarcorr80100),unp.nominal_values(mstarcorr80100)+unp.std_devs(mstarcorr80100))
LCBGFIT80100DOWN=schechter_func_scipy_onethree(LCBG_Range,unp.nominal_values(phistarcorr80100)-unp.std_devs(phistarcorr80100),unp.nominal_values(mstarcorr80100)-unp.std_devs(mstarcorr80100))

#***************************************************
#*******        z=0.8-1.0 GAL Error        ********
#***************************************************

GALFIT80100UP=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr80100)+unp.std_devs(galphistarcorr80100),unp.nominal_values(galmstarcorr80100)+unp.std_devs(galmstarcorr80100))
GALFIT80100DOWN=schechter_func_scipy_oneone(LCBG_Range,unp.nominal_values(galphistarcorr80100)-unp.std_devs(galphistarcorr80100),unp.nominal_values(galmstarcorr80100)-unp.std_devs(galmstarcorr80100))

#c3LCBGFIT80100,cLCBGMASK80100=ap_fitting_onethree(LCBGPARAMS80100,MASKARRAY=[7])
#c2LCBGFIT80100,cLCBGMASK80100=ap_fitting_onetwo(LCBGPARAMS80100,MASKARRAY=[7])
#c1LCBGFIT80100,cLCBGMASK80100=ap_fitting_oneone(LCBGPARAMS80100,MASKARRAY=[7])

f=plt.figure(figsize=(24,13.5))

ax1=plt.subplot2grid((9,18),(0,0),colspan=6,rowspan=3)
ax2=plt.subplot2grid((9,18),(0,6),colspan=6,rowspan=3)
ax3=plt.subplot2grid((9,18),(0,12),colspan=6,rowspan=3)
ax4=plt.subplot2grid((9,18),(3,0),colspan=6)
ax5=plt.subplot2grid((9,18),(3,6),colspan=6)
ax6=plt.subplot2grid((9,18),(3,12),colspan=6)
ax7=plt.subplot2grid((9,18),(5,3),colspan=6,rowspan=3)
ax8=plt.subplot2grid((9,18),(5,9),colspan=6,rowspan=3)
ax9=plt.subplot2grid((9,18),(8,3),colspan=6)
ax10=plt.subplot2grid((9,18),(8,9),colspan=6)



axes=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,2],[ax9,ax10,2]])

#*************************************************
#*************************************************
#*******        z=0.01-0.2 Plotting        ********
#*************************************************
#*************************************************

LCBG020code=ax1.errorbar(LCBGPARAMS020[1],np.log10(LCBGPARAMS020[2]),yerr=LCBGPARAMS020[4],xerr=[abs(LCBGPARAMS020[0]-0.5*(LCBGPARAMS020[0][2]-LCBGPARAMS020[0][1])-LCBGPARAMS020[1]),abs(LCBGPARAMS020[0]+0.5*(LCBGPARAMS020[0][2]-LCBGPARAMS020[0][1])-LCBGPARAMS020[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL020code=ax1.errorbar(GALPARAMS020[1],np.log10(GALPARAMS020[2]),yerr=GALPARAMS020[4],xerr=[GALPARAMS020[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS020[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
ax1.errorbar(GALPARAMS020[1][cGALMASK020],np.log10(GALPARAMS020[2][cGALMASK020]),yerr=GALPARAMS020[4][cGALMASK020],fmt='x',color='black')
ax1.errorbar(LCBGPARAMS020[1][cLCBGMASK020],np.log10(LCBGPARAMS020[2][cLCBGMASK020]),yerr=LCBGPARAMS020[4][cLCBGMASK020],fmt='x',label='1/V$_{MAX}$ code',color='blue')
GALS020hist=ax4.bar(GALPARAMS020[0],np.log10(GALPARAMS020[5]),float(GALSPECVALS020[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG020hist=ax4.bar(LCBGPARAMS020[0],np.log10(LCBGPARAMS020[5]),float(LCBGSPECVALS020[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
ax1.plot(LCBG_Range,np.log10(cLCBGFIT020(LCBG_Range)),color='blue')
ax1.plot(LCBG_Range,np.log10(cGALFIT020(LCBG_Range)),color='black')
ax1.fill_between(LCBG_Range,np.log10(LCBGFIT020UP),np.log10(LCBGFIT020DOWN),alpha=0.25,color='blue')
ax1.fill_between(LCBG_Range,np.log10(GALFIT020UP),np.log10(GALFIT020DOWN),alpha=0.25,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
ax1.set_xticks(x)
ax1.set_xticks(xminor,minor=True)
ax1.set_yticks(y)
ax1.set_yticks(yminor,minor=True)
ax1.set_ylim([-7.5,-0.5])
ax4.set_yticks([3,2,1,0])
ax4.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax4.set_ylim([0,4])
autolabel(LCBG020hist,'black',1,0)
autolabel(GALS020hist,'black',1,0)

#*************************************************
#*************************************************
#*******        z=0.2-0.4 Plotting        ********
#*************************************************
#*************************************************

LCBG2040code=ax2.errorbar(LCBGPARAMS2040[1],np.log10(LCBGPARAMS2040[2]),yerr=LCBGPARAMS2040[4],xerr=[abs(LCBGPARAMS2040[0]-0.5*(LCBGPARAMS2040[0][2]-LCBGPARAMS2040[0][1])-LCBGPARAMS2040[1]),abs(LCBGPARAMS2040[0]+0.5*(LCBGPARAMS2040[0][2]-LCBGPARAMS2040[0][1])-LCBGPARAMS2040[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL2040code=ax2.errorbar(GALPARAMS2040[1],np.log10(GALPARAMS2040[2]),yerr=GALPARAMS2040[4],xerr=[GALPARAMS2040[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS2040[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
ax2.errorbar(GALPARAMS2040[1][cGALMASK2040],np.log10(GALPARAMS2040[2][cGALMASK2040]),yerr=GALPARAMS2040[4][cGALMASK2040],fmt='x',color='black')
ax2.errorbar(LCBGPARAMS2040[1][cLCBGMASK2040],np.log10(LCBGPARAMS2040[2][cLCBGMASK2040]),yerr=LCBGPARAMS2040[4][cLCBGMASK2040],fmt='x',label='1/V$_{MAX}$ code',color='blue')
GALS2040hist=ax5.bar(GALPARAMS2040[0],np.log10(GALPARAMS2040[5]),float(GALSPECVALS2040[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG2040hist=ax5.bar(LCBGPARAMS2040[0],np.log10(LCBGPARAMS2040[5]),float(LCBGSPECVALS2040[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
ax2.plot(LCBG_Range,np.log10(cLCBGFIT2040(LCBG_Range)),color='blue')
ax2.plot(LCBG_Range,np.log10(cGALFIT2040(LCBG_Range)),color='black')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
ax2.fill_between(LCBG_Range,np.log10(LCBGFIT2040UP),np.log10(LCBGFIT2040DOWN),alpha=0.25,color='blue')
ax2.fill_between(LCBG_Range,np.log10(GALFIT2040UP),np.log10(GALFIT2040DOWN),alpha=0.25,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
ax2.set_xticks(x)
ax2.set_xticks(xminor,minor=True)
ax2.set_yticks(y)
ax2.set_yticks(yminor,minor=True)
ax2.set_ylim([-7.5,-0.5])
ax5.set_yticks([3,2,1,0])
ax5.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax5.set_ylim([0,4])
autolabel(LCBG2040hist,'black',1,1)
autolabel(GALS2040hist,'black',1,1)

#*************************************************
#*************************************************
#*******        z=0.4-0.6 Plotting        ********
#*************************************************
#*************************************************

LCBG4060code=ax3.errorbar(LCBGPARAMS4060[1],np.log10(LCBGPARAMS4060[2]),yerr=LCBGPARAMS4060[4],xerr=[abs(LCBGPARAMS4060[0]-0.5*(LCBGPARAMS4060[0][2]-LCBGPARAMS4060[0][1])-LCBGPARAMS4060[1]),abs(LCBGPARAMS4060[0]+0.5*(LCBGPARAMS4060[0][2]-LCBGPARAMS4060[0][1])-LCBGPARAMS4060[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL4060code=ax3.errorbar(GALPARAMS4060[1],np.log10(GALPARAMS4060[2]),yerr=GALPARAMS4060[4],xerr=[GALPARAMS4060[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS4060[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
ax3.errorbar(GALPARAMS4060[1][cGALMASK4060],np.log10(GALPARAMS4060[2][cGALMASK4060]),yerr=GALPARAMS4060[4][cGALMASK4060],fmt='x',color='black')
ax3.errorbar(LCBGPARAMS4060[1][cLCBGMASK4060],np.log10(LCBGPARAMS4060[2][cLCBGMASK4060]),yerr=LCBGPARAMS4060[4][cLCBGMASK4060],fmt='x',label='1/V$_{MAX}$ code',color='blue')
GALS4060hist=ax6.bar(GALPARAMS4060[0],np.log10(GALPARAMS4060[5]),float(GALSPECVALS4060[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG4060hist=ax6.bar(LCBGPARAMS4060[0],np.log10(LCBGPARAMS4060[5]),float(LCBGSPECVALS4060[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
ax3.plot(LCBG_Range,np.log10(cLCBGFIT4060(LCBG_Range)),color='blue')
ax3.plot(LCBG_Range,np.log10(cGALFIT4060(LCBG_Range)),color='black')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
ax3.fill_between(LCBG_Range,np.log10(LCBGFIT4060UP),np.log10(LCBGFIT4060DOWN),alpha=0.25,color='blue')
ax3.fill_between(LCBG_Range,np.log10(GALFIT4060UP),np.log10(GALFIT4060DOWN),alpha=0.25,color='black')
plt.subplots_adjust(hspace=0,wspace=0)
ax3.set_xticks(x)
ax3.set_xticks(xminor,minor=True)
ax3.set_yticks(y)
ax3.set_yticks(yminor,minor=True)
ax3.set_ylim([-7.5,-0.5])
ax6.set_yticks([3,2,1,0])
ax6.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax6.set_ylim([0,4])
autolabel(LCBG4060hist,'black',1,2)
autolabel(GALS4060hist,'black',1,2)

#*************************************************
#*************************************************
#*******        z=0.6-0.8 Plotting        ********
#*************************************************
#*************************************************

LCBG6080code=ax7.errorbar(LCBGPARAMS6080[1],np.log10(LCBGPARAMS6080[2]),yerr=LCBGPARAMS6080[4],xerr=[abs(LCBGPARAMS6080[0]-0.5*(LCBGPARAMS6080[0][2]-LCBGPARAMS6080[0][1])-LCBGPARAMS6080[1]),abs(LCBGPARAMS6080[0]+0.5*(LCBGPARAMS6080[0][2]-LCBGPARAMS6080[0][1])-LCBGPARAMS6080[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL6080code=ax7.errorbar(GALPARAMS6080[1],np.log10(GALPARAMS6080[2]),yerr=GALPARAMS6080[4],xerr=[GALPARAMS6080[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS6080[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
ax7.errorbar(GALPARAMS6080[1][cGALMASK6080],np.log10(GALPARAMS6080[2][cGALMASK6080]),yerr=GALPARAMS6080[4][cGALMASK6080],fmt='x',color='black')
ax7.errorbar(LCBGPARAMS6080[1][cLCBGMASK6080],np.log10(LCBGPARAMS6080[2][cLCBGMASK6080]),yerr=LCBGPARAMS6080[4][cLCBGMASK6080],fmt='x',label='1/V$_{MAX}$ code',color='blue')
GALS6080hist=ax9.bar(GALPARAMS6080[0],np.log10(GALPARAMS6080[5]),float(GALSPECVALS6080[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG6080hist=ax9.bar(LCBGPARAMS6080[0],np.log10(LCBGPARAMS6080[5]),float(LCBGSPECVALS6080[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
ax7.plot(LCBG_Range,np.log10(cLCBGFIT6080(LCBG_Range)),color='blue')
ax7.plot(LCBG_Range,np.log10(cGALFIT6080(LCBG_Range)),color='black')
ax7.fill_between(LCBG_Range,np.log10(LCBGFIT6080UP),np.log10(LCBGFIT6080DOWN),alpha=0.25,color='blue')
ax7.fill_between(LCBG_Range,np.log10(GALFIT6080UP),np.log10(GALFIT6080DOWN),alpha=0.25,color='black')

#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
ax7.set_xticks(x)
ax7.set_xticks(xminor,minor=True)
ax7.set_yticks(y)
ax7.set_yticks(yminor,minor=True)
ax7.set_ylim([-7.5,-0.5])
ax9.set_yticks([3,2,1,0])
ax9.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax9.set_ylim([0,4])
autolabel(LCBG6080hist,'black',3,0)
autolabel(GALS6080hist,'black',3,0)

#*************************************************
#*************************************************
#*******        z=0.8-1.0 Plotting        ********
#*************************************************
#*************************************************

LCBG80100code=ax8.errorbar(LCBGPARAMS80100[1],np.log10(LCBGPARAMS80100[2]),yerr=LCBGPARAMS80100[4],xerr=[abs(LCBGPARAMS80100[0]-0.5*(LCBGPARAMS80100[0][2]-LCBGPARAMS80100[0][1])-LCBGPARAMS80100[1]),abs(LCBGPARAMS80100[0]+0.5*(LCBGPARAMS80100[0][2]-LCBGPARAMS80100[0][1])-LCBGPARAMS80100[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL80100code=ax8.errorbar(GALPARAMS80100[1],np.log10(GALPARAMS80100[2]),yerr=GALPARAMS80100[4],xerr=[GALPARAMS80100[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS80100[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
ax8.errorbar(GALPARAMS80100[1][cGALMASK80100],np.log10(GALPARAMS80100[2][cGALMASK80100]),yerr=GALPARAMS80100[4][cGALMASK80100],fmt='x',color='black')
ax8.errorbar(LCBGPARAMS80100[1][cLCBGMASK80100],np.log10(LCBGPARAMS80100[2][cLCBGMASK80100]),yerr=LCBGPARAMS80100[4][cLCBGMASK80100],fmt='x',label='1/V$_{MAX}$ code',color='blue')
GALS80100hist=ax10.bar(GALPARAMS80100[0],np.log10(GALPARAMS80100[5]),float(GALSPECVALS80100[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG80100hist=ax10.bar(LCBGPARAMS80100[0],np.log10(LCBGPARAMS80100[5]),float(LCBGSPECVALS80100[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
ax8.plot(LCBG_Range,np.log10(cLCBGFIT80100(LCBG_Range)),color='blue')
ax8.plot(LCBG_Range,np.log10(cGALFIT80100(LCBG_Range)),color='black')
ax8.fill_between(LCBG_Range,np.log10(LCBGFIT80100UP),np.log10(LCBGFIT80100DOWN),alpha=0.25,color='blue')
ax8.fill_between(LCBG_Range,np.log10(GALFIT80100UP),np.log10(GALFIT80100DOWN),alpha=0.25,color='black')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
#axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
ax8.set_xticks(x)
ax8.set_xticks(xminor,minor=True)
ax8.set_yticks(y)
ax8.set_yticks(yminor,minor=True)
ax8.set_ylim([-7.5,-0.5])
ax10.set_yticks([3,2,1,0])
ax10.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax10.set_ylim([0,4])
autolabel(LCBG80100hist,'black',3,1)
autolabel(GALS80100hist,'black',3,1)

ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
ax8.set_yticklabels([])
#axes[2][2].set_yticklabels([])
ax10.set_yticklabels([])
#axes[3][2].set_yticklabels([])



ax1.set_yticklabels([-7,-6,-5,-4,-3,-2,-1])
ax7.set_yticklabels([-7,-6,-5,-4,-3,-2,-1])
ax4.set_yticklabels([3,2,1,0])
ax9.set_yticklabels([3,2,1,0])
ax4.set_xticklabels(['',-23,-22,-21,-20,-19,-18,-17,-16])
ax5.set_xticklabels(['',-23,-22,-21,-20,-19,-18,-17,-16])
ax6.set_xticklabels(['',-23,-22,-21,-20,-19,-18,-17,-16])
ax9.set_xticklabels(['',-23,-22,-21,-20,-19,-18,-17,-16])
ax10.set_xticklabels(['',-23,-22,-21,-20,-19,-18,-17,-16])

ax1.set_xlim([-24,-15])
ax2.set_xlim([-24,-15])
ax3.set_xlim([-24,-15])
ax4.set_xlim([-24,-15])
ax5.set_xlim([-24,-15])
ax6.set_xlim([-24,-15])
ax7.set_xlim([-24,-15])
ax8.set_xlim([-24,-15])
ax9.set_xlim([-24,-15])
ax10.set_xlim([-24,-15])
              
ax1.text(-23.5,-1,'z=0.01-0.2',verticalalignment='center',fontsize=16)
ax2.text(-23.5,-1,'z=0.2-0.4',verticalalignment='center',fontsize=16)
ax3.text(-23.5,-1,'z=0.4-0.6',verticalalignment='center',fontsize=16)
ax7.text(-23.5,-1,'z=0.6-0.8',verticalalignment='center',fontsize=16)
ax8.text(-23.5,-1,'z=0.8-1',verticalalignment='center',fontsize=16)
f.text(0.52,0.03,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=18)
f.text(0.02,0.84,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=18)
f.text(0.18,0.32,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=18)
f.text(0.02,0.63,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=18)
f.text(0.18,0.11,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=18)

plt.subplots_adjust(left=0.04,right=1,bottom=0.06,top=1)

plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LCBG_LF_COMPARE.pdf')

def autolabel(rects,thecolor,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)


#**************************************************
#**************************************************
#*******        z=0.01-0.2 Plotting        ********
#**************************************************
#**************************************************
f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
LCBG020code=axes[0].errorbar(LCBGPARAMS020[1],np.log10(LCBGPARAMS020[2]),yerr=LCBGPARAMS020[4],xerr=[abs(LCBGPARAMS020[0]-0.5*(LCBGPARAMS020[0][2]-LCBGPARAMS020[0][1])-LCBGPARAMS020[1]),abs(LCBGPARAMS020[0]+0.5*(LCBGPARAMS020[0][2]-LCBGPARAMS020[0][1])-LCBGPARAMS020[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL020code=axes[0].errorbar(GALPARAMS020[1],np.log10(GALPARAMS020[2]),yerr=GALPARAMS020[4],xerr=[GALPARAMS020[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS020[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
axes[0].errorbar(GALPARAMS020[1][cGALMASK020],np.log10(GALPARAMS020[2][cGALMASK020]),yerr=GALPARAMS020[4][cGALMASK020],fmt='x',color='red')
axes[0].errorbar(LCBGPARAMS020[1][cLCBGMASK020],np.log10(LCBGPARAMS020[2][cLCBGMASK020]),yerr=LCBGPARAMS020[4][cLCBGMASK020],fmt='x',label='1/V$_{MAX}$ code',color='red')
GALS020hist=axes[1].bar(GALPARAMS020[0],np.log10(GALPARAMS020[5]),float(GALSPECVALS020[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG020hist=axes[1].bar(LCBGPARAMS020[0],np.log10(LCBGPARAMS020[5]),float(LCBGSPECVALS020[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0].plot(LCBG_Range,np.log10(cLCBGFIT020(LCBG_Range)),color='blue')
axes[0].plot(LCBG_Range,np.log10(cGALFIT020(LCBG_Range)),color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0].set_xticks(x)
axes[0].set_xticks(xminor,minor=True)
axes[0].set_yticks(y)
axes[0].set_yticks(yminor,minor=True)
axes[0].set_ylim([-7.5,-0.5])
axes[1].set_yticks([3,2,1,0])
axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1].set_ylim([0,4])
axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
f.suptitle('z=0.01-0.2',fontsize=18)
autolabel(LCBG020hist,'black',1)
autolabel(GALS020hist,'black',1)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC020.png')


#*************************************************
#*************************************************
#*******        z=0.2-0.4 Plotting        ********
#*************************************************
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
LCBG2040code=axes[0].errorbar(LCBGPARAMS2040[1],np.log10(LCBGPARAMS2040[2]),yerr=LCBGPARAMS2040[4],xerr=[abs(LCBGPARAMS2040[0]-0.5*(LCBGPARAMS2040[0][2]-LCBGPARAMS2040[0][1])-LCBGPARAMS2040[1]),abs(LCBGPARAMS2040[0]+0.5*(LCBGPARAMS2040[0][2]-LCBGPARAMS2040[0][1])-LCBGPARAMS2040[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL2040code=axes[0].errorbar(GALPARAMS2040[1],np.log10(GALPARAMS2040[2]),yerr=GALPARAMS2040[4],xerr=[GALPARAMS2040[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS2040[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
axes[0].errorbar(GALPARAMS2040[1][cGALMASK2040],np.log10(GALPARAMS2040[2][cGALMASK2040]),yerr=GALPARAMS2040[4][cGALMASK2040],fmt='x',color='red')
axes[0].errorbar(LCBGPARAMS2040[1][cLCBGMASK2040],np.log10(LCBGPARAMS2040[2][cLCBGMASK2040]),yerr=LCBGPARAMS2040[4][cLCBGMASK2040],fmt='x',label='1/V$_{MAX}$ code',color='red')
GALS2040hist=axes[1].bar(GALPARAMS2040[0],np.log10(GALPARAMS2040[5]),float(GALSPECVALS2040[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG2040hist=axes[1].bar(LCBGPARAMS2040[0],np.log10(LCBGPARAMS2040[5]),float(LCBGSPECVALS2040[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0].plot(LCBG_Range,np.log10(cLCBGFIT2040(LCBG_Range)),color='blue')
axes[0].plot(LCBG_Range,np.log10(cGALFIT2040(LCBG_Range)),color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0].set_xticks(x)
axes[0].set_xticks(xminor,minor=True)
axes[0].set_yticks(y)
axes[0].set_yticks(yminor,minor=True)
axes[0].set_ylim([-7.5,-0.5])
axes[1].set_yticks([3,2,1,0])
axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1].set_ylim([0,4])
axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
f.suptitle('z=0.2-0.4',fontsize=18)
autolabel(LCBG2040hist,'black',1)
autolabel(GALS2040hist,'black',1)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC2040.png')

#*************************************************
#*************************************************
#*******        z=0.4-0.6 Plotting        ********
#*************************************************
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
LCBG4060code=axes[0].errorbar(LCBGPARAMS4060[1],np.log10(LCBGPARAMS4060[2]),yerr=LCBGPARAMS4060[4],xerr=[abs(LCBGPARAMS4060[0]-0.5*(LCBGPARAMS4060[0][2]-LCBGPARAMS4060[0][1])-LCBGPARAMS4060[1]),abs(LCBGPARAMS4060[0]+0.5*(LCBGPARAMS4060[0][2]-LCBGPARAMS4060[0][1])-LCBGPARAMS4060[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL4060code=axes[0].errorbar(GALPARAMS4060[1],np.log10(GALPARAMS4060[2]),yerr=GALPARAMS4060[4],xerr=[GALPARAMS4060[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS4060[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
axes[0].errorbar(GALPARAMS4060[1][cGALMASK4060],np.log10(GALPARAMS4060[2][cGALMASK4060]),yerr=GALPARAMS4060[4][cGALMASK4060],fmt='x',color='red')
axes[0].errorbar(LCBGPARAMS4060[1][cLCBGMASK4060],np.log10(LCBGPARAMS4060[2][cLCBGMASK4060]),yerr=LCBGPARAMS4060[4][cLCBGMASK4060],fmt='x',label='1/V$_{MAX}$ code',color='red')
GALS4060hist=axes[1].bar(GALPARAMS4060[0],np.log10(GALPARAMS4060[5]),float(GALSPECVALS4060[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG4060hist=axes[1].bar(LCBGPARAMS4060[0],np.log10(LCBGPARAMS4060[5]),float(LCBGSPECVALS4060[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0].plot(LCBG_Range,np.log10(cLCBGFIT4060(LCBG_Range)),color='blue')
axes[0].plot(LCBG_Range,np.log10(cGALFIT4060(LCBG_Range)),color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0].set_xticks(x)
axes[0].set_xticks(xminor,minor=True)
axes[0].set_yticks(y)
axes[0].set_yticks(yminor,minor=True)
axes[0].set_ylim([-7.5,-0.5])
axes[1].set_yticks([3,2,1,0])
axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1].set_ylim([0,4])
axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
f.suptitle('z=0.4-0.6',fontsize=18)
autolabel(LCBG4060hist,'black',1)
autolabel(GALS4060hist,'black',1)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC4060.png')

#*************************************************
#*************************************************
#*******	z=0.6-0.8 Plotting	********
#*************************************************
#*************************************************
f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
LCBG6080code=axes[0].errorbar(LCBGPARAMS6080[1],np.log10(LCBGPARAMS6080[2]),yerr=LCBGPARAMS6080[4],xerr=[abs(LCBGPARAMS6080[0]-0.5*(LCBGPARAMS6080[0][2]-LCBGPARAMS6080[0][1])-LCBGPARAMS6080[1]),abs(LCBGPARAMS6080[0]+0.5*(LCBGPARAMS6080[0][2]-LCBGPARAMS6080[0][1])-LCBGPARAMS6080[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL6080code=axes[0].errorbar(GALPARAMS6080[1],np.log10(GALPARAMS6080[2]),yerr=GALPARAMS6080[4],xerr=[GALPARAMS6080[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS6080[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
axes[0].errorbar(GALPARAMS6080[1][cGALMASK6080],np.log10(GALPARAMS6080[2][cGALMASK6080]),yerr=GALPARAMS6080[4][cGALMASK6080],fmt='x',color='red')
axes[0].errorbar(LCBGPARAMS6080[1][cLCBGMASK6080],np.log10(LCBGPARAMS6080[2][cLCBGMASK6080]),yerr=LCBGPARAMS6080[4][cLCBGMASK6080],fmt='x',label='1/V$_{MAX}$ code',color='red')
GALS6080hist=axes[1].bar(GALPARAMS6080[0],np.log10(GALPARAMS6080[5]),float(GALSPECVALS6080[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG6080hist=axes[1].bar(LCBGPARAMS6080[0],np.log10(LCBGPARAMS6080[5]),float(LCBGSPECVALS6080[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0].plot(LCBG_Range,np.log10(cLCBGFIT6080(LCBG_Range)),color='blue')
axes[0].plot(LCBG_Range,np.log10(cGALFIT6080(LCBG_Range)),color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0].set_xticks(x)
axes[0].set_xticks(xminor,minor=True)
axes[0].set_yticks(y)
axes[0].set_yticks(yminor,minor=True)
axes[0].set_ylim([-7.5,-0.5])
axes[1].set_yticks([3,2,1,0])
axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1].set_ylim([0,4])
axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
f.suptitle('z=0.6-0.8',fontsize=18)
autolabel(LCBG6080hist,'black',1)
autolabel(GALS6080hist,'black',1)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC6080.png')

#*************************************************
#*************************************************
#*******        z=0.8-1.0 Plotting        ********
#*************************************************
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]},figsize=(8,8))
LCBG80100code=axes[0].errorbar(LCBGPARAMS80100[1],np.log10(LCBGPARAMS80100[2]),yerr=LCBGPARAMS80100[4],xerr=[abs(LCBGPARAMS80100[0]-0.5*(LCBGPARAMS80100[0][2]-LCBGPARAMS80100[0][1])-LCBGPARAMS80100[1]),abs(LCBGPARAMS80100[0]+0.5*(LCBGPARAMS80100[0][2]-LCBGPARAMS80100[0][1])-LCBGPARAMS80100[1])],fmt=',',label='1/V$_{MAX}$ code',color='blue')
GAL80100code=axes[0].errorbar(GALPARAMS80100[1],np.log10(GALPARAMS80100[2]),yerr=GALPARAMS80100[4],xerr=[GALPARAMS80100[1]-np.linspace(-24,-15.5,18),np.linspace(-23.5,-15,18)-GALPARAMS80100[1]],fmt=',',label='1/V$_{MAX}$ code',color='black')
axes[0].errorbar(GALPARAMS80100[1][cGALMASK80100],np.log10(GALPARAMS80100[2][cGALMASK80100]),yerr=GALPARAMS80100[4][cGALMASK80100],fmt='x',color='red')
axes[0].errorbar(LCBGPARAMS80100[1][cLCBGMASK80100],np.log10(LCBGPARAMS80100[2][cLCBGMASK80100]),yerr=LCBGPARAMS80100[4][cLCBGMASK80100],fmt='x',label='1/V$_{MAX}$ code',color='red')
GALS80100hist=axes[1].bar(GALPARAMS80100[0],np.log10(GALPARAMS80100[5]),float(GALSPECVALS80100[4]),align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
LCBG80100hist=axes[1].bar(LCBGPARAMS80100[0],np.log10(LCBGPARAMS80100[5]),float(LCBGSPECVALS80100[4]),align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0].plot(LCBG_Range,np.log10(cLCBGFIT80100(LCBG_Range)),color='blue')
axes[0].plot(LCBG_Range,np.log10(cGALFIT80100(LCBG_Range)),color='black')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0].set_xticks(x)
axes[0].set_xticks(xminor,minor=True)
axes[0].set_yticks(y)
axes[0].set_yticks(yminor,minor=True)
axes[0].set_ylim([-7.5,-0.5])
axes[1].set_yticks([3,2,1,0])
axes[1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1].set_ylim([0,4])
axes[1].set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',fontsize=16)
axes[0].set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=16)
axes[1].set_ylabel('Log$_{10}$(N)',fontsize=16)
f.suptitle('z=0.8-1.0',fontsize=18)
autolabel(LCBG80100hist,'black',1)
autolabel(GALS80100hist,'black',1)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMFUNC80100.png')

#*************************************************
#*************************************************
#*******        Overall formatting        ********
#*************************************************
#*************************************************

#axes[0][1].set_yticklabels([])
#axes[0][2].set_yticklabels([])
#axes[1][1].set_yticklabels([])
#axes[1][2].set_yticklabels([])
#axes[2][1].set_yticklabels([])
#axes[2][2].set_yticklabels([])
#axes[3][1].set_yticklabels([])
#axes[3][2].set_yticklabels([])
#axes[0][0].text(-23.5,-1,'z=0.01-0.2',verticalalignment='center')
#axes[0][1].text(-23.5,-1,'z=0.2-0.4',verticalalignment='center')
#axes[0][2].text(-23.5,-1,'z=0.4-0.6',verticalalignment='center')
#axes[2][0].text(-23.5,-1,'z=0.6-0.8',verticalalignment='center')
#axes[2][2].text(-23.5,-1,'z=0.3-0.8',verticalalignment='center')
#axes[2][1].text(-23.5,-1,'z=0.8-1.0',verticalalignment='center')
#f.text(0.52,0.05,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center')
#f.text(0.05,0.75,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical')
#f.text(0.05,0.35,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical')
#f.text(0.05,0.55,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical')
#f.text(0.05,0.15,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical')

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
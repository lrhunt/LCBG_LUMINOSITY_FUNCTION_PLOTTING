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

def LF_FITTING(lcbgfile,galfile,LCBG_Range=np.linspace(-24,-15,30),init_vals=None,LCBGMASK=None,GALMASK=None):
	LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(lcbgfile,unpack=True,skiprows=1)
	MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(galfile,unpack=True,skiprows=1)
	LCBGPARAMS=np.stack((LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight))
	GALPARAMS=np.stack((MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight))
	if init_vals==None:
		init_vals=[0.0056,-21,-1.03]
	if LCBGMASK==None:
		LCBGMASK=[]
	if GALMASK==None:
		GALMASK=[]
	print(init_vals)
	with open(lcbgfile,'r') as lf:
		LSPECVALS=lf.readline().strip().split()
	with open(galfile,'r') as gf:
		SPECVALS=gf.readline().strip().split()
	for i in range(1,len(SPECVALS)):
		LSPECVALS[i]=float(LSPECVALS[i])
		SPECVALS[i]=float(SPECVALS[i])
	LCBGFIT_init=schechter_func()
	GALFIT_init=schechter_func()
	#Creating Mask Arrays (to easily remove points that are generally off
	LLumFunc2=np.ma.array(LCBGPARAMS[2],mask=False)
	LMBINAVE2=np.ma.array(LCBGPARAMS[1],mask=False)
	LLumFuncErr2=np.ma.array(LCBGPARAMS[3],mask=False)
	LumFunc2=np.ma.array(GALPARAMS[2],mask=False)
	MBINAVE2=np.ma.array(GALPARAMS[1],mask=False)
	LumFuncErr2=np.ma.array(GALPARAMS[3],mask=False)
	#Masking zeros in LCBG Luminosity Function
	LLumFunc2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
	LMBINAVE2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
	LLumFuncErr2.mask[np.where(LCBGPARAMS[2]==0)[0]]=True
	#Masking zeros in Luminosity Function
	LumFunc2.mask[np.where(GALPARAMS[2]==0)[0]]=True
	MBINAVE2.mask[np.where(GALPARAMS[2]==0)[0]]=True
	LumFuncErr2.mask[np.where(GALPARAMS[2]==0)[0]]=True
	#Masking errant points in LCBG Luminosity Function
	LLumFunc2.mask[LCBGMASK]=True
	LMBINAVE2.mask[LCBGMASK]=True
	LLumFuncErr2.mask[LCBGMASK]=True
	LumFunc2.mask[GALMASK]=True
	MBINAVE2.mask[GALMASK]=True
	LumFuncErr2.mask[GALMASK]=True
	#Astropy Modelling
	LCBG_FIT020=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
	LUMFUNC_FIT020=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
	#Scipy Modelling
	scipy_LCBG_020_fit,scipy_LCBG_020_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
	scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())
	return LCBG_FIT020,LUMFUNC_FIT020,scipy_LCBG_020_fit,scipy_LCBG_020_cov,scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov,LCBGPARAMS,GALPARAMS,LSPECVALS,SPECVALS,LLumFunc2.mask,LumFunc2.mask

@uncertainties.wrap
def integratethings(ps,ms,al,limit):
    def integrand(X):
        return (0.4*np.log(10)*ps)*(10**(0.4*(al+1)*(ms-X)))*(np.e**(-np.power(10,0.4*(ms-X))))
    integral,abserr=sp.integrate.quad(integrand,-100,limit)
    return integral

fit=LevMarLSQFitter()
redshiftrange=np.array([0.1,0.3,0.5,0.7,0.9])

LCBGFIT020,GALFIT020,spLCBGFIT020,spLCBGCOV020,spGALFIT020,spGALCOV020,LCBGPARAMS020,GALPARAMS020,LCBGSPECVALS020,GALSPECVALS020,LCBGMASK020,GALMASK020=LF_FITTING('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_0_20.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_0_20_FULL.txt',LCBGMASK=[6,8],GALMASK=[2,3])
LCBGFIT2040,GALFIT2040,spLCBGFIT2040,spLCBGCOV2040,spGALFIT2040,spGALCOV2040,LCBGPARAMS2040,GALPARAMS2040,LCBGSPECVALS2040,GALSPECVALS2040,LCBGMASK2040,GALMASK2040=LF_FITTING('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_20_40.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_20_40_FULL.txt',LCBGMASK=[2,11],GALMASK=[1,2,14,15])
LCBGFIT4060,GALFIT4060,spLCBGFIT4060,spLCBGCOV4060,spGALFIT4060,spGALCOV4060,LCBGPARAMS4060,GALPARAMS4060,LCBGSPECVALS4060,GALSPECVALS4060,LCBGMASK4060,GALMASK4060=LF_FITTING('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_60.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_40_60_FULL.txt',GALMASK=[11,12,13,14])
LCBGFIT6080,GALFIT6080,spLCBGFIT6080,spLCBGCOV6080,spGALFIT6080,spGALCOV6080,LCBGPARAMS6080,GALPARAMS6080,LCBGSPECVALS6080,GALSPECVALS6080,LCBGMASK6080,GALMASK6080=LF_FITTING('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_60_80.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_60_80_FULL.txt',LCBGMASK=[0,1,2,9],GALMASK=[9])
LCBGFIT80100,GALFIT80100,spLCBGFIT80100,spLCBGCOV80100,spGALFIT80100,spGALCOV80100,LCBGPARAMS80100,GALPARAMS80100,LCBGSPECVALS80100,GALSPECVALS80100,LCBGMASK80100,GALMASK80100=LF_FITTING('/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_80_100.txt','/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_80_100_FULL.txt',LCBGMASK=[7],GALMASK=[7])

#********************************************************************************************
#********************************************************************************************
#*******        Using uncertainties package to define LF parameters in z bins        ********
#********************************************************************************************
#********************************************************************************************

phistarcorr020,mstarcorr020,alphacorr020=uncertainties.correlated_values(spLCBGFIT020,spLCBGCOV020)
phistarcorr2040,mstarcorr2040,alphacorr2040=uncertainties.correlated_values(spLCBGFIT2040,spLCBGCOV2040)
phistarcorr4060,mstarcorr4060,alphacorr4060=uncertainties.correlated_values(spLCBGFIT4060,spLCBGCOV4060)
phistarcorr6080,mstarcorr6080,alphacorr6080=uncertainties.correlated_values(spLCBGFIT6080,spLCBGCOV6080)
phistarcorr80100,mstarcorr80100,alphacorr80100=uncertainties.correlated_values(spLCBGFIT80100,spLCBGCOV80100)
galphistarcorr020,galmstarcorr020,galalphacorr020=uncertainties.correlated_values(spGALFIT020,spGALCOV020)
galphistarcorr2040,galmstarcorr2040,galalphacorr2040=uncertainties.correlated_values(spGALFIT2040,spGALCOV2040)
galphistarcorr4060,galmstarcorr4060,galalphacorr4060=uncertainties.correlated_values(spGALFIT4060,spGALCOV4060)
galphistarcorr6080,galmstarcorr6080,galalphacorr6080=uncertainties.correlated_values(spGALFIT6080,spGALCOV6080)
galphistarcorr80100,galmstarcorr80100,galalphacorr80100=uncertainties.correlated_values(spGALFIT80100,spGALCOV80100)

#******************************************************
#******************************************************
#*******        Making evolution arrays        ********
#******************************************************
#******************************************************

lcbg_phistar_evo=np.array([phistarcorr020,phistarcorr2040,phistarcorr4060,phistarcorr6080,phistarcorr80100])
lcbg_mstar_evo=np.array([mstarcorr020,mstarcorr2040,mstarcorr4060,mstarcorr6080,mstarcorr80100])
lcbg_alpha_evo=np.array([alphacorr020,alphacorr2040,alphacorr4060,alphacorr6080,alphacorr80100])
gal_phistar_evo=np.array([galphistarcorr020,galphistarcorr2040,galphistarcorr4060,galphistarcorr6080,galphistarcorr80100])
gal_mstar_evo=np.array([galmstarcorr020,galmstarcorr2040,galmstarcorr4060,galmstarcorr6080,galmstarcorr80100])
gal_alpha_evo=np.array([galalphacorr020,galalphacorr2040,galalphacorr4060,galalphacorr6080,galalphacorr80100])

#*****************************************************************************************
#*****************************************************************************************
#*******        Calculating number density through uncertainty integration        ********
#*****************************************************************************************
#*****************************************************************************************

lcbg_density_err=np.array([integratethings(phistarcorr020,mstarcorr020,alphacorr020,-18.5),integratethings(phistarcorr2040,mstarcorr2040,alphacorr2040,-18.5),integratethings(phistarcorr4060,mstarcorr4060,alphacorr4060,-18.5),integratethings(phistarcorr6080,mstarcorr6080,alphacorr6080,-18.5),integratethings(phistarcorr80100,mstarcorr80100,alphacorr80100,-18.5)])
gal_densityeighteen_err=np.array([integratethings(galphistarcorr020,galmstarcorr020,galalphacorr020,-18.5),integratethings(galphistarcorr2040,galmstarcorr2040,galalphacorr2040,-18.5),integratethings(galphistarcorr4060,galmstarcorr4060,galalphacorr4060,-18.5),integratethings(galphistarcorr6080,galmstarcorr6080,galalphacorr6080,-18.5),integratethings(galphistarcorr80100,galmstarcorr80100,galalphacorr80100,-18.5)])
gal_densityfifteen_err=np.array([integratethings(galphistarcorr020,galmstarcorr020,galalphacorr020,-15),integratethings(galphistarcorr2040,galmstarcorr2040,galalphacorr2040,-15),integratethings(galphistarcorr4060,galmstarcorr4060,galalphacorr4060,-15),integratethings(galphistarcorr6080,galmstarcorr6080,galalphacorr6080,-15),integratethings(galphistarcorr80100,galmstarcorr80100,galalphacorr80100,-15)])

#******************************************************************
#******************************************************************
#*******        Output evolution parameters to file        ********
#******************************************************************
#******************************************************************

NumDens=np.stack((redshiftrange[0:5],unp.nominal_values(gal_densityeighteen_err),unp.std_devs(gal_densityeighteen_err),unp.nominal_values(gal_densityfifteen_err),unp.std_devs(gal_densityfifteen_err),unp.nominal_values(lcbg_density_err),unp.std_devs(lcbg_density_err)),axis=-1)
LCBG_LF_PARAM_EVO=np.stack((redshiftrange,unp.nominal_values(lcbg_phistar_evo),unp.std_devs(lcbg_phistar_evo),unp.nominal_values(lcbg_mstar_evo),unp.std_devs(lcbg_mstar_evo),unp.nominal_values(lcbg_alpha_evo),unp.std_devs(lcbg_alpha_evo)),axis=-1)
GAL_LF_PARAM_EVO=np.stack((redshiftrange,unp.nominal_values(gal_phistar_evo),unp.std_devs(gal_phistar_evo),unp.nominal_values(gal_mstar_evo),unp.std_devs(gal_mstar_evo),unp.nominal_values(gal_alpha_evo),unp.std_devs(gal_alpha_evo)),axis=-1)
np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/NumDens_test.txt',NumDens,header='#z	galden18	galden15	lcbgden')



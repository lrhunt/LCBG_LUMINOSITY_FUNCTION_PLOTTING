import numpy as np
import astropy as ap
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages
from uncertainties import unumpy as unp
import uncertainties as unc 
import scipy as sp

plt.style.use(astropy_mpl_style)

#*****************************
#This is used to plot number of galaxies above a histogram autmatically
#*****************************

def autolabel(rects,thecolor,row,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[row][col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(round(np.power(10,height)))),ha='center',va='bottom',fontsize=7,color=thecolor)

#*****************************
#Scipy and astropy definitions of Schechter Function
#*****************************

@custom_model
def schechter_func(x,phistar=0.0056,mstar=-21):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.1+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.1+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

#*****************************
#Defining Schechter Function and range of parameters
#*****************************

LCBGFIT_init=schechter_func()
LCBG_Range=np.linspace(-24,-15,30)

#*****************************
#Creating figure and defining subplots
#*****************************

f=plt.figure(figsize=(24,13.5))

init_vals=[0.0056,-21]	#Best guess at initial values, needed for scipy fitting
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

x=np.linspace(-23,-16,8)
xminor=np.linspace(-23.5,-15.5,9)
y=np.linspace(-7,-1,7)
yminor=np.linspace(-6.5,-1.5,6)

#f,axes=plt.subplots(nrows=4,ncols=3,sharex=True,gridspec_kw={'height_ratios':[3,1,3,1]})

#Zucca 0.1<z<0.35
zuccaM=np.array([-17.11,-18.04,-18.93,-19.85,-20.77,-21.58])
zuccaPHI=np.array([-2.1,-2.168,-2.288,-2.448,-2.708,-3.344])
zuccaPHIERR=np.array([0.06,0.036,0.002,0.036,0.052,0.08])
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_ZUCCA_10_35.txt'
zfilein='/home/lrhunt/LUM_FUNC/ZUCCA_DATA/VMAX/LF_Vmax_B_tot_FIXA0.dat'

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
MBINAVEz,LogLumFuncz,LogErrzup,LogErrzdown=np.loadtxt(zfilein,unpack=True)
zuccaerr=np.power(10,LogLumFuncz[0:5])*np.log(10)*LogErrzdown[0:5]

with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[11:17]=True
MBINAVE2.mask[11:17]=True
LumFuncErr2.mask[11:17]=True

LUMFUNC_ZUCCA_FIT1035=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LUMFUNC_ZUCCA_FIT1035z=fit(LCBGFIT_init,MBINAVEz[0:5],np.power(10,LogLumFuncz[0:5]),weights=1/zuccaerr[0:5])

#Scipy Modelling

scipy_LUMFUNC_ZUCCA_1035_fit,scipy_LUMFUNC_ZUCCA_1035_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())
scipy_LUMFUNC_ZUCCA_1035_fitz,scipy_LUMFUNC_ZUCCA_1035_covz=sp.optimize.curve_fit(schechter_func_scipy,MBINAVEz[0:5],np.power(10,LogLumFuncz[0:5]),p0=init_vals,sigma=1/zuccaerr[0:5])

#Defining Errors on fit parameters from scipy

phistarcorr1035,mstarcorr1035=unc.correlated_values(scipy_LUMFUNC_ZUCCA_1035_fit,scipy_LUMFUNC_ZUCCA_1035_cov)
phistarcorr1035z,mstarcorr1035z=unc.correlated_values(scipy_LUMFUNC_ZUCCA_1035_fitz,scipy_LUMFUNC_ZUCCA_1035_covz)

#LUMFUNCZUCCA1035ERRORS=np.array([np.sqrt(scipy_LUMFUNC_ZUCCA_1035_cov[0][0]),np.sqrt(scipy_LUMFUNC_ZUCCA_1035_cov[1][1]),np.sqrt(scipy_LUMFUNC_ZUCCA_1035_cov[2][2])])

def schechter_fit(sample_M, phi=0.4*np.log(10)*0.00645, M_star=-20.73, alpha=-1.03, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter

code=ax1.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This Study',color='blue')
#code=ax1.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
gals=ax4.bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
ax1.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT1035(LCBG_Range)),'-.',color='blue')
ax1.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
zucca=ax1.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
plt.subplots_adjust(hspace=0,wspace=0)
ax1.set_xticks(x)
ax1.set_xticks(xminor,minor=True)
ax1.set_yticks(y)
ax1.set_yticks(yminor,minor=True)
ax1.set_ylim([-7.5,-0.5])
ax4.set_yticks([3,2,1,0])
ax4.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax4.set_xticks(x)
ax4.set_xticks(xminor,minor=True)
ax4.set_ylim([0,4.2])
autolabel(gals,'black',1,0)

#CODE TO MAKE INDIVIDUAL PLOT (COMMENTED OUT FOR NOW)

#f1,ax=plt.subplots(figsize=(8,8))
#code=ax.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax2.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
#ax.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT1035(LCBG_Range)),'-.',color='blue')
#ax.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
#zucca=ax.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
#plt.subplots_adjust(hspace=0,wspace=0)
#ax.set_xticks(x)
#ax.set_xticks(xminor,minor=True)
#ax.set_yticks(y)
#ax.set_yticks(yminor,minor=True)
#ax.set_ylim([-7.5,-0.5])
#ax.set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=14)
#ax.set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=14)
#ax.set_title('0.1<z<0.35')
#plt.legend(loc=4,fontsize='small')
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/ZUCCA_LF_COMPARE_1035.png')


#ZUCCA 0.35<z<0.55
zuccaM=np.array([-19.26,-19.83,-20.43,-21.01,-21.6,-22.23])
zuccaPHI=np.array([-2.424,-2.44,-2.62,-2.868,-3.308,-4.096])
zuccaPHIERR=np.array([0.028,0.028,0.032,0.052,0.068,0.12])
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_ZUCCA_35_55.txt'
zfilein='/home/lrhunt/LUM_FUNC/ZUCCA_DATA/VMAX/LF_Vmax_B_tot_FIXA1.dat'

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
MBINAVEz,LogLumFuncz,LogErrzup,LogErrzdown=np.loadtxt(zfilein,unpack=True)
zuccaerr=np.power(10,LogLumFuncz)*np.log(10)*LogErrzdown

with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[11:17]=True
MBINAVE2.mask[11:17]=True
LumFuncErr2.mask[11:17]=True

LUMFUNC_ZUCCA_FIT3555=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LUMFUNC_ZUCCA_FIT3555z=fit(LCBGFIT_init,MBINAVEz[0:4],np.power(10,LogLumFuncz[0:4]),weights=1/zuccaerr[0:4])

#Scipy Modelling

scipy_LUMFUNC_ZUCCA_3555_fit,scipy_LUMFUNC_ZUCCA_3555_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

phistarcorr3555,mstarcorr3555=unc.correlated_values(scipy_LUMFUNC_ZUCCA_3555_fit,scipy_LUMFUNC_ZUCCA_3555_cov)
#LUMFUNCZUCCA3555ERRORS=np.array([np.sqrt(scipy_LUMFUNC_ZUCCA_3555_cov[0][0]),np.sqrt(scipy_LUMFUNC_ZUCCA_3555_cov[1][1]),np.sqrt(scipy_LUMFUNC_ZUCCA_3555_cov[2][2])])

def schechter_fit(sample_M, phi=0.4*np.log(10)*0.0049, M_star=-20.91, alpha=-1.03, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter

code=ax2.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax2.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
gals=ax5.bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
ax2.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT3555(LCBG_Range)),'-.',color='blue')
ax2.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
zucca=ax2.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
autolabel(gals,'black',1,1)
plt.subplots_adjust(hspace=0,wspace=0)
ax2.set_xticks(x)
ax2.set_xticks(xminor,minor=True)
ax2.set_yticks(y)
ax2.set_yticks(yminor,minor=True)
ax2.set_ylim([-7.5,-0.5])
ax5.set_yticks([3,2,1,0])
ax5.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax5.set_xticks(x)
ax5.set_xticks(xminor,minor=True)
ax5.set_ylim([0,4.2])

#CODE TO MAKE INDIVIDUAL PLOT (COMMENTED OUT FOR NOW)

#f1,ax=plt.subplots(figsize=(8,8))
#code=ax.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax2.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
#ax.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT3555(LCBG_Range)),'-.',color='blue')
#ax.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
#zucca=ax.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
#plt.subplots_adjust(hspace=0,wspace=0)
#ax.set_xticks(x)
#ax.set_xticks(xminor,minor=True)
#ax.set_yticks(y)
#ax.set_yticks(yminor,minor=True)
#ax.set_ylim([-7.5,-0.5])
#ax.set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=14)
#ax.set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=14)
#ax.set_title('0.35<z<0.55')
#plt.legend(loc=4,fontsize='small')
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/ZUCCA_LF_COMPARE_3555.png')



#ZUCCA 0.55<z<0.75
zuccaM=np.array([-20.13,-20.63,-21.21,-21.77,-22.36,-22.98])
zuccaPHI=np.array([-2.42,-2.568,-2.764,-3.152,-3.784,-4.92])
zuccaPHIERR=np.array([0.028,0.032,0.036,0.052,0.124,0.18])
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_ZUCCA_55_75.txt'
zfilein='/home/lrhunt/LUM_FUNC/ZUCCA_DATA/VMAX/LF_Vmax_B_tot_FIXA2.dat'

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
MBINAVEz,LogLumFuncz,LogErrzup,LogErrzdown=np.loadtxt(zfilein,unpack=True)
zuccaerr=np.power(10,LogLumFuncz)*np.log(10)*LogErrzdown

with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[5]=True
MBINAVE2.mask[5]=True
LumFuncErr2.mask[5]=True

LUMFUNC_ZUCCA_FIT5575=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LUMFUNC_ZUCCA_FIT3555z=fit(LCBGFIT_init,MBINAVEz[0:4],np.power(10,LogLumFuncz[0:4]),weights=1/zuccaerr[0:4])

#Scipy Modelling

scipy_LUMFUNC_ZUCCA_5575_fit,scipy_LUMFUNC_ZUCCA_5575_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())
scipy_LUMFUNC_ZUCCA_5575_fitz,scipy_LUMFUNC_ZUCCA_5575_covz=sp.optimize.curve_fit(schechter_func_scipy,MBINAVEz[0:4],np.power(10,LogLumFuncz[0:4]),p0=init_vals,sigma=1/zuccaerr[0:4])

#Defining Errors on fit parameters from scipy

phistarcorr5575,mstarcorr5575=unc.correlated_values(scipy_LUMFUNC_ZUCCA_5575_fit,scipy_LUMFUNC_ZUCCA_5575_cov)
phistarcorr5575z,mstarcorr5575z=unc.correlated_values(scipy_LUMFUNC_ZUCCA_5575_fitz,scipy_LUMFUNC_ZUCCA_5575_covz)
#LUMFUNCZUCCA5575ERRORS=np.array([np.sqrt(scipy_LUMFUNC_ZUCCA_5575_cov[0][0]),np.sqrt(scipy_LUMFUNC_ZUCCA_5575_cov[1][1]),np.sqrt(scipy_LUMFUNC_ZUCCA_5575_cov[2][2])])

def schechter_fit(sample_M, phi=0.4*np.log(10)*0.00557, M_star=-21.14, alpha=-1.03, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter

code=ax3.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax3.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
gals=ax6.bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
ax3.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT5575(LCBG_Range)),'-.',color='blue')
ax3.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
zucca=ax3.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
autolabel(gals,'black',1,2)
plt.subplots_adjust(hspace=0,wspace=0)
ax3.set_xticks(x)
ax3.set_xticks(xminor,minor=True)
ax3.set_yticks(y)
ax3.set_yticks(yminor,minor=True)
ax3.set_ylim([-7.5,-0.5])
ax6.set_yticks([3,2,1,0])
ax6.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax6.set_xticks(x)
ax6.set_xticks(xminor,minor=True)
ax6.set_ylim([0,4.2])

#CODE TO MAKE INDIVIDUAL PLOT (COMMENTED OUT FOR NOW)

#f1,ax=plt.subplots(figsize=(8,8))
#code=ax.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax2.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
#ax.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT5575(LCBG_Range)),'-.',color='blue')
#ax.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
#zucca=ax.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
#plt.subplots_adjust(hspace=0,wspace=0)
#ax.set_xticks(x)
#ax.set_xticks(xminor,minor=True)
#ax.set_yticks(y)
#ax.set_yticks(yminor,minor=True)
#ax.set_ylim([-7.5,-0.5])
#ax.set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=14)
#ax.set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=14)
#ax.set_title('0.55<z<0.75')
#plt.legend(loc=4,fontsize='small')
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/ZUCCA_LF_COMPARE_5575.png')


#ZUCCA 0.75<z<1

zuccaM=np.array([-20.68,-21.17,-21.7,-22.22,-22.77,-23.4])
zuccaPHI=np.array([-2.48,-2.588,-2.976,-3.372,-4.12,-5.084])
zuccaPHIERR=np.array([0.080,0.052,0.024,0.068,0.12,0.24])
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_ZUCCA_75_100_newbins.txt'
zfilein='/home/lrhunt/LUM_FUNC/ZUCCA_DATA/VMAX/LF_Vmax_B_tot_FIXA3.dat'

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
MBINAVEz,LogLumFuncz,LogErrzup,LogErrzdown=np.loadtxt(zfilein,unpack=True)
zuccaerr=np.power(10,LogLumFuncz)*np.log(10)*LogErrzdown

with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[4:6]=True
MBINAVE2.mask[4:6]=True
LumFuncErr2.mask[4:6]=True
LogErrzdown[4]=0.312

LUMFUNC_ZUCCA_FIT75100=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LUMFUNC_ZUCCA_75100_fit,scipy_LUMFUNC_ZUCCA_75100_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

phistarcorr75100,mstarcorr75100=unc.correlated_values(scipy_LUMFUNC_ZUCCA_75100_fit,scipy_LUMFUNC_ZUCCA_75100_cov)
#LUMFUNCZUCCA75100ERRORS=np.array([np.sqrt(scipy_LUMFUNC_ZUCCA_75100_cov[0][0]),np.sqrt(scipy_LUMFUNC_ZUCCA_75100_cov[1][1]),np.sqrt(scipy_LUMFUNC_ZUCCA_75100_cov[2][2])])

def schechter_fit(sample_M, phi=0.4*np.log(10)*0.00715, M_star=-21.17, alpha=-1.03, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter

code=ax7.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax7.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
gals=ax9.bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
ax7.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT75100(LCBG_Range)),'-.',color='blue')
ax7.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
zucca=ax7.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
autolabel(gals,'black',3,0)
ax7.legend(loc=4,fontsize=12)
plt.subplots_adjust(hspace=0,wspace=0)
ax7.set_xticks(x)
ax7.set_xticks(xminor,minor=True)
ax7.set_yticks(y)
ax7.set_yticks(yminor,minor=True)
ax7.set_ylim([-7.5,-0.5])
ax9.set_yticks([3,2,1,0])
ax9.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax9.set_xticks(x)
ax9.set_xticks(xminor,minor=True)
ax9.set_ylim([0,4.2])


#CODE TO MAKE INDIVIDUAL PLOT (COMMENTED OUT FOR NOW)

#f1,ax=plt.subplots(figsize=(8,8))
#code=ax.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax2.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
#ax.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT75100(LCBG_Range)),'-.',color='blue')
#ax.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
#zucca=ax.errorbar(zuccaM,zuccaPHI,yerr=zuccaPHIERR,fmt=',',label='Zucca C+',color='black')
#plt.subplots_adjust(hspace=0,wspace=0)
#ax.set_xticks(x)
#ax.set_xticks(xminor,minor=True)
#ax.set_yticks(y)
#ax.set_yticks(yminor,minor=True)
#ax.set_ylim([-7.5,-0.5])
#ax.set_xlabel('M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=14)
#ax.set_ylabel('Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',fontsize=14)
#ax.set_title('0.75<z<1.0')
#plt.legend(loc=4,fontsize='small')
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/ZUCCA_LF_COMPARE_75100.png')

#ZUCCA 0.3<z<0.8

sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_ZUCCA_30_80.txt'
zfilein='/home/lrhunt/LUM_FUNC/ZUCCA_DATA/VMAX/LF_Vmax_B_tot_FIXA4.dat'

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
MBINAVEz,LogLumFuncz,LogErrzup,LogErrzdown=np.loadtxt(zfilein,unpack=True)
zuccaerr=np.power(10,LogLumFuncz)*np.log(10)*LogErrzdown

with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True


LUMFUNC_ZUCCA_FIT3080=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LUMFUNC_ZUCCA_FIT3080z=fit(LCBGFIT_init,MBINAVEz,np.power(10,LogLumFuncz),weights=1/zuccaerr)

#Scipy Modelling

scipy_LUMFUNC_ZUCCA_3080_fit,scipy_LUMFUNC_ZUCCA_3080_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

#LUMFUNCZUCCA3080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_ZUCCA_3080_cov[0][0]),np.sqrt(scipy_LUMFUNC_ZUCCA_3080_cov[1][1]),np.sqrt(scipy_LUMFUNC_ZUCCA_3080_cov[2][2])])
phistarcorr3080,mstarcorr3080=unc.correlated_values(scipy_LUMFUNC_ZUCCA_3080_fit,scipy_LUMFUNC_ZUCCA_3080_cov)

def schechter_fit(sample_M, phi=0.4*np.log(10)*0.00542, M_star=-21.02, alpha=-1.03, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter

code=ax8.errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='This study',color='blue')
#code=ax8.errorbar(MBINAVEz,LogLumFuncz,yerr=[LogErrzup,LogErrzdown],fmt=',',label='This study',color='blue')
gals=ax10.bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
ax8.plot(LCBG_Range,np.log10(LUMFUNC_ZUCCA_FIT3080(LCBG_Range)),'-.',color='blue')
ax8.plot(LCBG_Range,np.log10(schechter_fit(LCBG_Range)),'--',color='black')
plt.subplots_adjust(hspace=0,wspace=0)
ax8.set_xticks(x)
ax8.set_xticks(xminor,minor=True)
ax8.set_yticks(y)
ax8.set_yticks(yminor,minor=True)
ax8.set_ylim([-7.5,-0.5])
ax10.set_yticks([3,2,1,0])
ax10.set_yticks([3.5,2.5,1.5,0.5],minor=True)
ax10.set_xticks(x)
ax10.set_xticks(xminor,minor=True)
ax10.set_ylim([0,4.2])
autolabel(gals,'black',3,1)

ZUCCA_PHISTAR=np.array([phistarcorr1035,phistarcorr3555,phistarcorr5575,phistarcorr75100,phistarcorr3080])
ZUCCA_MSTAR=np.array([mstarcorr1035,mstarcorr3555,mstarcorr5575,mstarcorr75100,mstarcorr3080])

ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax5.set_yticklabels([])
ax6.set_yticklabels([])
ax8.set_yticklabels([])
#axes[2][2].set_yticklabels([])
ax10.set_yticklabels([])
#axes[3][2].set_yticklabels([])

ax1.set_yticklabels([-7,-6,-5,-4,-3,-2,-1],fontsize=16)
ax7.set_yticklabels([-7,-6,-5,-4,-3,-2,-1],fontsize=16)
ax4.set_yticklabels([3,2,1,0],fontsize=16)
ax9.set_yticklabels([3,2,1,0],fontsize=16)
ax4.set_xticklabels([-23,-22,-21,-20,-19,-18,-17,-16],fontsize=16)
ax5.set_xticklabels([-23,-22,-21,-20,-19,-18,-17,-16],fontsize=16)
ax6.set_xticklabels([-23,-22,-21,-20,-19,-18,-17,-16],fontsize=16)
ax9.set_xticklabels([-23,-22,-21,-20,-19,-18,-17,-16],fontsize=16)
ax10.set_xticklabels([-23,-22,-21,-20,-19,-18,-17,-16],fontsize=16)

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
              
ax1.text(-23.5,-1,'z=0.1-0.35',verticalalignment='center',fontsize=16)
ax2.text(-23.5,-1,'z=0.35-0.55',verticalalignment='center',fontsize=16)
ax3.text(-23.5,-1,'z=0.55-0.75',verticalalignment='center',fontsize=16)
ax7.text(-23.5,-1,'z=0.75-1.0',verticalalignment='center',fontsize=16)
ax8.text(-23.5,-1,'z=0.3-0.8',verticalalignment='center',fontsize=16)
f.text(0.52,0.03,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=16)
f.text(0.02,0.84,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.18,0.32,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.02,0.63,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.18,0.11,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=16)

plt.subplots_adjust(left=0.04,right=1,bottom=0.06,top=1)

plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/ZUCCA_LF_COMPARE.pdf')

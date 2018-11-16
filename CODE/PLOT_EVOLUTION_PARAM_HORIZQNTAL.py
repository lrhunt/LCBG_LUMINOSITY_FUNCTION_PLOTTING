import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages
from scipy import special
import uncertainties as unc
from uncertainties import unumpy as unp


#*******************************
#*******************************
#*******************************
# Plotting Evolution of Parameters
#*******************************
#*******************************
#*******************************

LFfittingparams=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LFfitparams.txt')
LCBGfittingparams=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LCBGfitparams.txt')
WILLMER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/WILLMER.txt')
FABER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FABER.txt')
COOL=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/COOL.txt')
ZUCCA=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA.txt')
FRITZ=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FRITZ.txt')
BEARE=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE.txt')

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,4],yerr=LFfittingparams[0:5,5],fmt='-o',color='black',label='This Work')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,4],yerr=WILLMER[:,5],color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,4],yerr=FABER[:,5],color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,4],yerr=ZUCCA[:,5],color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,4],yerr=COOL[:,5],color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,4],yerr=FRITZ[:,5],color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,4],yerr=BEARE[:,5],color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,4],yerr=LCBGfittingparams[0:5,5],color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([-19.6,-21.8])
axes[1].set_yticks([-20,-20.5,-21,-21.5])
plt.subplots_adjust(wspace=0,left=0.1,right=0.95)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,-21.6,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,-21.6,'LCBG',fontsize=12,ha='center',va='center')
axes[0].set_ylabel('M$^{*}$-5log(h$_{70}$)',fontsize=14)
axes[0].set_xlabel('z',fontsize=14)
axes[1].set_xlabel('z',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVMSTAR.png')

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,6]/1000.,yerr=LFfittingparams[0:5,7]/1000.,color='black',label='This Work',fmt='-o')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,6]/1000.,yerr=WILLMER[:,7]/1000.,color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,6]/1000.,yerr=FABER[:,7]/1000.,color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,6]/1000.,yerr=ZUCCA[:,7]/1000.,color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,6]/1000.,yerr=COOL[:,7]/1000.,color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,6]/1000.,yerr=FRITZ[:,7]/1000.,color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,6]/1000.,yerr=BEARE[:,7]/1000.,color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,6]/1000.,yerr=LCBGfittingparams[0:5,7]/1000.,color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
plt.subplots_adjust(wspace=0,left=0.1,right=0.95)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,0.009,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,0.009,'LCBG',fontsize=12,ha='center',va='center')
axes[0].set_ylabel('$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',fontsize=14)
axes[0].set_xlabel('z',fontsize=14)
axes[1].set_xlabel('z',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVPHISTAR.png')

lcbgmstaru=unp.uarray([LCBGfittingparams[0:5,4],LCBGfittingparams[0:5,5]])
lcbgphistaru=unp.uarray([LCBGfittingparams[0:5,6],LCBGfittingparams[0:5,7]])
allmstaru=unp.uarray([LFfittingparams[0:5,4],LFfittingparams[0:5,5]])
allphistaru=unp.uarray([LFfittingparams[0:5,6],LFfittingparams[0:5,7]])
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

lcbgju=lcbgphistaru/1000*np.power(10,(lcbgmstaru-5.48)/-2.5)*scipy.special.gammainc(LCBGfittingparams[0:5,2]+2,np.power(10,(LCBGfittingparams[0:5,4]+18.5)/2.5))
galju=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*scipy.special.gammainc(2+LFfittingparams[0:5,2],np.power(10,(LFfittingparams[0:5,4]+18.5)/2.5))
BEAREju=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*scipy.special.gammainc(2+BEARE[:,2],np.power(10,(BEARE[:,4]+18.5)/2.5))
WILLMERju=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*scipy.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+18.5)/2.5))
FABERju=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*scipy.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+18.5)/2.5))
COOLju=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*scipy.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+18.5)/2.5))
ZUCCAju=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*scipy.special.gammainc(2+ZUCCA[:,2],np.power(10,(ZUCCA[:,4]+18.5)/2.5))
FRITZju=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*scipy.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+18.5)/2.5))

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/10000000,yerr=unp.std_devs(galju)[0:5]/10000000,color='black')
axes[0].errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/10000000,yerr=unp.std_devs(BEAREju)/10000000,color='slateblue',label='Beare, 2015')
axes[0].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/10000000,yerr=unp.std_devs(WILLMERju)/10000000,color='yellow',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],unp.nominal_values(FABERju)/10000000,yerr=unp.std_devs(FABERju)/10000000,color='green', label='Faber, 2007')
axes[0].errorbar(COOL[:,0],unp.nominal_values(COOLju)/10000000,yerr=unp.std_devs(COOLju)/10000000,color='red',label='Cool, 2012')
axes[0].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/10000000,yerr=unp.std_devs(ZUCCAju)/10000000,color='grey',label='Zucca, 2009')
axes[0].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/10000000,yerr=unp.std_devs(FRITZju)/10000000,color='purple', label='Fritz, 2014')
axes[1].errorbar(LFfittingparams[0:5,0],unp.nominal_values(lcbgju)[0:5]/10000000,yerr=unp.std_devs(lcbgju)[0:5]/10000000,color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_ylim([0,10])
axes[1].set_yticks([1,2,3,4,5,6,7,8,9])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
plt.subplots_adjust(wspace=0,left=0.1,right=0.95)
axes[0].grid()
axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,4,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,4,'LCBG',fontsize=12,ha='center',va='center')
axes[0].set_ylabel('$j_{B}$ (10$^{7}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',fontsize=14)
axes[0].set_xlabel('z',fontsize=14)
axes[1].set_xlabel('z',fontsize=14)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.png')

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

galju2=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*scipy.special.gamma(2+allalpha)
BEAREju2=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*scipy.special.gamma(2+BEARE[:,2])
WILLMERju2=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*scipy.special.gamma(2+WILLMER[:,2])
FABERju2=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*scipy.special.gamma(2+FABER[:,2])
COOLju2=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*scipy.special.gamma(2+COOL[:,2])
ZUCCAju2=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*scipy.special.gamma(2+ZUCCA[:,2])
FRITZju2=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*scipy.special.gamma(2+FRITZ[:,2])

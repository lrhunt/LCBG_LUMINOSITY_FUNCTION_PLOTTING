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
from astropy.visualization import astropy_mpl_style

#*******************************
#*******************************
#*******************************
#Setting Up and collecting data
#*******************************
#*******************************
#*******************************

plt.style.use(astropy_mpl_style)

#*************************************************
#redshift and galaxy numbers
#*************************************************

redshiftrange=np.array([0.1,0.3,0.5,0.7,0.9])

nlcbg=np.array([sum(LCBGPARAMS020[5]),sum(LCBGPARAMS2040[5]),sum(LCBGPARAMS4060[5]),sum(LCBGPARAMS6080[5]),sum(LCBGPARAMS80100[5])])

ngal=np.array([sum(GALPARAMS020[5]),sum(GALPARAMS2040[5]),sum(GALPARAMS4060[5]),sum(GALPARAMS6080[5]),sum(GALPARAMS80100[5])])

#*************************************************
#Calculate phistar and mstar from covariant matrix
#*************************************************

phistarcorr020,mstarcorr020=uncertainties.correlated_values(cspLCBGFIT020,cspLCBGCOV020)
phistarcorr2040,mstarcorr2040=uncertainties.correlated_values(cspLCBGFIT2040,cspLCBGCOV2040)
phistarcorr4060,mstarcorr4060=uncertainties.correlated_values(cspLCBGFIT4060,cspLCBGCOV4060)
phistarcorr6080,mstarcorr6080=uncertainties.correlated_values(cspLCBGFIT6080,cspLCBGCOV6080)
phistarcorr80100,mstarcorr80100=uncertainties.correlated_values(cspLCBGFIT80100,cspLCBGCOV80100)
galphistarcorr020,galmstarcorr020=uncertainties.correlated_values(cspGALFIT020,cspGALCOV020)
galphistarcorr2040,galmstarcorr2040=uncertainties.correlated_values(cspGALFIT2040,cspGALCOV2040)
galphistarcorr4060,galmstarcorr4060=uncertainties.correlated_values(cspGALFIT4060,cspGALCOV4060)
galphistarcorr6080,galmstarcorr6080=uncertainties.correlated_values(cspGALFIT6080,cspGALCOV6080)
galphistarcorr80100,galmstarcorr80100=uncertainties.correlated_values(cspGALFIT80100,cspGALCOV80100)

galmstar=np.array([galmstarcorr020,galmstarcorr2040,galmstarcorr4060,galmstarcorr6080,galmstarcorr80100])
galphistar=np.array([galphistarcorr020,galphistarcorr2040,galphistarcorr4060,galphistarcorr6080,galphistarcorr80100])
lcbgmstar=np.array([mstarcorr020,mstarcorr2040,mstarcorr4060,mstarcorr6080,mstarcorr80100])
lcbgphistar=np.array([phistarcorr020,phistarcorr2040,phistarcorr4060,phistarcorr6080,phistarcorr80100])



@uncertainties.wrap
def integratethings(ps,ms,al=None,m=None):
	if al==None:
		al=-1.1
	if m==None:
		m=-18.5
	def integrand(X):
		return (0.4*np.log(10)*ps)*(10**(0.4*(al+1)*(ms-X)))*(np.e**(-np.power(10,0.4*(ms-X))))
	integral,abserr=sp.integrate.quad(integrand,-100,m)
	return integral



lcbg_density_err=np.array([integratethings(phistarcorr020,mstarcorr020,al=-1.3,m=-18.5),integratethings(phistarcorr2040,mstarcorr2040,al=-1.3,m=-18.5),integratethings(phistarcorr4060,mstarcorr4060,al=-1.3,m=-18.5),integratethings(phistarcorr6080,mstarcorr6080,al=-1.3,m=-18.5),integratethings(phistarcorr80100,mstarcorr80100,al=-1.3,m=-18.5)])

gal_densityeighteen_err=np.array([integratethings(galphistarcorr020,galmstarcorr020,al=-1.1,m=-18.5),integratethings(galphistarcorr2040,galmstarcorr2040,al=-1.1,m=-18.5),integratethings(galphistarcorr4060,galmstarcorr4060,al=-1.1,m=-18.5),integratethings(galphistarcorr6080,galmstarcorr6080,al=-1.1,m=-18.5),integratethings(galphistarcorr80100,galmstarcorr80100,al=-1.1,m=-18.5)])

gal_densityfifteen_err=np.array([integratethings(galphistarcorr020,galmstarcorr020,al=-1.1,m=-16),integratethings(galphistarcorr2040,galmstarcorr2040,al=-1.1,m=-16),integratethings(galphistarcorr4060,galmstarcorr4060,al=-1.1,m=-16),integratethings(galphistarcorr6080,galmstarcorr6080,al=-1.1,m=-16),integratethings(galphistarcorr80100,galmstarcorr80100,al=-1.1,m=-16)])

WILLMER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/WILLMER.txt')
FABER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FABER.txt')
COOL=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/COOL.txt')
ZUCCA=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA.txt')
FRITZ=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FRITZ.txt')
BEARE=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE.txt')
BEAREALPH=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE_CONSTANTALPHA.txt')
ZUCCAALPH=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA_CONSTANTALPHA.txt')

lcbgmstaru=unp.uarray([unp.nominal_values(lcbgmstar),unp.std_devs(lcbgmstar)])
lcbgphistaru=unp.uarray([unp.nominal_values(lcbgphistar),unp.std_devs(lcbgphistar)])
allmstaru=unp.uarray([unp.nominal_values(galmstar),unp.std_devs(galmstar)])
allphistaru=unp.uarray([unp.nominal_values(galphistar),unp.std_devs(galphistar)])
BEAREmstaru=unp.uarray([BEAREALPH[:,4],BEAREALPH[:,5]])
BEAREphistaru=unp.uarray([BEAREALPH[:,6],BEAREALPH[:,7]])
WILLMERmstaru=unp.uarray([WILLMER[:,4],WILLMER[:,5]])
WILLMERphistaru=unp.uarray([WILLMER[:,6],WILLMER[:,7]])
FABERmstaru=unp.uarray([FABER[:,4],FABER[:,5]])
FABERphistaru=unp.uarray([FABER[:,6],FABER[:,7]])
COOLmstaru=unp.uarray([COOL[:,4],COOL[:,5]])
COOLphistaru=unp.uarray([COOL[:,6],COOL[:,7]])
ZUCCAmstaru=unp.uarray([ZUCCAALPH[:,4],ZUCCAALPH[:,5]])
ZUCCAphistaru=unp.uarray([ZUCCAALPH[:,6],ZUCCAALPH[:,7]])
FRITZmstaru=unp.uarray([FRITZ[:,4],FRITZ[:,5]])
FRITZphistaru=unp.uarray([FRITZ[:,6],FRITZ[:,7]])

lcbgalpha=np.array([-1.3,-1.3,-1.3,-1.3,-1.3])
allalpha=np.array([-1.1,-1.1,-1.1,-1.1,-1.1])

lcbgju=lcbgphistaru*np.power(10,(lcbgmstaru-5.48)/-2.5)*(sp.special.gammainc(lcbgalpha+2,np.power(10,(unp.nominal_values(lcbgmstar)+23.5)/2.5))-sp.special.gammainc(lcbgalpha+2,np.power(10,(unp.nominal_values(lcbgmstar)+18.5)/2.5)))
galju=allphistaru*np.power(10,(allmstaru-5.48)/-2.5)*(sp.special.gammainc(allalpha+2,np.power(10,(unp.nominal_values(galmstar)+23.5)/2.5))-sp.special.gammainc(2+allalpha,np.power(10,(unp.nominal_values(galmstar)+18.5)/2.5)))
BEAREju=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*(sp.special.gammainc(2+BEAREALPH[:,2],np.power(10,(BEAREALPH[:,4]+23.5)/2.5))-sp.special.gammainc(2+BEAREALPH[:,2],np.power(10,(BEAREALPH[:,4]+18.5)/2.5)))
WILLMERju=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*(sp.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+23.5)/2.5))-sp.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+18.5)/2.5)))
FABERju=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*(sp.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+23.5)/2.5))-sp.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+18.5)/2.5)))
COOLju=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*(sp.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+23.5)/2.5))-sp.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+18.5)/2.5)))
ZUCCAju=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*(sp.special.gammainc(2+ZUCCAALPH[:,2],np.power(10,(ZUCCAALPH[:,4]+23.5)/2.5))-sp.special.gammainc(2+ZUCCAALPH[:,2],np.power(10,(ZUCCAALPH[:,4]+18.5)/2.5)))
FRITZju=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*(sp.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+23.5)/2.5))-sp.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+18.5)/2.5)))


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

fractionLCBGS=nlcbg/ngal
FractionalError=fractionLCBGS*np.sqrt((nlcbg/np.power(nlcbg,2))+(ngal/np.power(ngal,2)))
guzman=12.0/21.0*102.0/301.0
guzmanerror=12.0/21.0*102.0/301.0*np.sqrt(12.0/12.0**2+21.0/21.0**2+102.0/102.0**2+301.0/301.0**2)
tolerud=199/1744.
toleruderror=199./1744.*np.sqrt(199./199.**2+1744./1744.**2)

#*******************************
#*******************************
#*******************************
# Plotting Evolution of Parameters (VERTICAL/PAPER)
#*******************************
#*******************************
#*******************************

#*************************************************
#Evolution of M*
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(redshiftrange,unp.nominal_values(galmstar),yerr=unp.std_devs(galmstar),fmt='-o',color='black',label='This Work')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,4],yerr=WILLMER[:,5],color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,4],yerr=FABER[:,5],color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCAALPH[:,0],ZUCCAALPH[:,4],yerr=ZUCCAALPH[:,5],color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,4],yerr=COOL[:,5],color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,4],yerr=FRITZ[:,5],color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEAREALPH[:,0],BEAREALPH[:,4],yerr=BEAREALPH[:,5],color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(redshiftrange,unp.nominal_values(lcbgmstar),yerr=unp.std_devs(lcbgmstar),color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([-19.6,-21.8])
axes[1].set_yticks([-20,-20.5,-21,-21.5])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
#axes[0].grid()
#axes[1].grid()
axes[0].text(0.5,-21.6,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,-21.6,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'M$^{*}$-5log(h$_{70}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.55,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(right=0.99,left=0.13,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVMSTAR.pdf')

#*************************************************
#Evolution of /phi*
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(redshiftrange,unp.nominal_values(galphistar),yerr=unp.std_devs(galphistar),color='black',label='This Work',fmt='-o')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,6]/1000.,yerr=WILLMER[:,7]/1000.,color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,6]/1000.,yerr=FABER[:,7]/1000.,color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCAALPH[:,0],ZUCCAALPH[:,6]/1000.,yerr=ZUCCAALPH[:,7]/1000.,color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,6]/1000.,yerr=COOL[:,7]/1000.,color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,6]/1000.,yerr=FRITZ[:,7]/1000.,color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEAREALPH[:,0],BEAREALPH[:,6]/1000.,yerr=BEAREALPH[:,7]/1000.,color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(redshiftrange,unp.nominal_values(lcbgphistar),yerr=unp.std_devs(lcbgphistar),color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
#axes[0].grid()
#axes[1].grid()
axes[0].text(0.5,0.009,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,0.009,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.55,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(right=0.99,left=0.14,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVPHISTAR.pdf')


#*************************************************
#Evolution of j*
#*************************************************

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(redshiftrange,unp.nominal_values(galju)[0:5]/100000000,yerr=unp.std_devs(galju)[0:5]/100000000,color='blue',label='This Work',fmt='-o')
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
#axes[0].grid()
#axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,3.2,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,3.2,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'$j_{B}$ (10$^{8}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.55,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(right=0.99,left=0.09,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.pdf')

#*************************************************
#Fraction of LCBGS
#*************************************************

f,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1,1]},figsize=(8,12))
rectseighteen=axes[1].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityeighteen_err),ecolor='black')
rects=axes[0].bar(redshiftrange,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='0.85',ecolor='black')
rectsfifteen=axes[2].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityfifteen_err),ecolor='black')
axes[0].text(0.05,0.6,'(a) $N_{LCBG}/N_{GAL}$ COSMOS',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.05,0.6,'(b) $\phi_{LCBG}$/$\phi_{GAL,M=-18.5}$',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[2].text(0.05,0.6,'(c) $\phi_{LCBG}$/$\phi_{GAL,M=-15}$',va='center',bbox={'facecolor':'white'},fontsize=14)
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
plt.xlabel('Redshift',fontsize=16)
f.text(0.03,0.5,'Fraction of LCBGS',va='center',rotation='vertical',fontsize=16)
plt.ylim(0,0.7)
autolabel(rects,'black',nlcbg,ngal,0)
autolabel2(rectseighteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),4),1)
autolabel2(rectsfifteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),4),2)
plt.subplots_adjust(right=0.98,left=0.11,top=0.97,bottom=0.06)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/FractionLCBGPLOT.pdf')

#*************************************************
#LCBG Number Density
#*************************************************

f,ax=plt.subplots(figsize=(8.5,8))
ax.errorbar(redshiftrange,unp.nominal_values(lcbg_density_err),yerr=unp.std_devs(lcbg_density_err),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='blue',marker='s',ls='none',label='This Study')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='black',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='black',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.set_xlim([0,1])
plt.xlabel('Redshift',fontsize=16)	
plt.ylabel('N (Mpc$^{-3}$)',fontsize=16)
plt.legend(loc='2')
plt.subplots_adjust(right=0.98,left=0.13,top=0.97,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityEvolution.png')

f,ax=plt.subplots(figsize=(8.5,8))
ax.errorbar(redshiftrange,unp.nominal_values(lcbg_density_err),yerr=unp.std_devs(lcbg_density_err),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='blue',marker='s',ls='none',label='This Study')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='black',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='black',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.set_xlim([0,1])
plt.xlabel('Redshift')
plt.ylabel('N (Mpc$^{-3}$)')
plt.legend(loc='2')
plt.subplots_adjust(right=0.92)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityEvolution.pdf')

#*************************************************
#[O II] Number Density
#*************************************************


#*******************************
#*******************************
#*******************************
# Plotting Evolution of Parameters (HORIZONTAL,TALK)
#*******************************
#*******************************
#*******************************

#*************************************************
#Evolution of M*
#*************************************************

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(redshiftrange,unp.nominal_values(galmstar),yerr=unp.std_devs(galmstar),fmt='-o',color='blue',label='This Work')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,4],yerr=WILLMER[:,5],color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,4],yerr=FABER[:,5],color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCAALPH[:,0],ZUCCAALPH[:,4],yerr=ZUCCAALPH[:,5],color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,4],yerr=COOL[:,5],color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,4],yerr=FRITZ[:,5],color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEAREALPH[:,0],BEAREALPH[:,4],yerr=BEAREALPH[:,5],color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(redshiftrange,unp.nominal_values(lcbgmstar),yerr=unp.std_devs(lcbgmstar),color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([-19.6,-21.8])
axes[1].set_yticks([-20,-20.5,-21,-21.5])
axes[0].legend(fontsize='small')
#axes[0].grid()
#axes[1].grid()
axes[0].text(0.5,-21.6,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,-21.6,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'M$^{*}$-5log(h$_{70}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.545,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(wspace=0,right=0.99,left=0.1,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVMSTAR.png')

#*************************************************
#Evolution of /phi*
#*************************************************

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(redshiftrange,unp.nominal_values(galphistar),yerr=unp.std_devs(galphistar),color='blue',label='This Work',fmt='-o')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,6]/1000.,yerr=WILLMER[:,7]/1000.,color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,6]/1000.,yerr=FABER[:,7]/1000.,color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCAALPH[:,0],ZUCCAALPH[:,6]/1000.,yerr=ZUCCAALPH[:,7]/1000.,color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,6]/1000.,yerr=COOL[:,7]/1000.,color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,6]/1000.,yerr=FRITZ[:,7]/1000.,color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEAREALPH[:,0],BEAREALPH[:,6]/1000.,yerr=BEAREALPH[:,7]/1000.,color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(redshiftrange,unp.nominal_values(lcbgphistar),yerr=unp.std_devs(lcbgphistar),color='blue',fmt='-o')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
axes[0].legend(fontsize='small')
#axes[0].grid()
#axes[1].grid()
axes[0].text(0.5,0.009,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,0.009,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.545,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(wspace=0,right=0.99,left=0.10,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVPHISTAR.png')


#*************************************************
#Evolution of j*
#*************************************************

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(12.5,6))
axes[0].errorbar(redshiftrange,unp.nominal_values(galju)[0:5]/100000000,yerr=unp.std_devs(galju)[0:5]/100000000,color='blue',label='This Work',fmt='-o')
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
#axes[0].grid()
#axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,3.2,'All',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.5,3.2,'LCBG',ha='center',va='center',bbox={'facecolor':'white'},fontsize=14)
f.text(0.03,0.55,'$j_{B}$ (10$^{8}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',ha='center',va='center',rotation='vertical',fontsize=16)
f.text(0.545,0.03,'z',ha='center',va='center',fontsize=16)
plt.subplots_adjust(wspace=0,right=0.99,left=0.1,top=0.99,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.png')

#*************************************************
#Fraction of LCBGS
#*************************************************

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
			axes[ax].text(rect.get_x() + rect.get_width()/2.,height+0.03,'{}'.format(fraction[i]) ,ha='center',va='bottom',fontsize='small')
			i=i+1

f,axes=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1]},figsize=(16,9))
rectseighteen=axes[1].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityeighteen_err),ecolor='black')
rects=axes[0].bar(redshiftrange,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='0.85',ecolor='black')
#rectsfifteen=axes[2].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityfifteen_err),ecolor='black')
axes[0].text(0.05,0.6,'(a) $N_{LCBG}/N_{GAL}$ COSMOS',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.05,0.6,'(b) $\phi_{LCBG}$/$\phi_{GAL,M=-18.5}$',va='center',bbox={'facecolor':'white'},fontsize=14)
#axes[2].text(0.05,0.6,'(c) $\phi_{LCBG}$/$\phi_{GAL,M=-16}$',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[0].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
axes[1].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
#axes[2].errorbar(0.594,guzman,yerr=guzmanerror,xerr=[[0.594-0.4],[1-0.594]],label='Guzm$\`{a}$n, 1997',color='blue')
asymmetric_error=[0.1,0.5]
axes[0].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')
axes[1].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud,2010',color='green')
#axes[2].errorbar(0.5,tolerud,yerr=toleruderror,xerr=[[0.1],[0.5]],label='Tollerud, 2010',color='green')
axes[0].legend(loc=6,fontsize='small')
axes[1].legend(loc=6,fontsize='small')
#axes[2].legend(loc=6,fontsize='small')
axes[1].set_xlabel('Redshift',fontsize=16)
axes[0].set_xticks([0.1,0.3,0.5,0.7,0.9])
axes[1].set_xticks([0.1,0.3,0.5,0.7,0.9])
#axes[2].set_xticks([0.1,0.3,0.5,0.7,0.9])
axes[0].set_ylabel('Fraction of LCBGS',fontsize=16)
#f.text(0.03,0.5,'Fraction of LCBGS',va='center',rotation='vertical',fontsize=16)
plt.ylim(0,0.7)
autolabel2(rects,np.round(fractionLCBGS,4),0)
autolabel2(rectseighteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),4),1)
#autolabel2(rectsfifteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),4),2)
plt.subplots_adjust(wspace=0,right=0.99,left=0.06,top=0.99,bottom=0.09)




f,axes=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,gridspec_kw={'width_ratios':[1,1,1]},figsize=(16,7))
rectseighteen=axes[1].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityeighteen_err),ecolor='black')
rects=axes[0].bar(redshiftrange,fractionLCBGS,0.2,align='center',yerr=FractionalError,color='0.85',ecolor='black')
rectsfifteen=axes[2].bar(redshiftrange,unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),0.2,align='center',color='0.85',yerr=unp.std_devs(lcbg_density_err/gal_densityfifteen_err),ecolor='black')
axes[0].text(0.05,0.6,'(a) $N_{LCBG}/N_{GAL}$ COSMOS',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[1].text(0.05,0.6,'(b) $\phi_{LCBG}$/$\phi_{GAL,M=-18.5}$',va='center',bbox={'facecolor':'white'},fontsize=14)
axes[2].text(0.05,0.6,'(c) $\phi_{LCBG}$/$\phi_{GAL,M=-15}$',va='center',bbox={'facecolor':'white'},fontsize=14)
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
plt.subplots_adjust(wspace=0)
axes[1].set_xlabel('Redshift',fontsize=16)
axes[0].set_ylabel('Fraction of LCBGS',va='center',rotation='vertical',fontsize=16)
axes[0].set_xticks([0.1,0.3,0.5,0.7,0.9])
axes[1].set_xticks([0.1,0.3,0.5,0.7,0.9])
axes[2].set_xticks([0.1,0.3,0.5,0.7,0.9])
plt.ylim(0,0.7)
autolabel2(rects,np.round(fractionLCBGS,4),0)
autolabel2(rectseighteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityeighteen_err),4),1)
autolabel2(rectsfifteen,np.round(unp.nominal_values(lcbg_density_err/gal_densityfifteen_err),4),2)
plt.subplots_adjust(right=0.98,left=0.07,top=0.97,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/FractionLCBGPLOT.png')

#*************************************************
#LCBG Number Density
#*************************************************

f,ax=plt.subplots(figsize=(8.5,8))
ax.errorbar(redshiftrange,unp.nominal_values(lcbg_density_err),yerr=unp.std_devs(lcbg_density_err),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='blue',marker='s',ls='none',label='This Study')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.set_xlim([0,1])
plt.xlabel('Redshift',fontsize=16)	
plt.ylabel('N (Mpc$^{-3}$)',fontsize=16)
plt.legend(loc='2')
plt.subplots_adjust(right=0.98,left=0.13,top=0.97,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityEvolution.png')

f,ax=plt.subplots(figsize=(8.5,8))
ax.errorbar(redshiftrange,unp.nominal_values(lcbg_density_err),yerr=unp.std_devs(lcbg_density_err),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='blue',marker='s',ls='none',label='This Study')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.set_xlim([0,1])
plt.xlabel('Redshift',fontsize=16)
plt.ylabel('N (Mpc$^{-3}$)',fontsize=16)
plt.legend(loc='2')
plt.subplots_adjust(right=0.92)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityEvolution.png')

#*************************************************
#Comparing to other sources
#*************************************************

execfile('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/CODE/FITTING_AND_PARAMETER_CODE/OIILuminosityDensity.py')
execfile('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/CODE/FITTING_AND_PARAMETER_CODE/ZUCCA_GAL_TYPES.py')
IRRedshift=np.array([0,0.25,0.55,0.85])
lirgs=unp.uarray([7,22,180,260],[1,10,80,62])
ulirgs=unp.uarray([0.5,1.5,18,38],[0.05,0.6,10,29])
lirgn=unp.log10(lirgs*10**6/(10**11.27))
ulirgn=unp.log10(ulirgs*10**6/(10**12.3))

f,ax=plt.subplots(figsize=(8.5,8))
ax.errorbar(redshiftrange,unp.nominal_values(unp.log10(lcbg_density_err)),yerr=unp.std_devs(unp.log10(lcbg_density_err)),xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='blue',marker='s',ls='none',label='This Study')
ax.errorbar(ZUCCA_z,unp.nominal_values(unp.log10(ZUCCA_Irr_N)),unp.std_devs(unp.log10(ZUCCA_Irr_N)),xerr=[0.125,0.1,0.1,0.125],color='blue',marker='8',ls='none',label='Zucca, 2009 (Irregular)')
ax.errorbar(ZUCCA_z,unp.nominal_values(unp.log10(ZUCCA_SP_N)),unp.std_devs(unp.log10(ZUCCA_SP_N)),xerr=[0.125,0.1,0.1,0.125],color='red',marker='H',ls='none',label='Zucca, 2009 (Spiral)')
ax.errorbar(simredshift,unp.nominal_values(unp.log10(OIINUMDENS)),yerr=unp.std_devs(unp.log10(OIINUMDENS)),xerr=[[0.1,0.1,0.15,0.2],[0.1,0.15,0.2,0.2]],color='green',marker='*',label='KwangHo, 2015',ls='none')
ax.errorbar(IRRedshift,unp.nominal_values(lirgn),yerr=unp.std_devs(lirgn),xerr=[0,0.15,0.15,0.15],color='orange',marker='>',ls='none',label='Magnelli,2013 (LIRGS)')
ax.errorbar(IRRedshift,unp.nominal_values(ulirgn),yerr=unp.std_devs(ulirgn),xerr=[0,0.15,0.15,0.15],color='yellow',marker='<',ls='none',label='Magnelli,2013 (ULIRGS)')
#ax.errorbar(0.55,np.log10(0.0022*(70./75.)**3),color='blue',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
#ax.errorbar(0.85,np.log10(0.0088*(70./75.)**3),color='blue',marker='x',xerr=0.15,ls='none')
#ax.errorbar(0.023,np.log10(0.00054*(70./75.)**3),color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.set_xlim([0,1.1])
plt.xlabel('Redshift',fontsize=16)	
plt.ylabel('N (Mpc$^{-3}$)',fontsize=16)
plt.legend(loc='2',fontsize=12)
plt.subplots_adjust(right=0.98,left=0.13,top=0.97,bottom=0.07)
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/CompareNumberDensity.pdf')


#*************************************************
#Fitting line to LCBG Number Density
#*************************************************

def line_scipy(x,slope,intercept):
    return (slope*x+intercept)
logoneplusz=np.linspace(np.log10(1),np.log10(3),30)
line_fit,line_cov=sp.optimize.curve_fit(line_scipy,np.log10(1+redshiftrange),unp.nominal_values(unp.log10(lcbg_density_err)),p0=[0.5,1],sigma=unp.std_devs(unp.log10(lcbg_density_err)))
line_uncertainties=uncertainties.correlated_values(line_fit,line_cov)
f,ax1=plt.subplots(figsize=(8,8))
ax1.errorbar(np.log10(1+redshiftrange),unp.nominal_values(unp.log10(lcbg_density_err)),fmt='-o',yerr=unp.std_devs(unp.log10(lcbg_density_err)),label='points',linestyle='None',color='blue')
#ax1.fill(line
ax1.plot(logoneplusz,line_fit[0]*logoneplusz+line_fit[1],label='Fit',color='blue')
plt.legend(loc=2)
ax1.set_xlim(0,0.3)
ax1.set_ylim(-3.3,-1.95)
ax1.set_xlabel('log$_{10}$(1+z)',fontsize=16)
ax1.set_ylabel('log$_{10}$(N) (mpc$^{-3}$)',fontsize=16)
ax2=ax1.twiny()
ax2.semilogx(1+redshiftrange,unp.nominal_values(unp.log10(lcbg_density_err)),color='blue',linestyle='None')
ax2.set_xticks(1+redshiftrange)
ax2.tick_params(axis='x',length=4,direction='in',width=3)
ax2.set_xticklabels([0.1,0.3,0.5,0.7,0.9])
ax2.set_xlim(1,2)
ax2.set_ylim(-3.3,-1.95)
ax2.grid()
ax2.set_xlabel('Redshift',fontsize=16)
plt.subplots_adjust(right=0.98,top=0.92,bottom=0.09)
fit_up=line_scipy(logoneplusz,unp.nominal_values(line_uncertainties[0])+unp.std_devs(line_uncertainties[0]),unp.nominal_values(line_uncertainties[1])+unp.std_devs(line_uncertainties[1]))
fit_dw=line_scipy(logoneplusz,unp.nominal_values(line_uncertainties[0])-unp.std_devs(line_uncertainties[0]),unp.nominal_values(line_uncertainties[1])-unp.std_devs(line_uncertainties[1]))
ax1.fill_between(logoneplusz,fit_up,fit_dw,alpha=0.25,color='blue')
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityTrend.pdf')
plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/NumberDensityTrend.png')

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

@uncertainties.wrap
def integratelow(al=None,be=None,l=None,ls=None):
	if al==None:
		al=-2.6
	if be==None:
		be=46.96
	if l==None:
		l=42
	if ls==None:
		ls=42
	def integrand(X):
		return np.power(10,((al+1)*(X-ls)+be))
	integral,abserr=sp.integrate.quad(integrand,40.6,l)
	return integral

@uncertainties.wrap
def integratehigh(al=None,be=None,ls=None,l=None):
	if al==None:
		al=-1.59
	if l==None:
		l=50
	if be==None:
		be=-4.1
	if ls==None:
		ls=43.06
	def integrand(X):
		return np.power(10,((al+1)*np.log10(10**X/10**ls)+be))*np.e**(-10**X/10**ls)
	integral,abserr=sp.integrate.quad(integrand,42,l)
	return integral

OIINumberDens1=integratelow(al=uncertainties.ufloat(-2.21,0.06),be=uncertainties.ufloat(46.96,2.6))+integratehigh(al=uncertainties.ufloat(-1.59,0.15),be=uncertainties.ufloat(-4.1,0.22),ls=uncertainties.ufloat(43.060,14),l=44)

def integ(x,al,bl):
    return 10**((al+1)*(x-42)+bl)

def integ2(x,al,bl):
    return 10**((al+1)*(x-41.5)+bl)

x=[40.9,41.1,41.3,41.5,41.7]
y10=np.array([10**-1.97,10**-2.6,10**-2.8])
y06=np.array([10**-2.57,10**-2.88,10**-2.9,10**-2.99,10**-3.35])
y03=np.array([10**-2.6,10**-2.88,10**-2.94,10**-3.1,10**-3.5])
y01=np.array([10**-2.8,10**-2.95,10**-3.2,10**-3.6,10**-3.8])

alph01,beta01=uncertainties.correlated_values(sp.optimize.curve_fit(integ,x,y01,p0=[-2.21,43.6])[0],sp.optimize.curve_fit(integ,x,y01,p0=[-2.21,43.6])[1])
alph03,beta03=uncertainties.correlated_values(sp.optimize.curve_fit(integ,x,y03,p0=[-2.21,43.6])[0],sp.optimize.curve_fit(integ,x,y03,p0=[-2.21,43.6])[1])
alph06,beta06=uncertainties.correlated_values(sp.optimize.curve_fit(integ,x,y06,p0=[-2.21,43.6])[0],sp.optimize.curve_fit(integ,x,y06,p0=[-2.21,43.6])[1])
alph1,beta1=uncertainties.correlated_values(sp.optimize.curve_fit(integ2,x[0:3],y10,p0=[-2.21,43.6])[0],sp.optimize.curve_fit(integ2,x[0:3],y10,p0=[-2.21,43.6])[1])

OIINumberDens01=integratelow(al=alph01,be=beta01)+integratehigh(al=uncertainties.ufloat(-1.59,0.15),be=uncertainties.ufloat(-4.1,0.22),ls=uncertainties.ufloat(43.060,0.14),l=45)

OIINumberDens03=integratelow(al=alph03,be=beta03)+integratehigh(al=uncertainties.ufloat(-1.91,0.13),be=uncertainties.ufloat(-4.8,0.34),ls=uncertainties.ufloat(43.4,0.21),l=45)

OIINumberDens06=integratelow(al=alph06,be=beta06)+integratehigh(al=uncertainties.ufloat(-1.88,0.14),be=uncertainties.ufloat(-4.56,0.30),ls=uncertainties.ufloat(43.16,0.18),l=45)

OIINumberDens1=integratelow(al=alph1,be=beta1,ls=41.5,l=41.5)+integratehigh(al=uncertainties.ufloat(-2.05,0.11),be=uncertainties.ufloat(-4.13,0.34),ls=uncertainties.ufloat(42.98,0.21),l=45)

simredshift=np.array([0.1,0.3,0.6,1.0])
OIINUMDENS=np.array([OIINumberDens01,OIINumberDens03,OIINumberDens06,OIINumberDens1])


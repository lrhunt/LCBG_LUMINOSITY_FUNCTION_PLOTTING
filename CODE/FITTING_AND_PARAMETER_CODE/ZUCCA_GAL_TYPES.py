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
def integratethings(ps,ms,al=None,m=None):
	if al==None:
		al=-1.1
	if m==None:
		m=-18.5
	def integrand(X):
		return (0.4*np.log(10)*ps)*(10**(0.4*(al+1)*(ms-X)))*(np.e**(-np.power(10,0.4*(ms-X))))
	integral,abserr=sp.integrate.quad(integrand,-100,m)
	return integral

ZUCCA_Irr_M=unp.uarray([-20.23,-20.73,-20.88,-21.08],[0.18,0.19,0.1,0.07])
ZUCCA_Irr_phi=unp.uarray([0.67e-3,0.47e-3,1.15e-3,2.22e-3],[0.05e-3,0.04e-3,0.06e-3,0.1e-3])

ZUCCA_SP_M=unp.uarray([-20.57,-20.86,-21.14,-21.2],[0.1,0.07,0.06,0.06])
ZUCCA_SP_phi=unp.uarray([2.5e-3,2.36e-3,2.68e-3,2.84e-3],[0.08e-3,0.07e-3,0.08e-3,0.1e-3])

ZUCCA_z=np.array([0.225,0.45,0.65,0.875])

ZUCCA_Irr_N=np.array([integratethings(ZUCCA_Irr_phi[0],ZUCCA_Irr_M[0],al=-1.2,m=-18.5),integratethings(ZUCCA_Irr_phi[1],ZUCCA_Irr_M[1],al=-1.2,m=-18.5),integratethings(ZUCCA_Irr_phi[2],ZUCCA_Irr_M[2],al=-1.2,m=-18.5),integratethings(ZUCCA_Irr_phi[3],ZUCCA_Irr_M[3],al=-1.2,m=-18.5)])
ZUCCA_SP_N=np.array([integratethings(ZUCCA_SP_phi[0],ZUCCA_SP_M[0],al=-1.2,m=-18.5),integratethings(ZUCCA_SP_phi[1],ZUCCA_SP_M[1],al=-1.2,m=-18.5),integratethings(ZUCCA_SP_phi[2],ZUCCA_SP_M[2],al=-1.2,m=-18.5),integratethings(ZUCCA_SP_phi[3],ZUCCA_SP_M[3],al=-1.2,m=-18.5)])
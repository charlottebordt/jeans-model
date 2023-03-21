# This script is used to determine the tNFW parameters (rs, rhos) and (M200, c) for a given total mass
# C. Bordt, June 2022, charlottebordt@gmail.com
import numpy as np
import astropy.units as ut
import astropy.constants as ct
from astropy.units.quantity import Quantity
from scipy.optimize import brentq

from profiles import tNFW

class parameters_from_Mtot:
    '''
    Calculates paraneters for given total mass
    delta_c values: c+: delta_c = 10**(0.3), c-: 10**(-0.3), 
                    c: do not specify or set delta_c = 1 (default option)
    '''
    def __init__(self, M_tot: Quantity, delta_c = 1, debugging = False):
        if debugging: print("Caltulating the parameter sets for the enclosed mass:", M_tot, "\nA truncated NFW profile is used.")
        self.M_tot = M_tot
        self.delta_c = delta_c
        H0 = 70. * ut.km / (ut.Mpc * ut.s)
        self.rho_crit = (3*H0**2 / (8*np.pi*ct.G)).to('M_sun/ pc3')

        self.M200 = self.__finding_M200()
        self.c = self.c_MCR(self.M200)
        if debugging and delta_c != 1: print("delta_c is included in c! \ndelta_c =", delta_c)
        self.rs, self.rhos = self.scale_parameters(self.M200)

    def c_MCR(self, M200, h = 0.7, z = 0):
        M200 = Quantity(M200, 'M_sun')
        delta_c = self.delta_c

        a = 0.520 + (0.905-0.520) * np.exp(-0.617*z**1.21)
        b = -0.101 + 0.026*z
        log10c = a + b * (np.log10(M200.value) - np.log10(1E12 * h**(-1.)))
 
        c=10.**log10c

        return c * delta_c

    def scale_parameters(self, M200):
        M200 = Quantity(M200, 'M_sun')
        c = self.c_MCR(M200) 
        
        rhos = (200*self.rho_crit/3) * c**3/(np.log(1+c) - c/(1+c))
        rs = np.cbrt(3*M200/(4*np.pi*200*self.rho_crit))/c

        return rs.to('kpc'), rhos.to('M_sun/pc3')

    def __finding_M200(self, tol=1e-5):
       
        def problem(M200):
            M_tot = self.M_tot
            
            rs, rhos = self.scale_parameters(M200)
            profile_nfw = tNFW(rs, rhos)
            M = profile_nfw.mass(1E15*ut.kpc)

            return float((M - M_tot)/M_tot)

        m200_min, m200_max = 1E5, 10.**10.5
        M200 = brentq(problem, m200_min, m200_max)
        if abs(float(((tNFW((self.scale_parameters(M200))[0], (self.scale_parameters(M200)[1]))).mass(1E15*ut.kpc) - self.M_tot)/self.M_tot)) > tol:
            print("Root finding was unsuccessful.")

        return Quantity(M200, 'M_sun')

if __name__ == "__main__":
    M_tot = Quantity(5E6, 'M_sun')
    ts = parameters_from_Mtot(M_tot, delta_c = 10**(0.3), debugging = True)
    rhos, rs = ts.rhos, ts.rs

    print("\nrs:", rs)
    print("rhos:", rhos)   
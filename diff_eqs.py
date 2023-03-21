import numpy as np
import astropy.constants as ct

class reduced_jeansModel:
    def __init__(self, x_range = [1e-21, 120.0], refinment=1000):
        self.x_range = x_range
        self.refinment = refinment
        print("Refinment used to be 12000. Please check if refinment = {} is reasonable.".format(self.refinment))
                
    def diff_iso(x, h, z):
        dh = z 
        ddh = -2./x * dh - np.exp(h)        
        return [dh, ddh]

    def boundary_cond(x_min):
        h_0 = 0.
        dh_0 = -x_min/3.
        return [h_0, dh_0]

class reduced_extended_jeansModel:
    def __init__(self, r_scale, nu0_guess, r1, rho1, m1, nu1, refinment=500):
        self.r1 = r1
        self.x_range = [float(r1/r_scale), 1e-21]
        self.refinment = refinment

        self.rho1 = rho1
        self.m1 = m1
        self.nu1 = nu1
        self.nu0 = nu0_guess
        self.mu = - 1./r1 * np.log(nu1/nu0_guess)
        self.rho_tilde = float(np.pi*ct.G*self.rho1/(self.mu**2 * self.nu0**2))

    def diff_iso(self, x, h, z):
        dh = z 
        ddh = - np.exp(h+x) - 2./x * (dh - 1.) + dh - 1.    
        return [dh, ddh]

    def boundary_cond(self, x_1):
        h_1 = np.log(float(self.rho1/self.rho_tilde))
        dh_1 = 1. + (2.*self.mu*np.exp(x_1)*ct.G*self.m1)/(x_1*self.nu0)**2
        return [h_1, dh_1]

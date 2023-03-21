# C. Bordt, 2022
# charlottebordt@gmail.com
import numpy as np
from astropy.units.quantity import Quantity

class NFW:
    '''
    NFW profile of CDM
    '''
    def __init__(self,  r_s: Quantity, rho_s: Quantity):
        try:
            self.r_s = r_s.to('kpc')
            self.rho_s = rho_s.to('solMass pc-3')
        except:
            print("Error: Make sure that all input parameters are in the proper units and in the correct order.") 
            quit()

    def density(self, r: Quantity):
        r = r.to('kpc')
        return self.rho_s/(r/self.r_s * (1 + r/self.r_s)**2)

    def mass(self, r: Quantity):
        r = r.to('kpc')
        x = r/self.r_s
        if x > 1e-5:
            m = 4.*np.pi * self.r_s**3 * self.rho_s * (np.log(1.+x)-x/(1.+x))
        else:
            # Taylor expansion approx for small x
            m = 4.*np.pi * self.rho_s * self.r_s**3 * (0.5*x**2 - (2./3.)*x**3 + 0.75*x**4 - 0.8*x**5)    
        return Quantity(m, 'solMass')  

    def rho(self, r): return self.density(r)    
    def m(self, r): return self.mass(r)       

class tNFW:
    '''
    tNFW profile of CDM
    '''
    def __init__(self, r_s: Quantity, rho_s: Quantity, xt=3,p=5):
        try:
            self.r_s = r_s.to('kpc')
            self.rho_s = rho_s.to('solMass pc-3')
        except:
            print("Error: Make sure that all input parameters are in the proper units and in the correct order.") 
            quit()
        self.xt = xt
        self.p = p
        self.profile_nfw = NFW(r_s, rho_s)
         
    def density(self, r: Quantity):
        rt = self.xt*self.r_s
        if r < rt:
            return self.profile_nfw.density(r)
        else:
            return self.profile_nfw.density(rt)*(rt/r)**self.p
    
    def mass(self, r: Quantity):
        rt = self.xt*self.r_s
        if r < rt:
            return self.profile_nfw.mass(r)
        else:
            return self.profile_nfw.mass(rt) + 4*np.pi*rt**3*self.profile_nfw.density(rt)*(1-(rt/r)**(self.p-3))/(self.p-3) 

    def rho(self, r): return self.density(r)    
    def m(self, r): return self.mass(r)           

class Henquist_Bonaca:
    def __init__(self, a = Quantity(0.01, 'kpc'), M_Hern = Quantity(5e6, 'M_sun')):
        self.a = a
        self.M_Hern = M_Hern

    def density(self, r: Quantity):
        r = r.to('kpc')
        return Quantity(self.M_Hern * self.a/(2*np.pi*r)/(r+self.a)**3, 'solMass pc-3')

    def mass(self, r: Quantity):
        r = r.to('kpc') 
        return self.M_Hern * r ** 2 * (self.a + r)**(-2)   

    def rho(self, r): return self.density(r)    
    def m(self, r): return self.mass(r)       
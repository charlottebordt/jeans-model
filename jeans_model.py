# C. Bordt, 2022-2023
# charlottebordt@gmail.com
import os
import pickle
import numpy as np
import astropy.units as ut
import astropy.constants as ct
from astropy.units.quantity import Quantity
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root
from scipy.integrate import quad

from profiles import NFW, tNFW
from diff_eqs import reduced_jeansModel, reduced_extended_jeansModel
from numerical_methods import RK4


class SIDM:
    '''

    This class will return the mass, mass density and velocity distribution modelling a SIDM halo. 

    - r_s [length], rho_s[mass over volume], sigma_over_m[area over mass], t_age[time] are the defining halo quantities 
    - the truncated flg enables matching to a tNFW profile instead of an NFW profile
    - debugging returns more information as the halo is being computed
    - The class decides whether to use the regular or the extended version of the JM. If you persist on computing only the 
        regular JM use force_classicalJM. The collapsing flag might be of interest to you as well if two solutions exist.
    - The halo will be saved in a data folder. If a solution already exists, it will load said solution. However, you are able 
        to make some adjustments:
        saving can be turned off with saveHalo = False
        you can repress loading the halo with repress_loadHalo = True (you might want to turn saveing off too)
        the output folder can be renamed with the dir_output option
        the filename can be chosen manually with the filename option
        (Please make sure when loading a file with a manually chosen filename that all other parameters loaded into the halo align.)
    
    '''
    def __init__(self, r_s: Quantity, rho_s: Quantity, sigma_over_m: Quantity, t_age: Quantity,
                    truncated: bool = False, debugging: bool = False, force_classicJM: bool = False, collapsing = None,
                    saveHalo: bool = True, repress_loadHalo: bool = False, dir_output = None, filename = None) -> None:
        # Constants
        try:
            self.sigma_m = sigma_over_m.to('cm2 g-1')
            self.t_age = t_age.to('Gyr')
            self.r_s = r_s.to('kpc')
            self.rho_s = rho_s.to('solMass pc-3')
        except:
            print("Error: Make sure that all input parameters are in the proper units and in the correct order.")
            print("Expected order: r_s, rho_s, sigma_over_m, t_age")
            quit() 
        
        self.debugging = debugging 

        self.truncated = truncated
        self.force_classicJM = force_classicJM        
        self.collapsing_intend = collapsing 

        # saving and loading the halo 
        if dir_output is not None: self.dir_output = dir_output
        else: self.dir_output = 'data'
        if filename is not None: self.fname = filename
        else: self.fname = self.__get_filename()
        self.repress_loadHalo = repress_loadHalo
        self.saveHalo = saveHalo
        
        # To-Do-List: 
        if False:# self.debugging: 
            print('''To-Do-List:
            - check if both tolerances make sense
            - nu0_guess for different r_s, rho_s
            - for later: hdf5 instaed of pickle\n''')

        # Constructing the halo    
        if self.truncated: self.profile_nfw = tNFW(r_s, rho_s)
        else: self.profile_nfw = NFW(r_s, rho_s) 

        self.__solving_jeansEquations()  
        if self.saveHalo: self.__save_solution()

    def __nu0_guess(self, sigma_m, t):
        t_tilde = (t*sigma_m).to('Gyr cm2 g-1').value

        data = pickle.load(open('nu0_evolution.pickle', 'rb'))
        sigma_0 = interp1d(data['t_times_sigmam'], data['sigma0'], kind='cubic', fill_value='extrapolate')

        return Quantity(sigma_0(t_tilde), 'km s-1')

    def _rate_equation(self, r1, nu_r1= None, nfw_only=False):
        r1 = Quantity(r1, 'kpc')
        if self.flag_extended:
            return 1. - float(self.profile_nfw.density(r1) * 4./np.sqrt(np.pi) * self.velocity_dispersion(r1, nfw_only=nfw_only) * self.sigma_m * self.t_age)
        else:
            return 1. - float(self.profile_nfw.density(r1) * 4./np.sqrt(np.pi) * nu_r1 * self.sigma_m * self.t_age)

    def __finding_r1(self):
        if self.flag_extended: 
            rate_eq = lambda r: self._rate_equation(r, nfw_only=True)
        else: 
            rate_eq = lambda r: self._rate_equation(r, nu_r1=self.nu0)
        
        r1 = brentq(rate_eq, 1e-4*self.r_s.value, 1e2*self.r_s.value)
        return Quantity(r1, 'kpc')

    def __solving_classicalJE(self):
        self.flag_extended = False

        if os.path.exists("reduced_jeansModel.pickle"):
            data = pickle.load(open('reduced_jeansModel.pickle', 'rb'))
            self.x_list, self.h_list, self.dh_list, self.j_list = data['x_list'], data['h_list'], data['dh_list'], data['j_list']
        else:
            jeans_ode = reduced_jeansModel()
            self.x_list, self.h_list, self.dh_list = self.__solving_DGLs(jeans_ode)
            self.j_list = np.log(-3./np.array(self.x_list) * np.array(self.dh_list)) 

            # saving the solution for later uses
            data = {'x_list': self.x_list, 'h_list': self.h_list, 'dh_list': self.dh_list, 'j_list': self.j_list}

            with open("reduced_jeansModel.pickle", 'wb') as fopen:
                pickle.dump(data, fopen)    

        self.h = interp1d(self.x_list, self.h_list, kind='cubic')
        self.dh = interp1d(self.x_list, self.dh_list, kind='cubic')
        self.j = interp1d(self.x_list, self.j_list, kind='cubic')

        return None

    def __solving_extendedJE(self, nu0, nu1):
        self.flag_extended = True
        self.r_scale = -self.r1/(2.*np.log(nu1/nu0)) #-2./self.r1*np.log(nu1/self.nu0)
        self.rho_characteristic = self.profile_nfw.density(self.r1)
        
        jeans_ode = reduced_extended_jeansModel(self.r_scale, nu0, self.r1, self.profile_nfw.density(self.r1), self.profile_nfw.mass(self.r1),  nu1)
        self.x_list, self.h_list, dh_list = self.__solving_DGLs(jeans_ode)
        self.h = interp1d(self.x_list, self.h_list, kind='cubic')

        return None

    def __solving_DGLs(self, ode):
        x_list, h_list, dh_list = RK4.RK4_2nd(ode.diff_iso, ode.x_range[0], ode.boundary_cond(ode.x_range[0])[0], 
                                ode.boundary_cond(ode.x_range[0])[1], ode.x_range[1], ode.refinment, log_steps=True)
        return  x_list, h_list, dh_list

    def __transforming_classicalJE(self):
                    
        def rate_eq(r1, nu0): 
            return self._rate_equation(r1*ut.kpc, nu_r1=nu0*ut.km/ut.s)
                    
        def scale(r1, x1, nu0):
            r1, nu0 = Quantity(r1, 'kpc'), Quantity(nu0, 'km s-1')
            rho_0 = self.profile_nfw.density(r1) * np.exp(-self.h(x1))
            return  x1 / float(r1/(nu0/np.sqrt(4.*np.pi*ct.G*rho_0))) - 1.
                    
        def ratio(r1, x1): 
            r1 = Quantity(r1, 'kpc')
            ratio_iso = - 1./x1 * self.dh(x1) * np.exp(-self.h(x1))
            ratio_nfw = self.profile_nfw.mass(r1)/(4.*np.pi * self.profile_nfw.density(r1)*r1**3)
            return ratio_iso - ratio_nfw

        #par = [r1, x1, nu0]
        def problem(par): return [rate_eq(par[0], par[2]), scale(par[0], par[1], par[2]), ratio(par[0], par[1])]

        # guess to 3D problem
        help_function = lambda x: ratio(self.r1.value, x)
        
        try: 
            x1_guess = brentq(help_function, self.x_list[0], self.x_list[-1])
        except:   
            if self.collapsing_intend:
                x1_guess = brentq(help_function, 22.544, self.x_list[-1])
            else:
                x1_guess = brentq(help_function, self.x_list[0], 22.544)
        
        guess = [self.r1.value, x1_guess, self.nu0_guess.value]

        try:
            sol = root(problem, guess, method='lm')
        except:
            if self.debugging: print("linearmixing failed.")
            try:
                sol = root(problem, guess, method='broyden1')
            except:
                if self.debugging: print("Broyden1 failed.")
                try: 
                    sol = root(problem, guess, method='broyden2')
                except:  
                    if self.debugging: print("Broyden2 failed.")
                    try: 
                        sol = root(problem, guess, method='diagbroyden') 
                    except:
                        print("Both root finding methods are unsucessful, please choose another one manually.")
                        quit() 

        r1, x1, nu0 = sol.x
        if not sol.success: print("Root finding success:", sol.success)

        self.r1, self.x1, self.nu0 = Quantity(r1, 'kpc'), x1, Quantity(nu0, 'km s-1')
        
        self.rho_characteristic = self.profile_nfw.density(self.r1) * np.exp(-self.h(x1))
        self.r_scale = (self.nu0/np.sqrt(4.*np.pi*ct.G*self.rho_characteristic)).to('kpc')

        return None

    def __solving_jeansEquations(self):

        if os.path.exists(self.fname) and not self.repress_loadHalo:
            self.saveHalo = False
            self.__load_solution()

            self.h = interp1d(self.x_list, self.h_list, kind='cubic')
            if self.flag_extended: 
                self.dh = interp1d(self.x_list, self.dh_list, kind='cubic')
                self.j = interp1d(self.x_list, self.j_list, kind='cubic')

        else:
            self.flag_extended = True
            self.nu0_guess = self.__nu0_guess(self.sigma_m, self.t_age)       

            self.r1 = self.__finding_r1()
            nu1 = self.velocity_dispersion(self.r1, nfw_only=True)

            # deciding whether classical or regular jeans model will be solved
            if float(nu1/self.nu0_guess) > 0.95 or self.force_classicJM:
                if self.debugging and not self.force_classicJM:
                    print("The velocity dispersion can be modelled as constant. Thus, the regular Jeans Model will be used.")  
                self.__solving_classicalJE()
                self.__transforming_classicalJE()  

                if self.velocity_dispersion(self.r1) > 0.95 * self.velocity_dispersion(1e-10*self.r_s):
                    if self.debugging: print("Successfully found a solution to the regular Jeans equations.")            
                else: print("Try the extended version of the Jeans equations. Currently unsupported loop.") 
                    
            else:
                self.__solving_extendedJE(self.nu0_guess, nu1)
                self.nu0 = self.velocity_dispersion(1e-10*self.r_s) # place holder 
                print('Root- finding option is currently unsopported. Needs to be extended to {rho_0, nu_0}_guess')

                # Shooting algorithm
                if False:
                    def shooting_over_nu0(nu0):
                        nu0 = Quantity(nu0, 'km s-1')
                        nu1 = self.velocity_dispersion(self.r1, nfw_only=True)
                        
                        self.__solving_extendedJE(nu0, nu1)
                        nu0_comp = self.velocity_dispersion(1e-10*self.r_s)

                        return float((nu0_comp - nu0)/nu0)
                    
                    self.nu0 = brentq(shooting_over_nu0, self.nu0_guess + 3.*ut.km/ut.s, self.nu0_guess - 3.*ut.km/ut.s)

            # check if found solution is valid + characterise growing/collapsing
            self.pde_check = self.__check_PDEs() 
            self.collapsing = self.__check_collapse()                                              

        return None

    def _delta_U(self, r):
        # Potential
        r = Quantity(r, 'kpc')
        integrand = lambda r: (- ct.G/2. * (self.mass(r*ut.kpc)**2 - self.profile_nfw.mass(r*ut.kpc)**2)/(r*ut.kpc)**2).to('solMass kpc/s2').value
        integral = quad(integrand, 0., r.value)
        if integral[1] != 0 and integral[1]/integral[0] > 1e-3:
            print("Estimation of delta U has an error of {:.2f}%.".format(integral[1]/integral[0]))

        return Quantity(integral[0], 'solMass kpc2/s2')
    
    # return bool: True == core collapsing, False == core growing
    def __check_collapse(self):
        if self._delta_U(self.r1).value < 0.:
            if self.debugging: print("Found solution is collapsing.")
            col = True
        else:
            if self.debugging: print("This is a core growing solution.")
            col = False
        
        if self.collapsing_intend != None and col != self.collapsing_intend:
            print("The found solution is collapsing: {}. You intended to find a solution that is collapsing: {}".format(col, self.collapsing_intend))
        return col   

    def __check_PDEs(self, delta_r=1e-5*ut.pc, tol=1e-3):   
        r_range = np.logspace(-3., np.log10(self.r1.value), num=100)*ut.kpc

        def first_pde(r):
            left_side = (self.density(r + delta_r)*self.velocity_dispersion(r + delta_r) 
                         - self.density(r - delta_r)*self.velocity_dispersion(r - delta_r) )/(2.*delta_r)
            #left_side = self.density(r)*self.dh(r*self.x1/self.r1)
            right_side = - ct.G * self.density(r) * self.mass(r) / r**2
            diff = float(left_side / right_side) - 1. 
            return np.absolute(diff)
        
        def second_pde(r): 
            left_side = (self.mass(r + delta_r) - self.mass(r - delta_r))/(2.*delta_r)
            right_side = 4.*np.pi * r**2 * self.density(r)
            diff = float(left_side / right_side) - 1. 
            return np.absolute(diff)

        diff_1, diff_2 = [], []
        for ri in r_range:
            diff_1.append(first_pde(ri).value)
            diff_2.append(second_pde(ri).value)      

        if self.debugging:
                print("PDE 1 diff = ", np.amax(diff_1))
                print("PDE 2 diff = ", np.amax(diff_2))

        if np.amax(diff_1) < tol and np.amax(diff_2) < tol:           
            return True
        else:
            print("Error: The found solutions are not solutions to the Jeans equations.")
            return False

    def __get_filename(self):
        prefix = ""
        if self.force_classicJM: prefix = prefix + "fcJM_"
        if self.truncated: prefix = prefix + "tNFW_"
        
        def replacing_dots(val): return format(val).replace('.', 'd')

        fname = os.path.join(self.dir_output, prefix + 'sigmam{}_t{}_rs{}_rhos{}.pickle'.format(int(self.sigma_m.value), 
                    replacing_dots(self.t_age.value), replacing_dots(self.r_s.value), replacing_dots(self.rho_s.value)))         
        return fname

    def __save_solution(self):
        os.makedirs(self.dir_output, exist_ok=True)
        
        data = {
            't': self.t_age, 'sigma_over_m': self.sigma_m, 'r_s': self.r_s, 'rho_s': self.rho_s,
            'r1': self.r1, 'r_scale': self.r_scale, 'rho_characteristic': self.rho_characteristic, 'nu_0': self.nu0, 
            'truncated': self.truncated, 'collapsing': self.collapsing, 
            'force_classicJM': self.force_classicJM, 'extended_JM': self.flag_extended,
            'x_list': self.x_list, 'h_list': self.h_list
            }

        if not self.flag_extended:
            data['dh_list'] = self.dh_list
            data['j_list'] = self.j_list

        with open(self.fname, 'wb') as fopen:
            pickle.dump(data, fopen)
        
        print("The solution saved at", self.fname)
        return None

    def __load_solution(self):
        if self.debugging: print("Loading solution:", self.fname)

        data = pickle.load(open(self.fname, 'rb'))
        if data['r_s'] != self.r_s or data['rho_s'] != self.rho_s or data['sigma_over_m'] != self.sigma_m or data['t'] != self.t_age:
            print('''Error: Loaded vs specified defining halo parameters are unequal.
                    Loaded parameters are:''', data['r_s'], data['rho_s'], data['sigma_over_m'], data['t'])
            print("Specified parameters are:", self.r_s, self.rho_s, self.sigma_m, self.t_age)
            quit()

        self.r1, self.r_scale, self.rho_characteristic, self.nu0, self.flag_extended, self.x_list, self.h_list, self.collapsing = data['r1'], \
            data['r_scale'], data['rho_characteristic'], data['nu_0'], data['extended_JM'], data['x_list'], data['h_list'], data['collapsing']
        if self.debugging: print("Missing in Loading Halo: PDE check.")

        if not self.flag_extended:
            #self.dh_list = data['dh_list']
            self.j_list = data['j_list']

        return None

    def density(self, r:Quantity):
        r = r.to('kpc')
        if r < self.r1:
            return self.rho_characteristic * np.exp(self.h(float(r/self.r_scale)))
        else:
            return self.profile_nfw.density(r) 

    def mass(self, r:Quantity):
        r = r.to('kpc')
        if r < self.r1:
            if self.flag_extended:                
                def integrand(r):
                    r = Quantity(r, 'kpc')
                    return (4.*np.pi * r**2 * self.density(r)).to('solMass kpc-1').value
                mnum = quad(integrand, 0., r.value)
                if mnum[1] != 0 and mnum[1]/mnum[0] > 1e-3:
                    print("Estimation of mass at r = {} has an error of {:.2f}%.".format(r, mnum[1]/mnum[0]))
                return Quantity(mnum[0], 'solMass')             
            
            else:
                return Quantity(4.*np.pi/3. * self.rho_characteristic * r**3 * np.exp(self.j(r/self.r_scale)), 'solMass')
        else:
            return self.profile_nfw.mass(r) 

    def velocity_dispersion(self, r:Quantity, nfw_only=False):
        r = r.to('kpc')
        if nfw_only == True:
            mass = lambda r: self.profile_nfw.mass(r)
            density = lambda r: self.profile_nfw.density(r)
        else:
            mass = lambda r: self.mass(r)
            density = lambda r: self.density(r)
        
        def p(r):
            rmax = np.inf*ut.kpc#1E15*ut.kpc
            def integrand(r):
                r = Quantity(r, 'kpc')
                return ((ct.G*density(r)*mass(r))/r**2).to('solMass kpc-2 s-2').value      
            pnum = Quantity(quad(integrand,r.value,rmax.value)[0], 'solMass kpc-1 s-2')
            return pnum  
        
        return np.sqrt((p(r)/(density(r))).to('km2 s-2'))

    def nu_ansatz(self, r):
        r = r.to('kpc')

        nu1 = self.velocity_dispersion(self.r1, nfw_only=True)
        mu = -1/self.r1 * np.log(nu1/self.nu0)        
        ansatz = self.nu0 * np.exp(-mu*r)
        
        return Quantity(ansatz, 'km s-1')

    def rho(self, r): return self.density(r)    
    def m(self, r): return self.mass(r)    
    def nu(self, r): return self.velocity_dispersion(r)    


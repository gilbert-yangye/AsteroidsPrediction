import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import math
plt.style.use('ggplot')

def plot_analytical(numerical=False,num_result=None):
    if numerical:
        z_position= num_result.altitude #altitude
    else:
        z_position=np.linspace(0,100000,1001) 
    v_position=np.zeros(501) #velocity
    energy_position=np.zeros(501) #energy
    dv_dz_position=np.zeros(501) #dv/dt
    dE_dz_position=np.zeros(501) #dE/dz

    test_v0 = 21000 #velocity
    test_theta0 = 45 #angle/radian
    test_r0 = 10 #radius
    test_density=3000 #density
    test_m0 = test_density*4/3*np.pi*test_r0**3 #mass

    test_list0= np.array([test_v0, test_m0, test_theta0, test_r0]) #test example

    def v_z(test_list, z, Cd=1, rho0=1.2, H=8000, angle = True):
        # v(z) Calculating velocity from altitude.

        rhoa = rho0*np.exp(-z/H) # atmospheric density
        A = np.pi*test_list[3]**2 # area

        # angle/radian transformation
        if angle:
            radian=test_list[2]*np.pi/180.
        else:
            radian=test_list[2]

        return test_list[0]*np.exp(-H*Cd*rho0*np.exp(-z/H)*A/(2*test_list[1]*np.sin(radian)))



    def dv_dz(test_list, z, Cd=1, rho0=1.2, H=8000, angle = True):
        # dv/dz Calculating derivative of velocity from altitude

        rhoa = rho0*np.exp(-z/H) # atmospheric density
        A = np.pi*test_list[3]**2 # area

        # angle/radian transformation
        if angle:
            radian=test_list[2]*np.pi/180.
        else:
            radian=test_list[2]

        return v_z(test_list, z)*Cd*rho0*np.exp(-z/H)*A/(2*test_list[1]*np.sin(radian))

    def dE_dz(test_list, z, Cd=1, rho0=1.2, H=8000, angle = True):
        # dE/dz Calculating derivative of energy from altitude - chain rule

        rhoa = rho0*np.exp(-z/H) # atmospheric density
        A = np.pi*test_list[3]**2 # area

        # angle/radian transformation
        if angle:
            radian=test_list[2]*np.pi/180.
        else:
            radian=test_list[2]

        return dv_dz(test_list, z)*v_z(test_list, z)*test_list[1]/4.184e9
    
    v_position, dE_dz_position = v_z(test_list0, z_position),dE_dz(test_list0, z_position)
    
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(111)
    ax1.plot(v_position, z_position, label='analytical data')
    if numerical:
        ax1.plot(num_result.velocity,num_result.altitude, label= 'numerical data')
        ax1.set_title('Analytical Solution V.S Numerical Solution\n Velocity-Altitude', fontsize=16)
    else:
        ax1.set_title('Analytical Solution\n Velocity-Altitude', fontsize=16)
    ax1.set_xlabel(r'$v, velocity$', fontsize=14)
    ax1.set_ylabel(r'$z, Altitude$', fontsize=14)
    
    ax1.legend(loc='best', fontsize=14)


    fig = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(111)
    ax2.plot(dE_dz_position, z_position, label='analytical data' )
    if numerical:
        ax2.plot(num_result.dedz,num_result.altitude, label= 'numerical data')
        ax2.set_title('Analytical Solution V.S Numerical Solution\n Energy Loss-Altitude', fontsize=16)
    else:
        ax2.set_title('Analytical Solution\n Energy Loss-Altitude', fontsize=16)
    ax2.set_xlabel(r'$dEdz, Energy Change per unit Height$', fontsize=14)
    ax2.set_ylabel(r'$z, Altitude$', fontsize=14)
    
    ax2.legend(loc='best', fontsize=14)
    print('Analytical Solution under the conditions:\ng=0; R_P=infinity; C_L=0; no ablation; no fragmentation')
    
    plt.show()
    return v_position,dE_dz_position
    
    

class scipy_solution():
    def __init__(self,setting,input_data):
        self.alldata = {**setting, **input_data}
        if self.alldata['radians']:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                  self.alldata['angle'],self.alldata['init_altitude'],0.,self.alldata['radius']])
        else:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                  self.deg_to_rad(self.alldata['angle']),self.alldata['init_altitude'],0.,self.alldata['radius']])
        
    def deg_to_rad(self,deg):
        """
        Returns an angle in radians
        for a given angle in degrees
        """
        return deg*np.pi/180

    def rad_to_deg(self,rad):
        """
        Returns an angle in degrees
        for a given angle in radians
        """
        return rad*180 / np.pi
    
    def simulation(self,t,parameters):

        Cd,g,Ch,Q,Cl,Rp,alpha,rhom,rho0,H,Y = (self.alldata['Cd'],self.alldata['g'],self.alldata['Ch'],self.alldata['Q'],self.alldata['Cl'],self.alldata['Rp'],
                                                self.alldata['alpha'],self.alldata['density'],self.alldata['rho0'],self.alldata['H'],self.alldata['strength'])

        v,m,theta,z,x,r = parameters
        A = np.pi*r**2
        rhoa = rho0*np.exp(-z/H)

        return np.array([-Cd*rhoa*A*v**2/(2*m)+g*np.sin(theta),
                         -Ch*rhoa*A*v**3/(2*Q),
                         g*np.cos(theta)/v-Cl*rhoa*A*v/(2*m)-v*np.cos(theta)/(Rp+z),
                         -v*np.sin(theta),
                         v*np.cos(theta)/(1+z/Rp),
                         (np.sqrt(7/2*alpha*(rhoa/rhom))*v if rhoa*v**2 >= Y else 0)])
    
    def sci_entry(self):
        
        t_0=self.alldata['t_0']
        t_end = self.alldata['t_end']
        t_step = self.alldata['dt']
        tol=self.alldata['tol']

        sci_result = solve_ivp(self.simulation, [t_0,t_end],self.initial_condition,
                         t_eval = np.arange(t_0,t_end,t_step),method='RK45',atol=tol,rtol=tol)

        sci_result = pd.DataFrame({'time':sci_result.t, 'velocity':sci_result.y[0],'mass':sci_result.y[1],
                            'angle':self.rad_to_deg(sci_result.y[2]),'altitude':sci_result.y[3],'distance':sci_result.y[4],'radius':sci_result.y[5]})


        Index_label = sci_result.query('velocity < 0').index.tolist()
        Index_label += sci_result.query('altitude < 0').index.tolist()
        Index_label += sci_result.query('mass < 0').index.tolist()
        Index_label += [1000000]
        min_index = min(Index_label)
        sci_result = sci_result.drop(sci_result.index[min_index:])


        if sci_result.altitude.nunique() != len(sci_result.altitude):
            sci_result = sci_result.drop_duplicates(subset='altitude', keep="first")
            sci_result = sci_result.reset_index()

        e = 0.5*sci_result['mass']*sci_result['velocity']**2

        dedz = e.diff(2)/sci_result.altitude.diff(2)/(4.184*1e9) #center approximation, with 2 nan, filled with 0 later
        dedz = dedz.shift(-1)
        dedz = dedz.fillna(0)

        sci_result = sci_result.copy()
        sci_result.insert(len(sci_result.columns),'dedz', dedz)
        sci_result = sci_result.drop(sci_result.index[-1])
        return sci_result
    
def plot_single(result,title):
        
    fig = plt.figure(figsize=(15,10))

    fig.suptitle(title, fontsize=16)
        
    plt.subplot(331)
    plt.plot(result.time,result.velocity)# t-v
    plt.title('t-v', fontsize=16)

    plt.subplot(332)
    plt.plot(result.time,result.mass) # t-m
    plt.title('t-m', fontsize=16)

    plt.subplot(333)
    plt.plot(result.time,result.angle) # t-theta
    plt.title('t-angle', fontsize=16)

    plt.subplot(334)
    plt.plot(result.time,result.altitude) # t-z
    plt.title('t-z', fontsize=16)

    plt.subplot(335)
    plt.plot(result.time,result.distance) # t-x
    plt.title('t-x', fontsize=16)

    plt.subplot(336)
    plt.plot(result.time,result.radius) # t-r
    plt.title('t-r', fontsize=16)

    plt.subplot(337)
    plt.plot(result.altitude,result.velocity) # z-v
    plt.title('z-v', fontsize=16)

    plt.subplot(338)
    plt.plot(result.dedz,result.altitude) # energy-z
    plt.title('de-dz', fontsize=16)

    plt.subplot(339)
    plt.plot(result.distance,result.altitude) # x-z
    plt.title('x-z', fontsize=16)

    plt.show()
    
def plot_contrast(result1,result2, title = 'SolScipy solution V.S Solver'):
    
    fig = plt.figure(figsize=(15,10))

    fig.suptitle(title, fontsize=16)

    plt.subplot(331)
    plt.plot(result1.time,result1.velocity,'r',label = 'solver')# t-v
    plt.plot(result2.time,result2.velocity,'b',label = 'scipy')
    plt.legend()
    plt.title('t-v', fontsize=16)

    plt.subplot(332)
    plt.plot(result1.time,result1.mass,'r',label = 'solver')
    plt.plot(result2.time,result2.mass,'b',label = 'scipy')# t-m
    plt.legend()
    plt.title('t-m', fontsize=16)

    plt.subplot(333)
    plt.plot(result1.time,result1.angle,'r',label = 'solver') # t-theta
    plt.plot(result2.time,result2.angle,'b',label = 'scipy')
    plt.legend()
    plt.title('t-angle', fontsize=16)

    plt.subplot(334)
    plt.plot(result1.time,result1.altitude,'r',label = 'solver') # t-z
    plt.plot(result2.time,result2.altitude,'b',label = 'scipy')
    plt.legend()
    plt.title('t-z', fontsize=16)

    plt.subplot(335)
    plt.plot(result1.time,result1.distance,'r',label = 'solver') # t-x
    plt.plot(result2.time,result2.distance,'b',label = 'scipy')
    plt.legend()
    plt.title('t-x', fontsize=16)

    plt.subplot(336)
    plt.plot(result1.time,result1.radius,'r',label = 'solver') # t-r
    plt.plot(result2.time,result2.radius,'b',label = 'scipy')
    plt.legend()
    plt.title('t-r', fontsize=16)

    plt.subplot(337)
    plt.plot(result1.altitude,result1.velocity,'r',label = 'solver') # z-v
    plt.plot(result2.altitude,result2.velocity,'b',label = 'scipy')
    plt.legend()
    plt.title('z-v', fontsize=16)

    plt.subplot(338)
    plt.plot(result1.dedz,result1.altitude,'r',label = 'solver') # energy-z
    plt.plot(result2.dedz,result2.altitude,'b',label = 'scipy')
    plt.legend()
    plt.title('dedz-z', fontsize=16)

    plt.subplot(339)
    plt.plot(result1.distance,result1.altitude,'r',label = 'solver') # x-z
    plt.plot(result2.distance,result2.altitude,'b',label = 'scipy')
    plt.legend()
    plt.title('x-z', fontsize=16)

    plt.show()



class scipy_solution_mars():
    def __init__(self,setting,input_data):
        self.alldata = {**setting, **input_data}
        if self.alldata['radians']:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                  self.alldata['angle'],self.alldata['init_altitude'],0.,self.alldata['radius']])
        else:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                  self.deg_to_rad(self.alldata['angle']),self.alldata['init_altitude'],0.,self.alldata['radius']])
        
    def deg_to_rad(self,deg):
        """
        Returns an angle in radians
        for a given angle in degrees
        """
        return deg*np.pi/180

    def rad_to_deg(self,rad):
        """
        Returns an angle in degrees
        for a given angle in radians
        """
        return rad*180 / np.pi
    
    def simulation(self,t,parameters):

        Cd,g,Ch,Q,Cl,Rp,alpha,rhom,rho0,H,Y = (self.alldata['Cd'],self.alldata['g'],self.alldata['Ch'],self.alldata['Q'],self.alldata['Cl'],self.alldata['Rp'],
                                                self.alldata['alpha'],self.alldata['density'],self.alldata['rho0'],self.alldata['H'],self.alldata['strength'])

        v,m,theta,z,x,r = parameters
        A = np.pi*r**2
        rhoa = 0.699*np.exp(-0.00009*z)/(0.1921*(249.7-0.00222*z)) if z >= 7000 else 0.699*np.exp(-0.00009*z)/(0.1921*(242.1-0.000998*z))

        return np.array([-Cd*rhoa*A*v**2/(2*m)+g*np.sin(theta),
                         -Ch*rhoa*A*v**3/(2*Q),
                         g*np.cos(theta)/v-Cl*rhoa*A*v/(2*m)-v*np.cos(theta)/(Rp+z),
                         -v*np.sin(theta),
                         v*np.cos(theta)/(1+z/Rp),
                         (np.sqrt(7/2*alpha*(rhoa/rhom))*v if rhoa*v**2 >= Y else 0)])
    
    def sci_entry(self):
        
        t_0=self.alldata['t_0']
        t_end = self.alldata['t_end']
        t_step = self.alldata['dt']
        tol=self.alldata['tol']

        sci_result = solve_ivp(self.simulation, [t_0,t_end],self.initial_condition,
                         t_eval = np.arange(t_0,t_end,t_step),method='RK45',atol=tol,rtol=tol)

        sci_result = pd.DataFrame({'time':sci_result.t, 'velocity':sci_result.y[0],'mass':sci_result.y[1],
                            'angle':self.rad_to_deg(sci_result.y[2]),'altitude':sci_result.y[3],'distance':sci_result.y[4],'radius':sci_result.y[5]})


        Index_label = sci_result.query('velocity < 0').index.tolist()
        Index_label += sci_result.query('altitude < 0').index.tolist()
        Index_label += sci_result.query('mass < 0').index.tolist()
        Index_label += [1000000]
        min_index = min(Index_label)
        sci_result = sci_result.drop(sci_result.index[min_index:])


        if sci_result.altitude.nunique() != len(sci_result.altitude):
            sci_result = sci_result.drop_duplicates(subset='altitude', keep="first")
            sci_result = sci_result.reset_index()

        e = 0.5*sci_result['mass']*sci_result['velocity']**2

        dedz = e.diff(2)/sci_result.altitude.diff(2)/(4.184*1e9) #center approximation, with 2 nan, filled with 0 later
        dedz = dedz.shift(-1)
        dedz = dedz.fillna(0)

        sci_result = sci_result.copy()
        sci_result.insert(len(sci_result.columns),'dedz', dedz)
        sci_result = sci_result.drop(sci_result.index[-1])
        return sci_result
    
    
    
class scipy_solution_constant():
    def __init__(self,setting,input_data):
        self.alldata = {**setting, **input_data}
        if self.alldata['radians']:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                self.alldata['angle'],self.alldata['init_altitude'],0.,self.alldata['radius']])
        else:
            self.initial_condition = np.array([self.alldata['velocity'],self.alldata['density']*self.alldata['radius']**3*4/3*np.pi,
                                self.deg_to_rad(self.alldata['angle']),self.alldata['init_altitude'],0.,self.alldata['radius']])
        
    def deg_to_rad(self,deg):
        """
        Returns an angle in radians
        for a given angle in degrees
        """
        return deg*np.pi/180

    def rad_to_deg(self,rad):
        """
        Returns an angle in degrees
        for a given angle in radians
        """
        return rad*180 / np.pi
    
    def simulation(self,t,parameters):

        Cd,g,Ch,Q,Cl,Rp,alpha,rhom,rho0,H,Y = (self.alldata['Cd'],self.alldata['g'],self.alldata['Ch'],self.alldata['Q'],self.alldata['Cl'],self.alldata['Rp'],
                                                self.alldata['alpha'],self.alldata['density'],self.alldata['rho0'],self.alldata['H'],self.alldata['strength'])

        v,m,theta,z,x,r = parameters
        A = np.pi*r**2
        rhoa = rho0

        return np.array([-Cd*rhoa*A*v**2/(2*m)+g*np.sin(theta),
                        -Ch*rhoa*A*v**3/(2*Q),
                        g*np.cos(theta)/v-Cl*rhoa*A*v/(2*m)-v*np.cos(theta)/(Rp+z),
                        -v*np.sin(theta),
                        v*np.cos(theta)/(1+z/Rp),
                        (np.sqrt(7/2*alpha*(rhoa/rhom))*v if rhoa*v**2 >= Y else 0)])
    
    def sci_entry(self):
        
        t_0=self.alldata['t_0']
        t_end = self.alldata['t_end']
        t_step = self.alldata['dt']
        tol=self.alldata['tol']

        sci_result = solve_ivp(self.simulation, [t_0,t_end],self.initial_condition,
                        t_eval = np.arange(t_0,t_end,t_step),method='RK45',atol=tol,rtol=tol)

        sci_result = pd.DataFrame({'time':sci_result.t, 'velocity':sci_result.y[0],'mass':sci_result.y[1],
                            'angle':self.rad_to_deg(sci_result.y[2]),'altitude':sci_result.y[3],'distance':sci_result.y[4],'radius':sci_result.y[5]})


        Index_label = sci_result.query('velocity < 0').index.tolist()
        Index_label += sci_result.query('altitude < 0').index.tolist()
        Index_label += sci_result.query('mass < 0').index.tolist()
        Index_label += [1000000]
        min_index = min(Index_label)
        sci_result = sci_result.drop(sci_result.index[min_index:])


        if sci_result.altitude.nunique() != len(sci_result.altitude):
            sci_result = sci_result.drop_duplicates(subset='altitude', keep="first")
            sci_result = sci_result.reset_index()

        e = 0.5*sci_result['mass']*sci_result['velocity']**2

        dedz = e.diff(2)/sci_result.altitude.diff(2)/(4.184*1e9) #center approximation, with 2 nan, filled with 0 later
        dedz = dedz.shift(-1)
        dedz = dedz.fillna(0)

        sci_result = sci_result.copy()
        sci_result.insert(len(sci_result.columns),'dedz', dedz)
        sci_result = sci_result.drop(sci_result.index[-1])
        return sci_result
    
    
def get_solver_result(alldata,solver):
    earth = solver.Planet(atmos_func=alldata['atmos_func'], atmos_filename=alldata['atmos_filename'],
                 Cd=alldata['Cd'], Ch=alldata['Ch'], Q=alldata['Q'], Cl=alldata['Cl'], alpha=alldata['alpha'], 
                      Rp=alldata['Rp'],g=alldata['g'], H=alldata['H'], rho0=alldata['rho0'])
    result = earth.solve_atmospheric_entry(
        radius=alldata['radius'], angle=alldata['angle'], strength=alldata['strength'], 
        velocity=alldata['velocity'], density=alldata['density'],dt=alldata['dt'],radians=alldata['radians'],init_altitude=alldata['init_altitude'])

    result1 = earth.calculate_energy(result)
    result1 = result1.drop(result1.index[-1])
    return result1


def get_solver_result_analytical(solver):
    earth = solver.Planet(Rp=math.inf,Q=math.inf)

    result = earth.solve_atmospheric_entry(
        radius=10., angle=45., strength=math.inf, 
        velocity=21000, density=3000,dt=0.01,radians=False)
    sol = earth.calculate_energy(result)
    sol.drop(sol.tail(1).index,inplace=True)
    return sol



def sci_result(setting, input_data):
    func = setting['atmos_func']
    
    if func == 'exponential':
        return scipy_solution(setting=setting,input_data=input_data).sci_entry()
        
    elif func == 'constant':
        return scipy_solution_constant(setting=setting,input_data=input_data).sci_entry()
        
    elif func == 'mars':
        return scipy_solution_mars(setting=setting,input_data=input_data).sci_entry()
    
    else:
        print('Errored: please enter one of below in atmos_func\n','exponential constant mars')
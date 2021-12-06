import numpy as np
import pandas as pd


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------

        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function ``rho = rho0 exp(-z/H)``.
            Options are ``exponential``, ``tabular``, ``constant`` and ``mars``

        atmos_filename : string, optional
            If ``atmos_func`` = ``'tabular'``, then set the filename of the table
            to be read in here.

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        Returns
        -------

        None
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.tabular_dict = {}

        if atmos_func == 'exponential':
            def rhoa(z):
                return self.rho0*np.exp(-z/self.H)
            self.rhoa = rhoa
        elif atmos_func == 'tabular':

            adt_df = pd.read_csv(atmos_filename, sep=' ', skiprows=6, names=['z_i', 'rho_i', 'h_i'])
    
            tabular_dict = dict(zip(adt_df.z_i,zip(adt_df.rho_i,adt_df.h_i)))
            def rhoa(z):
                index = float(int(z/10)*10)
                return tabular_dict[index][0]*np.exp((index - z) \
                                /tabular_dict[index][1])
            self.rhoa = rhoa
            
        elif atmos_func == 'mars':
            def rhoa(z):
                p = 0.699 * np.exp(-0.00009 * z)
                if z >= 7000:
                    T = 249.7 - 0.00222 * z
                else:
                    T = 242.1 - 0.000998 * z
                return p / (0.1921 * T)
            self.rhoa = rhoa
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact event.
        Also calculates the kinetic energy lost per unit altitude and
        analyses the result to determine the outcome of the impact.

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------

        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``, ``dedz``

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        result = self.solve_atmospheric_entry(
                radius=radius, angle=angle, strength=strength, velocity=velocity, density=density, init_altitude=init_altitude, dt=dt, radians=radians)
        result = self.calculate_energy(result)
        outcome = self.analyse_outcome(result)
        
        return result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``
        """
        
        # Enter your code here to solve the differential equations
        tmax = 500 #10000
        if radians == False:
            angle = self.deg_to_rad(angle)
            
        u0 = np.array([velocity, density*4/3*np.pi*radius**3, angle, init_altitude, 0, radius])
        u_all, t_all = self.RK4(self.f_ode, u0, 0, tmax, dt, density, strength)
        
        if radians == False:
            u_all[:, 2] = self.rad_to_deg(u_all[:, 2])
            
        return pd.DataFrame({'velocity': u_all[:, 0],
                             'mass': u_all[:, 1],
                             'angle': u_all[:, 2],
                             'altitude': u_all[:, 3],
                             'distance': u_all[:, 4],
                             'radius': u_all[:, 5],
                             'time': t_all}, index=range(u_all.shape[0]))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------

        result : DataFrame
            A pandas DataFrame with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        -------

        Result : DataFrame
            Returns the DataFrame with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        if result.altitude.nunique() != len(result.altitude):
            result = result.drop_duplicates(subset='altitude', keep="first")
            result = result.reset_index()

        e = 0.5*result['mass']*result['velocity']**2

        dedz = e.diff(2)/result.altitude.diff(2)/(4.184*1e9) #center approximation, with 2 nan
        dedz = dedz.shift(-1)
        dedz = dedz.fillna(0)

        result = result.copy()
        result.insert(len(result.columns),
                            'dedz', dedz)
        return result


    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats

        Parameters
        ----------

        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """

        # Enter your code here to process the result DataFrame and
        # populate the outcome dictionary.
        outcome={}

        max_row = len(np.array(result.dedz))
        max_dedz_index = result.dedz.idxmax()
        bur_altitude = result.altitude[max_dedz_index]
        if bur_altitude > 5000:
            outcome['outcome'] = "Airburst"
            outcome['burst_peak_dedz'] = result.dedz[max_dedz_index]
            outcome['burst_altitude'] = bur_altitude
            total_lost = np.abs(0.5*result.mass[max_dedz_index]*result.velocity[max_dedz_index]**2-0.5*result.mass[0]*result.velocity[0]**2)
            outcome['burst_total_ke_lost'] = total_lost/(4.184*10**12)
        if max_dedz_index == max_row-1 :
            outcome['outcome'] = "Cratering"
            outcome['impact_time'] = result.time[max_row-1]
            outcome['impact_mass'] = result.mass[max_row-1]
            outcome['impact_speed'] = result.velocity[max_row-1]
        if bur_altitude > 0 and bur_altitude <= 5000:
            outcome['outcome'] = "Airburst and cratering"
            outcome['burst_peak_dedz'] = result.dedz[max_dedz_index]
            outcome['burst_altitude'] = bur_altitude
            total_lost = np.abs(0.5*result.mass[max_dedz_index]*result.velocity[max_dedz_index]**2-0.5*result.mass[0]*result.velocity[0]**2)
            outcome['burst_total_ke_lost'] = total_lost/(4.184*10**12)
            outcome['impact_time'] = result.time[max_row-1]
            outcome['impact_mass'] = result.mass[max_row-1]
            outcome['impact_speed'] = result.velocity[max_row-1]
        return outcome

    
    def f_ode(self, t, u, density, strength):
        '''
        Derivative function of velocity, mass, angle, altitude, distance, radius
        
        Parameters
        ----------
        t : time-step
        
        u : state sarray or list 
        current condition of velocity, mass, angle, altitude, distance, radius (in order) for every time step calculation

        Returns
        -------------
        a Numpy array next time-step result of velocity, mass, angle, altitude, distance, radius
        '''
        #u[0] is velocity
        #u[1] is mass
        #u[2] is angle
        #u[3] is altitude
        #u[4] is distance
        #u[5] is radius
        val = np.zeros_like(u)
        A = np.pi * u[5]**2
        rho_a = self.rhoa(u[3])
               
        val[0] = (-self.Cd*rho_a*A*(u[0])**2)/(2*(u[1])) + self.g*np.sin(u[2])
        val[1] = (-self.Ch*rho_a*A*(u[0])**3)/(2*self.Q)
        val[2] = self.g*np.cos(u[2])/(u[0]) - self.Cl*rho_a*A*(u[0])/(2*u[1]) \
                 - u[0]*np.cos(u[2])/(self.Rp+u[3])
        val[3] = -u[0]*np.sin(u[2])
        val[4] = u[0]*np.cos(u[2])/(1 + u[3]/self.Rp)
        val[5] = (np.sqrt((7/2*self.alpha*rho_a/density))*u[0] if rho_a*u[0]**2 >= strength else 0)            
        return val

    def RK4(self, f, u0, t0, t_max, dt, density, strength):
        '''
        Return velocity, mass, angle, altitude, distance, radius (in order) calculation for each time step 
        Using Runge_Kutta 4th order method
        Consider burst effect in dr/dt calculation

        Parameters
        ----------
        f : function
            derivative function of velocity, mass, angle, altitude, distance, radius   

        uo = initial state : array or list
            initial condition of velocity, mass, angle, altitude, distance, radius 

        t0 = float
            initial time

        t_max = float
            maximum time

        dt = float, optional
            time step, in s

        Returns
        -------------
        a Numpy array containing velocity, mass, angle, altitude, distance, radius
        '''
            
        u = np.array(u0)
        t = np.array(t0)
        u_all = [u0]
        t_all = [t0]
        while t < t_max and min(u) >= 0 :
            k1 = dt*f(t, u, density, strength)
            k2 = dt*f(t + 0.5*dt, u + 0.5*k1, density, strength)
            k3 = dt*f(t + 0.5*dt, u + 0.5*k2, density, strength)
            k4 = dt*f(t + dt, u + k3, density, strength)
            u = u + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
            u_all.append(u)
            t = t + dt
            t_all.append(t)
        return np.array(u_all[:-1]), np.array(t_all[:-1])
    
    def deg_to_rad(self, deg):
        """
        Returns an angle in radians
        for a given angle in degrees
        Parameters
        ----------
        deg: float
            An angle in degrees
        Returns
        -------
        float
            The same angle converted into radians
        """
        return deg / 180 * np.pi 
    
    def rad_to_deg(self, rad):
        """
        Returns an angle in degrees
        for a given angle in radians
        Parameters
        ----------
        rad: float
            An angle in radians
        Returns
        -------
        float
            The same angle converted into degrees
        """

        return rad / np.pi * 180
    

   


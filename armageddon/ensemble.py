import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask import delayed, compute

g_variables = ['radius', 'angle', 'strength', 'velocity', 'density']

def solve_ensemble(
        planet,
        fiducial_impact,
        variables,
        radians=False,
        rmin=8, rmax=12,
        num = 10
        ):
    """
    Run asteroid simulation for a distribution of initial conditions and
    find the burst distribution

    Parameters
    ----------

    planet : object
        The Planet class instance on which to perform the ensemble calculation

    fiducial_impact : dict
        Dictionary of the fiducial values of radius, angle, strength, velocity
        and density

    variables : list
        List of strings of all impact parameters to be varied in the ensemble
        calculation

    rmin : float, optional
        Minimum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    rmax : float, optional
        Maximum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """
    
    global g_variables
    g_variables = variables
    
#     num = 100
    random_array = distribution_input(num, fiducial_impact)

    output_pd = pd.DataFrame(columns=variables+['burst_altitude'], index=range(num))

    if 'velocity' in variables:            
        vesc = 11.2
        v_real = np.sqrt(vesc**2 + random_array[3, :]**2)*1e3  # m/s
    else:
        v_real = random_array[3, :]
       

    ba_np = [0]*num
    for i in range(0, random_array.shape[1]):
        
        result, outcome = delayed(planet.impact, nout = 2)(
            radius=random_array[0, i], angle=random_array[1, i], strength=random_array[2, i], velocity=v_real[i], density=random_array[4, i])
        ba_np[i] = outcome.get('burst_altitude', 0)
        
    ba_np = compute(*ba_np, sheduler='processes', optimize_graph = True)
   
    for i in output_pd.keys():
        if i == 'radius':
            output_pd.radius = random_array[0]
        if i == 'angle':
            output_pd.angle = random_array[1]
        if i == 'strength':
            output_pd.strength = random_array[2]
        if i == 'velocity':
            output_pd.velocity = random_array[3]
        if i == 'density':
            output_pd.density = random_array[4]
        if i == 'burst_altitude':
            output_pd.burst_altitude = ba_np #ba_np

#     plot_output(output_pd.burst_altitude)
#     plot_distribution_5(random_array)

    return output_pd

def distribution_input(num, fiducial_impact):
    
    global g_variables
    
    num_ = 100000
    random_array = np.zeros((5, num))

    if 'radius' in g_variables:
        rmin, rmax = 8, 12
        r = np.linspace(rmin, rmax, num_)
        random_array[0] = np.random.choice(r, size=num, replace=True, p=None)
    else:
        random_array[0] = np.array(fiducial_impact['radius'])
    if 'angle' in g_variables:
        theta = np.linspace(0, np.pi/2, num_)
        p_theta = 2*np.cos(theta)*np.sin(theta)
        random_array[1] = np.random.choice(np.degrees(theta), size=num, replace=True, p=p_theta/sum(p_theta))
    else:
        random_array[1] = np.array(fiducial_impact['angle'])
    if 'strength' in g_variables:
        Ymin, Ymax = 1e3, 1e7
        Y = np.linspace(np.log10(Ymin), np.log10(Ymax), num_)
        random_array[2] = np.random.choice(Y, size=num, replace=True, p=None)
    else:
        random_array[2] = np.array(fiducial_impact['strength'])
    if 'velocity' in g_variables:
        a = 11  #km/s
        vesc = 11.2  #km/s
        v =  np.linspace(0, 50, num_)
        p_v = v**2/a**3*np.sqrt(2/np.pi)*np.exp(-v**2/(2*a**2))
        random_array[3] = np.random.choice(v, size=num, replace=True, p=p_v/sum(p_v))
        vi = np.sqrt(vesc**2 + v**2)*1e3  # m/s
    else:
        random_array[3] = np.array(fiducial_impact['velocity'])
    if 'density' in g_variables:
        rhom = 3000  #kg/m3
        sigma_rho = 1000 #kg/m3
        rho =  np.linspace(0, 6500, num_)
        p_rho = 1/(sigma_rho*np.sqrt(2*np.pi))*np.exp(-((rho-rhom)/(sigma_rho*np.sqrt(2)))**2)
        random_array[4] = np.random.choice(rho, size=num, replace=True, p=p_rho/sum(p_rho))
    else:
        random_array[4] = np.array(fiducial_impact['density'])
    return random_array


def plot_distribution_5(random_array):
    """
    Plot input distrbution
    ----------
    1. radius
    2. angle
    3. strength
    4. velocity
    5. density
    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(231)
    ax1.set_title("Distribution of radius")
    ax1.set_xlabel('Radius (m)')
    ax1.set_ylabel('Probability')
    ax1.set_yticks([])
    ax1.hist(random_array[0], bins=20)

    ax2 = fig.add_subplot(232) 
    ax2.set_title("Distribution of angle")
    ax2.set_xlabel('Trajectory angle (degrees)')
    ax2.set_ylabel('Probability')
    ax2.set_yticks([])
    ax2.hist(random_array[1], bins=25)

    ax3 = fig.add_subplot(233)
    ax3.set_title("Distribution of strength")
    ax3.set_xlabel('log(Strength)(Pa)')
    ax3.set_ylabel('Probability')
    ax3.set_yticks([])
    ax3.hist(random_array[2], bins=10)

    ax4 = fig.add_subplot(234)
    ax4.set_title("Distribution of Velocity")
    ax4.set_xlabel('Velocity (km/s)')
    ax4.set_ylabel('Probability')
    ax4.set_yticks([])
    ax4.hist(random_array[3], bins=25)

    ax5 = fig.add_subplot(235)
    ax5.set_title("Distribution of density")
    ax5.set_xlabel('Density (kg/m^3)')
    ax5.set_ylabel('Probability')
    ax5.set_yticks([])
    ax5.hist(random_array[4], bins=20)

    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.show()
    return None


def plot_output(ba):
    """
    Plot output distrbution
    ----------
    Burst altitude
    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(111)
    ax1.set_title("Distribution of burst altitude")
    ax1.set_xlabel('Burst altitude (m)')
    ax1.set_ylabel('Probability')
    ax1.set_yticks([])
    ax1.hist(ba, bins=20)

    plt.show()
    return None

def plot_inputs5(fiducial_impact):
    """
    Plot input distrbution
    ----------
    1. radius
    2. angle
    3. strength
    4. velocity
    5. density
    Returns
    -------
    None
    """
    random_array = distribution_input(10000, fiducial_impact)
    plot_distribution_5(random_array)
    return None


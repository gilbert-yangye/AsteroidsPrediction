import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def deg_to_rad(deg):
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
    return deg*np.pi/180

def data_for_scipy(rad=10., vel=21000, den=3000., stre=1e5, theta=45., dt=0.05):
    """
    Create the data necessary for the scipy solver

    Parameters
    ----------

    rad : float, optional
        The radius of the asteroid in meters. Default value is 10 meters.

    vel : float, optional
        The entery speed of the asteroid in meters/second.
        Default value is 21000 meters/second.

    den : float, optional
        The density of the asteroid in kg/m^3. Default value is 3000 kg/m^3.

    stre : float, optional
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa). Default value is 10^5 N/m^2.

    theta : float, optional
        The initial trajectory angle of the asteroid to the horizontal
        By default, input is in degrees. Default value is 45 degrees.

    dt : float, optional
        The output timestep, in s. Default value is 0.05 seconds.

    Returns
    -------

    dict
        A dictionary with all the values needed for the scipy solution,
        including initial conditions and constants
    """
    input_data = {'radius': rad,
                  'velocity': vel,
                  'density': den,
                  'strength': stre,
                  'angle': theta,
                  'init_altitude':100000,
                  'dt': dt,
                  'radians': False}

    #data within the Planet class, can be customized
    setting = {'atmos_func':'constant', 'atmos_filename':None,
               'Cd':1., 'Ch':0.1, 'Q':1e7, 'Cl':1e-3,
               'alpha':0.3, 'Rp':6371e3,
               'g':9.81, 'H':8000., 'rho0':1.2}

    #integrate
    alldata = {**setting, **input_data}

    return alldata

def initial_for_scipy(rad=10., vel=21000, den=3000., stre=1e5, theta=45.):
    """
    Returns the initial condition for the scipy solver

    Parameters
    ----------

    rad : float, optional
        The radius of the asteroid in meters. Default value is 10 meters.

    vel : float, optional
        The entery speed of the asteroid in meters/second.
        Default value is 21000 meters/second.

    den : float, optional
        The density of the asteroid in kg/m^3. Default value is 3000 kg/m^3.

    stre : float, optional
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa). Default value is 10^5 N/m^2.

    theta : float, optional
        The initial trajectory angle of the asteroid to the horizontal
        By default, input is in degrees. Default value is 45 degrees.

    Returns
    -------

    np.array
        An array with the initial conditions for velocity, mass, angle,
        altitude, distance and radius of the asteroid.
    """
    alldata = data_for_scipy(rad, vel, den, stre, theta)
    #set initial_condition for scipy ODE solver
    initial_condition = np.array([alldata['velocity'],
                                  alldata['density']*alldata['radius']**3*4/3*np.pi,
                                  deg_to_rad(alldata['angle']), alldata['init_altitude'],
                                  0., alldata['radius']])

    return initial_condition

def sci_result(initial_condition, t_0=0, t_end=500, t_step=0.05, tol=1e-4,
               rad=10., vel=21000, den=3000., stre=1e5, theta=45.):
    """
    Returns the solution of the ODE solver for the equations of the asteroid's motion.

    Parameters
    ----------

    initial_condition: np.array
        An array with the initial conditions for velocity, mass, angle,
        altitude, distance and radius of the asteroid.

    t_0: float, optional
        The initial time of the simulation in seconds. Default value is 0 seconds.

    t_end: float, optional
        The final time of the simulation in seconds. Default value is 500 seconds.

    tol: float, optional
        The tolerance of the calculation. Default value is 10^(-4).

    rad : float, optional
        The radius of the asteroid in meters. Default value is 10 meters.

    vel : float, optional
        The entery speed of the asteroid in meters/second.
        Default value is 21000 meters/second.

    den : float, optional
        The density of the asteroid in kg/m^3. Default value is 3000 kg/m^3.

    stre : float, optional
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa). Default value is 10^5 N/m^2.

    theta : float, optional
        The initial trajectory angle of the asteroid to the horizontal
        By default, input is in degrees. Default value is 45 degrees.

    Returns
    -------

    pandas.core.frame.DataFrame
        A dataframe with the values of velocity, mass, angle,
        altitude, distance and radius of the asteroid during the simulation time.
    """
    alldata = data_for_scipy(rad, vel, den, stre, theta, t_step)
    rhom = alldata['density']
    Y = alldata['strength']

    def simulation(t, parameters):
        v, m, theta, z, x, r = parameters
        alldata = data_for_scipy(r, v, theta=theta)
        Cd, g, Ch, Q, Cl, Rp, alpha, rho0, H = (alldata['Cd'], alldata['g'], alldata['Ch'],
                                                alldata['Q'], alldata['Cl'], alldata['Rp'],
                                                alldata['alpha'], alldata['rho0'], alldata['H'])

        A = np.pi*r**2
        rhoa = rho0*np.exp(-z/H)

        return np.array([-Cd*rhoa*A*v**2/(2*m)+g*np.sin(theta),
                         -Ch*rhoa*A*v**3/(2*Q),
                         g*np.cos(theta)/v-Cl*rhoa*A*v/(2*m)-v*np.cos(theta)/(Rp+z),
                         -v*np.sin(theta),
                         v*np.cos(theta)/(1+z/Rp),
                         (np.sqrt(7/2*alpha*(rhoa/rhom))*v if rhoa*v**2 >= Y else 0)])

    sci_result = solve_ivp(simulation, [t_0, t_end], initial_condition,
                           t_eval=np.arange(t_0, t_end, t_step), method='RK45', atol=tol, rtol=tol)

    sci_result = pd.DataFrame({'time':sci_result.t, 'velocity':sci_result.y[0],
                               'mass':sci_result.y[1], 'angle':sci_result.y[2],
                               'altitude':sci_result.y[3], 'distance':sci_result.y[4],
                               'radius':sci_result.y[5]})

    sci_result = sci_result.drop(sci_result[sci_result.velocity <= 0].index)
    sci_result = sci_result.drop(sci_result[sci_result.altitude <= 0].index)
    sci_result = sci_result.drop(sci_result[sci_result.mass <= 0].index)

    return sci_result

def compute_errors(planet, r=10., a=45., s=1e5, v=21000, den=3000., fragmentation=True):
    """
    Computes the errors between the RK4 solution and the scipy solution for different time-steps.
    The error for a single time-step is calculated as the mean of the relative errors in every point
    of the simulation, in percentage and absolute value. Besides plotting this results,
    the function also plots a loglog graph with the norm of the arrays with those relative errors,
    in relationship to different time-step choices.

    Parameters
    ----------

    planet: class
        Class from the file solver.py

    r : float, optional
        The radius of the asteroid in meters. Default value is 10 meters.

    v : float, optional
        The entery speed of the asteroid in meters/second.
        Default value is 21000 meters/second.

    den : float, optional
        The density of the asteroid in kg/m^3. Default value is 3000 kg/m^3.

    s : float, optional
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa). Default value is 10^5 N/m^2.

    a : float, optional
        The initial trajectory angle of the asteroid to the horizontal
        By default, input is in degrees. Default value is 45 degrees.

    fragmentation: Boolean, optional
        Set if the asteroid is mooving with or without fragmentation (that is,
        dr/dt = 0).

    Returns
    -------

    pandas.core.frame.DataFrame
        A dataframe with the errors for the variables computed with the two methods.
    """
    errors1 = []
    errors2 = []
    errors3 = []
    errors4 = []
    errors5 = []

    rel_errors1 = []
    rel_errors2 = []
    rel_errors3 = []
    rel_errors4 = []
    rel_errors5 = []

    errors_v = []
    errors_m = []
    errors_theta = []
    errors_z = []
    errors_x = []

    rel_errors_v = []
    rel_errors_m = []
    rel_errors_theta = []
    rel_errors_z = []
    rel_errors_x = []

    if fragmentation:
        errors6 = []
        rel_errors6 = []
        errors_r = []
        rel_errors_r = []
    else:
        s = 1e100

    dt_array = []
    d_t = 1.

    finisher = 0

    while d_t >= 0.01:

        dt_array.append(d_t)
        result = planet.solve_atmospheric_entry(radius=r, angle=a, strength=s,
                                                velocity=v, density=den, dt=d_t)
        scipy = sci_result(initial_for_scipy(r, v, den, s, a),
                           rad=r, vel=v, den=den, stre=s, theta=a, t_step=d_t)

        l1 = min(len(scipy.velocity), len(result.velocity))
        l2 = min(len(scipy.mass), len(result.mass))
        l3 = min(len(scipy.angle), len(result.angle))
        l4 = min(len(scipy.altitude), len(result.altitude))
        l5 = min(len(scipy.distance), len(result.distance))

        for i in range(l1):
            errors1.append(abs(result.velocity[i] - scipy.velocity[i]))
            rel_errors1.append(errors1[i] / scipy.velocity[i] * 100)

        for i in range(l2):
            errors2.append(abs(result.mass[i] - scipy.mass[i]))
            rel_errors2.append(errors2[i] / scipy.mass[i] * 100)

        for i in range(l3):
            errors3.append(abs(deg_to_rad(result.angle[i]) - scipy.angle[i]))
            rel_errors3.append(errors3[i] / scipy.angle[i] * 100)

        for i in range(l4):
            errors4.append(abs(result.altitude[i] - scipy.altitude[i]))
            rel_errors4.append(errors4[i] / scipy.altitude[i] * 100)

        for i in range(l5):
            errors5.append(abs(result.distance[i] - scipy.distance[i]))
            if scipy.distance[i] == 0:
                rel_errors5.append(0.)
            else:
                rel_errors5.append(errors5[i] / scipy.distance[i] * 100)

        errors_v.append(np.linalg.norm(errors1))
        errors_m.append(np.linalg.norm(errors2))
        errors_theta.append(np.linalg.norm(errors3))
        errors_z.append(np.linalg.norm(errors4))
        errors_x.append(np.linalg.norm(errors5))

        rel_errors_v.append(np.mean(rel_errors1))
        rel_errors_m.append(np.mean(rel_errors2))
        rel_errors_theta.append(np.mean(rel_errors3))
        rel_errors_z.append(np.mean(rel_errors4))
        rel_errors_x.append(np.mean(rel_errors5))

        errors1 = []
        errors2 = []
        errors3 = []
        errors4 = []
        errors5 = []

        rel_errors1 = []
        rel_errors2 = []
        rel_errors3 = []
        rel_errors4 = []
        rel_errors5 = []

        if fragmentation:
            l6 = min(len(scipy.radius), len(result.radius))
            for i in range(l6):
                errors6.append(abs(result.radius[i] - scipy.radius[i]))
                rel_errors6.append(errors6[i] / scipy.radius[i] * 100)
            errors_r.append(np.linalg.norm(errors6))
            rel_errors_r.append(np.mean(rel_errors6))
            errors6 = []
            rel_errors6 = []
        else:
            rel_errors_r = np.zeros_like(dt_array)

        if d_t <= 0.053:
            d_t = 0.05 - finisher*0.02
            finisher += 1
        else:
            d_t *= 0.9

    overall_errors = np.zeros_like(rel_errors_v)
    for j in range(len(overall_errors)):
        overall_errors[j] = rel_errors_v[j] + rel_errors_m[j] + rel_errors_theta[j] + \
                            rel_errors_z[j] + rel_errors_x[j]
        if fragmentation:
            overall_errors[j] += rel_errors_r[j]
            overall_errors[j] = overall_errors[j] / 6
        else:
            overall_errors[j] = overall_errors[j] / 5

    d = {'dt': dt_array,
         'velocity error (%)': rel_errors_v,
         'mass error (%)': rel_errors_m,
         'angle error (%)': rel_errors_theta,
         'altitude error (%)': rel_errors_z,
         'distance error (%)': rel_errors_x,
         'radius error (%)': rel_errors_r,
         'overall error (%)': overall_errors}
    dataframe = pd.DataFrame(data=d)

    if fragmentation:
        y_zoom_end = 4.
        y_zoom_zoom_end = 0.3
    else:
        y_zoom_end = 0.005
        y_zoom_zoom_end = 0.0025

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 30))
    ax1.loglog(dt_array, errors_v, 'b.-', label='velocity')
    ax1.loglog(dt_array, errors_m, 'k.-', label='mass')
    ax1.loglog(dt_array, errors_theta, 'r.-', label='angle')
    ax1.loglog(dt_array, errors_z, 'g.-', label='altitude')
    ax1.loglog(dt_array, errors_x, 'y.-', label='distance')
    ax1.set_xlabel('$\Delta t$', fontsize=16)
    ax1.set_ylabel('error', fontsize=16)
    ax1.legend()
    ax1.set_title('Compare our solver and scipy solver')
    ax2.plot(dt_array, rel_errors_v, 'b.-', label='velocity')
    ax2.plot(dt_array, rel_errors_m, 'k.-', label='mass')
    ax2.plot(dt_array, rel_errors_theta, 'r.-', label='angle')
    ax2.plot(dt_array, rel_errors_z, 'g.-', label='altitude')
    ax2.plot(dt_array, rel_errors_x, 'y.-', label='distance')
    ax2.set_xlabel('$\Delta t$', fontsize=16)
    ax2.set_ylabel('error (%)', fontsize=16)
    ax2.legend()
    ax2.set_title('Mean of relative errors')
    ax3.plot(dt_array, rel_errors_v, 'b.-', label='velocity')
    ax3.plot(dt_array, rel_errors_m, 'k.-', label='mass')
    ax3.plot(dt_array, rel_errors_theta, 'r.-', label='angle')
    ax3.plot(dt_array, rel_errors_z, 'g.-', label='altitude')
    ax3.plot(dt_array, rel_errors_x, 'y.-', label='distance')
    ax3.axis([0.005, 0.3, -0.001, y_zoom_end])
    ax3.set_xlabel('$\Delta t$', fontsize=16)
    ax3.set_ylabel('error (%)', fontsize=16)
    ax3.legend()
    ax3.set_title('Mean of relative errors - zoomed in')
    ax4.plot(dt_array, rel_errors_v, 'b.-', label='velocity')
    ax4.plot(dt_array, rel_errors_m, 'k.-', label='mass')
    ax4.plot(dt_array, rel_errors_theta, 'r.-', label='angle')
    ax4.plot(dt_array, rel_errors_z, 'g.-', label='altitude')
    ax4.plot(dt_array, rel_errors_x, 'y.-', label='distance')
    ax4.axis([0.009, 0.1, -0.0001, y_zoom_zoom_end])
    ax4.set_xlabel('$\Delta t$', fontsize=16)
    ax4.set_ylabel('error (%)', fontsize=16)
    ax4.legend()
    ax4.set_title('Mean of relative errors - zoomed in x 2')
    if fragmentation:
        ax1.loglog(dt_array, errors_r, 'c.-', label='radius')
        ax2.plot(dt_array, rel_errors_r, 'c.-', label='radius')
        ax3.plot(dt_array, rel_errors_r, 'c.-', label='radius')
        ax4.plot(dt_array, rel_errors_r, 'c.-', label='radius')

    return dataframe

def errors_for_comparison(planet, r=10., a=45., s=1e5, v=21000, den=3000., fragmentation=True):
    """
    Computes the errors between the RK4 solution and the scipy solution for different time-steps.
    The error for a single time-step is calculated as the mean of the relative errors in every point
    of the simulation, in percentage and absolute value.

    Parameters
    ----------

    planet: class
        Class from the file solver.py

    r : float, optional
        The radius of the asteroid in meters. Default value is 10 meters.

    v : float, optional
        The entery speed of the asteroid in meters/second.
        Default value is 21000 meters/second.

    den : float, optional
        The density of the asteroid in kg/m^3. Default value is 3000 kg/m^3.

    s : float, optional
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa). Default value is 10^5 N/m^2.

    a : float, optional
        The initial trajectory angle of the asteroid to the horizontal
        By default, input is in degrees. Default value is 45 degrees.

    fragmentation: Boolean, optional
        Set if the asteroid is mooving with or without fragmentation (that is,
        dr/dt = 0).

    Returns
    -------

    array_like
        An array with the overall errors for different time-steps.
        The overall errors are calculated as the mean of the errors for any variable.
    """
    errors1 = []
    errors2 = []
    errors3 = []
    errors4 = []
    errors5 = []

    rel_errors1 = []
    rel_errors2 = []
    rel_errors3 = []
    rel_errors4 = []
    rel_errors5 = []

    errors_v = []
    errors_m = []
    errors_theta = []
    errors_z = []
    errors_x = []

    rel_errors_v = []
    rel_errors_m = []
    rel_errors_theta = []
    rel_errors_z = []
    rel_errors_x = []

    if fragmentation:
        errors6 = []
        rel_errors6 = []
        errors_r = []
        rel_errors_r = []
    else:
        s = 1e100

    dt_array = []
    d_t = 0.1

    while d_t >= 0.01:

        dt_array.append(d_t)
        result = planet.solve_atmospheric_entry(radius=r, angle=a, strength=s,
                                                velocity=v, density=den, dt=d_t)
        scipy = sci_result(initial_for_scipy(r, v, den, s, a),
                           rad=r, vel=v, den=den, stre=s, theta=a, t_step=d_t)

        l1 = min(len(scipy.velocity), len(result.velocity))
        l2 = min(len(scipy.mass), len(result.mass))
        l3 = min(len(scipy.angle), len(result.angle))
        l4 = min(len(scipy.altitude), len(result.altitude))
        l5 = min(len(scipy.distance), len(result.distance))

        for i in range(l1):
            errors1.append(abs(result.velocity[i] - scipy.velocity[i]))
            rel_errors1.append(errors1[i] / scipy.velocity[i] * 100)

        for i in range(l2):
            errors2.append(abs(result.mass[i] - scipy.mass[i]))
            rel_errors2.append(errors2[i] / scipy.mass[i] * 100)

        for i in range(l3):
            errors3.append(abs(deg_to_rad(result.angle[i]) - scipy.angle[i]))
            rel_errors3.append(errors3[i] / scipy.angle[i] * 100)

        for i in range(l4):
            errors4.append(abs(result.altitude[i] - scipy.altitude[i]))
            rel_errors4.append(errors4[i] / scipy.altitude[i] * 100)

        for i in range(l5):
            errors5.append(abs(result.distance[i] - scipy.distance[i]))
            if scipy.distance[i] == 0:
                rel_errors5.append(0.)
            else:
                rel_errors5.append(errors5[i] / scipy.distance[i] * 100)

        errors_v.append(np.linalg.norm(errors1))
        errors_m.append(np.linalg.norm(errors2))
        errors_theta.append(np.linalg.norm(errors3))
        errors_z.append(np.linalg.norm(errors4))
        errors_x.append(np.linalg.norm(errors5))

        rel_errors_v.append(np.mean(rel_errors1))
        rel_errors_m.append(np.mean(rel_errors2))
        rel_errors_theta.append(np.mean(rel_errors3))
        rel_errors_z.append(np.mean(rel_errors4))
        rel_errors_x.append(np.mean(rel_errors5))

        errors1 = []
        errors2 = []
        errors3 = []
        errors4 = []
        errors5 = []

        rel_errors1 = []
        rel_errors2 = []
        rel_errors3 = []
        rel_errors4 = []
        rel_errors5 = []

        if fragmentation:
            l6 = min(len(scipy.radius), len(result.radius))
            for i in range(l6):
                errors6.append(abs(result.radius[i] - scipy.radius[i]))
                rel_errors6.append(errors6[i] / scipy.radius[i] * 100)
            errors_r.append(np.linalg.norm(errors6))
            rel_errors_r.append(np.mean(rel_errors6))
            errors6 = []
            rel_errors6 = []
        else:
            rel_errors_r = np.zeros_like(dt_array)

        d_t -= 0.01

    overall_errors = np.zeros_like(rel_errors_v)
    for j in range(len(overall_errors)):
        overall_errors[j] = rel_errors_v[j] + rel_errors_m[j] + rel_errors_theta[j] + \
                            rel_errors_z[j] + rel_errors_x[j]
        if fragmentation:
            overall_errors[j] += rel_errors_r[j]
            overall_errors[j] = overall_errors[j] / 6
        else:
            overall_errors[j] = overall_errors[j] / 5

    return overall_errors

def dt_comparison(planet):
    """
    Compares the errors of different time-steps for different initial conditions.

    Parameters
    ----------

    planet: class
        Class from the file solver.py

    Returns
    -------

    pandas.core.frame.DataFrame
        A dataframe with the overall errors comparing the two methods.
    """
    dt = np.array([round(0.1-i*0.01, 2) for i in range(10)])
    r = np.random.randint(8, 13, size=10)
    a = np.random.randint(15, 75, size=10)
    s = np.random.randint(3, 8, size=10)
    s[:] = 10**s[:]
    v = np.random.randint(14, 30, size=10)*1000
    d = np.random.randint(1500, 4500, size=10)
    situations = np.array([i for i in range(1,11)])

    errors_0 = np.array(errors_for_comparison(planet, r[0], a[0], s[0], v[0], d[0]))
    errors_1 = np.array(errors_for_comparison(planet, r[1], a[1], s[1], v[1], d[1]))
    errors_2 = np.array(errors_for_comparison(planet, r[2], a[2], s[2], v[2], d[2]))
    errors_3 = np.array(errors_for_comparison(planet, r[3], a[3], s[3], v[3], d[3]))
    errors_4 = np.array(errors_for_comparison(planet, r[4], a[4], s[4], v[4], d[4]))
    errors_5 = np.array(errors_for_comparison(planet, r[5], a[5], s[5], v[5], d[5]))
    errors_6 = np.array(errors_for_comparison(planet, r[6], a[6], s[6], v[6], d[6]))
    errors_7 = np.array(errors_for_comparison(planet, r[7], a[7], s[7], v[7], d[7]))
    errors_8 = np.array(errors_for_comparison(planet, r[8], a[8], s[8], v[8], d[8]))
    errors_9 = np.array(errors_for_comparison(planet, r[9], a[9], s[9], v[9], d[9]))

    full_errors = np.array([errors_0, errors_1, errors_2, errors_3, errors_4, errors_5,
                           errors_6, errors_7, errors_8, errors_9])

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    X,Y = np.meshgrid(dt, situations)
    Z = full_errors
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.xlabel("Time-steps", fontsize=12)
    plt.ylabel("Situation n.", fontsize=12)
    plt.title("Comparing different time-step choices with different initial conditions", fontsize=16)
    plt.show()

    mean_errors = np.zeros_like(dt)
    max_errors = np.zeros_like(dt)
    min_errors = np.zeros_like(dt)
    mean_errors[:] = errors_1[:] + errors_2[:] + errors_3[:] + errors_4[:] + errors_5[:] + \
                    errors_6[:] + errors_7[:] + errors_8[:] + errors_9[:] + errors_0[:]
    mean_errors[:] = mean_errors[:] / 10
    for i in range(len(dt)):
        max_errors[i] = max(errors_1[i], errors_2[i], errors_3[i], errors_4[i], errors_5[i], \
                        errors_6[i], errors_7[i], errors_8[i], errors_9[i], errors_0[i])
        min_errors[i] = min(errors_1[i], errors_2[i], errors_3[i], errors_4[i], errors_5[i], \
                        errors_6[i], errors_7[i], errors_8[i], errors_9[i], errors_0[i])

    times = np.zeros_like(dt)
    rows = np.zeros_like(dt)
    for i in range(len(times)):
        start = time.time()
        result = planet.solve_atmospheric_entry(
                        radius=10, angle=45, strength=1e5, velocity=21e3, density=3000, dt=dt[i])
        end = time.time()
        times[i] = end-start
        rows[i] = result.count()[0]

    change_in_error = np.zeros_like(dt)
    change_in_time = np.zeros_like(dt)
    change_in_space = np.zeros_like(dt)
    for i in range(len(dt)):
        change_in_error[i] = (mean_errors[i]-mean_errors[5])/mean_errors[5]*100
        change_in_time[i] = (times[i]-times[5])/times[5]*100
        change_in_space[i] = (rows[i]-rows[5])/rows[5]*100

    d = {'dt': dt,
         'Mean error (%)': mean_errors,
         'Max error (%)': max_errors,
         'Min error (%)': min_errors,
         'Change in error (%)': change_in_error,
         'Time for num. solution': times,
         'Change in time (%)': change_in_time,
         'Space used (in rows)': rows,
         'Change in space (%)': change_in_space}
    dataframe = pd.DataFrame(data=d)
    return dataframe

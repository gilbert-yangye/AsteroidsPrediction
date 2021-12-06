import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
"""
Find the best fit curve for the Chelyabinsk airburst
by changing asteroid radius and strength
"""
def f_fit_event(filename):
    """
    Find fit function for event
    Parameters
    ----------
    filename : the directory of ChelyabinskEnergyAltitude.csv
    Returns
    -------
    f_event : the function of Clelyabinsk
    height : the height of Clelyabinsk
    
    """
    event = pd.read_csv(filename, skiprows = 1, names=["Height","dedz"])
    height = np.array(event.Height)
    dedz = np.array(event.dedz)
    f_event = interp1d(height, dedz)
    return f_event, height

def get_best_fit(planet, f_event, height, range_radius, range_strength):
    """
    Get the best fit curve
    Parameters
    ----------
    planet : the class we call
    f_event : the function of Clelyabinsk
    height : the height of Clelyabinsk
    radiu : initial guess for the radius
    range_radius : the range of iteration radius
    range_strength : the range of iteration strength
    Returns
    -------
    radiu_final : the radius which is the best fit
    stre_final : the strength which is the best fit
    error_final : the best fit's error
    """
    all1 = []
    for radiu in range_radius:
        for stre in range_strength:
            result = planet.solve_atmospheric_entry(
                radius=radiu, angle=18.3, strength=stre, velocity=19200, density=3300)
            result = planet.calculate_energy(result)
            height_solver = np.array(result.altitude/1000)
            dedz_solver = np.array(result.dedz)
            slice1 = np.where(height_solver >= height[-1])[0][-1]
            slice2 = np.where(height_solver <= height[0])[0][0]
            height_new = height_solver[slice2:slice1]
            dedz_new = dedz_solver[slice2:slice1]
            dedz_final = f_event(height_new)

            error = mean_squared_error(dedz_new, dedz_final)
            error = np.sqrt(error)
#             print("radius:", radiu, "strength:", stre, "error:", error)
            lis = [radiu, stre, error]
            all1.append(lis)
    all1 = np.array(all1)
    index1 = np.argmin(all1[:,2])
    radiu_final = all1[index1,0]
    stre_final = all1[index1,1]
    error_final = all1[index1,2]

    return radiu_final, stre_final, error_final

def plot_pre3(planet, f_event, height, radiu_final, stre_final):
    """
    Plot
    Parameters
    ----------
    planet : the class we call
    f_event : the function of Clelyabinsk
    height : the height of Clelyabinsk
    radiu : initial guess for the radius
    range_radius : the range of iteration radius
    range_strength : the range of iteration strength
    Returns
    -------
    None
    """
    
    result = planet.solve_atmospheric_entry(
        radius=radiu_final, angle=18.3, strength=stre_final, velocity=19200, density=3300)
    result = planet.calculate_energy(result)
    height_solver = np.array(result.altitude/1000)
    dedz_solver = np.array(result.dedz)
    slice1 = np.where(height_solver >= height[-1])[0][-1]
    slice2 = np.where(height_solver <= height[0])[0][0]
    height_new = height_solver[slice2:slice1]
    dedz_new = dedz_solver[slice2:slice1]
    dedz_final = f_event(height_new)
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)

    ax1.plot(height_new, dedz_final,'b', label = "event fit")

    ax1.plot(height_new, dedz_new, 'r', label = "numerical ")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel("Height(km)")
    ax1.set_ylabel("Energy per unit length(kt Km^-1)")

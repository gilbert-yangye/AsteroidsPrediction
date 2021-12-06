"""Python asteroid airburst calculator"""
import matplotlib.pyplot as plt

def plot_element_6(result):
    """
    Plot result
    ----------
    1. velocity with time 
    2. mass with timem
    3. theta with time
    4. altitude vs velocity
    5. distance with time 
    6. radius with time
    7. Altitude with Energy per unit height
    Returns
    -------
    None
    """
    e = (0.5*result.mass*result.velocity**2).to_numpy()
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(6, 20))
    fig.tight_layout(w_pad=5)  # add some padding otherwise the axes labels overlap

    ax1.plot(result.time, result.velocity, 'b', label='v')
    ax1.set_ylabel('velocity', fontsize=16)
    ax1.set_xlabel('$t$', fontsize=16)
    ax2.plot(result.time, result.mass, 'b', label='m')
    ax2.set_ylabel('mass', fontsize=16)
    ax2.set_xlabel('$t$', fontsize=16)
    ax3.plot(result.time, result.angle, 'b', label='theta')
    ax3.set_ylabel('theta', fontsize=16)
    ax3.set_xlabel('$t$', fontsize=16)
    ax4.plot(result.altitude, result.velocity, 'b', label='z vs v')
    ax4.set_ylabel('altitude', fontsize=16)
    ax4.set_xlabel('velocity', fontsize=16)
    ax5.plot(result.time,  result.distance, 'b', label='x')
    ax5.set_ylabel('distance', fontsize=16)
    ax5.set_xlabel('$t$', fontsize=16)
    ax6.plot(result.time,  result.radius, 'b', label='r')
    ax6.set_ylabel('radius', fontsize=16)
    ax6.set_xlabel('$t$', fontsize=16)
    ax7.plot(result.dedz, result.altitude, 'b', label='energy vs z')
    ax7.set_ylabel('Altitude', fontsize=16)
    ax7.set_xlabel('Energy per unit height', fontsize=16)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.show()
    return None

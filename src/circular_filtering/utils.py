import numpy as np
import matplotlib.pyplot as plt

'''Utility functions for circular filtering techniques

Methods
-------
get_bessel_ratio(x, v, N)
    Compute I_{v+1}(x)/I_v(x)
get_vm_concentration_param(m1, eps, v, N)
    Invert I_{v+1}(k)/I_v(k) = abs(m1) to precision +- eps
'''


def get_bessel_ratio(x : float, v : int = 0, N : int = 10 ) -> float:
    '''Compute I_{v+1}(x)/I_v(x)
    I_v is the vth modified Bessel function of the first kind

    Method as given by D. E. Amos in "Computation of Modified Bessel Functions and Their Ratios",
    and converted to pseudocode by Kurz et. al in "Recursive Nonlinear Filtering for Angular Data
    Based on Circular Distributions".
    
    Parameters
    ----------
    x : float
        Point at which to evaluate the ratio. x > 0
    v : int, optional
        Determines the orders of the Bessel functions in the ratio. Default value
        is 0, so gives a value for the magnitude of the first circular moment of 
        the von Mises distribution with concentration parameter x.
    N : int, optional
        Number of discretisation steps

    Returns
    -------
    float : 
        Value of I_{v+1}(x)/I_v(x)
    '''

    o = min(v, 10)
    r = np.zeros(N +1)
    for i in range(N + 1):
        r[i] = x/(o + i + 0.5 + np.sqrt((o + i + 1.5)**2 + x**2))
    for i in range(1,N+1):
        for k in range(N-i+1):
            r[k] = x/(o + k + 1 + np.sqrt((o + k + 1)**2 + (r[k+1]/r[k])*(x**2)))
    y = r[0]
    i = o
    while i > v:
        y = np.power((2*i/x) + y ,-1)
        i -= 1

    return y

def get_vm_concentration_param(m1: complex, eps: float = pow(2,-15), v : int = 0, N : int = 10) -> float:
    '''Invert I_{v+1}(k)/I_v(k) = abs(m1) to precision +- eps

    Parameters
    ----------
    m1 : complex
        First circular moment of von Mises distribution. 
        m1 = exp(i mu) * I_{v+1}(k)/I_v(k).
    eps : float
        Precision to return k to. 
    v : int, optional
        Inverts the ratio for higher order Bessel functions. Default is 0 for the first
        circular moment of von Mises distribution.
    N : int, optional
        Number of discretisations steps for calculating the Bessel ratio. Default is 10.

    Returns
    -------
    float : 
        For the interval [k_l, k_u] return (k_l + k_u)/2, where k_u - k_l < eps.
    '''
    
    r = abs(m1)
    concentration_param_interval = np.array([0,1],dtype=np.float32)
    found_upper_limit = False

    while concentration_param_interval[1] - concentration_param_interval[0] > eps:
        k_trial = concentration_param_interval.mean()
        r_trial = get_bessel_ratio(k_trial, v, N)

        # Increase the upper limit of the interval until we contain k
        if not found_upper_limit:
            if r_trial > r:
                found_upper_limit = True
                concentration_param_interval[1] = k_trial
            else:
                concentration_param_interval[0] = concentration_param_interval[1]
                concentration_param_interval[1] = 2*concentration_param_interval[1]
        else:
            if r_trial > r:
                concentration_param_interval[1] = k_trial
            else:
                concentration_param_interval[0] = k_trial

    return concentration_param_interval.mean()


if __name__ == "__main__":
    x = np.linspace(0.1, 100, 1000)
    y = np.zeros(x.size)

    for i in range(y.size):
        y[i] = get_bessel_ratio(x[i])

    plt.plot(x,y)
    plt.show()

    x = np.linspace(0.05, 0.995, 100)
    y = np.zeros(x.size)


    for i in range(y.size):
        y[i] = get_vm_concentration_param(x[i],0.01)

    plt.plot(x,y)
    plt.show()

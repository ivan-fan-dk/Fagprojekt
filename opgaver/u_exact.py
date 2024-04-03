import numpy as np
def u_exact(x, t, v:float, A:float, c:float, c1:float, c2:float, x0:float): 
    """
    params:
    x: any
    t: any
    v: float, velocity
    A: float, amplitude
    c: float, phase
    c1: float, constant
    c2: float, constant

    for example: 
        ```
        v, A, c = 0, 2, 0
        c1, c2 = 1/2, 1
        x0 = 0
        ```
    return: any
    """
    B = np.sqrt(c2*A**2/2*c1)
    a = v/(2*c1)
    b = c1*(B**2-a**2)
    xi = lambda x, t: x - x0 - v*t
    rho = lambda x, t: A/np.cosh(B*xi(x, t))
    phi = lambda x, t: a*x + b*t + c

    return rho(x, t)*np.exp(1j*phi(x, t))
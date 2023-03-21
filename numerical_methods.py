import numpy as np  
class RK4:
    # 4th-order Runge Kutta
    def RK4_1st(fun, t0, x0, t, h):
        n = round((t - t0) / h)
        x = x0

        t_arr, x_arr = [t0], [x0]
        for i in range(1, n + 1) :
            k1 = fun(t0, x)
            k2 = fun(t0 + h/2, x + h/2*k1)
            k3 = fun(t0 + h/2, x + h/2*k2)
            k4 = fun(t0 + h, x + h*k3)

            # Update next value of x
            x = x + h/6. * (k1 + 2.*k2 + 2*k3 + k4)
            x_arr.append(x)
    
            # Update next value of t
            t0 = t0 + h
            t_arr.append(t0)

        return [t_arr, x_arr]

    # Runge Kutta for second order diff. eq. (4th-order)
    # gives back [t, x, v]
    #def RK4_2nd(fun, t0, x0, v0, t, h):
    #    n = round((t - t0) / h)
    def RK4_2nd(fun, t0, x0, v0, t, n, log_steps=False):
        if log_steps:
            grid = np.logspace(np.log10(t0), np.log10(t), num=n+1)
            h = np.diff(grid)
        else:  
            h = np.empty(n); h.fill((t - t0) / n)   
        
        x = x0
        v = v0

        t_arr, x_arr, v_arr = [t0], [x0], [v0]
        for i in range(0, n):
            k1 = fun(t0, x, v)[0]
            l1 = fun(t0, x, v)[1]

            k2 = fun(t0 + h[i]/2, x + h[i]/2*k1, v + h[i]/2*l1)[0]
            l2 = fun(t0 + h[i]/2, x + h[i]/2*k1, v + h[i]/2*l1)[1]

            k3 = fun(t0 + h[i]/2, x + h[i]/2*k2, v + h[i]/2*l2)[0]
            l3 = fun(t0 + h[i]/2, x + h[i]/2*k2, v + h[i]/2*l2)[1]

            k4 = fun(t0 + h[i], x + h[i]*k3, v + h[i]*l3)[0]
            l4 = fun(t0 + h[i], x + h[i]*k3, v + h[i]*l3)[1]

            # Update next value of x
            x = x + h[i]/6. * (k1 + 2.*k2 + 2*k3 + k4)
            x_arr.append(x)
    
            # Update next value of v
            v = v + h[i]/6. * (l1 + 2.*l2 + 2*l3 + l4)
            v_arr.append(v)

            # Update next value of t
            t0 = t0 + h[i]
            t_arr.append(t0)

        return [t_arr, x_arr, v_arr]


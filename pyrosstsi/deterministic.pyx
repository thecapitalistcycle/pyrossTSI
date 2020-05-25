import  numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
import pyross.utils
import warnings

DTYPE   = np.float




cdef class IntegratorsClass:
    """
    List of all integrator used by various deterministic models listed below.

    Methods
    -------
    simulateRHS: Performs numerical integration.

    simulator: interface for user to call simulateRHS

    set_contactMatrix: setting contact matrix
    """

    def simulateRHS(self, rhs0, x0, Ti, Tf, Nf, integrator, maxNumSteps, **kwargs):
        """
        Performs numerical integration

        Parameters
        ----------
        rhs0: python function(x,t)
            Input function of current state and time x, t
            returns dx/dt
        x0: np.array
            Initial state vector.
        Ti: float
            Start time for integrator.
        Tf: float
            End time for integrator.
        Nf: Int
            Number of time points to evaluate at.
        integrator: string, optional
            Selects which integration method to use. The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator is allowed to take
            to obtain a solution. The default is 100000.
        **kwargs: optional kwargs to be passed to the IntegratorsClass

        Raises
        ------
        Exception
            If integration fails.

        Returns
        -------
        X: np.array(len(t), len(x0))
            Numerical integration solution.
        time_points : np.array
            Corresponding times at which X is evaluated at.

        """

        if integrator=='solve_ivp':
            from scipy.integrate import solve_ivp
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            X = solve_ivp(lambda t, xt: rhs0(xt,t), [Ti,Tf], x0, t_eval=time_points, **kwargs).y.T

        elif integrator=='odeint':
            from scipy.integrate import odeint
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            X = odeint(rhs0, x0, time_points, mxstep=maxNumSteps, **kwargs)

        elif integrator=='odespy' or integrator=='odespy-vode':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=maxNumSteps)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        elif integrator=='odespy-rkf45':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RKF45(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        elif integrator=='odespy-rk4':
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.RK4(rhs0)
            solver.set_initial_condition(x0)
            X, time_points = solver.solve(time_points, **kwargs)

        else:
            raise Exception("Error: Integration method not found! \n \
                            Please set integrator='odeint' to use the scipy's odeint (Default). \n \
                            Use integrator='odespy-vode' to use vode from odespy (github.com/rajeshrinet/odespy). \n \
                            Use integrator='odespy-rkf45' to use RKF45 from odespy (github.com/rajeshrinet/odespy). \n \
                            Use integrator='odespy-rk4' to use RK4 from odespy (github.com/rajeshrinet/odespy). \n \
                            Alternatively, write your own integrator to evolve the system in time \n")
        return X, time_points


    cpdef set_contactMatrix(self, double t, contactMatrix):
        self.CM=contactMatrix(t)


    def simulator(self, x0, contactMatrix, Tf, Nf, Ti=0, integrator='odeint', maxNumSteps=100000, **kwargs):
        """
        Parameters
        ----------
        x0: np.array
            Initial number of compartment values.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        integrator: TYPE, optional
            Integrator to use either from scipy.integrate or odespy.
            The default is 'odeint'.
        maxNumSteps: int, optional
            maximum number of steps the integrator can take. The default is 100000.
        **kwargs: kwargs for integrator

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.

        """

        def rhs0(xt, t):
            self.CM = contactMatrix(t)
            self.rhs(xt, t)
            return self.dxdt

        X, time_points = self.simulateRHS(rhs0, x0 , Ti, Tf, Nf, integrator, maxNumSteps, **kwargs)

        data     = {'X':X, 't':time_points, 'Ni':self.Ni, 'M':self.M}
        data_out = data.copy()
        data_out.update(self.paramList)
        return data_out

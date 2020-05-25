import  numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
import pyross.utils
import warnings
from scipy.special import legendre
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import nlopt

DTYPE   = np.float




cdef class Simulator:
    """

    Methods
    -------
    solve_Predictor_Corrector: 

    solve_Galerkin: 
    """
    cdef:
        dict params
        str method
        str integrator 
        list IC 
        np.ndarray phi_alpha, p_alpha

    def __init__(self, parameters, IC, subclasses, method='Galerkin', integrator='odeint'):
        self.params     = parameters
        self.method     = method
        self.integrator = integrator 
        self.IC         = IC


        #Define sub-classes of infecteds - optional
        #subclasses = ['Recovered', 'Hospitalized', 'Deceased'] #e.g. Recovered, Hospitalized, Deceased
        Nc = len(subclasses)
        
        #Describe the dynamics of how people move in and out of subclasses.  See section 2.2 of report
        if Nc > 0:
            M = parameters['M']
            #define the probability of eventually having membership in one
            pR = 0.99*np.ones(M);  #probability of eventually recovering for each age class
            pH = 0.05*np.ones(M);  #probability of needing hospitalization for each age class
            pD = 1-pR;             #probability of death for each age class
        
            #prepare for a linear interpolating function evaluated at times:
            tsi_sc = parameters['tsi_sc']
            #tsi_sc  =  np.array([0,   3.,    6.,    9.,   12,    T])  #For convenience, we say that you reach your final state at time T
                                                                      #if this is not OK, adjust previous section accordingly
        
            #probability density function (arbitrary units) for transferring to each of the defined subclasses at tsi
            #once again, the 'shape' of these curves is assumed to be same for all age classes.
            phiR     = np.array([0,    0,    0.5,   3,     2,     0])  #rate of transferring to 'recovered' (arbitrary units)
            phiH_in  = np.array([0,    0,    1,     1,     0,     0])  #rate that people enter hospital     (arbitrary units)
            phiH_out = np.array([0,    0,    0,     1,     1,     0])  #rate that people enter hospital     (arbitrary units)
            phiD     = np.array([0,    0,    0,     1,     1,    .5])  #times at which a person dies        (arbitrary units)
        
            #combine hospital in/out to a single function for net change in hospitalized cases
            phiH = np.add(-phiH_out/np.trapz(phiH_out,tsi_sc),phiH_in/np.trapz(phiH_in,tsi_sc))
        
            #normalize all to one -- can then be rescaled by approprate pR, pH, pD, etc. at a later time
            phiR = phiR/np.trapz(phiR,   tsi_sc)
            phiH = phiH/np.trapz(phiH_in,tsi_sc)
            phiD = phiD/np.trapz(phiD,   tsi_sc)
        
            #group them all together for later processing
            self.phi_alpha = np.array([phiR, phiH, phiD])
            self.p_alpha = np.array([pR, pH, pD])
        else:
            raise Exception('number of E stages should be greater than zero, kE>0')



    
    def solve_Predictor_Corrector(self, parameters, contactMatrix, hybrid=False, tstart=0):
        ''' Predictor/Corrector is a finite difference method described in the TSI report, section 2.5
             It has good properties for speed and accuracy and should be preferred in most applications
             Notable disadvantage is a lack of flexibility in time-stepping -- you must increment by
             the same time step every time.  Function evaluations at intermediate times can be found by
             interpolation.
        '''
        M  = self.params['M']                  
        Nc = self.params['Nc']                   
        Nk = self.params['Nk']                   
        Tf = self.params['Tf']                   
       
        tsi       = self.params['tsi']
        beta      = self.params['beta']                  
        tsi_sc    = self.params['tsi_sc']                  
        
        p_alpha   = self.p_alpha
        phi_alpha = self.phi_alpha


        Cij_t = contactMatrix

        S_0, I_0, Ic_0 = self.IC
    
        #set up the discretization in s
        s = np.linspace(-1,1,Nk)
        h = 2/(Nk - 1)
    
        #find the timesteps
        nt = int(np.round(Tf/h)) + 1
        t = h*np.linspace(0,nt-1,nt)
    
        #weighted betas for numerical integration
        beta_n = h*np.interp(s,tsi,beta)
        beta_n[[0, Nk-1]] = beta_n[[0, Nk-1]]/2
    
        #weighted phi_alpha for numerical integration
        phi_alpha_n = np.zeros((Nc, Nk))
        for i in range(Nc):
            phi_alpha_n[i,:] = h*np.interp(s,tsi_sc,phi_alpha[i,:])
            phi_alpha_n[i,[0,Nk-1]] = phi_alpha_n[i,[0,Nk-1]]/2
    
        #weights for generic trapezoid integration
        w = h*np.ones(Nk)
        w[[0, Nk-1]] = w[[0, Nk-1]]/2
    
        #initialize variables
        S = S_0 + 0
        I = I_0 + 0
        Ic = Ic_0 + 0
    
        #initialize output vectors:
        S_t = np.zeros((M, nt))
        I_t = np.zeros((M, nt))
        Ic_t = np.zeros((Nc,M,nt))
    
        #set their starting values:
        S_t[:,0] = S_0
        I_t[:,0] = np.matmul(w,I_0)
        Ic_t[:,:,0] = Ic_0
    
        #initialize a few variables
        dIc_dt_e = np.zeros((Nc,M))
        dIc_dt_i = np.zeros((Nc,M))
    
        for i in (1 + np.arange(nt-1)):
    
            #explicit time step
            dSdt_e = -np.matmul(np.matmul(np.diag(S),Cij_t(tstart + t[i-1])),np.matmul(beta_n,I))
            Sp = S + h*dSdt_e
    
            for j in range(Nc):
                dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]
    
            I[1:Nk,:] = I[0:(Nk-1),:]
            I[0,:] = -dSdt_e
    
            #'implicit' step
            dSdt_i = -np.matmul(np.matmul(np.diag(Sp),Cij_t(tstart + t[i])),np.matmul(beta_n,I))
            S = S + h/2*(dSdt_e + dSdt_i)
    
            for j in range(Nc):
                dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],I)*p_alpha[j,:]
    
            Ic = Ic + h/2*(dIc_dt_e + dIc_dt_i)
    
            I[0,:] = -dSdt_i
    
            #remember this timestep
            S_t[:,i]     = S
            I_t[:,i]     = np.matmul(w,I)
            Ic_t[:,:, i] = Ic
    
        if not hybrid:
            return t, S_t, I_t, Ic_t
        else:
            return t, S_t, I_t, Ic_t, [S, I, Ic]
    
    
    def solve_Galerkin(self, contactMatrix, hybrid=False, tstart=0):
        '''The Galerkin method is defined in the TSI report, section 2.6
         It spectral accuracy in s and allows for adatptive timestepping in t
         For constant contact matrix, use 'odeint', otherise use 'Crank Nicolson'
             -Notable advantage over predictor/corrector is flexibility in time-stepping
             -Notable disadvantages include:
              (1) ill-suited to non-smooth dynamics (like most spectral methods)
              (2) must be solved as DAE when contact matrix is time-dependent (slow)
        
        For most practical purposes, we regard predictor/corrector as the preferred choice.
        '''
        M  = self.params['M']                  
        Nc = self.params['Nc']                   
        Nk = self.params['Nk']                   
        NL = self.params['NL']                   
        Tf = self.params['Tf']                   
       
        tsi       = self.params['tsi']
        beta      = self.params['beta']                  
        tsi_sc    = self.params['tsi_sc']                  
        
        p_alpha   = self.p_alpha
        phi_alpha = self.phi_alpha


        Cij_t = contactMatrix
        G_method = self.method

        S_0, I_0, Ic_0 = self.IC
    
        #set up the discretization in s
        s = np.linspace(-1,1,1000)
        sk = np.linspace(-1,1,Nk)
    
    
        #initialize first timestep
        h = 1/Nk
    
        #weighted betas for numerical integration
        beta_n = np.zeros(NL)
        for i in range(NL):
            Pn = legendre(i)
            beta_n[i] = np.trapz(Pn(s)*np.interp(s,tsi,beta),s)
    
        #weighted phi_alpha for numerical integration
        phi_alpha_n = np.zeros((Nc, NL))
        for i in range(Nc):
            for j in range(NL):
                Pn = legendre(j)
                phi_alpha_n[i,j] = np.trapz(Pn(s)*np.interp(s,tsi_sc,phi_alpha[i,:]),s)
    
        #advection matrix
        b_nk = np.zeros((NL,NL))
        for i in range(NL):
            for j in range(NL):
                if np.mod(i + j,2) == 1 and j > i:
                    b_nk[i,j] = 2*i + 1
    
        #initialize variables
        S = S_0 + 0
        I = np.zeros((NL,M))
        for i in range(NL):
            Pn = legendre(i)
            for j in range(M):
                I[i,j] = np.trapz(Pn(s)*np.interp(s,sk,I_0[:,j]),s)*(2*i + 1)/2
    
        Ic = Ic_0 + 0
    
        #set their starting values:
        S_t = S_0
        I_t = 2*I[0]
        Ic_t = Ic_0
    
        if G_method == 'Crank Nicolson':
    
            #define a function to spit out time derivatives
            def get_dxdt(x, t):
                Cij = Cij_t(tstart + t)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                dxdt = np.zeros(len(x))
                for i in range(M):
                    dxdt[i] = -S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    dxdt[M + i*NL + np.array(range(NL))] = -np.matmul(b_nk,I[:,i])
                return dxdt
    
            #define a function to give 'residuals' based on current estimate of next time step
            def get_res(x, xp, t, dt):
                Cij = Cij_t(tstart + t)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                Sp = xp[:M]
                Ip = np.transpose(np.reshape(xp[M:],(M,NL)))
                dxdt_e = get_dxdt(x,t)
                dxdt_i = get_dxdt(xp,t+dt)
                res = xp - x - dt/2*(dxdt_e + dxdt_i)
                for i in range(M):
                    res[M + i*NL + NL - 1] = np.matmul((-1.)**np.array(range(NL)),Ip[:,i]) + dxdt_i[i]
                return res
    
            ##EVERYTHING'S WORKING EXCEPT THE JACOBIAN HERE
    
            #define a function to compute Jacobians.
            def get_J(x, t, dt):
                Cij = Cij_t(tstart + t)
                S = x[:M]
                I = np.transpose(np.reshape(x[M:],(M,NL)))
                J = np.zeros((len(x),len(x)))
                for i in range(M):
                    J[i,i] = 1 + dt/2*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    for j in range(M):
                        for k in range(NL):
                            J[i,M + j*NL + k] = dt/2*S[i]*Cij[i,j]*beta_n[k]
                for i in range(M):
                    q = np.identity(NL) + dt/2*b_nk
                    for j in range(NL - 1):
                        J[M + NL*i + j, (i*NL + M):(i*NL + M + NL)] = q[j,:]
                for i in range(M):
                    J[M + i*NL + NL - 1,i] = -np.matmul(Cij[i,:],np.matmul(beta_n,I))
                    J[M + i*NL + NL - 1,(M + i*NL):(M + i*NL + NL)] = (-1)**np.array(range(NL))
                    for j in range(M):
                        J[M + i*NL + NL - 1,(M + j*NL):(M + j*NL + NL)] += - S[i]*Cij[i,j]*beta_n
                return J
    
            t_t = np.array([0])
            t = 0
    
            x = np.append(S,np.transpose(I))
            xh2 = x + 0;
            res = get_res(x, xh2, t, h)
    
            def get_next_step(x,x0,t,dt):
                xp = x0 + 0
                res = get_res(x,xp,t,dt)
                ep = np.amax(np.abs(res))
                tol = 10**-8  #error tolerance of root finding
                count = 0
                maxiter = 100
                while ep > tol and count < maxiter:
                    J = get_J(xp,t,dt)
                    dx = np.linalg.solve(J,res)
                    xp += -dx
                    res = get_res(x, xp, t, dt)
                    ep = np.amax(np.abs(res))
                    count += 1
                if count == maxiter:
                    print('solver maxed out')
                return xp
    
            etol = 10**-4 #absolute error tolerance of CN method
            rtol = 10**-3 #relative error tolerance
            while t < Tf:
                e_abs = 100
                e_rel = 100
                count = 0
                maxiter = 20
                x = xh2
    
                while (e_abs > etol or e_rel > rtol) and count < maxiter:
                    count+=1
                    if t + h > Tf:
                        h = Tf - t
                    #full step
                    xf  = get_next_step(x,x,t,h)
    
                    #half steps
                    xh1 = get_next_step(x,1/2*(x + xf),t, h/2)
                    xh2 = get_next_step(xh1, xf,t +h/2, h/2)
    
                    #compare predictions
                    err = np.abs(xh2 - xf)
                    e_abs = np.amax(err)
                    e_rel = np.amax(err/np.abs(xh2))
    
                    #adaptive time stepping
                    #reduce time step if not meeting error tolerances
                    if e_abs > etol:
                        h = 0.8*h*(etol/e_abs)**.5
                    elif e_rel > rtol:
                        h = 0.8*h*(rtol/e_rel)**.5
    
                #increment
                t_t = np.append(t_t,np.array([t + h/2, t + h]))
                t +=h
                count +=1
    
                #remember some of the results
                S_t = np.append(S_t, xh1[:M])
                S_t = np.append(S_t, xh2[:M])
                I_t = np.append(I_t, 2*xh1[M + NL*np.arange(M)])
                I_t = np.append(I_t, 2*xh2[M + NL*np.arange(M)])
                #subclasses of infecteds:
                Ii = np.transpose(np.reshape(xh1[M:],(M,NL)))
                Ie = np.transpose(np.reshape(x[M:],(M,NL)))
                dIc_dt_i = np.zeros((Nc,M))
                dIc_dt_e = np.zeros((Nc,M))
                for j in range(Nc):
                    dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],Ii)*p_alpha[j,:]
                    dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],Ie)*p_alpha[j,:]
                Ic += h/4*(dIc_dt_e + dIc_dt_i)
                if t == h:
                    Ic_t = np.concatenate(([Ic_0], [Ic]))
                else:
                    Ic_t = np.concatenate((Ic_t, [Ic]))
    
                Ie = Ii + 0
                Ii = np.transpose(np.reshape(xh2[M:],(M,NL)))
                for j in range(Nc):
                    dIc_dt_i[j,:] = np.matmul(phi_alpha_n[j,:],Ii)*p_alpha[j,:]
                    dIc_dt_e[j,:] = np.matmul(phi_alpha_n[j,:],Ie)*p_alpha[j,:]
                Ic += h/4*(dIc_dt_e + dIc_dt_i)
                Ic_t = np.concatenate((Ic_t, [Ic]))
    
                #adaptive time stepping
                #bump time step if well below threshold.
                if e_abs == 0:
                    h = 2*h
                else:
                    if e_rel/rtol >= e_abs/etol:
                        h = 0.8*h*(rtol/e_rel)**.5
                    else:
                        h = 0.8*h*(etol/e_abs)**.5
    
                if count == maxiter:
                    print('CN solver failed')
                    t = Tf
    
            #reshape the output of infecteds class:
            Ic_t_reshape = np.zeros((Nc,M,len(t_t)))
            for i in range(len(t_t)):
                Ic_t_reshape[:,:,i] = Ic_t[i,:,:]
    
            if not hybrid:
                return t_t, np.transpose(np.reshape(S_t,(len(t_t),M))), np.transpose(np.reshape(I_t,(len(t_t),M))), Ic_t_reshape
            else:
                #compute IC for next run from final point of current run.
                S_0 = xh2[:M]
                Ic_0 = Ic
                I_0 = np.zeros((Nk,M))
                I_end = np.reshape(xh2[M:],(M,NL))
                for i in range(M):
                    for j in range(NL):
                        Pn = legendre(j)
                        I_0[:,i] += Pn(sk)*I_end[i,j]
                return t_t, np.transpose(np.reshape(S_t,(len(t_t),M))), np.transpose(np.reshape(I_t,(len(t_t),M))), Ic_t_reshape, [S_0, I_0, Ic_0]
    
        elif G_method == 'odeint':
            import odepsy
            #compute time derivative w. explicit treatment of BC
            def get_dxdt(x, t):
                Cij = Cij_t(tstart + t)
    
                S = x[:M]
                I = np.zeros((NL,M))
                I[:(NL-1), :] = np.transpose(np.reshape(x[M:(M + M*(NL-1))],(M,NL-1)))
    
                #solve for the highest Legendre polynomial coefficients:
                A = np.identity(M)*(-1)**(NL-1) - np.matmul(np.diag(S),Cij)*beta_n[NL-1]
                b = np.zeros(M)
                for i in range(M):
                    b[i] = S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I)) - np.sum(I[:,i]*(-1)**np.arange(NL))
                IN = np.linalg.solve(A,b)
                I[NL-1,:] = IN
    
                #compute derivs for S, I:
                dxdt = np.zeros(len(x))
                for i in range(M):
                    dxdt[i] = -S[i]*np.matmul(Cij[i,:],np.matmul(beta_n,I))
                for i in range(M):
                    dxdt[M + i*(NL-1) + np.array(range(NL-1))] = -np.matmul(b_nk[:(NL-1),:],I[:,i])
    
                #compute derivs for subclasses of infecteds:
                dIc_dt = np.zeros((Nc,M))
                for i in range(Nc):
                    dIc_dt[i,:] = np.matmul(phi_alpha_n[i,:],I)*p_alpha[i,:]
                dxdt[(M + (M)*(NL-1)):] = np.reshape(dIc_dt,(1,Nc*M))
    
                return dxdt
    
            #set initial condition
            x0 = np.append(S,np.transpose(I[:(NL-1),:]))
            x0 = np.append(x0,Ic)
    
            #choose times for reporting output
            t = np.linspace(0, Tf, int(Tf*2*Nk));
            nt = len(t)
    
            #get solution
            u = odeint(get_dxdt, x0, t)
    
            #transform solution to get outputs
            S_t = np.transpose(u[:,:M])
            I_t = 2*np.transpose(u[:,M + (NL - 1)*np.arange(M)])
            Ic_t = np.zeros((Nc,M,nt))
            for i in range(nt):
                Ic_t[:,:,i] = np.reshape(u[i,(M + (NL-1)*(M)):],(Nc,M))
    
            if not hybrid:
                return t, S_t, I_t, Ic_t
            else:
                #compute IC for next run from final point of current run.
                S_0 = S_t[:,nt-1]
                Ic_0 = Ic_t[:,:,nt-1]
                I_0 = np.zeros((Nk,M))
                I_end = np.zeros((M, NL))
                I_end[:,:(NL-1)] = np.reshape(u[nt-1,M:(M + (NL-1)*M)],(M,NL-1))
    
                A = np.identity(M)*(-1)**(NL-1) - np.matmul(np.diag(S_t[:,nt-1]),Cij_t(t[nt - 1]))*beta_n[NL-1]
                b = np.zeros(M)
                for i in range(M):
                    b[i] = S_t[i,nt-1]*np.matmul(Cij_t(t[nt - 1])[i,:],np.matmul(beta_n,np.transpose(I_end))) - np.matmul(I_end[i,:],(-1)**np.arange(NL))
                IN = np.linalg.solve(A,b)
                I_end[:,NL-1] = IN
    
                for i in range(M):
                    for j in range(NL):
                        Pn = legendre(j)
                        I_0[:,i] += Pn(sk)*I_end[i,j]
                return t, S_t, I_t, Ic_t, [S_0, I_0, Ic_0]
        else:
            print('please choose a valid method for solving Galerkin -- Crank Nicolson or odeint')



    def integrate(self, contactMatrix):
        M  = self.params['M']                  
        T  = self.params['T']                   
        Nc = self.params['Nc']                   
        Nk = self.params['Nk']                   
        NL = self.params['NL']                   
        Tf = self.params['Tf']                   
        
        tsi       = self.params['tsi']
        beta      = self.params['beta']                  
        beta      = self.params['beta']                  
        tsi_sc    = self.params['tsi_sc']                  
        phi_alpha = self.params['phi_alpha']                  
        p_alpha   = self.params['p_alpha']                  
        tswap     = self.params['tswap']                  
        method    = self.method
        G_method  = self.integrator
        
        
        if method == 'Predictor_Corrector':
            t, S_t, I_t, Ic_t = self.solve_Predictor_Corrector(contactMatrix)

        elif method == 'Galerkin':
            t, S_t, I_t, Ic_t = self.solve_Galerkin(contactMatrix)
        
        elif method == 'Hybrid':
            tc = 0
            count = 0
            while tc < Tf:
                
                #run the next simulation
                if tc < tswap[count]:
                    tstep = tswap[count] - tc
                    self.params['Tf'] = tstep
                    sol = self.solve_Galerkin(contactMatrix,True, tc)
                else:
                    tstep = 2
                    self.params['Tf'] = tstep
                    sol = self.solve_Predictor_Corrector(contactMatrix, True, tc)
                    if count < len(tswap)-1:
                        count += 1
                
                #unpack results
                if count == 0 or (count == 1 and tc == 0):
                    t, S_t, I_t, Ic_t, self.IC = sol
                else:
                    t  = np.concatenate((t,tc + sol[0]))
                    nt = len(t); S_t_new = np.zeros((M,nt)); I_t_new = np.zeros((M,nt)); Ic_t_new = np.zeros((Nc,M,nt))
                    for i in range(M):
                        S_t_new[i,:] = np.append(S_t[i,:],sol[1][i,:])
                    for i in range(M):
                        I_t_new[i,:] = np.append(I_t[i,:],sol[2][i,:])
                    for i in range(Nc):
                        for j in range(M):
                            Ic_t_new[i,j,:] = np.append(Ic_t[i,j,:],sol[3][i,j,:])
                    self.IC = sol[4]
                    S_t, I_t, Ic_t = [S_t_new, I_t_new, Ic_t_new]
                
                #prepare for next loop
                tc = tc + tstep
                #print(IC_t)
        

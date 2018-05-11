# -*- coding: utf-8 -*-

# May 2018
# Ethan Campbell
# University of Washington
# School of Oceanography
#
# Implements first-order local turbulence closure for evolution of stably stratified boundary layer.
#
# Borrows framework from Python script by Emily Ramnarine for dry convective boundary layer (ATM S 547, April 2016).

from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# constants
u_0_val = 10     # initial geostrophic u-wind speed, uniform with height (units: m s^-1)
v_0_val = 0      # initial v-wind speed, zero at all heights (units: m s^-1)
rho_0 = 1.2      # air density (units: kg m-3)
c_p = 1000       # heat capacity of air (units: J kg-1 K-1)
k = 0.4          # Von Karman constant (unitless)
g = 9.8          # gravitational acceleration (units: m s^-2)
u_star = 0.2     # given friction velocity (units m s^-1)
theta_surf = 290 # constant surface temperature for constant buoyancy flux (units: K)
f = 10**-4       # Coriolis frequency (units: s^-1)

# constants changing between runs 1-3
H_0_all = [-10,-10,-25]   # surface sensible heat flux (units: W m-2)
lamb_all = [10,5,5]       # asymptotic Blackadar lengthscale (units: m)

# conduct runs 1-3
for run_idx in range(3):
    H_0 = H_0_all[run_idx]
    lamb = lamb_all[run_idx]

    # derived quantities
    theta_flux_0 = H_0 / (rho_0 * c_p)   # surface theta flux (= overline{w'theta'}) (units: K m s-1)

    # model setup (height)
    z_top = 150     # upper z level (units: m)
    dz = 10         # level spacing (units: m)
    z_int = arange(0, z_top+dz, dz)              # (N_z+1) layer interfaces (for computing turbulent fluxes)
    z_mid = arange(dz/2, z_top - dz/2 + dz, dz)  # (N_z)   layer centers ("half-levels") (for computing u, v, theta)

    # model setup (time)
    t_f = 3 * 60 * 60    # integration time (= 3 hours -> units: s)
    dt = 10              # integration time step (units: s)
    t_span = arange(0, t_f+dt, dt)

    def create_state_vec(theta,u,v):
        return concatenate((theta,u,v))

    def unpack_state_vec(state_vec):
        state_matrix = state_vec.reshape((3,len(z_mid)))
        return state_matrix[0], state_matrix[1], state_matrix[2]

    def unpack_state_vec_evolution(state_vec_evolution):
        state_matrix = state_vec_evolution.reshape((len(t_span),3,len(z_mid)))
        return state_matrix[:,0,:], state_matrix[:,1,:], state_matrix[:,2,:]

    # initial profiles
    theta_0 = 290 + 0.01 * z_mid             # initial theta profile (units: K)
    u_0 = tile(u_0_val,len(z_mid))           # initial u-wind profile (units: m s^-1)
    v_0 = tile(v_0_val,len(z_mid))           # initial v-wind profile (units: m s^-1)
    state_vec_0 = create_state_vec(theta_0,u_0,v_0)   # concatenate theta, u, v
    l = lamb / (1 + lamb / (k * z_int[1:len(z_mid)])) # (N_z-1) Blackadar length scale for interior flux levels only
                                                      # (units: m)

    def d_state_vec_dt(state_vec, t, extras_out=False):

        """
        Calculates time derivatives (tendencies) of theta, u, and v.

        Arguments:
            state_vec:  (N_z x 3) concatenation of theta, u, v
            extras_out: (boolean) whether Ri, SHF, and K_h are returned (see inside function for details)

        Returns:
            state_vec_tendency: (N_z x 3) tendencies d(theta)/dt, du/dt, dv/dt at layer centers

        """

        # unpack state vector
        theta, u, v = unpack_state_vec(state_vec)

        # derived quantities at each time step (assuming theta(z=0) evolves...?)
        B_0 = (g * H_0) / (rho_0 * c_p * theta_surf)   # surface buoyancy flux (units: m^2 s^-3)
        L = -1 * u_star**3 / (k * B_0)                 # Monin-Obukhov length (units: m)
        U = sqrt(u**2 + v**2)                          # wind speed (units: m s^-1)
        s = abs(diff(U) / diff(z_mid))                 # wind shear (units: s^-1)

        # calculate K_a
        # for both heat (theta) and momentum (u, v), K_a = l^2 * s * (1 + 5*zeta)^(-2), with zeta = z/L
        z = z_int[1:len(z_mid)]
        K_a = l**2 * s * (1 + 5*z/L)**(-2)

        # calculate vertical derivatives
        dtheta_dz = diff(theta) / diff(z_mid)
        du_dz = diff(u) / diff(z_mid)
        dv_dz = diff(v) / diff(z_mid)

        # calculate fluxes, overbar{w'a'} for a = theta, u, v
        # first order closure: overbar{w'a'} = -K_a * da/dz
        flux_theta, flux_u, flux_v = zeros(len(z_int)), zeros(len(z_int)), zeros(len(z_int))

        # –––> set surface values
        flux_theta[0] = theta_flux_0
        flux_u[0] = -1 * u_star**2
        flux_v[0] = -1 * u_star**2

        # –––> set interior values
        flux_theta[1:len(z_mid)] = -1 * K_a * dtheta_dz
        flux_u[1:len(z_mid)] = -1 * K_a * du_dz
        flux_v[1:len(z_mid)] = -1 * K_a * dv_dz

        # –––> set upper boundary values
        flux_theta[len(z_mid)] = 0
        flux_u[len(z_mid)] = 0
        flux_v[len(z_mid)] = 0

        # calculate tendencies
        # dtheta(z)/dt = -d(overline{w'theta'})/dz, du(z)/dt = -d(overline{w'u'})/dz, dv(z)/dt = -d(overline{w'v'})/dz
        dtheta_dt = -1 * diff(flux_theta) / diff(z_int)   # note: could simply use dz instead of diff(z_int)
        du_dt = -1 * diff(flux_u) / diff(z_int)
        dv_dt = -1 * diff(flux_v) / diff(z_int)

        # rotate u, v vectors, according to Ekman balance: du/dt = fv and dv/dt = -fu
        du_dt += f * v
        dv_dt -= f * u

        state_vec_tendency = create_state_vec(dtheta_dt,du_dt,dv_dt)

        if extras_out:
            # calculate extra profiles to output
            N_squared = (g / mean(theta)) * dtheta_dz  # (N_z-1) buoyancy frequency squared (units: s^-2)
            s_squared = s**2
            s_squared[s_squared == 0] = NaN   # note: to avoid dividing by zero
            Ri = N_squared / s_squared        # (N_z-1) Richardson number (unitless)
            shf = rho_0 * c_p * flux_theta    # (N_z+1) turbulent sensible heat flux (units: W m^-2)
            K_h = K_a                         # (N_z-1) eddy heat diffusivity (units: m s^-2)
            return state_vec_tendency, Ri, shf, K_h

        else:
            return state_vec_tendency


    # use ODE solver to integrate coupled equations
    state_vec_evolution = odeint(d_state_vec_dt, state_vec_0, t_span)  # note: could add args=(...)
    theta, u, v = unpack_state_vec_evolution(state_vec_evolution)
    t, z = broadcast_arrays(t_span[:,None], z_mid[None,:])             # note: sort of like meshgrid

    # re-calculate hourly profiles of Ri, SHF, and K_h
    profs_Ri, profs_shf, profs_K_h, profs_time_strings = [], [], [], []
    profs_times = arange(0, t_f+1, 3600)   # hourly
    for t_idx, time in enumerate(t_span):
        if time in profs_times:
            state_vec = create_state_vec(theta[t_idx,:],u[t_idx,:],v[t_idx,:])
            _, Ri, shf, K_h = d_state_vec_dt(state_vec,time,extras_out=True)
            profs_time_strings.append('{:.0f} hour(s)'.format(time/3600))
            profs_Ri.append(Ri)
            profs_shf.append(shf)
            profs_K_h.append(K_h)

    # plots
    plt.figure(figsize=(7,9))

    # ———> time-height evolution of theta, u, v
    plt.subplot(3,2,1)
    im = plt.pcolormesh(t/3600,z,theta,cmap='viridis',shading='gouraud',vmin=283.5,vmax=291.5)
    plt.contour(t/3600,z,theta,10,colors='k',linewidths=0.5)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)
    plt.ylabel('z (m)')
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title(r'Evolution of $\theta(z)$')

    plt.subplot(3,2,3)
    im = plt.pcolormesh(t/3600,z,u,cmap='RdGy',shading='gouraud',vmin=-15,vmax=15)
    plt.contour(t/3600,z,u,10,colors='k',linewidths=0.5)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)
    plt.ylabel('z (m)')
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title(r'Evolution of $u(z)$')

    plt.subplot(3,2,5)
    im = plt.pcolormesh(t/3600,z,v,cmap='RdGy',shading='gouraud',vmin=-15,vmax=15)
    plt.contour(t/3600,z,v,10,colors='k',linewidths=0.5)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel('time (hours)')
    plt.ylabel('z (m)')
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title(r'Evolution of $v(z)$')

    # ———> hourly profiles of Ri, SHF, K_h

    plt.subplot(3,2,2)
    for t in range(len(profs_Ri)):
        plt.plot(profs_Ri[t],z_int[1:-1],label=profs_time_strings[t])
    plt.legend(fontsize=8,frameon=False)
    plt.xlabel('Ri')
    plt.xlim(0,0.23)
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title('Hourly profiles of Ri')

    plt.subplot(3,2,4)
    for t in range(len(profs_shf)):
        plt.plot(profs_shf[t],z_int,label=profs_time_strings[t])
    plt.legend(fontsize=8,frameon=False)
    plt.xlabel(r'Turbulent sensible heat flux (W m$^{-2}$)')
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title('Hourly profiles of SHF')

    plt.subplot(3,2,6)
    for t in range(len(profs_K_h)):
        plt.plot(profs_K_h[t],z_int[1:-1],label=profs_time_strings[t])
    plt.legend(fontsize=8,frameon=False)
    plt.xlabel(r'Eddy heat diffusivity (m s$^{-2}$)')
    plt.ylim(0,z_top)
    plt.tick_params(axis='both',labelsize=8)
    plt.title('Hourly profiles of K_h')

    plt.tight_layout()
    plt.savefig('/Users/Ethan/Desktop/hw4_run_{0}.pdf'.format(run_idx+1))

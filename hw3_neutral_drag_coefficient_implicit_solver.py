from numpy import *
import matplotlib.pyplot as plt

# constants
g = 9.81     # m s-2
a_c = 0.016  # unitless
z_R = 10.0   # m
k = 0.4      # unitless

# data vectors
u10_vec = arange(4,20.1,0.1)
C_DN = zeros(len(u10_vec))

# implicit solver
# matches Charnock's formula and bulk aerodynamic formula
for u10_idx, u10 in enumerate(u10_vec):
    C_DN_valid_range = arange(0.1e-3,5.0e-3,0.001e-3)
    C_DN_abs_diff = zeros(len(C_DN_valid_range))
    # match sides of the equation (C_DN_try and C_DN_computed)
    for C_DN_idx, C_DN_try in enumerate(C_DN_valid_range):
        z_0 = z_R / exp((k**2 / C_DN_try)**0.5)
        u_star_sq = z_0 * g / a_c
        C_DN_computed = u_star_sq / u10**2
        C_DN_abs_diff[C_DN_idx] = abs(C_DN_computed - C_DN_try)
    # choose closest match
    C_DN[u10_idx] = C_DN_valid_range[argmin(C_DN_abs_diff)]

# approximation for C_DN
C_DN_approx = (0.75 + 0.067*u10_vec) * 10**-3  # Eq. 5.11, online notes

# plot
plt.figure(figsize=(5,3))
plt.plot(u10_vec,C_DN,c='k',label='Computed implicitly')
plt.plot(u10_vec,C_DN_approx,c='b',ls='--',label='Approximation (Eq. 5.11)')
plt.legend(loc='upper left',frameon=False)
plt.xlabel(r'$u_{10}$ (m/s)')
plt.ylabel(r'$C_{DN}$')
plt.tight_layout()
plt.savefig('/Users/Ethan/Documents/UW/By course/2018-03 - ATM S 547 (Christopher S. Bretherton)/'
            '2018-04-26 - problem set 3/q2.pdf')

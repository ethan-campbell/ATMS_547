from numpy import *
import scipy.io as sio
from scipy import stats
from scipy import signal
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# prep: load data
# note column names: ['u(m/s)', 'v(m/s)', 'w(m/s)', 't(oC)', 'q(g/kg)', 'p(hp)']
hw_dir = '/Users/Ethan/Documents/UW/By course/2018-03 - ATM S 547 (Christopher S. Bretherton)/' \
         '2018-04-12 - problem set 1/'
data = pd.read_csv(hw_dir + 'rf18L1.txt',header=1,delim_whitespace=True,index_col=0)
data.index.name = 'time(s)'

# prep: set first time to zero
data.index -= data.index[0]
max_time = max(data.index)

# Q1: plot u, v, w
plt.figure(figsize=(7.5,3.0))
plt.plot(data['u(m/s)'],lw=0.25,c='navy',label='u')
plt.plot(data['v(m/s)'],lw=0.25,c='darkgreen',label='v')
plt.plot(data['w(m/s)'],lw=0.25,c='k',label='w')
plt.xlim([0,max_time])
plt.ylim([plt.ylim()[0],plt.ylim()[1]*1.15])
leg = plt.legend(frameon=False,ncol=3,loc='upper right')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
plt.xlabel('Time (s)')
plt.ylabel('Wind speed (m/s)')
plt.tight_layout()
plt.savefig(hw_dir + 'fig_q1.jpg',dpi=300)
plt.close()

# Q2: spectral analysis of w and u
window = 4096  # samples per data window
rate = 25.0    # sampling frequency (Hz)
U_0 = 100.0    # aircraft speed (m/s)
freq_u, power_uu = signal.welch(data['u(m/s)'],fs=rate,window='hanning',nperseg=window,noverlap=window/2)
freq_w, power_ww = signal.welch(data['w(m/s)'],fs=rate,window='hanning',nperseg=window,noverlap=window/2)

# Q2: generation frequencies
freq_u_max_idx = argmax(freq_u * power_uu)
freq_w_max_idx = argmax(freq_w * power_ww)
freq_u_max = freq_u[freq_u_max_idx]
power_uu_max = freq_u_max * power_uu[freq_u_max_idx]
eddy_len_u = U_0 / freq_u_max
freq_w_max = freq_w[freq_w_max_idx]
power_ww_max = freq_w_max * power_ww[freq_w_max_idx]
eddy_len_w = U_0 / freq_w_max

# Q3: eyeball fit to spectra
C_u = 0.04
C_w = 0.03
freq_vec_u = arange(freq_u_max,10**1,0.0001)
freq_vec_w = arange(freq_w_max,10**1,0.0001)
power_uu_fit = freq_vec_u * C_u * freq_vec_u**(-5/3)
power_ww_fit = freq_vec_w * C_w * freq_vec_w**(-5/3)
eps_u = (2*pi/U_0) * ((C_u/0.8)**(3/2))
eps_w = (2*pi/U_0) * ((C_w/0.8)**(3/2))

# Q2 and Q3: plot of spectra
plt.figure(figsize=(7.5,3.5))

plt.subplot(1,2,1)
plt.loglog(freq_u,freq_u * power_uu,c='k',lw=0.5)
plt.loglog(freq_vec_u,power_uu_fit,c='darkred',lw=1.25,
           label=r'$P_{uu}\approx$' + str(C_u) + r'$f^{-5/3}$' + '\n' + r'$\epsilon \approx$'
                 + '{0:.1e}'.format(eps_u) + r' m$^2$ s$^{-3}$')
plt.ylim([0.2*10**-2,0.4*10**0])
plt.loglog([freq_u_max,freq_u_max],[plt.ylim()[0],power_uu_max-0.001],c='b',ls='--',lw=0.75)
plt.text(freq_u_max,0.008,'Eddy length  \nscale: {0:.1f} m'.format(eddy_len_u),color='b',
         va='center',ha='right',rotation='vertical')
plt.legend(loc='upper right',frameon=False)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Scaled power spectral density\n' + r'$fP_{uu}$ (m$^2$ s$^{-2}$)')
plt.title('E-W velocity component $u$')

plt.subplot(1,2,2)
plt.loglog(freq_w,freq_w * power_ww,c='k',lw=0.5)
plt.loglog(freq_vec_w,power_ww_fit,c='darkred',lw=1.25,
           label=r'$P_{ww}\approx$' + str(C_w) + r'$f^{-5/3}$' + '\n' + r'$\epsilon \approx$'
                 + '{0:.1e}'.format(eps_w) + r' m$^2$ s$^{-3}$')
plt.ylim([0.2*10**-2,0.4*10**0])
plt.loglog([freq_w_max,freq_w_max],[plt.ylim()[0],power_ww_max-0.001],c='b',ls='--',lw=0.75)
plt.text(freq_w_max,0.008,'Eddy length  \nscale: {0:.1f} m'.format(eddy_len_w),color='b',
         va='center',ha='right',rotation='vertical')
plt.legend(loc='upper right',frameon=False)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$fP_{ww}$ (m$^2$ s$^{-2}$)')
plt.title('Vertical velocity component $w$')
plt.tight_layout()
plt.savefig(hw_dir + 'fig_q2.jpg',dpi=300)
plt.close()

# Q4: high-pass filtering to isolate turbulence in signal
butter_order = 4  # Butterworth filter order
cutoff_freq = 0.05 * 2  # Hz (not sure why factor of 2 is necessary)
init = round(rate / cutoff_freq)  # discard first N samples for initialization
def hp_filt(series,new_series_name):
    b, a = signal.butter(N=butter_order,Wn=cutoff_freq / rate,btype='highpass')
    data[new_series_name] = signal.filtfilt(b=b,a=a,x=series)
    data[new_series_name].iloc[:init] = NaN
hp_filt(data['w(m/s)'],'w_filt(m/s)')
w_high_pass = data['w_filt(m/s)'].values[init:]
t_high_pass = data.index.values[init:]
print('std(w, original): ' + str(std(data['w(m/s)'])))
print('std(w, high-passed): ' + str(std(w_high_pass)))

# Q4: analyze and plot filtered spectrum
freq_w_hp, power_ww_hp = signal.welch(w_high_pass,fs=rate,window='hanning',nperseg=window,noverlap=window/2)
plt.figure(figsize=(4.0,3.75))
plt.loglog(freq_w_hp,freq_w_hp * power_ww_hp,c='k',lw=0.5)
plt.loglog(freq_vec_w,power_ww_fit,c='darkred',lw=1.25)
plt.ylim([0.2*10**-2,0.4*10**0])
plt.loglog([freq_w_max,freq_w_max],[plt.ylim()[0],power_ww_max-0.001],c='b',ls='--',lw=0.75)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Scaled power spectral density\n' + r'$fP_{ww}$ (m$^2$ s$^{-2}$)')
plt.title('Vertical velocity component $w$,\nhigh-pass filtered')
plt.tight_layout()
plt.savefig(hw_dir + 'fig_q4.jpg',dpi=300)
plt.close()

# Q5: autocorrelation timescale of w
max_lag_t = 10.0  # seconds
lag_n = arange(0,max_lag_t*rate).astype(int)
lag_t = arange(0,max_lag_t*rate) / rate
autocorrs_w_filt = array([data['w_filt(m/s)'].iloc[init:].autocorr(lag=shift) for shift in lag_n])
autocorrs_w_orig = array([data['w(m/s)'].autocorr(lag=shift) for shift in lag_n])
tau_idx_filt = argmin(abs(autocorrs_w_filt - 1/e))
tau_filt = lag_t[tau_idx_filt]
tau_idx_orig = argmin(abs(autocorrs_w_orig - 1/e))
tau_orig = lag_t[tau_idx_orig]
print('characteristic updraft scale (filtered w): ' + str(2 * tau_filt * U_0))
print('characteristic updraft scale (original w): ' + str(2 * tau_orig * U_0))

# Q5: plot autocorrelation sequence
plt.figure(figsize=(7.5,4.0))
plt.plot([0,10],[0,0],c='k',lw=0.5)
plt.plot([0,10],[1/e,1/e],c='darkred',lw=1.0,ls='--',label=r'$R=1/e$')
plt.plot(lag_t,autocorrs_w_orig,c='k',lw=1.5,ls='--',label='original $w$')
plt.plot([tau_orig,tau_orig],[-0.2,1],c='navy',lw=1.0,ls='--',label=r'  $\tau=$' + ' {0:.2f} s'.format(tau_orig))
plt.plot(lag_t,autocorrs_w_filt,c='k',lw=1.5,label='high-pass filtered $w$')
plt.plot([tau_filt,tau_filt],[-0.2,1],c='navy',lw=1.0,label=r'  $\tau=$' + ' {0:.2f} s'.format(tau_filt))
plt.xlim([0,10])
plt.ylim([-0.2,1])
plt.legend(loc='upper right',frameon=False)
plt.xlabel('Lag (s)')
plt.ylabel('Lagged autocorrelation')
plt.title('Vertical velocity component $w$')
plt.tight_layout()
plt.savefig(hw_dir + 'fig_q5.jpg',dpi=300)
plt.close()

# Q6: high-pass filter other variables; discard data in initialization period for good
hp_filt(data['u(m/s)'],'u_filt(m/s)')
hp_filt(data['v(m/s)'],'v_filt(m/s)')
hp_filt(data['t(oC)'],'t_filt(oC)')
data['q(kg/kg)'] = data['q(g/kg)'] / 1000
hp_filt(data['q(kg/kg)'],'q_filt(kg/kg)')
data = data.iloc[init:]

# Q6: statistics
print('variances (w, u, v, t, q):')
var_w = data['w_filt(m/s)'].std() ** 2;  print(var_w)
var_u = data['u_filt(m/s)'].std() ** 2;  print(var_u)
var_v = data['v_filt(m/s)'].std() ** 2;  print(var_v)
var_t = data['t_filt(oC)'].std() ** 2;   print(var_t)
var_q = data['q_filt(kg/kg)'].std() ** 2; print(var_q)
print('correlations between w and u, v, t, q:')
corr_w_u = data['w_filt(m/s)'].corr(data['u_filt(m/s)']);  print(corr_w_u)
corr_w_v = data['w_filt(m/s)'].corr(data['v_filt(m/s)']);  print(corr_w_v)
corr_w_t = data['w_filt(m/s)'].corr(data['t_filt(oC)']);   print(corr_w_t)
corr_w_q = data['w_filt(m/s)'].corr(data['q_filt(kg/kg)']); print(corr_w_q)
cov_w_u = corr_w_u * ((var_w ** 0.5) * (var_u ** 0.5))
cov_w_v = corr_w_v * ((var_w ** 0.5) * (var_v ** 0.5))
cov_w_t = corr_w_t * ((var_w ** 0.5) * (var_t ** 0.5))
cov_w_q = corr_w_q * ((var_w ** 0.5) * (var_q ** 0.5))

# Q6: flux calculations
print('flux calculations (mom-u, mom-v, sensible, latent, buoyancy):')
rho_0 = 1.21  # kg m-3
C_p = 1004.0  # J kg K-1
L_v = 2.5e6   # J kg-1
g = 9.8       # m s-2
T_0 = data['t(oC)'].mean() + 273.15  # K
mom_flux_u = rho_0 * cov_w_u; print(mom_flux_u)  # Pa
mom_flux_v = rho_0 * cov_w_v; print(mom_flux_v) # Pa
sens_heat_flux = rho_0 * C_p * cov_w_t; print(sens_heat_flux)  # W m-2
ltnt_heat_flux = rho_0 * L_v * cov_w_q; print(ltnt_heat_flux)  # W m-2
buoyancy_flux = (g / (rho_0 * C_p * T_0)) * (sens_heat_flux + 0.07 * ltnt_heat_flux)  # m2 s-3 ? (formula is approx.)
print(buoyancy_flux)
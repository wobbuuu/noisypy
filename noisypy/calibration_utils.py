import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import csaps

from .noisy_utils import det2power
from .settings import *


__all__ = ['calibrate', 'fit_calibration']


def calibrate(vbs, res_inds, det_inds, skip=[], name='./data/cond', both=True,  det_ind=2,\
              shift=1, r_set=varz.r_set, preamp=varz.preamp, r_wires=215/2, coef=varz.coef,\
              r0=varz.r0, vmax=1e-3, sg_rpar=None, sg_det=None, sg_power=(9, 1), cs_smooth=None):
    """
    ADD AUTOEXCLUSION OF BAD CURVES
    Provides calibration data and calibration function
    DC measurement in 3 probe geometry.
    If 1 terminal of sample is grounded (DC), r_wires must be 0
    """
    r_set = np.atleast_1d(r_set)
    if len(r_set) == 1:
        r_set = r_set*np.ones(len(res_inds))
        
    # Resistance vs Vtg
    res = []
    for j, ind in enumerate(res_inds):        
        vb = vbs[j]
        df = pd.read_csv(name+str(ind)+'.dat', sep=' ', header=None)
        if both:
            df.loc[1::2,2::] = df.loc[1::2,:2:-1].values
        for k in range(len(df)):
            v = np.array(df.loc[k,2:].values/preamp, dtype='float64')
            # Shift typical occurs even in DC measurements
            v = pd.Series(v).shift(-shift)
            # Subtract preamp offset
            v -= v[np.abs(vb).argmin()] 
            i = (vb*coef - v) / r_set[j]
            inds = np.abs(v) < vmax
            r = np.nan
            if len(i[inds]) > 3:
                r = np.polyfit(i[inds], v[inds], 1)[0]
            res.append(r)            
    res = np.array(res)
    
    # Detector vs Vtg
    det = pd.DataFrame()
    det_inds = set(det_inds) - set(skip)
    for ind in det_inds:
        df = pd.read_csv(name+str(ind)+'.dat', sep=' ', header=None)
        df.set_index(1, inplace=True)
        det = pd.concat([det, df[det_ind]], axis=1, ignore_index=True)
    
    # Always start measurement from smallest resistance (largest detector signal)
#     if det.index[0] < det.index[-1]:
#         det = det[::-1]
    # This is sometimes wrong
  
    vtg = det.index
    det = det.mean(axis=1)
    
    # Subtract wires (3-probe measurements)
    r_sample = max(res)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        r_tr = res*r_sample / (r_sample-res)
        rpar = 1 / (1/r_tr + 1/(r_sample-r_wires) + 2/r0)

    calib_tg = pd.DataFrame({'vtg':vtg, 'res':res, 'rpar':rpar, 'det':det,\
                             'power':det2power(det)})
    if calib_tg['res'].isna().any():
        print("Following transistor gate values are skipped due to vmax threshold:\n[",\
              end='')
        to_print = calib_tg[calib_tg['res'].isna()]['vtg'].values
        [print('{:.4g}'.format(value), end=' ') for value in to_print]
        print("]")
        calib_tg.dropna(inplace=True)
        
    # subtract offset
    fit_inds = calib_tg['rpar'] < 300
    if len(calib_tg.loc[fit_inds, 'rpar']) > 2:
        _, P0 = np.polyfit(calib_tg.loc[fit_inds, 'rpar'], calib_tg.loc[fit_inds, 'power'], 1)
        calib_tg['power'] -= P0
    
    fig, ax = plt.subplots(1, 3, figsize=varz.figsize13)
    ax[0].plot(calib_tg['vtg'], calib_tg['rpar'], '.')
    ax[1].plot(calib_tg['vtg'], calib_tg['det'], '.')
    ax[2].plot(calib_tg['rpar'], calib_tg['power'], '.', ms=3)
    
    # savgol_filter on rpar vs Vtg
    if sg_rpar:
        calib_tg.sort_values(by='vtg', inplace=True)
        vtg_reg = np.arange(calib_tg['vtg'].min(), calib_tg['vtg'].max(),\
                            calib_tg['vtg'].diff().min())
        rpar_smooth = savgol_filter(np.interp(vtg_reg, calib_tg['vtg'],\
                                              calib_tg['rpar']), *sg_rpar)
        calib_tg['rpar'] = np.interp(calib_tg['vtg'], vtg_reg, rpar_smooth)
        ax[0].plot(calib_tg['vtg'], calib_tg['rpar'])
    
    # savgol_filter on det vs Vtg
    if sg_det:
        calib_tg.sort_values(by='vtg', inplace=True)
        vtg_reg = np.arange(calib_tg['vtg'].min(), calib_tg['vtg'].max(),\
                            calib_tg['vtg'].diff().min())
        det_smooth = savgol_filter(np.interp(vtg_reg, calib_tg['vtg'],\
                                             calib_tg['det']), *sg_det)
        calib_tg['det'] = np.interp(calib_tg['vtg'], vtg_reg, det_smooth)
        ax[1].plot(calib_tg['vtg'], calib_tg['det'])
    
    if sg_det or sg_rpar:
        calib_tg['power'] = det2power(calib_tg['det'])
        ax[2].plot(calib_tg['rpar'], calib_tg['power'], '.', markersize=3)
    
    # savgol_filter on power vs rpar or csaps smoothingspline
    calib_tg.sort_values(by='rpar', inplace=True)
    rpar_reg = np.linspace(calib_tg['rpar'].min(), calib_tg['rpar'].max(), 500)
    if not cs_smooth:
        power = np.interp(rpar_reg, calib_tg['rpar'], calib_tg['power'])
        smoothed_power = savgol_filter(power, *sg_power)
        calib = sp.interpolate.interp1d(rpar_reg, smoothed_power, fill_value='extrapolate')
    else:
        calib = csaps.CubicSmoothingSpline(calib_tg['rpar'], calib_tg['power'],\
                                           smooth=cs_smooth)

    calib_data = calib_tg.loc[calib_tg['rpar']>0, :]   
    ax[2].plot(rpar_reg, calib(rpar_reg))
    
    ax[0].set_xlabel(get_label('Vtg'))
    ax[0].set_ylabel(get_label('rpar'))    
    ax[1].set_xlabel(get_label('Vtg'))
    ax[1].set_ylabel(get_label('Vdet'))
    ax[2].set_xlabel(get_label('rpar'))
    ax[2].set_ylabel(get_label('P'))
    
    return calib_data, calib



def fit_calibration(calib_data, filt, fitrange=[], p0_dict={'C':18e-12, 'f0':22.5e6,\
                    'G':1.4e7, 'S_amp':5e-27, 'P0':0, 'T':4.2}, fixed=['f0', 'T'], kwargs={},\
                    names={'r_par':'rpar', 'power':'power', 'f':'f', 'Tr':'Tr'}, plot=True):
    
    if {'C', 'f0', 'G', 'S_amp', 'P0', 'T'} -\
       set(p0_dict.keys()) - set(fixed) != set():
        raise Exception
    
    if len(fitrange) == 2:
        inds = (calib_data[names['r_par']] > fitrange[0]) &\
               (calib_data[names['r_par']] < fitrange[1])
    else:
        inds = calib_data.index
    r_par_fit = calib_data.loc[inds, names['r_par']].values
    power_fit = calib_data.loc[inds, names['power']].values
    
    r_par = calib_data[names['r_par']].values
    power = calib_data[names['power']].values
    f = filt[names['f']].values
    Tr = filt[names['Tr']].values
    
    x0 = []
    param_inds = {}
    for k, v in p0_dict.items():
        if k not in fixed:
            x0.append(v)
            param_inds[k] = len(x0) - 1
            
            
    def _params2dict(params):
        p_dict = {k:params[param_inds[k]] for k in param_inds}
        p_dict.update({k:p0_dict[k] for k in fixed})
        return p_dict
    
    
    def _G(r_par, p_dict):
        r_par = np.atleast_1d(r_par)
        p_dict['L'] = 1 / (4* np.pi**2 * p_dict['C'] * p_dict['f0']**2)        
        shape = (len(r_par), len(f))
        mesh = dict()
        mesh['r_par'] = np.array([r_par for i in range(shape[1])]).T
        mesh['f'] = np.array([f for i in range(shape[0])])
        mesh['Tr'] = np.array([Tr for i in range(shape[0])])
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            mesh['z_lc'] = np.abs(1 / (2j*np.pi*mesh['f']*p_dict['C'] +\
                                       1/(2j*np.pi*mesh['f']*p_dict['L'])))
        integral = np.trapz(mesh['Tr']*p_dict['G']/\
                (mesh['r_par']**(-2) + mesh['z_lc']**(-2))/50, f, axis=1)
        return integral
    
    
    def _get_power(r_par, p_dict):
        r_par = np.atleast_1d(r_par)
        return (4*kB*p_dict['T']/r_par +\
                p_dict['S_amp'])*_G(r_par, p_dict) + p_dict['P0']
    
    
    def _power_fitting_min(params, r_par, power):
        p_dict = _params2dict(params)
        return np.trapz((_get_power(r_par, p_dict) - power)**2, r_par)
    
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        result = sp.optimize.minimize(_power_fitting_min, x0=x0, method='Nelder-Mead',\
                                      args=(r_par_fit, power_fit,), **kwargs)
    p_dict = _params2dict(result.x)
    [print(str(k)+': '+'{:.3g}'.format(v), end='; ') for k,v in p_dict.items()]
    print()
    if plot:
        fig, ax = plt.subplots()
        ax.plot(r_par, _get_power(r_par, p0_dict), lw=1, label='x0')
        ax.plot(r_par, power, 'black', marker='.', ms=2, lw=0, label='data')
        ax.plot(r_par, _get_power(r_par, p_dict), lw=1, label='Nelder-Mead')
        ax.set_xlabel(get_label('rpar'))
        ax.set_ylabel(get_label('G'))
        ax.legend();
    return lambda x: _get_power(x, _params2dict(result.x)),\
           lambda x: _G(x, _params2dict(result.x)), p_dict

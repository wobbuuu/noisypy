import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from .settings import *
from .plot_utils import *

__all__ = ['four_probe', 'conv_preamp', 'spec_density', 'spec_fitting', 'det2power',\
           'plot_noise', 'plot_aux', 'plot_fanos', 'get_noise', 'four_probe_cur_offset',\
	   'conv_two_probe']


def det2power(v_det):
    """
    Calculate power in W from array-like detector voltage
    
    Parameters
    ----------
    v_det: array-like
        detector voltage, V
        
    Returns
    -------
    power: array-like
        noise power, W
    """
    power = np.arange(14, -22, -4) - 30
    v_power = [1.0499, 1.1476, 1.2429, 1.3380, 1.4324, 1.526, 1.6184, 1.7078, 1.7933]
    return np.power(10, (np.polyval(np.polyfit(v_power, power, 4), v_det) - 30) / 10)


def spec_density(rdif, v, i, F, T, offset=0):
    """
    Calculate noise spectral density with crossover from thermal to shot noise.
    
    Parameters
    ----------
    rdif: array-like
        differential resistance, Ohm
    v: array-like
        Bias voltage, V
    i: array-like
        Current, A
    F: float
        Fano factor
    T: float
        Temperature, K
    Returns
    -------
    array-like, spectral density, A^2/Hz
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        ksi = e*np.abs(v) / (2*kB*T)
        spec = 4*kB*T/rdif + 2*e*np.abs(i)*F*(1/np.tanh(ksi) - 1/ksi)
    return pd.Series(spec).interpolate(limit_direction='both') + offset


def spec_fitting(F, rdif, v, i, T, spec_exp):
    """
    Cost function for spectral density fitting.
    Parameters
    ----------
    F: float
        Fano factor
    rdif: array-like
        Differential resistance, Ohm
    v: array-like
        Bias voltage, V
    i: array-like
        Current, A
    T: float
        Temperature, K
    spec_exp: array-like
        Experimental spectral density, A^2/Hz
    Returns
    -------
    array-like, diff between exp and fitted, 1e27*A^2/Hz
    
    """
    return 1e27*(spec_density(rdif, v, i, F, T) -\
                 pd.Series(spec_exp).interpolate(limit_direction='both'))
                 
                 
def _spec_fitting_min_offset(fitting, rdif, v, i, T, spec_exp):
	"""
	Cost function for spectral density fitting.
	Parameters
	----------
	F: float
		Fano factor
	rdif: array-like
		Differential resistance, Ohm
	v: array-like
		Bias voltage, V
	i: array-like
		Current, A
	T: float
		Temperature, K
	spec_exp: array-like
		Experimental spectral density, A^2/Hz
	Returns
	-------
	array-like, diff between exp and fitted, 1e27*A^2/Hz
	
	"""
	return np.trapz((spec_density(rdif, v, i, fitting[0], T, fitting[1]) -\
					 pd.Series(spec_exp).interpolate(limit_direction='both'))**2, i)
        
        
def _spec_fitting_min(F, rdif, v, i, T, spec_exp):
	"""
	Cost function for spectral density fitting.
	Parameters
	----------
	F: float
		Fano factor
	rdif: array-like
		Differential resistance, Ohm
	v: array-like
		Bias voltage, V
	i: array-like
		Current, A
	T: float
		Temperature, K
	spec_exp: array-like
		Experimental spectral density, A^2/Hz
	Returns
	-------
	array-like, diff between exp and fitted, 1e27*A^2/Hz
	
	"""
	return np.trapz((spec_density(rdif, v, i, F, T) -\
					 pd.Series(spec_exp).interpolate(limit_direction='both'))**2, i)



def four_probe(df, cur=1, volt=2, r_series=0, r_set=varz.r_set,\
               coef=varz.coef, preamp=varz.preamp, shift=1):
    """
    Returns correctly calculated current and voltage from data obtained 
    via 2-probe or 4-probe DC measurements
    """
    vb = df[cur] * coef
    v = df[volt] / preamp
    # Typical ~1 point delay in fast DC measurements
    v = pd.Series(v).shift(-shift)
    
    # Remove preamp offset (supposed that 0 is in vb)
    v -= v[np.abs(vb).argmin()]
    
    i = (vb - v) / r_set
    v -= i * r_series
    return i, v


def four_probe_cur_offset(df, cur=1, volt=2, r_series=0, r_set=varz.r_set,\
               coef=varz.coef, preamp=varz.preamp, shift=1):
    """
    Returns correctly calculated current and voltage from data obtained 
    via 2-probe or 4-probe DC measurements
    """
    vb = df[cur] * coef
    v = df[volt] / preamp
    # Typical ~1 point delay in fast DC measurements
    v = pd.Series(v).shift(-shift)

    i = (vb - v) / r_set
    # Remove preamp current offset (supposed that 0 is in vb)
    i -= i[np.abs(v).argmin()]
    
    v -= i * r_series
    return i, v


def conv_preamp(df, cur=3, volt=2, r_conv=varz.r_conv, preamp=varz.preamp,
                shift=1):
    """
    Returns correctly calculated current and voltage from data obtained 
    via simultaneusly measurement of voltage (preamp) and current (i-v converter)
    """
    v = df[volt] / preamp
    i = -df[cur] / r_conv
    
    # Typical ~1 point delay in fast DC measurements
    v = pd.Series(v).shift(-shift)
    i = pd.Series(i).shift(-shift)
    
    # Remove preamp offset (supposed that 0 is in vb)
    v -= v[np.abs(df[1]).argmin()]
    i -= i[np.abs(df[1]).argmin()]
    return i, v


def conv_two_probe(df, cur=2, volt=1, r_conv=varz.r_conv, div_c=0.01,
		   r_series=0, shift=1):
    """
    Returns correctly calculated current and voltage from data obtained 
    via simultaneusly measurement of voltage (preamp) and current (i-v converter)
    """
    v = df[volt] * div_c
    i = (df[cur] - v) / r_conv
    v -= i * r_series
	
    # Typical ~1 point delay in fast DC measurements
    i = pd.Series(i).shift(-shift)
    
    # Remove preamp offset (supposed that 0 is in vb)
    i -= i[np.abs(v).argmin()]
    return i, v


def plot_aux(noise, fanos, rdifs, calib, dep='v', calib_range=0.5, plot=None,\
             labels=None, legend=True, figname=None, kwargs={}):

    if not noise:
        return
    if labels is None:
        labels = np.arange(len(noise))
    labels = labels[:len(noise)]
    if plot is None:
        plot = labels
        
    fig, ax = plt.subplots(1, 3, figsize=varz.figsize13)
    ax_coef = [1e3] * 2
    ax[0].set_xlabel(get_label('V'))
    ax[0].set_ylabel(get_label('r'))
    ax[1].set_xlabel(get_label('V'))
    ax[1].set_ylabel(get_label('S_I'))
    ax[2].set_xlabel(get_label('rpar'))
    ax[2].set_ylabel(get_label('P'))
    
    if dep == 'i':
        dep = ['i'] * 2
        ax_coef = [1e9] * 2
        ax[0].set_xlabel(get_label('I'))
        ax[1].set_xlabel(get_label('I'))
    elif type(dep) == list:
        if dep[0] == 'i':
            ax_coef[0] = 1e9
            ax[0].set_xlabel(get_label('I'))
        if dep[1] == 'i':
            ax_coef[1] = 1e9
            ax[1].set_xlabel(get_label('I'))
    else:
        dep = ['v'] * 2
        
    for k in range(len(noise)):
        if (labels[k] in plot) or (plot is None):
            if type(labels[k]) == str:
                label_now = labels[k]
            else:
                label_now = '{:.2g}'.format(labels[k])
            line = ax[0].plot(noise[k][dep[0]]*ax_coef[0], noise[k]['rdif_clean'], '.',\
                              alpha=0.6, ms=3, label=label_now)
            ax[0].plot(noise[k][dep[0]]*ax_coef[0], noise[k]['rdif'],\
                       color=line[0].get_color(), linewidth=1.5, label='_nolegend_')
            
            line = ax[1].plot(noise[k][dep[1]]*ax_coef[1], noise[k]['noise'], '.',\
                              alpha=0.6, ms=3, label='{:.2g}'.format(fanos[k]))
            ax[1].plot(noise[k][dep[1]]*ax_coef[1], noise[k]['fitted'],\
                       color=line[0].get_color(), linewidth=1.5, label='_nolegend_')
            
            line = ax[2].plot(noise[k]['rpar'], noise[k]['power'], '.',\
                              alpha=0.6, ms=3, label=label_now)
            if type(calib_range) == list:
                rpar = np.linspace(calib_range[0], calib_range[1], 201)
            else:
                d = noise[k]['rpar'].max() - noise[k]['rpar'].min()
                rpar = np.linspace(noise[k]['rpar'].min() - d*calib_range,\
                                   noise[k]['rpar'].max() + d*calib_range, 201)
            ax[2].plot(rpar, calib(rpar), color=line[0].get_color(),\
                       linewidth=1.5, label='_nolegend_')
    if legend:
        [axis.legend() for axis in ax]
    ax[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0));
    if figname is not None:
        fig.savefig(figname, **kwargs)
    return fig, ax

    
def plot_noise(noise, fanos, rdifs, calib, dep='v', calib_range=0.5, plot=None,\
               labels=None, legend=True, leg_title=None, fitted=False, figname=None, kwargs={}):
    
    if labels is None:
        labels = np.arange(len(noise))
    labels = labels[:len(noise)]
    if plot is None:
        plot = labels
        
    fig, ax = plt.subplots(1, 2, figsize=varz.figsize12)
    ax_coef = [1e3] * 2
    ax[0].set_xlabel(get_label('V'))
    ax[0].set_ylabel(get_label('r'))
    ax[1].set_xlabel(get_label('V'))
    ax[1].set_ylabel(get_label('S_I'))
    
    if dep == 'i':
        dep = ['i'] * 2
        ax_coef = [1e9] * 2
        ax[0].set_xlabel(get_label('I'))
        ax[1].set_xlabel(get_label('I'))
    elif type(dep) == list:
        if dep[0] == 'i':
            ax_coef[0] = 1e9
            ax[0].set_xlabel(get_label('I'))
        if dep[1] == 'i':
            ax_coef[1] = 1e9
            ax[1].set_xlabel(get_label('I'))
    else:
        dep = ['v'] * 2
        
    for k in range(len(noise)):
        if labels[k] in plot or (plot is None):
            if type(labels[k]) == str:
                label_now = labels[k]
            else:
                label_now = '{:.2g}'.format(labels[k])
                
            noise[k].sort_values(by=dep[0], inplace=True)
            ax[0].plot(noise[k].loc[noise[k]['rdif']>0, dep[0]]*ax_coef[0],\
                       noise[k].loc[noise[k]['rdif']>0, 'rdif'], '.',  ms=3,\
                       label=label_now)
            
            noise[k].sort_values(by=dep[1], inplace=True)
            ax[1].plot(noise[k][dep[1]]*ax_coef[1], noise[k]['noise'], '.', ms=3,\
                       label='{:.2g}'.format(fanos[k]))
            
            if fitted:
                ax[1].plot(noise[k][dep[1]]*ax_coef[1], noise[k]['fitted'], '--',\
                           color='black', lw=0.8, label='_nolegend_')
    if legend:        
        leg = ax[0].legend(loc=2, **varz.lkwargs, title=leg_title)
        leg.set_frame_on(True)
        leg = ax[1].legend(loc=2, **varz.lkwargs, title=r'$F$');
        leg.set_frame_on(True)
    if figname is not None:
        fig.savefig(figname, **kwargs)
    return fig, ax
        
        
def plot_fanos(fanos, rdifs, labels=None, xlabel=None, figname=None, kwargs={}):
    
    if labels is None:
        labels = np.arange(len(fanos))
    labels = labels[:len(fanos)]
        
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(labels, rdifs, '.C0') 
    ax2.plot(labels, fanos, '.C1')
    
    rdif_max = np.round(rdifs.max(), 2-int(np.floor(np.log10(np.abs(rdifs.max()))))-1)
    fanos_max = np.round(fanos.max(), 2-int(np.floor(np.log10(np.abs(fanos.max()))))-1)
    fanos_max = max(fanos_max, 1)
    align_axes(ax1, (0, rdif_max), ax2, (0, fanos_max) , 4)
    
    ax1.set_ylabel(get_label('r'), color='C0')
    ax2.set_ylabel(r'$F$', color='C1')
    ax1.spines['left'].set_color('C0')
    ax2.spines['left'].set_color('C0')
    ax1.spines['right'].set_color('C1')
    ax2.spines['right'].set_color('C1')
    ax1.tick_params('y', colors='C0')
    ax2.tick_params('y', colors='C1')
    
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    if figname is not None:
        fig.savefig(figname, **kwargs)    
        
                 
def get_noise(indexes, calib, G, name='./data/cond', iters=20, skip_indexes=[],\
              var='v', det_ind=3, get_r=four_probe, threshold={'v':1, 'i':1e-2, 'det':3},\
              reg_grid=None, max_deviation={'v':15e-6, 'i':4e-10, 'det':2.5e-3},\
              savgolize={'v':{'window_length':3,'polyorder':1,},}, savgol_v=[], savgol_i=[],\
              savgol_r=[3, 1], rpar=varz.r0/2, skip_points=(0, 0), T=4.2, fitrange=None,\
              aux_rdif=None, rkwargs={}, verbose=True, fit_offset=False, both=False,\
	      savgol_det=[3, 1]):
    """
    savgolize only dict of dicts
    
    """
    def _spec_fitting_min_offset(fitting, rdif, v, i, T, spec_exp):
        """
        Cost function for spectral density fitting.
        Parameters
        ----------
        F: float
            Fano factor
        rdif: array-like
            Differential resistance, Ohm
        v: array-like
            Bias voltage, V
        i: array-like
            Current, A
        T: float
            Temperature, K
        spec_exp: array-like
            Experimental spectral density, A^2/Hz
        Returns
        -------
        array-like, diff between exp and fitted, 1e27*A^2/Hz
        
        """
        return np.trapz((spec_density(rdif, v, i, fitting[0], T, fitting[1]) -\
                         pd.Series(spec_exp).interpolate(limit_direction='both'))**2, i)
        
        
    def _spec_fitting_min(F, rdif, v, i, T, spec_exp):
        """
        Cost function for spectral density fitting.
        Parameters
        ----------
        F: float
            Fano factor
        rdif: array-like
            Differential resistance, Ohm
        v: array-like
            Bias voltage, V
        i: array-like
            Current, A
        T: float
            Temperature, K
        spec_exp: array-like
            Experimental spectral density, A^2/Hz
        Returns
        -------
        array-like, diff between exp and fitted, 1e27*A^2/Hz
        
        """
        return np.trapz((spec_density(rdif, v, i, F, T) -\
                         pd.Series(spec_exp).interpolate(limit_direction='both'))**2, i)
        
    dep = 'v'
    if var=='v':
        dep = 'i'
        
    noise = []
    fanos, rdifs = np.zeros(len(indexes)), np.zeros(len(indexes))
    for k,i in enumerate(indexes):    
        cur = pd.DataFrame()
        det = pd.DataFrame()
        v = pd.DataFrame()
        for j in range(iters):
            df = pd.read_csv(name+str(i+j)+'.dat', sep=' ', header=None)
            current, voltage = get_r(df, **rkwargs)
            if i+j in skip_indexes:
                continue
            if both and (j%2 == 1):
                cur = pd.concat([cur, current[::-1].reset_index(drop=True)], axis=1, ignore_index=True)
                v = pd.concat([v, voltage[::-1].reset_index(drop=True)], axis=1, ignore_index=True)
                det = pd.concat([det, df[det_ind][::-1].reset_index(drop=True)], axis=1, ignore_index=True)   
            else:
                cur = pd.concat([cur, current], axis=1, ignore_index=True)
                v = pd.concat([v, voltage], axis=1, ignore_index=True)
                det = pd.concat([det, df[det_ind]], axis=1, ignore_index=True)   
        last_df = df
        
        good_inds = (v.abs() < threshold['v']) & (cur.abs() < threshold['i']) &\
                    (det.abs() < threshold['det'])
        v = v[good_inds]
        cur = cur[good_inds]
        det = det[good_inds]
        data = {'det':det, 'v':v, 'i':cur}
        for key, df in data.items():
            data[key] = df.interpolate(method='index', axis=0, limit_direction='both')
        
        # Reindexing to regulize grid
        if reg_grid is not None:
            index = last_df[reg_grid]
            grid = np.linspace(index.iloc[0], index.iloc[-1], len(index))
            for key, df in data.items():
                df['index'] = index
                df.set_index('index', inplace=True)
                df = df.reindex(df.index.union(grid))
                df = df.interpolate(method='index', axis=0, limit_direction='both')
                data[key] = df.reindex(index=grid)
        grid = data['v'].index
        
        # Check applicability of savgol_filter
        d_grid = np.diff(grid)
        savgol_ok = False
        if (np.abs((d_grid - d_grid.mean()) / d_grid.mean()) < 1e-10).all():
            savgol_ok = True
        if not savgol_ok and (savgol_i or savgol_v or savgol_r or savgolize):
            print('Savgol_filter can not be used')
        
        # Choosing nice iterations and applying savgol_filter on each
        for key, df in data.items():
            df_mean = df.mean(axis=1)
            df = df.loc[:, ((df.T - df_mean.values).T.abs() < max_deviation[key]).all()]
            if key in savgolize.keys() and savgol_ok:
                df = df.apply(savgol_filter, axis=0, **savgolize[key])
            if verbose:
                print(key, df.shape, end='; ')
            if df.shape[1] == 0:
                print('\nToo small max_deviation value of ' + str(key))
                return None, None, None
            data[key] = df.mean(axis=1)
        if verbose:
            print()
            
        data['rdif_clean'] = data['v'].diff() / data['i'].diff()
        
        # Applying savgol_filter on mean signal of v and i
        if savgol_i and savgol_ok:
            if type(savgol_i) == dict:
                data['i'] = pd.Series(savgol_filter(data['i'], **savgol_i))
            else:
                data['i'] = pd.Series(savgol_filter(data['i'], *savgol_i))
                
        if savgol_v and savgol_ok:
            if type(savgol_v) == dict:
                data['v'] = pd.Series(savgol_filter(data['v'], **savgol_v))
            else:
                data['v'] = pd.Series(savgol_filter(data['v'], *savgol_v))
        
        if savgol_det:
            if type(savgol_det) == dict:
                data['det'] = pd.Series(savgol_filter(data['det'], **savgol_det))
            else:
                data['det'] = pd.Series(savgol_filter(data['det'], *savgol_det))
		
        # Creating DataFrame and applying savgol_filter on rdif if necessary
        noise.append(pd.DataFrame({'v':data['v'].values, 'i':data['i'].values,\
                                   'rdif_clean':data['rdif_clean'].values,\
                                   'det':data['det'].values,\
                                   'power':det2power(data['det'].values)}))
        
        noise[k]['rdif_smooth'] = noise[k]['v'].diff() / noise[k]['i'].diff()
        noise[k]['rdif'] = noise[k]['rdif_smooth']
        if savgol_r and savgol_ok:
            if type(savgol_r) == dict:
                noise[k]['rdif'] = pd.Series(savgol_filter(noise[k]['rdif'], **savgol_r))
            else:
                noise[k]['rdif'] = pd.Series(savgol_filter(noise[k]['rdif'], *savgol_r))
        noise[k]['rpar'] = noise[k]['rdif']*rpar / (noise[k]['rdif']+rpar)
        rdifs[k] = noise[k].loc[noise[k][dep].abs().idxmin(), 'rdif']
        
        # Dropping edge points
        noise[k] = noise[k].iloc[skip_points[0]:len(noise[k])-skip_points[1]]
        
        # Power in point with smallest current (!= min power)
        # must be equal to thermal noise power
        ind_min = noise[k]['i'].abs().idxmin()
        noise[k]['power'] -= float(noise[k].loc[ind_min, 'power'] -\
                                   calib(noise[k].loc[ind_min, 'rpar']))

        noise[k]['noise'] = (noise[k]['power']-calib(noise[k]['rpar'])) / G(noise[k]['rpar']) +\
                            4*kB*T/noise[k]['rdif']
        
        if fitrange is not None:
            fitrange = np.atleast_1d(fitrange)
            if len(fitrange) == 2:
                inds = (np.abs(noise[k]['v']) > fitrange[0]) &\
                       (np.abs(noise[k]['v']) < fitrange[1])
            else:
                inds = (np.abs(noise[k]['v']) < fitrange[0])
        else:
            inds = noise[k].index
        
        x0, minimizer = 0.3, _spec_fitting_min
        if fit_offset:
            x0, minimizer = [0.3, 0], _spec_fitting_min_offset
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            result = sp.optimize.minimize(minimizer, x0=x0, method='Nelder-Mead',\
                                    args=(noise[k].loc[inds, 'rdif'], noise[k].loc[inds, 'v'],\
                                          noise[k].loc[inds, 'i'], T, noise[k].loc[inds, 'noise'],))
        
#         result = sp.optimize.least_squares(spec_fitting, x0=F,\
#                            args=(noise[k].loc[inds, 'rdif'], noise[k].loc[inds, 'v'],\
#                                  noise[k].loc[inds, 'i'], T, noise[k].loc[inds, 'noise'],),\
#                            gtol=1e-15, xtol=1e-15, ftol=1e-15)

        if fit_offset:
            fanos[k] = result.x[0]
            offset = result.x[1]
        else:
            fanos[k] = result.x
            offset = 0
        noise[k]['fitted'] = spec_density(noise[k]['rdif'], noise[k]['v'],\
                                          noise[k]['i'], fanos[k], T, offset)

    return noise, fanos, rdifs

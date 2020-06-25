from matplotlib import pyplot as plt
import pandas as pd

__all__ = ['kB', 'e', 'h', 'rq', 'varz', 'get_label']

# Physical constants are defined here
kB = 1.380649e-23
e = 1.602176634e-19
h = 6.62607015e-34
rq = h / e**2

# Initiallization of global constants
_labels_df = pd.DataFrame(\
              data={'plain':[r'$V$, mV',\
                             r'$I$, nA',\
                             r'$R$, $\Omega$',\
                             r'$V_\mathrm{G}$, V',\
                             r'$V_\mathrm{tg}$, V',\
                             r'$V_\mathrm{det}$, V',\
                             r'$R_{||}$, $\Omega$',\
                             r'$P$, W',\
                             r'$G$, $\Omega\cdot$Hz',\
                             r'$S_\mathrm{I}$, A$^2\cdot$Hz$^{-1}$',\
                             r'$B$, T',\
                             ],\
                    'latex':[r'$V$,\,\si{\mV}',\
                             r'$I$,\,\si{\nA}',\
                             r'$R$,\,\si{\ohm}',\
                             r'$V_\mathrm{G}$,\,\si{\V}',\
                             r'$V_\mathrm{tg}$,\,\si{\V}',\
                             r'$V_\mathrm{det}$,\,\si{\V}',\
                             r'$R_{||}$,\,\si{\ohm}',\
                             r'$P$,\,\si{\W}',\
                             r'$G$,\,\si{\ohm\Hz}',\
                             r'$S_\mathrm{I}$,\,\si{\A\squared\per\Hz}',\
                             r'$B$,\,\si{\tesla}',\
                            ]},\
                      index=['V', 'I', 'r', 'Vg', 'Vtg', 'Vdet', 'rpar',\
                             'P', 'G', 'S_I', 'B'])


def get_label(variable):
    if plt.rcParams['text.usetex']:
        return _labels_df.loc[variable, 'latex']
    return _labels_df.loc[variable, 'plain']


class Global_varz():
    """
    Global variables container
    """
    def __init__(self, r0=20e3, A2T=0.05137, preamp=106.1, r_set=2e6, r_conv=1e7,\
                 coef=0.22/(0.22+0.68), coef2=0.22*0.68/(0.68**2+2*0.22*0.68),\
                 figsize12=(5.8, 2.7), figsize13=(7, 2.5), figsize22=(5.8, 4.5),\
                 figsize23=(5.8, 3.2)):
        """
        Initializes global variables of experimental setup and output format.
        Paramenters
        -----------
        r0: float
            1/2 of parallel resistance, Ohm
        A2T: float
            Ampere to Tesla coef
        preamp: float
            Preamplifier coef
        r_set: float
            Resintance for setting current, Ohm
        r_conv: float
            Feedback resistance of transimpedance amplifier, Ohm
        coef: float
            Bias voltage division coef
        coef2: float
            Bias voltage division coef (symmetric box, two inputs)
        figsize12: tuple of two floats
            Size in inches for figure of 2 horizontal subplots
        figure13: tuple of two floats
            Size in inches for figure of 3 horizontal subplots
        figure22: tuple of two floats
            Size in inches for figure of 4 (2x2) subplots
        figure23: tuple of two floats
            Size in inches for figure of 6 (2x3) subplots
        """
        self.r0 = r0
        self.A2T = A2T
        self.preamp =  preamp
        self.r_set = r_set
        self.r_conv = r_conv
        self.coef = coef
        self.coef2 = coef2    
        self.figsize12 = figsize12
        self.figsize13 = figsize13
        self.figsize22 = figsize22
        self.figsize23 = figsize23
        self.lkwargs = {'bbox_to_anchor':(1.02, 1), 'borderaxespad':0.1, 'borderpad':0.5,\
                        'handletextpad':0.5, 'handlelength':0.5}
        
    def initialize(self, kwargs):
        self.__init__(**kwargs)
    
varz = Global_varz()

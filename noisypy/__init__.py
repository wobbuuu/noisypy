from .settings import *
from .plot_utils import *
from .noisy_utils import *
from .calibration_utils import *

from matplotlib import pyplot as plt


# Changing matplotlib rc defaults

plt.rcdefaults()
plt.rcParams['backend'] = 'module://ipykernel.pylab.backend_inline'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preview'] = True
plt.rcParams['text.latex.preamble'] = r'''\renewcommand{\familydefault}{\sfdefault}
    \usepackage[scaled=1]{helvet}
    \usepackage[helvet]{sfmath}
    \usepackage{siunitx}
    \sisetup{detect-family=true, detect-weight=true}
    \usepackage{amsmath}'''
plt.rcParams['figure.max_open_warning'] = 100
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.formatter.limits'] = (-3, 4)

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.borderaxespad'] = 0.3
plt.rcParams['legend.borderpad'] = 0.4
plt.rcParams['legend.columnspacing'] = 0.9
plt.rcParams['legend.handlelength'] = 0.5
plt.rcParams['legend.handletextpad'] = 0.3
plt.rcParams['legend.labelspacing'] = 0.4

plt.rcParams['lines.markeredgewidth'] = 0.0

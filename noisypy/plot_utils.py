import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


__all__ = ['align_axes', '_imkwargs', 'crop_to_aspect', 'scalebar', 'panel_label',\
           'average']
_imkwargs = dict(origin='lower', cmap='gray', vmin=0, vmax=255)


def align_axes(ax1, y1, ax2, y2, nticks=7):
    """
    Creates ticks on the first and second y-axis via np.linspace(yi[0], yi[1], nticks).
    Changes limits of both axis in order align grid lines.
    
    Parameters
    ----------
    ax1: pyplot.axis
        Left side axis
    y1: array of 2 floats
        Left side y axis values to align
    ax2: pyplot.axis
        Right side axis
    y2: array of 2 floats
        Right side y axis values to align
    nticks: int
        Number of ticks on y axis    
    """
    ax1_lims = ax1.get_ybound()
    ax2_lims = ax2.get_ybound()
    ax1.set_yticks(np.linspace(y1[0], y1[1], nticks))
    ax2.set_yticks(np.linspace(y2[0], y2[1], nticks))
    dy1 = ax1.get_yticks()[1] - ax1.get_yticks()[0]
    dy2 = ax2.get_yticks()[1] - ax2.get_yticks()[0]
    
    if (y1[0] - ax1_lims[0]) / dy1 > (y2[0] - ax2_lims[0]) / dy2:
        ax1.set_ylim([ax1_lims[0], None])
        ax2.set_ylim([y2[0] - (y1[0] - ax1_lims[0]) / dy1 * dy2, None])
    else:
        ax2.set_ylim([ax2_lims[0], None])
        ax1.set_ylim([y1[0] - (y2[0] - ax2_lims[0]) / dy2 * dy1, None])
    
    if (ax1_lims[1] - y1[1]) / dy1 > (ax2_lims[1] - y2[1]) / dy2:
        ax1.set_ylim([None, ax1_lims[1]])
        ax2.set_ylim([None, y2[1] + (ax1_lims[1] - y1[1]) / dy1 * dy2])
    else:
        ax2.set_ylim([None, ax2_lims[1]])
        ax1.set_ylim([None, y1[1] + (ax2_lims[1] - y2[1]) / dy2 * dy1])
    ax2.grid(None)
    

def crop_to_aspect(file, aspect=np.array([1,1]), size=None,\
                   shift=np.array([0,0]), annotation=0):
    """
    Crop the image to fit required aspect ratio.
    
    Parameters
    ----------
    file: string
        Image path
    aspect: array of 2 integers
        (vertical dimension, horizontal dimension)
    size: int or None
        size of the output image along greater side in pixels
        None is treated to take as many pixels as possible to fit desired aspect
    shift: array of 2 integers
        skip pixels starting from the bottom left corner
        (along vert dimension, along horz dimension)
    annotation: int or None
        skip several rows of pizels from bottom of the image
        for SEM image typical value is 96
        
    Returns
    -------
    img: 2D np.array
        cropped image
    """
    fig, ax = plt.subplots(1, 2, figsize=(4,2.5), tight_layout=True)
    shift = np.array(shift)
    aspect = np.array(aspect)/aspect[1]
    img = mpl.image.imread(file)
    if annotation != 0:
        img = img[:-annotation,:]
    img = img[::-1]
    ax[0].imshow(img, extent=(0, img.shape[1], 0, img.shape[0]), **_imkwargs)
    shape = img.shape[:2]
    if size is None:
        size = (shape-shift)[1]
    if ((shift+size*aspect) >= shape).any():
        size = np.min((shape-shift)/aspect)
    to = np.array(shift+size*aspect, dtype='int64')
    img = img[shift[0]:to[0], shift[1]:to[1]]
    ax[1].imshow(img, extent=(0, img.shape[1], 0, img.shape[0]), **_imkwargs)
    return img


def scalebar(ax, img, l, label=None, pos='right', delta=0, color='w', lw=1.2,\
             fontsize=plt.rcParams['font.size']):
    """
    Print scalebar and label on the image.
    
    Parameters
    ----------
    ax: pyplot.axis
        axis to put scalebar on
    img: 2D np.array
        image drawn on the axis
    l: float
        length of the scalebar in pizel units
    label: string, optional
        label for scalebar
    pos: string or array of 2 integers, optional
        position for scalebar
        'right' -- right bottom corner
        'left' -- left bottom corner
        (x, y) -- coordinates in pixel units starting from left bottom corner
    delta: float, optional
        shift scalebar horizontally away from axis border by delta pixels
    color: float, optional
        color for the scalebar and label
    lw: float, optional
        linewidth of the scalebar in pt units
    fontsize: int, optional
        fontsize of label in pt units    
    """
    if pos == 'right': 
        h = 0.04*img.shape[0]
        w = img.shape[1]-l-h-delta
    elif pos == 'left':
        h = 0.04*img.shape[0]
        w = h+delta
    else:
        h = pos[1]
        w = pos[0]
    ax.plot(np.array([0, l])+w, [h, h], color=color, linewidth=lw)
    ax.annotate(label, [w+l/2, 1.4*h], fontsize=fontsize, color=color,\
                va='bottom', ha='center')
    

def panel_label(ax, label, pos=(0.5, 0.5), ha='left', va='bottom',\
                fontsize=plt.rcParams['font.size']+1):
    """
    Print panel label for figure.
    
    Parameters
    ----------
    ax: pyplot.axis
        axis to put label on
    label: string
        label is printed in bold
    pos: array of 2 integers, optional
        position for label on the axis
        (x, y) -- coordinates in axis fraction units starting from left bottom corner
    ha: string, optional
        Determines the point of label to align horizontally with pos
        Possible values: 'left', 'center', 'right'
        Used directly in ax.annotate()
    va: string, optional
        Determines the point of label to align vertically with pos
        Possible values: 'bottom', 'center', 'top'
        Used directly in ax.annotate()
    fontsize: int, optional
        fontsize of label in pt units    
    """
    ax.axis('off')
    ax.annotate(r'\textbf{'+label+'}', pos, fontsize=fontsize,\
                transform=ax.transAxes, ha=ha, va=va)
    
    
def average(inds, name='./data/cond', cols={1:['vg', 1], 2:['v', 1/106.1], 3:['i', -1/1e7]},\
            both=False, plot={'x':1, 'y':2}, plot_inds=[], space=0, fig=None, ax=None,\
            figname=None, return_fig=False):
    """
    ...Function to be updated...
    e.g. plot={'x':1, 'y':3}
    
    """
    if plot and not ax:
        fig, ax = plt.subplots()
    for i, ind in enumerate(inds):
        df = pd.read_csv(name+str(ind)+'.dat', sep=' ', header=None)
        if both and i%2 == 1:
            df = df[::-1].values.T
        if i == 0:
            data = pd.DataFrame(np.zeros((len(df), len(cols))),\
                                columns=[col[0] for col in cols.values()])
        for k, col in cols.items():
            data[col[0]] += df[k]*col[1]
        if plot and ind in plot_inds:
            ax.plot(df[plot['x']]*cols[plot['x']][1],\
                    df[plot['y']]*cols[plot['y']][1] + space*i, lw=1, label=ind)
    data /= len(inds)
    if plot:
        ax.plot(data[cols[plot['x']][0]], data[cols[plot['y']][0]],\
                color='black', lw=1, label='average')
        ax.legend()
        if figname and fig:
            fig.savefig(figname)
    if return_fig:
        return data, fig, ax
    return data
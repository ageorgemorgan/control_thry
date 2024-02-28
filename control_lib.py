import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# First we use a plotting helper from the discussion at 
# https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib 
def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$-\frac{%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
        
# takes as input a transfer function as a sympy expression, and produces the Bode plot. 
# The following code is partly based on Carl Sandrock's notes: https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/8_Frequency_domain/Frequency%20response%20plots.html with the visuals gussied up

# TODO: some options for switching between gain/phase and |H|, argH for the labels? 
def bode(H, savefig=False): 

    omega = np.linspace(0,100, int(1e4))
    
    s = sp.symbols('s')
    
    G = sp.lambdify(s,H)(1j*omega)
    
    fig, [ax_mag, ax_phase] = plt.subplots(2, 1)
    ax_mag.semilogx(omega, 20.*np.log10(np.abs(G)), color='xkcd:slate grey')
    ax_mag.set_xlabel(r'$\omega$', fontsize=14)
    #ax_mag.set_ylabel(r'$\left|H\left(i\omega\right)\right|$', fontsize=14)
    ax_mag.set_ylabel(r'Gain (dB)', fontsize=14)
    ax_mag.set_xlim(1e-2, 1e2)
    ax_mag.tick_params(axis='both', labelsize=12)
    ax_mag.grid('on')
    
    ax_phase.semilogx(omega, np.angle(G), color='xkcd:deep magenta')
    ax_phase.set_xlabel(r'$\omega$', fontsize=14)
    #ax_phase.set_ylabel(r'$\mathrm{arg}H\left(i\omega\right)$', fontsize=14)
    ax_phase.set_ylabel(r'Phase (rad)', fontsize=14)
    ax_phase.set_xlim(1e-2, 1e2)
    #ax_phase.set_ylim(-1.1*np.pi/2, 1.1*np.pi/2)
    ax_phase.yaxis.set_major_locator(plt.MultipleLocator(0.5*np.pi))
    ax_phase.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax_phase.tick_params(axis='both', labelsize=12)
    ax_phase.grid('on')

    plt.tight_layout()

    if savefig == True:

        fignamepng = 'bode_plot' + '.jpg'
        plt.savefig(fignamepng, dpi=600)

    else: 

        pass 

# for orienting the Nyquist plots, I use a nice helper fnc heavily based on code
# found here: https://stackoverflow.com/questions/27614245/contour-curve-with-orientation       
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.5],
    arrowstyle='-|>', arrowsize=1, color='k', transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)
    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, color=color, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

# as with Bode, Nyquist plots are similar to Sandrock's at
# https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/8_Frequency_domain/Frequency%20response%20plots.html 
# but with more bells and whistles, argument being a SymPy expression instead of an
# array, and some more customizability.
def nyquist(H, oriented = True, markpt = False, savefig=False, numpts = 2e5):

    omega = np.linspace(0,1000, int(numpts))
    
    s = sp.symbols('s')
    
    G = sp.lambdify(s,H)(1j*omega)

    Greal = G.real
    Gimag = G.imag

    fig, ax = plt.subplots(1, 1)

    lineplus = plt.plot(Greal, Gimag ,color='xkcd:tree green')

    linemin = plt.plot(Greal, -Gimag, color='xkcd:blueberry', linestyle='dashed') # negative freq part

    if oriented: 
        
        add_arrow_to_line2D(ax, lineplus, arrow_locs=[0.5],
                    arrowsize=1.5, color='xkcd:tree green')

        add_arrow_to_line2D(ax, linemin, arrow_locs=[0.5],
                    arrowsize=1.5, arrowstyle='<|-', color='xkcd:blueberry')

    else:

        pass

    if markpt:
        
        plt.scatter(-1,0, marker='+', c='xkcd:cherry')

    else: 
        
        pass
        
    plt.xlabel(r'$\Re(s)$', fontsize=14)
    plt.ylabel(r'$\Im(s)$', fontsize=14)
    plt.xticks(fontsize=12, rotation=0, color = 'k')
    plt.yticks(fontsize=12, rotation=0, color = 'k')

    xmin = min(-1.1, 1.1*np.amin(Greal))
    xmax = max(0.1, 1.1*np.amax(Greal))

    plt.xlim([xmin, xmax])

    ymin = 1.1*np.amin(Gimag)
    ymax = 1.1*np.amax(-Gimag)

    plt.ylim([ymin, ymax])
    
    #plt.axis('equal')

    plt.tight_layout()
    
    if savefig == True:

        fignamepng = 'nyquist_plot' + '.jpg'
        plt.savefig(fignamepng, dpi=600)

    else: 

        pass 

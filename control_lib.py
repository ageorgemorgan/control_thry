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
def bode(H, savefig=False): 

    omega = np.linspace(0,100, int(1e4))
    
    s = sp.symbols('s')
    
    G = sp.lambdify(s,H)(1j*omega)
    
    fig, [ax_mag, ax_phase] = plt.subplots(2, 1)
    ax_mag.loglog(omega, np.abs(G), color='xkcd:slate grey')
    ax_mag.set_xlabel(r'$\omega$', fontsize=14)
    ax_mag.set_ylabel(r'$\left|H\left(i\omega\right)\right|$', fontsize=14)
    ax_mag.set_xlim(1e-2, 1e2)
    ax_mag.tick_params(axis='both', labelsize=12)
    ax_mag.grid('on')
    
    ax_phase.semilogx(omega, np.angle(G), color='xkcd:deep magenta')
    ax_phase.set_xlabel(r'$\omega$', fontsize=14)
    ax_phase.set_ylabel(r'$\mathrm{arg}H\left(i\omega\right)$', fontsize=14)
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


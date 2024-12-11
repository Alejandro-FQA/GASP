import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

import numpy as np
import os

import parameters as pm

# Enable LaTeX labels in Matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12  # Optional: set the font size for clarity

def evo_fig_params_simple(time, mesh, den, params, fig_path='output.pdf'):
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    if num_params == 1 and pm.architecture == 'GASP':
        ha = 'left'
        va = 'top'
        tx = 0.01
        ty = 0.95

        labels = [r'$z$']  # Labels for legend

    elif num_params == 2 and pm.architecture == 'GASP':
        ha = 'left'
        va = 'bottom'
        tx = 0.01
        ty = 0.05

        labels = [r'$\langle x \rangle$', r'$\langle p \rangle$']  # Labels for legend

    else:
        ha = 'left'
        va = 'bottom'
        tx = 0.01
        ty = 0.05

        labels = [r'$\theta_{%i}$' % j for j in range(num_params)]  # Labels for legend

    # Line styles and labels for each parameter line
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']  # Colors for each parameter
    line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 2, 5, 2, 5, 10)), (0, (5, 1, 3, 1, 3, 1))]  # Custom dash patterns

    # Create the figure and GridSpec layout
    fig, axs = plt.subplots(3,1, figsize=(4, 3), sharex=True, layout='constrained')

    # Top Panel
    pcm = axs[0].pcolor(time, mesh, den, cmap='viridis', shading='auto')
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].tick_params(labelbottom=False, length=5)
    axs[0].set_xticks(np.linspace(0, t_max, 6))
    axs[0].set_ylim([lim * 3 for lim in [-1.25, 1.25]])
    axs[0].set_yticks(np.linspace(-3, 3, 3))
    axs[0].text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[0].transAxes,
                fontsize=12,
                color='white')

    # Add colorbar to the side without affecting panel width
    cbar = fig.colorbar(pcm, ax=axs[0], pad=0.01, aspect=10, orientation='vertical')
    cbar.set_label(r'$|\psi(x,t)|^2$')
    cbar.set_ticks(np.round(np.linspace(0,max(map(max, den)),4),2))

    # Middle Panel
    for i in range(real_params.shape[1]):
        axs[1].plot(time, real_params[:,i], color=colors[i%7], linestyle=line_styles[i%7], label=labels[i])
    axs[1].set_ylabel(r'${\rm Re}(\theta_j)$')
    axs[1].tick_params(labelbottom=False, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    # axs[1].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
    
    # Bottom Panel
    for i in range(imag_params.shape[1]):
        axs[2].plot(time, imag_params[:,i], color=colors[i%7], linestyle=line_styles[i%7], label=labels[i])
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'${\rm Im}(\theta_j)$')
    axs[2].tick_params(labelbottom=True, length=5)
    # axs[2].set_xticks(np.linspace(0, t_max, 6)) 
    axs[2].set_xticks(np.round(np.linspace(0,max(time),6),1))
    # axs[2].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[2].text(tx, ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[2].transAxes,
                fontsize=12
                )
    # Collect handles and labels from either the middle or bottom panel for the legend
    handles, labels = axs[1].get_legend_handles_labels()

    # Add a figure-level legend below the colorbar
    fig.legend(handles, labels,
               loc='upper left',
               bbox_to_anchor=(0.8, 0.7),
               ncol=np.ceil(num_params / 7),
               fontsize=10, 
               frameon=False
               )
    
    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='png', bbox_inches='tight', transparent=False)

def evo_fig_params(time, mesh, den, params, fig_path='output.pdf'):
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)
    num_params = params.shape[1]

    # Panel label
    ha = 'left'
    va = 'bottom'
    tx = 0.01
    ty = 0.05

    # Create the figure and GridSpec layout
    fig, axs = plt.subplots(3,1, 
                            figsize=(4, 3), 
                            height_ratios=[1,1,1],
                            sharex=True, 
                            layout='constrained')

    # Top Panel
    pcm_0 = axs[0].pcolor(time, mesh, den, 
                          cmap='viridis', 
                          shading='auto')
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].tick_params(labelbottom=False, length=5)
    axs[0].set_xticks(np.linspace(0, t_max, 6))
    axs[0].set_ylim([lim * 3 for lim in [-1.25, 1.25]])
    axs[0].set_yticks(np.linspace(-3, 3, 3))
    axs[0].text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[0].transAxes,
                fontsize=12,
                color='white')

    # Add colorbar to the side without affecting panel width
    cbar_0 = fig.colorbar(pcm_0, ax=axs[0], pad=0.01, aspect=10, orientation='vertical')
    cbar_0.set_label(r'$|\psi(x,t)|^2$')
    cbar_0.set_ticks(np.round(np.linspace(0,max(map(max, den)),4),2))

    # Create a diverging colormap centered at zero
    cmap = 'coolwarm'
    vabs = np.max([np.max(np.abs(real_params)), np.max(np.abs(imag_params))])
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Middle Panel
    pcm_1 = axs[1].pcolor(time, np.arange(num_params), real_params.T, 
                          cmap=cmap, 
                          norm=norm, 
                          shading='auto')
    axs[1].set_ylabel(r'${\rm Re}(\theta_j)$')
    axs[1].set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    axs[1].tick_params(labelbottom=False, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
    
    # Bottom Panel
    axs[2].pcolor(time, np.arange(num_params), imag_params.T, 
                  cmap=cmap, 
                  norm=norm, 
                  shading='auto')
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'${\rm Im}(\theta_j)$')
    axs[2].tick_params(labelbottom=True, length=5)
    axs[2].set_xticks(np.round(np.linspace(0,max(time),6),1))
    axs[2].set_yticks(np.round(np.linspace(0, num_params-1, 3)))
    axs[2].text(tx, ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[2].transAxes,
                fontsize=12)
    
    # Add colorbar to the side without affecting panel width
    cbar_1 = fig.colorbar(pcm_1, ax=axs.ravel()[-2:].tolist(),
                          pad=0.01,
                          aspect=20,
                          orientation='vertical')
    cbar_1.set_ticks(np.round(np.linspace(-vabs*0.85, vabs*0.85, 5), 2))
    

    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='png', bbox_inches='tight', transparent=False)

def evo_fig_compare(time, mesh, den, energy, fig_path='compare.png'):
    
    # define variables
    t_max = max(time)

    ha = 'left'
    va = 'bottom'
    tx = 0.01
    ty = 0.05

    # Create the figure and GridSpec layout
    fig, axs = plt.subplots(2,1, figsize=(4, 3), sharex=True, layout='constrained')

    # Create a diverging colormap centered at zero
    cmap = 'coolwarm'
    vmin = np.min(den)
    vmax = np.max(den)
    norm = TwoSlopeNorm(vmin = vmin, vcenter=0, vmax=vmax)

    # Top Panel
    pcm = axs[0].pcolor(time, mesh, den, cmap=cmap, norm=norm, shading='auto')
    axs[0].set_ylabel(r'$x/a_0$')
    axs[0].tick_params(labelbottom=False, length=5)
    axs[0].set_xticks(np.linspace(0, t_max, 6))
    axs[0].set_ylim([lim * 3 for lim in [-1.25, 1.25]])
    axs[0].set_yticks(np.linspace(-3, 3, 3))
    axs[0].text(tx, ty, r"${\rm (a)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[0].transAxes,
                fontsize=12)

    # Add colorbar to the side without affecting panel width
    cbar = fig.colorbar(pcm, ax=axs[0], pad=0.01, aspect=10, orientation='vertical')
    cbar.set_label(r'$|\psi|^2-|\psi_0|^2$')
    # cbar.set_ticks(np.round(np.linspace(vmin, vmax, 5), 2))
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    # Set colorbar ticks to scientific notation
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    formatter = FormatStrFormatter('%.2e')
    cbar.ax.yaxis.set_major_formatter(formatter)

    # Bottom Panel
    axs[1].plot(time, energy)
    axs[1].set_xlabel(r'$t/\tau$')
    axs[1].set_ylabel(r'$\delta E$')
    axs[1].tick_params(labelbottom=True, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
        
    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    plt.show()

    # Ensure figs directory exists
    if not os.path.exists(pm.figs_dir):
        os.makedirs(pm.figs_dir)
    # Save the figure
    fig.savefig(fig_path, format='png', bbox_inches='tight', transparent=True)


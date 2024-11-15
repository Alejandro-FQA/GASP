import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Enable LaTeX labels in Matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12  # Optional: set the font size for clarity

def evo_fig_1_params(time, mesh, den, params, fig_name='output.pdf'):
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)

    # top labels
    ha = 'left'
    va = 'top'
    tx = 0.01
    ty = 0.95

    # bottom labels
    # ha = 'left'
    # va = 'bottom'
    # tx = 0.01
    # ty = 0.05

    # Line styles and labels for each parameter line
    colors = ['tab:blue', 'tab:orange']   # Colors for each parameter
    line_styles = ['-', '--']             # Line styles for each parameter
    labels = [r'$z$']  # Labels for legend

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
    cbar.set_ticks(np.linspace(0,round(max(map(max, den)),2),4))

    # Middle Panel
    for i in range(real_params.shape[1]):
        axs[1].plot(time, real_params[:,i], color=colors[i], linestyle=line_styles[i], label=labels[i])
    axs[1].set_ylabel(r'${\rm Re}(z)$')
    axs[1].tick_params(labelbottom=False, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    axs[1].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
    
    # Bottom Panel
    for i in range(imag_params.shape[1]):
        axs[2].plot(time, imag_params[:,i], color=colors[i], linestyle=line_styles[i], label=labels[i])
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'${\rm Im}(z)$')
    axs[2].tick_params(labelbottom=True, length=5)
    axs[2].set_xticks(np.linspace(0, t_max, 6)) 
    axs[2].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[2].text(tx, ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[2].transAxes,
                fontsize=12
                )

    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    plt.show()

    # Save the figure in vector format (SVG, PDF, or EPS)
    fig.savefig(fig_name, format='pdf', bbox_inches='tight', transparent=True)

def evo_fig_2_params(time, mesh, den, params, fig_name='output.pdf'):
    
    # define variables
    t_max = max(time)
    real_params = np.real(params)
    imag_params = np.imag(params)

    # top labels
    # ha = 'left'
    # va = 'top'
    # tx = 0.01
    # ty = 0.95

    # bottom labels
    ha = 'left'
    va = 'bottom'
    tx = 0.01
    ty = 0.05

    # Line styles and labels for each parameter line
    colors = ['tab:blue', 'tab:orange']   # Colors for each parameter
    line_styles = ['-', '--']             # Line styles for each parameter
    labels = [r'$\langle x \rangle$', r'$\langle p \rangle$']  # Labels for legend

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
    cbar.set_ticks(np.linspace(0,round(max(map(max, den)),2),4))

    # Middle Panel
    for i in range(real_params.shape[1]):
        axs[1].plot(time, real_params[:,i], color=colors[i], linestyle=line_styles[i], label=labels[i])
    axs[1].set_ylabel(r'${\rm Re}(\theta_j)$')
    axs[1].tick_params(labelbottom=False, length=5)
    axs[1].set_xticks(np.linspace(0, t_max, 6))
    axs[1].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[1].text(tx,ty, r"${\rm (b)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[1].transAxes,
                fontsize=12)
    # axs[1].legend(loc='upper right', fontsize=10)  # Add legend for real part
    
    # Bottom Panel
    for i in range(imag_params.shape[1]):
        axs[2].plot(time, imag_params[:,i], color=colors[i], linestyle=line_styles[i], label=labels[i])
    axs[2].set_xlabel(r'$t/\tau$')
    axs[2].set_ylabel(r'${\rm Im}(\theta_j)$')
    axs[2].tick_params(labelbottom=True, length=5)
    axs[2].set_xticks(np.linspace(0, t_max, 6)) 
    axs[2].set_ylim([lim * 1 for lim in [-1.25, 1.25]])
    axs[2].text(tx, ty, r"${\rm (c)}$",
                horizontalalignment=ha,
                verticalalignment=va,
                transform=axs[2].transAxes,
                fontsize=12
                )
    # axs[2].legend(fontsize=10, bbox_to_anchor=(1.05, 1.0))  # Add legend for imaginary part

    # Collect handles and labels from either the middle or bottom panel for the legend
    handles, labels = axs[1].get_legend_handles_labels()

    # Add a figure-level legend below the colorbar
    fig.legend(handles, labels,
               loc='upper left',
               bbox_to_anchor=(0.8, 0.525),
               ncol=1,
               fontsize=10, 
               )
    
    # Adjust vertical spacing
    fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    plt.show()

    # Save the figure in vector format (SVG, PDF, or EPS)
    fig.savefig(fig_name, format='pdf', bbox_inches='tight', transparent=True)

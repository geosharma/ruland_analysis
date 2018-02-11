# ! /usr/bin/python
# -*- coding: utf-8 -*-

# import numerical module numpy
import numpy as np

# import plotting module matplotlib
import matplotlib.pyplot as plt

# import operation system module os
import os

# import regex module
import re

# import cycler
from cycler import cycler

# import curve fitting model
from lmfit.models import LorentzianModel, ConstantModel, LinearModel

# Description: Read the polor image file from SAXSGUI and determine the
# scattering domain size and orientation angle based on Ruland streak method.
# Save the polar image in SAXSGUI as triplets .csv text file.
# The fits Lorentzian profile to the peaks.
# The folder structure: Parent folder: samplename folder /,
#                       Children folders: data, prog, plots
# the data file should be in the data folder, this script file is in the prog
# folder and the plot will be saved to the plots folder. These can be changed
# in the script below if so desired.
# Date: 01/28/2017

# Font settings
plt.rc(
       'font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'],
                  # 'monospace': ['Computer Modern Typewriter']
                  }
        )

# set some parameters for this plot
params = {'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'mathtext.default': 'regular',
          'text.latex.preamble': [r'\usepackage{siunitx}',
                                  r'\usepackage{amsmath}'],
          'text.usetex': True,  # use Tex to render all fonts
          # 'figure.figsize': fig_size,
          # set default color cycle
          'axes.prop_cycle': cycler(
                             'color', ['#332288', '#88CCEE', '#44AA99',
                                       '#117733', '#999933', '#DDCC77',
                                       '#CC6677', '#882255', '#AA4499']),
          'axes.unicode_minus': True}

# update parameters
plt.rcParams.update(params)

# Close all open figures
plt.close('all')

# custom colors from colorbrewer2.org
colors01 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf']

colors02 = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
            '#ffd92f', '#e5c494', '#b3b3b3']

colors03 = ['#ffd92f', '#b3b3b3']


# lorentzian fit function
def lorentzfit(x, c, p1, p2, p3):
    "Fits a multi-parameter Lorentzian function to data"
    return p1/((x - p2)**2 + p3) + c


def lorentzian(x, amp, cen, wid):
    "1-d lorentzian: lorentzian(x, amp, cen, wid)"
    return amp/np.pi * (wid/((x - cen)**2 + wid**2))


def line(x, slope, intercept):
    "line"
    return slope * x + intercept


# path for the input polarimage file
filepath = "./data/"

# path for the plots folder
plotpath = './plots/'

# name of the polar image input file
infilename = 'saxs_plrimg.txt'

# name of the output file
outfilename = "saxs"

# the polar image file saved from SAXSGUI generally starts from
# q = 0.00107, 0.00213, 0.00320, 0.00427.... 0.21113, 0.21220, 0.21326
# Dq is approximately 0.00107

# the step size of the azimuthal angle from the SAXSGUI is 2 degrees
# starting from 0 degrees
azi_stepsize = 2.0
n = int(360.0/azi_stepsize + 1.0)

# the SAXSGUI extracts azimuthal scan values from 0 - 360 degs at 2 degs
# interval. Choose the range of phi interval
# the range of the y-axis azimuthal angle is 0 to 360, however the 2D
# scattering pattern is symmetic therefore only part of the pattern can be
# used for further analysis
# the limits of the azimuthal angle for analysis
start_phi = 90
end_phi = 270

#  there are around 200 q slices in the polar image
# only a few initial q values are used in the Ruland streak analysis
# generally the q value ranges from 0.01 to 0.04 A^(-1)
# initially start with 20 to 40 slices and adjust the values
start_qslice = 0.065
end_qslice = 0.13

# complete paths with filename for the simulated and measured data
infile = filepath + infilename

# output filename without the extension
outfigname = outfilename + '_loren'
outfilename = outfilename + '_loren_summary.txt'
print('Summary file: ', outfilename)

# create plots folders if it does not exist
if not os.path.exists(plotpath):
    os.mkdir(plotpath)

# write the output to a text file
# table headers
fout = open(plotpath + outfilename, 'w')
print('{0:9s} {1:30s}'.format('Sample: ', infilename), file=fout)
print('{0:>8s}, {1:>8s}, {2:>7s}, {3:>7s}, {4:>7s}, {5:>7s},\
 {6:>12s}, {7:>10s}, {8:>15s}'.format(
 'q vector', 'Phi peak', 'Center', 'FWHM', 'Area',
 'Height', 'Int. breadth', 'Chi-square', 'Red. Chi-square'),
  file=fout)

# read the output file from SAXSGUI
# The output file is a polar image saved as a x,y,z triplets text file
# in comma separated variables format
# the input file is read into polarimage, a multidimensional array with
# three columns.
# plrimg[0]: is the scattering vector q, the x-axis of the polar image
# plrimg[1]: is the azimuthal angle, the y-axis of the polar image
# plrimg[2]: is the intensity values

plrimg = np.genfromtxt(infile, delimiter=',', skip_header=0)

# shape of the polar image
plrimgshp = np.shape(plrimg)

# the number of qslices
num_qslc = int(plrimgshp[0]/n)
print('Total number of q slices: ', num_qslc)

# separate each scan for a given q value
plrimg = np.array([plrimg[i:i + n] for i in range(0, plrimgshp[0], n)])

# these are the q values of the scan
qslc = plrimg[:, 0][:, 0]

# these are the phi values
phi = plrimg[0, :][:, 1]

# index for the starting q and phi value
idx_qstart = (np.abs(qslc - start_qslice)).argmin()
idx_phistart = (np.abs(phi - start_phi)).argmin()

# index for the the ending q value
idx_qend = (np.abs(qslc - end_qslice)).argmin()
idx_phiend = (np.abs(phi - end_phi)).argmin()

print('Starting q value: ', qslc[idx_qstart])
print('Ending q value: ', qslc[idx_qend - 1])
print('Starting phi value: ', phi[idx_phistart])
print('Ending phi value: ', phi[idx_phiend])

# total number of slices that will be written to separate files for
# further analysis
qnum = idx_qend - idx_qstart
print('Number of q slices analyzed: ', qnum)

# plot six patterns as subplot in the figure
# if the qslices chosen are less than six then plot each and every
# qslice, else divide the number of slices by six and pick few
# intermediate slices
if num_qslc < 6:
    fstep = 1
else:
    fstep = np.floor((qnum - 2)/6) + 1

figidx = [0, fstep, (2 * fstep) + 1, (3 * fstep) + 2,
          fstep * 4, qnum-1]

figscale = 1.5
# Figure 1: Plot the scan and the baseline
fig1, axs = plt.subplots(2, 3, figsize=(figscale*8, figscale*6))
axs = axs.ravel()

# axs.set_xlabel(r"$\phi$ ($^o$)")
# axs.set_ylabel(r"Intensity, $I$")

figscale = 0.65
# Figure 2: Plot s vs sB
fig2 = plt.figure(figsize=(figscale*8.0, figscale*6.0))
ax1 = fig2.add_subplot(111)
ax1.set_xlabel(r"$s$ $\left(\AA^{-1}\right)$")
ax1.set_ylabel(r"$s\: B_{obs}$ $\left(\AA^{-1}\right)$")

# array to store s and sB for later plotting
pltvar = np.zeros((qnum, 2))
print(np.shape(pltvar))
# params.add('height', value=1.0, expr='0.3183099*amplitude/max(1.e-15, sigma)')
# create the model class
peak = LorentzianModel()
peak.set_param_hint('height', expr='0.3183099*amplitude/sigma')
background = ConstantModel()
# background = LinearModel()


# write the phi angle and intensity pair values for each q value in
# a separate file for further analysis
for i in range(idx_qstart, idx_qend):
    # create a dummpy loop index
    lidx = i - idx_qstart
    # print(lidx)

    # the q value or q label for the data currently analyzed
    q = qslc[i]

    # get the data for the choosen q slice
    slc = plrimg[i]

    # print('slc: ', slc)

    # use only the portion of the phi range of interest
    slc = slc[idx_phistart:idx_phiend]

    # remove rows with intensity = nan
    slc = slc[~np.isnan(slc).any(1)]

    # remove rows with inf or -inf
    slc = slc[~np.isinf(slc).any(1)]

    # extract the azimuthal angles and intensities
    phi = slc[:, 1]
    intI = slc[:, 2]

    # using argpartition to get the indices of the three smallest values
    # of intensity for linear baseline determination
    # Requires NumPy version >= 1.8
    # min_idx = np.argpartition(intI, 3)[:5]

    # determine linear baseline and subtract the baseline
    # p = np.polyfit(phi[min_idx], intI[min_idx], 1)
    xbg = np.concatenate((phi[:3], phi[-3:]), axis=0)
    ybg = np.concatenate((intI[:3], intI[-3:]), axis=0)
    # generate a linear baseline
    p = np.polyfit(xbg, ybg, 1)
    # p = np.polyfit([phi[0], phi[-1]], [intI[0], intI[-1]], 1)
    baseline = np.polyval(p, phi)
    intI = intI
    print('max = ', max(intI))
    # find the maximum intensity after correcting for baseline
    intI = intI.clip(min=0)
    maxI = np.amax(intI)

    # find the index of the maximum intensity
    maxI_index = np.argmax(intI)

    # find the angle for maximum intensity
    angle_maxI = phi[maxI_index]

    # Normalized intensity
    normI = intI/maxI

    # define intial values
    # p3 = ((np.amax(phi) - np.amin(phi))/10)**2
    # p2 = (np.amax(phi) + np.amin(phi))/2.0
    # p1 = np.amax(normI) * p3
    # c = np.amin(normI)
    # compute initial parameters and then create a model
    pars = background.make_params(c=intI.min())
    # pars = linmod.make_params(intercept=intI.min(), slope=0)
    pars += peak.guess(intI, x=phi)
    # mod = lmod + linmod
    model = background + peak

    # result = mod.fit(normI, pars, x=phi)
    # curve fit Lorentzian to the azimuthal scan
    # popt, pconv = curve_fit(lorentzfit, phi, normI, p0=[c, p1, p2, p3])
    # print(popt)
    # yfit = lorentzfit(phi, popt[0], popt[1], popt[2], popt[3])
    result = model.fit(intI, pars, x=phi)
    # result.params.add('height', expr='0.3183099*amplitude/max(1.e-15, sigma)')
    print(result.fit_report(min_correl=0.5))
    # compute the integral breadth (Bobs)
    intb = np.radians(result.params['fwhm'].value) * np.pi/2

    # convert from q to s
    s = q/(2 * np.pi)

    # compute sBobs
    sintb = s * intb

    # save s and sB for plotting later
    pltvar[lidx, 0] = s
    pltvar[lidx, 1] = sintb

    print('{0:8.5f}, {1:8.2f}, {2:7.2f}, {3:7.3f}, \
{4:7.3f}, {5:7.3f}, {6:12.5f}, {7:10.5f}, {8:15.5f}'.format(
          q, angle_maxI, result.params['center'].value,
          result.params['fwhm'].value, result.params['amplitude'].value,
          0.3183099 * result.params['amplitude']/max(1.e-15, result.params['sigma']),
          intb, result.chisqr, result.redchi), file=fout)
    # ax1.plot(phi, normI, lw=1.0, label='Scan')
    # ax1.plot(phi, yfit, lw=1.0, label='baseline')

    if lidx in figidx:
        axidx = figidx.index(lidx)
        # print('Axis index: ', axidx)
        axs[axidx].plot(phi, intI, 'bo', mec='k', mfc=colors02[0])
        axs[axidx].plot(phi, result.best_fit, lw=1.5, color=colors02[1])
        axs[axidx].set_ylim(0, )
        axs[axidx].set_xlim(start_phi, end_phi)
        axs[axidx].set_xlabel(r"$\phi$ ($^o$)", fontsize=16)
        axs[axidx].set_ylabel(r"$I/I_{max}$", fontsize=16)
        axs[axidx].set_title(r"q = " + str(q))
        txtfwhm = 'fwhm = ' + str(round(result.params['fwhm'].value, 2))
        # axs[axidx].text(100, 0.8, txtfwhm)

# extract s and s*B
s = pltvar[:, 0]
sintb = pltvar[:, 1]

ax1.plot(s, sintb, ls='', marker='s', mec='k', mfc=colors02[0],
         label='Data')
# fit a linear line to the data
p = np.polyfit(s, sintb, 1)

# extract the slope and intercept from the fit and create the best
#  fit line
ax1.plot
ax1.set_xlim(0,)
ax1.set_ylim(0,)
axxlim = ax1.get_xlim()
axylim = ax1.get_ylim()
ax1.set_xlabel(r"$s$ $\left(\AA^{-1}\right)$")
ax1.set_ylabel(r"$s\: B_{obs}$ $\left(\AA^{-1}\right)$")

# for the seconday x-axis showing corresponding values of q
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
# print(ax1.get_xticks())
ax2labels = ax1.get_xticks() * 2 * np.pi
ax2labels = np.round(ax2labels, 3)
# print(ax2labels)
# ax2.set_xticks(ax2tickpos)
ax2.set_xticklabels(ax2labels)
ax2.set_xlabel(r"$q$ $\left(\AA^{-1}\right)$")

# extract the slope and intercept from the fit and create the best
#  fit line
slope = p[0]
print('Slope: ', slope)
intercept = p[1]
print('Intercept: ', intercept)
xs = np.linspace(0, axxlim[1], 10, endpoint=True)
ys = slope * xs + intercept

# compute the value for Rudland's streak analysis
# average longitudinal extension and misorientation

# plot this best fit line on the figure too
ax1.plot(xs, ys, ls='-', color=colors02[1])

xdist = axxlim[1] - axxlim[0]
ydist = axylim[1] - axylim[0]
s_l = r"$\langle L \rangle = $ " + str(round((1.0/intercept), 2))\
    + r" $\AA$"
s_bg = r"$B_g = $ " + str(round(np.degrees(slope), 2)) + r"$^o$"
ax1.text(axxlim[0] + 0.1 * xdist, axylim[0] + 0.80 * ydist, s_l)
ax1.text(axxlim[0] + 0.1 * xdist, axylim[0] + 0.73 * ydist, s_bg)
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, frameon=False)

# tight layout
fig1.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)
fig2.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)


# save the file
fig1.savefig(plotpath + outfigname + 'profiles', ext='png', dpi=300)
fig2.savefig(plotpath + outfigname, ext="png", dpi=300)
plt.show()

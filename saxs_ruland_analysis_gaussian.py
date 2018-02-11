# ! /usr/bin/python
# -*- coding: utf-8 -*-

# import numerical module numpy
import numpy as np

# import plotting module matplotlib
import matplotlib.pyplot as plt

# import operation system module os
import os

# import curve fitting model
from lmfit.models import GaussianModel, LinearModel

# Description: Read the polor image file from SAXSGUI and determine the
# scattering domain size and orientation angle based on Ruland streak method.
# Save the polar image in SAXSGUI as triplets .csv text file.
# Fits Gaussian profile to the peaks with linear background.
# The folder structure: Parent folder: samplename folder /,
#                       Children folders: data, prog, plots
# the data file should be in the data folder, this script file should be the
# prog folder and the plot will be saved to the plots folder.
# These can be changed in the script below if so desired.
# LATEX installation is required for plotting.
# Date: 01/28/2017

# path for the input polarimage file
filepath = "./data/"

# path for the plots folder
plotpath = './plots/'

# name of the polar image input file
infilename = 'saxs_plrimg.txt'

# for SAXS the polar image file saved from SAXSGUI generally starts from
# q = 0.00153, 0.00306, 0.00459, 0.00612.... 0.30289, 0.30442, 0.30595
# Dq is approximately 0.00153
# these will change for different q range

# the step size of the azimuthal angle from the SAXSGUI is 2 degrees
# starting from 0 degrees
azi_stepsize = 2.0
n = int(360.0/azi_stepsize + 1.0)

# the SAXSGUI extracts azimuthal scan values from 0 - 360 degs at 2 degs
# interval. Choose the range of azimuthal angle (phi) for analysis
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

# output filenames
outfilenamenoext = os.path.splitext(infilename)
outfigname = outfilenamenoext[0] + '_gauss'
outfilename = outfilenamenoext[0] + '_gauss_summary.txt'
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

# create the model class
peak = GaussianModel()
background = LinearModel()
model = background + peak
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

    # find the maximum intensity after correcting for baseline
    intI = intI.clip(min=0)
    maxI = np.amax(intI)

    # find the index of the maximum intensity
    maxI_index = np.argmax(intI)

    # find the angle for maximum intensity
    angle_maxI = phi[maxI_index]

    # compute initial parameters and then create a model
    # pars = background.make_params(c=intI.min())
    pars = background.make_params(intercept=intI.min(), slope=0)
    pars += peak.guess(intI, x=phi)
    result = model.fit(intI, pars, x=phi)
    calbkg = np.polyval([result.params['slope'],
                         result.params['intercept']], phi)

    # print(result.fit_report(min_correl=0.5))
    # compute the integral breadth (Bobs)
    intb = np.radians(np.sqrt(np.pi/np.log(2)) * result.params['fwhm'].value/2)

    # convert from q to s
    s = q/(2 * np.pi)

    # compute x and y axis values for Gaussian fit
    s2 = s**2
    s2intb2 = s2 * intb**2

    # save s and sB for plotting later
    pltvar[lidx, 0] = s2
    pltvar[lidx, 1] = s2intb2

    print('{0:8.5f}, {1:8.2f}, {2:7.2f}, {3:7.3f}, \
{4:7.3f}, {5:7.3f}, {6:12.5f}, {7:10.5f}, {8:15.5f}'.format(
          q, angle_maxI, result.params['center'].value,
          result.params['fwhm'].value, result.params['amplitude'].value,
          result.params['height'].value,
          intb, result.chisqr, result.redchi), file=fout)
    # ax1.plot(phi, normI, lw=1.0, label='Scan')
    # ax1.plot(phi, yfit, lw=1.0, label='baseline')

    if lidx in figidx:
        axidx = figidx.index(lidx)
        # print('Axis index: ', axidx)
        axs[axidx].plot(phi, intI, 'o')
        axs[axidx].plot(phi, result.best_fit, lw=1.5)
        axs[axidx].plot(phi, calbkg)
        # axs[axidx].set_ylim(-500, )
        axs[axidx].set_xlim(start_phi, end_phi)
        axs[axidx].set_xlabel(r"$\phi$ ($^o$)", fontsize=16)
        axs[axidx].set_ylabel(r"$I/I_{max}$", fontsize=16)
        axs[axidx].set_title(r"q = " + str(q))
        txtfwhm = 'fwhm = ' + str(round(result.params['fwhm'].value, 2))
        aylim = axs[axidx].get_ylim()
        axlim = axs[axidx].get_xlim()
        xtextloc = axlim[0] + 0.075 * (axlim[1] - axlim[0])
        ytextloc = aylim[0] + 0.80 * (aylim[1] - aylim[0])
        axs[axidx].text(xtextloc, ytextloc, txtfwhm)

# extract s and s*B
s = pltvar[:, 0]
sintb = pltvar[:, 1]

ax1.plot(s, sintb, ls='', marker='s', label='Data')
# fit a linear line to the data
p = np.polyfit(s, sintb, 1)

# extract the slope and intercept from the fit and create the best
#  fit line
ax1.plot
ax1.set_xlim(0,)
ax1.set_ylim(0,)
axxlim = ax1.get_xlim()
axylim = ax1.get_ylim()
ax1.set_xlabel(r"$s^2$ $\left(\AA^{-2}\right)$")
ax1.set_ylabel(r"$s^2\: B_{obs}^2$ $\left(\AA^{-2}\right)$")

# for the seconday x-axis showing corresponding values of q
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
# print(ax1.get_xticks())
ax2labels = np.sqrt(ax1.get_xticks()) * 2 * np.pi
ax2labels = np.round(ax2labels, 2)
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

# plot this best fit line on the figure too
ax1.plot(xs, ys, ls='-')

# compute the value for Rudland's streak analysis
# average longitudinal extension and misorientation
s_l = r"$\langle L \rangle = $ " + str(round((1.0/np.sqrt(intercept)), 2))\
    + r" $\AA$"
s_bg = r"$B_g = $ " + str(round(np.degrees(np.sqrt(slope)), 2)) + r"$^o$"

xdist = axxlim[1] - axxlim[0]
ydist = axylim[1] - axylim[0]
ax1.text(axxlim[0] + 0.1 * xdist, axylim[0] + 0.80 * ydist, s_l)
ax1.text(axxlim[0] + 0.1 * xdist, axylim[0] + 0.73 * ydist, s_bg)
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# move the exponent of the y-axis ticklabels towards the left of the
# of the y-axis
expoffset = ax1.yaxis.get_offset_text()
expoffset.set_x(-0.10)
# ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, frameon=False)

# tight layout
fig1.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)
fig2.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

# save the file
fig1.savefig(plotpath + outfigname + 'profiles', ext='png', dpi=300)
fig2.savefig(plotpath + outfigname, ext="png", dpi=300)
# plt.show()

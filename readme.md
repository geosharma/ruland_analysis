# Ruland's streak method
The python script reads the SAXSGUI polar image and uses Ruland's streak method to determine the length and and the orientation of the scattering domain (pores). The integral breadth is determined by fitting Gaussian to the azimuthal profile. The users chooses the azimuthal angle region and the q-range for the analysis.
## Directory structure
- project_folder
 * data
 * prog
 * plots

The *project_folder* is the parent folder and the rest are children folders within the project folder. The name of this folder should reflect the sample or as one chooses. The script file resides in the *prog* folder and the data files should be in the *data* folder. The *plot* folder will be created if the folder does not already exist.
## Input data
The input data is the polar transformation of 2D SAXS image. The polar image
should be saved as (x, y,z) triplets in SAXSGUI.

## User inputs
+ *infilename*: name of the polar image
+ *start_phi*: start azimuthal angle for analysis
+ *end_phi*: end azimuthal angle for analysis
+ *start_qslice*: start q value for analysis
+ *end_qslice*: end q value for analysis 

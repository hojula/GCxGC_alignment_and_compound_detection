Tool for GCXGC chromatography

Python: 3.11

Necessary libraries:
* torch
* numpy
* tqdm
* netCDF4
* cv2
* matplotlib
* argparse
* OmegaConf
  

Arguments are as follows:
* --config - name of the file with program configuration
* --input_cdf - path to cdf that should be processed
* --clear - 'yes' to clear tmp folder

config.yaml options:
* output_directory - directory where should be the output stored
* t1_shift - offset of machine
* system_number - '1/2/3'
* box_t1 - length in each direction t1 to be added to the expected position to find calibration compound
* box_t2 - length in each direction t2 to be added to the expected position to find calibration compound
* dot_product_threshold - the minimum value of the dot product at which the substance is declared to conform
* m_z_index - plot only specific m/z index if -1 then the sum is plotted
* calibration compounds - compounds used to find the shift

compounds_system1/2/3.yaml:
* contains compounds and their expected position as well as small box where after shift the compound is being searched

avg_spectrum:
* folder with spectrum saved in torch used for dot products

tmp:
* folder automatically generated to save loaded files for the next experiment with the same file (makes it faster)
* To clean call --clear

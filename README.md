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
* faiss
  

Arguments are as follows:
* --help - prints basic help for user
* --config - name of the file with program configuration
* --input_cdf - path to cdf that should be processed
* --clear - call to clear tmp folder [yes/no]
* --shift_method - determines method for alignment [avg/triangles] (default triangles)
* --debug_calibration - shows calibrations compounds and triangles if used [yes/no]
* --area - computes area for compounds [yes/no] (still work in progress)
* --spectrum - saves spectrum with times and plots the spectrum [yes/no]

config.yaml options:
* output_directory - directory where should be the output stored
* output_file_name - name of the output ['original']
* t1_shift - offset of machine [s]
* system_number - '1/2/3'
* box_t1 - length in each direction t1 to be added to the expected position to find calibration compound [px]
* box_t2 - length in each direction t2 to be added to the expected position to find calibration compound [px]
* m_z_importance_untill_100 - number of points in spectrum graph till m/z 100
* m_z_importance_over_100 - number of points in spectrum graph over m/z 100
* m_z_start - start of the spectrum graph
* m_z_end - end of the spectrum graph
* dot_product_threshold - the minimum value of the dot product at which the substance is declared to conform (normalized dot product used -> [0,1])
* m_z_index - plot only specific m/z index if -1 then the sum is plotted
* font_size_chromatogram
* font_thickness_chromatogram
* calibration compounds - compounds used to find the shift

compounds_system1/2/3.yaml:
* contains compounds and their expected position as well as small box where after shift the compound is being searched

compounds_numbers.yaml:
* contains compounds and their numbers in the outputted image

avg_spectrum:
* folder with spectrum saved in torch used for dot products

tmp:
* folder automatically generated to save loaded files for the next experiment with the same file (makes it faster)
* To clean call --clear

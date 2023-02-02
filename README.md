# Hydro-LSTM
Towards Interpretable LSTM Modeling of Hydrological Systems 

This repository is splited in 4 different sections.
  - Data
  - Codes
  - Results
  - Notebooks

### Data
This folder contains the three sources of data used in this paper (USGS, CAMELS attributes, CAMELS time series). That information is completely available in this repository, so you do not need to request or download information from another source. 

By using the CAMELS attributes in your publication(s), you agree to cite:

> *Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences, doi:10.5194/hess-2017-169, 2017.*

By using the CAMELS time series in your publication(s), you agree to cite:

> *Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T. and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance, Hydrology and Earth System Sciences, 19, 209–223, doi:10.5194/hess-19-209-2015, 2015.*

### Codes
This forlder has the 5 python code used in the training of the models.
  - main.py: This code is calling all the other files. This script has some parameterization such as gauge ID, #cells, #memory (days), learning rate, #epochs, and the model used (LSTM or HYDRO)
  - LSTM.py and Hydro_LSTM.py: They create a class with the specific structure. Its equation can be found in this script.
  - importing.py: This script create the dataset with the data from the thre sources.
  - utils.py: This script has some specific functions used to create the datset and train the model.
  
  ### Results
This folder has the summary results for each structure used. Hydro and LSTM refers to the experiment with 10 catchments. Hydro_CONUS refers to the experiment with 587 catchment using Hydro_LSTM. The folder has the actual model saved in a pkl file in the cases where the weight distribution is done.
  
  ### Notebooks
This folder has the files used to create each of the figures presented in the paper. All the figures are contained in the jupiter notebook Figures.ipynb.


  

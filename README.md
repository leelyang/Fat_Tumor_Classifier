# Fat_Tumor_Classifier
Using SVM and CNN to build a classifier which can determine the margin of Tumor based on Raman Spectroscopy data

### 1. Data
The data come from Raman spectroscopy which have already been preprocessed such as baseline correction, scatter correction and noise removal. Here, I used fluorescence corrected data. If you are interested in the data, please email me. 

Totally, I only have 209 samples. Only using 209 samples cannot build a robust classifier, so I need to do data augmentation to increase the number of samples and make model more strong.

### 2. Saving paras
In order to make our paras consistancy, I saved paras for each model. When you test the model, you can load the .json file make the preprocessing step same.

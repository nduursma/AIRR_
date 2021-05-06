# MeanBlue, SPARET and SPARCK AIRR Drone Race Gate Detection Methods

This repository contains the code for three methods to detect gates in AIRR drone races: MeanBlue, SPARET and SPARCK. MeanBlue is a filter-based method. SPARET runs the Extra Trees classifier on image patches. SPARCK runs a convolutional neural network on image patches.

## Installation

Clone this repository.
```bash
git clone https://github.com/nduursma/AIRR_gate_detection.git
```
Change your working directory.
```bash
cd AIRR_gate_detection
```

Place your images with the AIRR gates in the folder *images*. Note: do NOT place them in a subfolder and make sure that the image names are in the format *img\*.png* 

If you have corner annotations available, name this file "*corners.xlsx*" and place it also in the folder *images*. This file shoud contain the following column values: \
[img_name, x_top_left, y_top_left, x_top_right, y_top_right, x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left]

### AIRR Environment

Create a new environment and activate it.
```bash
conda env create -f AIRR.yml
conda activate AIRR
```
This might only work on Windows. Else, make sure to have the following modules available:

\- jupyter notebook\
\- numpy\
\- matplotlib\
\- pandas\
\- skimage\
\- cv2 (opencv)\
\- sklearn\
\- os\
\- glob\
\- torch (pytorch)\
\- torchsummary\
\- torchvision\
\- pickle \
\- xlsxwriter


## MeanBlue

MeanBlue is a filter based method, filtering out the vertical blue parts of an image to detect the gates. It works the best if there are two blue edges from an obstacle are visible and if there are no vertical blue bars visible in the environment. It can be used to detect the nearest gate only.

![MeanBlue Processing](https://github.com/nduursma/AIRR_gate_detection/blob/master/MeanBlue/fig.PNG)

### Corner Prediction
To run the MeanBlue algorithm, use: 
```bash
python MeanBlue
```

This will store the images with the detected corners, as well as an Excel file corners.xlsx with the predicted corner locations in a new folder *predictions_MeanBlue*. 


### Performance Evaluation
To evaluate the performance of the MeanBlue algorithm, open Jupyter Notebook.

```bash
jupyter notebook
```
Navigate in the folder *MeanBlue* to the file *MeanBlue.ipynb* and run all the cells. Scroll down to see the visualizations of the algorithm methodology, the last cell displays the accuracy score. 

The images with the predicted corners are stored in the folder *MeanBlue/perf_evaluation*. Images with correctly predicted gates contain the label *'T'*, images with wrong predicted gates contain the label *'F'*.

## SPARET
SPARET: Sample Patches And Run Extra Trees, works exactly as the name suggest. It samples 2500 grid points on an image and creates a 16x16 patch around them. Then an extra trees classifier classifies each point to be a gate corner or not. It works best if there are not many areas with neighbouring black and white squares in the image, except for the gate corners. It can be used to detect multiple gates within a certain range.

![SPARET Results](https://github.com/nduursma/AIRR_gate_detection/blob/master/SPARET/fig.jpg)

### Gate Location Prediction
To run the SPARET algorithm, use: 
```bash
python SPARET
```

This will store the images with the detected corners, as well as an Excel file gate_points.xlsx with the predicted gate point locations (x and y) in a new folder *predictions_SPARET*. 


### Performance Evaluation
To retrain or to evaluate the performance of the SPARET algorithm, open Jupyter Notebook.

```bash
jupyter notebook
```
Navigate in the folder *SPARET* to the file *SPARET.ipynb* and run all the cells. Scroll down to see the visualizations of the algorithm training, the last cell displays the accuracy score. 

The images with the predicted gate locations are stored in the folder *SPARET/perf_evaluation*. Images with correctly predicted gates, where six or more gate points lie on a real corner, are labelled with a *'T'*. If this is not the case, the image is labelled with *'F'*.


## SPARCK
SPARCK: Sample Patches And Run Convolution Kernels, samples 2500 grid points on an image and creates a 16x16 patch around them. Then a 3-layer convolutional neural network with kernel size 5x5, ReLU activation and 2x2 max pooling classifies each point to be a gate corner or not. It works best if there are not many areas with neighbouring black and white squares in the image, except for the gate corners. It can be used to detect multiple gates within a certain range.

![SPARCK Results](https://github.com/nduursma/AIRR_gate_detection/blob/master/SPARCK/fig.jpg)

### Gate Location Prediction
To run the SPARCK algorithm, use: 
```bash
python SPARCK
```

This will store the images with the detected corners, as well as an Excel file gate_points.xlsx with the predicted gate point locations (x and y) in a new folder *predictions_SPARCK*. 


### Performance Evaluation
To retrain or to evaluate the performance of the SPARCK algorithm, open Jupyter Notebook.

```bash
jupyter notebook
```
Navigate in the folder *SPARCK* to the file *SPARCK.ipynb* and run all the cells. Scroll down to see the visualizations of the algorithm training, the last cell displays the accuracy score. 

The images with the predicted gate locations are stored in the folder *SPARCK/perf_evaluation*. Images with correctly predicted gates, where six or more gate points lie on a real corner, are labelled with a *'T'*. If this is not the case, the image is labelled with *'F'*.

## AUTHOR RIGHTS
The methodology and the code of the algorithms cannot be reused without author's permission.

**Contact:** Nadine Duursma, email: N.A.Duursma@outlook.com

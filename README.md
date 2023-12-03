# VisiSolve Project

## Introduction
VisiSolve is a vision-based equation resolver that employs image processing and machine learning techniques to recognize handwritten mathematical equations. This project aims to overcome the challenges associated with interpreting handwritten symbols through a series of preprocessing steps and neural network evaluations.

## Preprocessing
### Image Cleaning
- **Noise Reduction**: Two versions developed; basic thresholding and blurring (Version 1), and advanced techniques including shadow removal and adaptive thresholding (Version 2). Command-line execution with `-n` parameter to select the version.
- **Brightness Adjustment**: Not included due to potential impairment of results.
- **Histogram Equalization**: Tested but not optimal for preprocessing as it could amplify unwanted background features.

### Segmentation
- **Symbol Extraction**: Implemented in two versions; Mean Shift Clustering (Version 1) and Contour Detection (Version 2). Command-line execution with `-s` parameter to select the version.
- **Results**: Version 2 (Contour Detection) proved more effective for symbol segmentation.

### Standardization
Transformed separated symbols into a consistent format for CNN processing, which involved centering and resizing symbols to a standard size of 28x28 pixels.

## Image Recognition
### Dataset
A collection of handwritten mathematical operators and numbers, comprising 554 images per symbol, publicly available for use.

### Experimentation & Analysis
Evaluated various neural network architectures:
- **Multi-layer Perceptrons (MLPs)**: Both shallow and deep architectures were tested.
- **Convolutional Neural Networks (CNNs)**: Shallow and deep architectures were evaluated, with the deep CNN emerging as the superior model due to better generalization capabilities.

## Conclusion
The project highlights the importance of advanced noise reduction, image preprocessing, and deep CNNs for the recognition of handwritten mathematical symbols. The deep CNN demonstrated the best performance in terms of accuracy and loss metrics.

## How to Run
Use the following command to execute the noise reduction and symbol extraction on the specified image:<br>
```python pipeline.py -n [Noise Reduction Version] -s [Symbol Extraction Version] -f [Image Path]```

Example:
```python pipeline.py -n 1 -s 2 -f ./eq_rejected.png```

## References
- Various references to methods and datasets used are included in the report.

## Documentation
- **Project Report**: [View the full project report here](https://github.com/engpap/visi-solve/blob/main/report.pdf)
- **Presentation Slides**: [Access the presentation slides here](https://github.com/engpap/visi-solve/blob/main/slides.pdf)


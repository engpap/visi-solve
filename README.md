
# VisiSolve: Vision-Based Equation Resolver

## Project Plan

### Data Collection

- **Dataset of numbers and operators:** ready
- **Dataset of equations:** can be done at the end

### Preprocessing (pt. 1)

1. **Histogram Equalization**: to increase contrast
2. **Thresholding:** to get black&white images
3. **Noise Reduction:** median filtering or gaussian filtering (compare results and choose the best one)
   
### Decomposition of equations into symbols

#### Method A

1. **Find contours:** 
   - [OpenCV Contours Documentation](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html)
2. **Loop through contours:** to get symbols

#### Method B

1. **Use edge detection**
2. **Separate symbols:** using clustering and previously obtained edges

> **Note:** It could be cool to apply both methods (A and B) and compare results.

### Preprocessing (pt. 2)

1. **Make the image squared**
2. **Center the symbols:** to improve training/prediction
3. **Rescaling image:** to get a standard nxn matrix
4. **Interpolation:** to make symbols denser, clearer for the NN (see if it's better here or in pt. 1)

### Symbol Recognition

Choose either one of the following methods:
- **Create CNN**
- **Create simple NN:** using multi-layer perceptron

### Final Computation

- **Calculate equation**

### Final Testing to Get Pipeline Accuracy

- **Write down some equations by hand and their associated results.** 
- **Calculate accuracy for calculating equations.**


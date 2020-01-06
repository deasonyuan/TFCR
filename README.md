## TFCR
Learning Target-focusing Convolutional Regression Model for Visual Object Tracking

was submited to Knowledge-Based Systems

The matlab code for TFCR tracker can be downloaded [here[google]](https://drive.google.com/open?id=1DCou-KvSj9joI68KwynWGIzJ3XY-lr06) or [here[baidu]]().

## Usage
### Tracking
1. If you want to compare our results in your experiment, just download the raw experimental results.
2. If you want to test our experiment:

   2.1 Download the code and unzip it in your computer.
   
   2.2 Run the demo.m to test a tracking sequence using a default model.
   
   2.3 Using run_TFCR.m to test the performance on OTB, TC or UAV benchmark.
3. Prerequisites: Ubuntu 18, Matlab R2017, GTX1080Ti, CUDA8.0.


## Results
| Dataste | OTB2013 | OTB2015 | TC128 | UAV123 |
| --------| --------| ------- | ------ | ----- | 
| Prec.   | 0.871   | 0.876   | 0.776  | 0.715 |
| AUC     | 0.671   | 0.665   | 0.564  | 0.512 | 


## Contact
Feedbacks and comments are welcome! Feel free to contact us via dyuanhit@gmail.com


## Acknowledgements
Some of the parameter settings and functions are borrowed from CREST(https://github.com/ybsong00/CREST-Release). 

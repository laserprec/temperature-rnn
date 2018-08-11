# Temperature Prediction using Recurrent Neural Network

## Goal:

Use LSTM to create a model make temperature prediction

## Requirements
- Python3
- Anaconda v4.5.9

## Setup

1. Unzip the data.zip file
2. run `python predict.py`

## Dataset

Dataset comes from https://www.bgc-jena.mpg.de/wetter/

## Discussion

The model is able to detect pattern over an extensive period of time. It is able to pick up the annual trend of increasing temperature as summer is approaching and decreasing temperature as winter arrives. What is more impressive is that the temperature is noisy and fluctuates, for example, on a week-to-week basis, and it did not prevent the model from capturing the long term patterns of temperature.

For more details, see the Jupyter Notebook [`pipeline.ipynb`](./pipeline.ipynb).

## References
- http://blogs.rstudio.com/tensorflow/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/
- https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- https://machinelearningmastery.com/multi-step-time-series-forecasting/
- http://adventuresinmachinelearning.com/keras-lstm-tutorial/
- https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf
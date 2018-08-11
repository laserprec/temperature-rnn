# Temperature Prediction using Recurrent Neural Network

## Goal:
Use LSTM to create a model make temperature prediction

## Dataset
Dataset comes from https://www.bgc-jena.mpg.de/wetter/

## Discussion

The model is able to detect pattern over an extensive period of time. It is able to pick up the annual trend of increasing temperature as summer is approaching and decreasing temperature as winter arrives. What is more impressive is that the temperature is noisy and fluctuates, for example, on a week-to-week basis, and it did not prevent the model from capturing the long term patterns of temperature.

For more details, see the Jupyter Notebook `pipeline.ipynb`.
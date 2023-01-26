# HeadDirectionPredNet
Code and data accompanying the paper "A Spiking Neural Network Model of Rodent Head Direction calibrated with Landmark Free Learning" by Rachael Stentiford, Thomas C. Knowles and Martin J. Pearson

Paper at: https://doi.org/10.3389/fnbot.2022.867019

This work was also involved in the paper "Integrating Spiking Neural Networks and Deep Learning Algorithms on the Neurorobotics Platform" by Rachael Stentiford, Thomas C. Knowles, Benedikt Feldoto, Deniz Ergene, Fabrice O. Morin and Martin J. Pearson

Paper at: https://link.springer.com/chapter/10.1007/978-3-031-20470-8_7

## Contents

### Data

Code to extract from ROSBags and process them into data suitable for the machine learning algorithms.

The datasets used in the paper can be found at: https://we.tl/t-JXgU04ZLmC

The predictions used in the paper can be found at:-

PCN Predictions: https://we.tl/t-UtTHeiKz9F

VAE Predictions: https://we.tl/t-YiJ5SMWSzK

CNN Predictions: https://we.tl/t-6YJ7kjhjMV

### Figures

Figures used in the paper, with code to generate where appropriate.

### NEST

Jupyter notebook files that can build and run the Ring Attractor Head Direction Cell model, outputting Idiothetic Estimates of head direction.

### Results

Script to analyse the Predictions compared to the ground truth data. This is needed for Figures 6 and 7 especially.

### Tensorflow

Python code to build and run the three machine learning models used in the paper; the Predictive Coding network, the Variational Autoencoder and the Convolutional Neural Network.


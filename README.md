# Try-Sparse-LSTM-Autoencoder-Implementation
Using LSTM autoencoder, L1 Regularization

## Purpose

* For anomaly detection, autoencoder is widely used.
* But using autoencoder, which have many variables with strong correlations, is said to cause a decline of detection power.
* To avoid the above problem, the technique to apply L1 regularization to LSTM autoencoder is advocated in the below paper.
>N. Gugulothu, P. Malhotra, L. Vig, and G. Shroff, “Sparse neural networks for anomaly detection in high-dimensional time series,” in AI4IOT Workshop in Conjunction with ICML, International Joint Conference on Artificial Intelligence and European Conference on Artificial Intelligence, Stockholm, Sweden, 2018.
* The point is to use L1 regularization at the second layer of sequence model(right under the input data).
## Algorithm and How to implement

* For the implementation, tensorflow and keras are used.
* First, 

## Results


## Conclustion


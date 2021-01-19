# Sparse-LSTM-Autoencoder-Implementation
Using LSTM autoencoder, L1 Regularization

## Purpose

* For anomaly detection, autoencoder is widely used.
* But using autoencoder, which have many variables with strong correlations, is said to cause a decline of detection power.
* To avoid the above problem, the technique to apply L1 regularization to LSTM autoencoder is advocated in the below paper.
>*N. Gugulothu, P. Malhotra, L. Vig, and G. Shroff, “[Sparse neural networks for anomaly detection in high-dimensional time series](https://www.researchgate.net/profile/Pankaj_Malhotra3/publication/326305246_Sparse_Neural_Networks_for_Anomaly_Detection_in_High-Dimensional_Time_Series/links/5b59f633aca272a2d66cbb98/Sparse-Neural-Networks-for-Anomaly-Detection-in-High-Dimensional-Time-Series.pdf),” in AI4IOT Workshop in Conjunction with ICML, International Joint Conference on Artificial Intelligence and European Conference on Artificial Intelligence, Stockholm, Sweden, 2018.*
![Extract the frame](https://github.com/takanyanta/Try-Sparse-LSTM-Autoencoder/blob/main/paper.png "process1")
* The point is to use L1 regularization at the second layer of sequence model(right under the input data).

## Algorithm and How to implement

* For the implementation, tensorflow and keras are used.
* At first, define "Standard RNN EncoderDecoder", then define "Sparse RNN Encoder-Decoder".
* Sparse RNN Encoder-Decoder is built by adding some changes to Standard RNN EncoderDecoder as below;
   * Flatten the input
   * Insert the custom layer(with L1 regularization)
   * Reshape the 2. output

* Structure of Layers
   * Standard RNN EncoderDecoder
 
| Seq | Layer | Input Shape | Output Shape |
----|----|----|----
| 1 | LSTM | (None, l, k) | (None, h) |
| 2 | RepeatVector | (None, h) | (None, l, h) |
| 3 | LSTM | (None, l, h) | (None, l, h) |
| 4 | TimeDistributed | (None, l, h) | (None, l ,k) |
   * Sparse RNN Encoder-Decoder
   
| Seq | Layer | Input Shape | Output Shape |
----|----|----|----
| 1 | Flatten | (None, l, k)| (None, l&times;k) |
| 2 | Custom | (None, l&times;k) | (None, l&times;k) |
| 3 | Reshape | (None, l&times;k) | (None, l, k) |
| 4 | LSTM | (None, l, k) | (None, h) |
| 5 | RepeatVector | (None, h) | (None, l, h) |
| 6 | LSTM | (None, l, h) | (None, l, h) |
| 7 | TimeDistributed | (None, l, h) | (None, l ,k) |

## Results


## Conclustion


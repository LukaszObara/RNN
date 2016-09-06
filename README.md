# RNN
A simple single layer recurrent neural network for character classification. The activation between nodes utilizes the tanh function and the outputs *o<sup>(t)</sup>* are used as the argument to the softmax function to obtain the vector y of the probabilities over the output. The loss is the negative log-likelihood of the true target y^{(t)} given the inputs so far  

The code works by first using Tex_vec.py to convert the characters in a .txt file into a character vectors of length 127 consisting of 0 except for the coordinate associated with the ASCII representation of the character. 

Once a text file has been converted it can be processed by RNN. RNN is initialized using the entire data and a user chosen integer value for the hidden connections. The network then proceeds to set the appropriate sizes for the input, hidden, and output matrices. 

## How It Works
To train the network is trained simple call `sgd(self, seq_length, epochs, eta, decay_rate=0.9, learning_decay=0.0).`

# RNN
A simple single layer recurrent neural network for character classification. The activation between nodes utilizes the tanh function and the outputs *o<sup>(t)</sup>* are used as the argument to the softmax function to obtain the vector *y* of the probabilities over the output. The loss is the negative log-likelihood of the true target *y<sup>(t)</sup>* given the inputs so far.  

The code works by first using `Tex_vec.py` to convert the characters in a `.txt` file into a character vectors of length 127 consisting of 0 except for the coordinate associated with the ASCII representation of the character. 

Once a text file has been converted it can be processed by RNN. RNN is initialized using the entire data and a user chosen integer value for the hidden connections. The network then proceeds to set the appropriate sizes for the input, hidden, and output matrices. 

## How It Works
The network works in two steps. The first step involves initializing the network and the second step involves training the network. 
<p>For step 1 we simply make the following declarion:
rnn = RNN(data, hidden_size, spec_rad = 1.01, eps=0.0001)

<p> Step 2 involes training the network and is performed by calling `train(self, seq_length, epochs, eta, decay_rate=0.9, learning_decay=0.0, randomize=False, print_final=True)` and selecting values for the parameters: 

<p><b>seq_length</b>: Integer value for the desired length of the subsubsequence<br> 
<b>epochs</b>: Integer value for the number of iteration to train over.<br>
<b>eta</b>: Learing rate for gradient descent.<br>
<b>decay_rate</b>: Decay parameter for the moving average. The value must lie between [0, 1) where smaller values indicate shorter memory. The default value is set to `0.9`<br>
<b>learning_decay</b>: Annealing parameter for the exponetial decay. The smaller values indicate milder annealing of the learning rate. The default value is set to `0.0`<br>
<b>randomize</b>: If set to `True` then the subsequences will be shuffle before beign processed further. The default value is set to `False`. <br>
<b>print_final</b>: Prints the final output at the end of evey epoch. The default value is set to `True`. <br>

# References
<ol>
<li>Goodfellow Ian, Bengio Yoshua, Courville Aaron, <em>Deep Learning</em>, MIT Press, 2016, http://www.deeplearningbook.org</li>
</ol>

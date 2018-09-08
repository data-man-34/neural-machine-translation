# NMT with Attention, step by step implementation 
## About
My experience using Tensor flow and it's keras API to implement a Neural Machine Translation (NMT) with Attention model. This article's focus is on implementing the model, if you are interested in its theory you can refer to [this article](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/). 

A little bit about me - I did maths in university and have completed Coursera's [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models). Through the course, I have gained a solid theoretical foundation in sequential models, but I still feel unprepared to implement models on my own, as I am still uncomfortable using ML libraries like TensorFlow or Keras. Therefore, I decided to get my hands dirty with code, starting with the official [TF tutorial on NMT](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb?linkId=53292082&utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=NLP%20News#scrollTo=tnxXKDjq3jEL) (opens in Google Colab)<sup>1</sup>.  I chose this tutorial because it can be a good template or starting point for more interesting models<sup>2</sup>. 

Before you proceed, here is what I wished someone told me about implementing AI models:

- While use cases of libraries like keras and TF are well documented, it is important to know how their APIs work in tandem to produce a desired architecture. 
- Keep track of input dimensions at every layer of the model!
- Be extremely familiar with [object oriented programming](https://python.swaroopch.com/oop.html)!


## Model architecture:
The model consists of an encoder and decoder. An attention mechanism decides how important each input is to the output. Here is a visual that I have drawn using [inkscape](https://inkscape.org/en/):


![NMT with attention](https://i.imgur.com/2tC6PPF.png)

### Encoder model
	
Full code, taken from the [TF tutorial on NMT](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb?linkId=53292082&utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=NLP%20News#scrollTo=tnxXKDjq3jEL):

```
1  Encoder(tf.keras.Model):
   
2    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
3        super(Encoder, self).__init__()
4        self.batch_sz = batch_sz
5        self.enc_units = enc_units
6        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
7        self.gru = gru(self.enc_units)
        
8    def call(self, x, hidden):
9        x = self.embedding(x)
10        output, state = self.gru(x, initial_state = hidden)        
11        return output, state
    
12    def initialize_hidden_state(self):
13       return tf.zeros((self.batch_sz, self.enc_units))
        
```
When I first read the code above, I was very confused. Soon I began to realise there are two stages in building AI models using ML libraries. One, we specify the architecture/blueprint of the model, i.e type of layers, dimensions of inputs/outputs etc. Two, we specify the model's forward pass, i.e the inputs and outputs of the model. 

In our case, we specify that the blueprint of the encoder in lines 2-7. Line 6 says, we design our encoder to have an embedding layer that takes in words mapped into indexes, and the values of the indices do not exceed `vocab_size`. We also specify the embedding layer to return a vector of size `embedding_dim`. 

In line 7, we specify that the encoder has a GRU layer, and one of its output dimension will be `enc_units` (more on that later). 

Now we have specified the encoder blueprint, we need to specify the encoder's forward pass. How? Line 1 tells us that our `Encoder` is a subclass inherits from `tf.keras.Model`. Owing to the properties of `tf.keras.Model`, we can specify the forward pass of our `Encoder` class in a `call` method (lines 8 - 11). 

In line 9, we pass the input tensor, which is of size **[batch\_size x max\_length]** through the embedding layer. Due to our specifications on line 6, it returns embeddings of size **[batch\_size x max\_length x embedding_dim]**.

In line 10, we pass the embeddings through a GRU. We want the GRU to return two things, and we can do so because we have have pre-defined the GRU architecture to be:

```
def gru(units):
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')

# remainder of code block is not shown for brevity
```
 	
 Above, we have specified that `return_sequences = True` and  `return_state = True`<sup>3</sup>. This means we want the GRU to return two things, which in line 10 we specify as `output` and `hidden`. It is important to understand what these two are: `output` is the GRU's output _at every time step_; `hidden` is the GRU's hidden state _at final time step_.
 
Owing to line 7, we expect the `output` to be of size **[batch\_size x max\_length x enc_units]**. The dimension **max\_length** is here, because `output` is the output of GRU across all time steps!

Also, owing to lines 12 and 13, the size of `hidden` is of size **[batch\_size x units]**. 
 

Something to note: by definition, the GRU hidden state _is_ the output (see this [tutorial](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)). Hence, `hidden`  is the final value of `output`.

Also, `units`, `hidden_size`, `enc_units` and `dec_units` refer to the same value in the colab notebook. 

### Decoder model

The decoder has a similar architecture to the encoder: it has an embedding and GRU layer. The decoder receives the following from the encoder:

- `enc_output`: Encoder output at each time step 
- `hidden`: Initialised with the encoder's final hidden state, which is used as the decoder GRU's hidden state at first time step.

Here is where the attention mechanism takes place. The decoder uses `enc_output` and `hidden` to compute `attention_weights`. These `attention_weights` are used to weigh each encoder output. The weighed encoder outputs are summed, and are known as the `context_vector` (refer to my visualisation above). 

The context vector and decoder input are combined into one before passing through a GRU. What is the decoder input? It depends. 

- training stage:  _teacher forcing_ is applied, i.e input at each time step is the target word
- translating stage: no _teacher forcing_, i.e input at each time step is the prediction at previous time step. This is done when we are evaluating the model.

During training: 

```
0 dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
1 for t in range(1, targ.shape[1]):
2                # passing enc_output to the decoder
3                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)                
4                loss += loss_function(targ[:, t], predictions)                
5                # using teacher forcing
6               dec_input = tf.expand_dims(targ[:, t], 1)

```
We start by feeding the start token to the decoder (line 0 and 3). During the t<sup>th</sup> forward pass, the decoder returns:

- `dec_hidden`: Hidden state of gru in decoder, for the (t+1)<sup>th</sup> pass
- `predictions`: Output of gru in decoder, passed through a feedforward network<sup>4</sup> such that its size is [batch\_size, target\_vocab\_size]. We take this as the unnormalised log-probabilities of the t<sup>th</sup> word in our output sentence. Therefore, during loss calculation, we use the `tf.nn.sparse_softmax_cross_entropy_with_logits` API, which has been designed to handle unnormalised log-probabilities. [More info](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits). 


## Evaluating model on validation set
I trained the NMT model using the default parameters:

```
num_examples = 30000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024

```
Next, I would like to see how the model could translate the Spanish sentences in validation set `input_tensor_val` into English. The `evaluate` function provided in the tutorial is designed to translate individual, user-specified sentences, so I rewrote it to take in tensors like `input_tensor_val` instead. See `evaluate.py`.

I then printed out some model predictions and compared them to their reference English translations in `target_tensor_val `. Here are some of my results:

![Results](https://i.imgur.com/oUzfOJP.png)

## Quantify quality of translations

I wrote `model_bleu.py` to evaluate the translation quality of our validation dataset using BLEU score. The `compute_bleu` function is taken from [here](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py).

Averaging the BLEU score over all examples, I arrived at a score of 0.64 (maximum BLEU score is 1).

## Potential areas for improvement
Here are some ways you can try to beat my BLEU score:

- Multilayer GRUs or LSTMs
- Bilayer GRUs or LSTMs
- Increase embedding dimension
- Increase GRU/LSTM output dimension 

## References
1. Tips for running notebook: Go to **Runtime** > **Change runtime type**, ensure that the code is run on python 3 (it doesn't work on python 2)

2. Here is another good tutorial: see [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)

3. For more information on parameters `return sequences` and `return states` in keras, see [this article](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/). You can experiment with GRU outputs by running the toy code below:

```
	from keras.models import Model
	from keras.layers import Input
	from keras.layers import GRU
	from numpy import array
	inputs1 = Input(shape=(3, 1))
	state_h, state_c = GRU(1, return_sequences=True, 	return_state=True)(inputs1)
	model = Model(inputs=inputs1, outputs=[state_h, state_c])
	data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
	print(model.predict(data))
```

4. 	Dense layers are typically used to change the shape of a tensor to a desired shape. The layer `keras.layers.Dense(units)` receives an input of size [batch\_size, ..., input\_dim] and returns an output of size [batch\_size, ..., units]



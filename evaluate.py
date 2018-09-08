def evaluate(encoder, decoder, val_input_tensor, val_target_tensor, targ_lang):
  	
  	# number of sentences in validation input and target must be equal
  	assert(val_input_tensor.shape[0] == val_target_tensor.shape[0])
  	
  	# maximum length of target sentence
    target_n = val_target_tensor.shape[1]
    
   # number of sentence examples 
    input_m = val_input_tensor.shape[0]
    
    # store model predictions in all_pred
    all_pred = np.zeros((input_m, target_n))

    for row in range(input_m):
      
      hidden = [tf.zeros((1,units))]
      
      # encoder input must be of size [1,1]
      inputs = tf.expand_dims(tf.convert_to_tensor(input_tensor_val[row]), 0)
      
      enc_out, enc_hidden = encoder(inputs, hidden)
      
      # enc_out will be [1 x max_length x units ]
      # enc_hidden will be [1 x units] 

      dec_hidden = enc_hidden
      dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
      # dec_input of dim 1 x 1 
      
      # in target tensor, the first token is always <start>, so we have (target_n - 1) tokens to predict 
      for t in range(target_n - 1):
          predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
          # predictions shape [1, vocab_target_size]
          
          # we have vocab_target_size classes, and the ith value of predictions is the unnormalised log-probability of the ith class. 
          predicted_id = tf.multinomial(predictions, num_samples=1)
          # predicted_id is now of tensor size [1, 1]
          
          # the predicted ID is fed back into the model
          dec_input = predicted_id
      
          # change predicted_id to integer
          predicted_id = predicted_id[0][0].numpy()
          
          # prediction for row th sentence, t th word
          all_pred[row,t] = predicted_id
          
          #stop predicting when we reach the '<end>' token
          if targ_lang.idx2word[predicted_id] == '<end>':
            break
      
    return all_pred

# to call the function:
# all_pred = evaluate(encoder, decoder, input_tensor_val, target_tensor_val, targ_lang)
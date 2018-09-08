def model_bleu(all_pred, target_tensor_val):
  assert(len(all_pred) == len(target_tensor_val))
  
  bleu_all_examples = 0
  for i in range(len(all_pred)):
    model_prediction = []  
    actual = []
    
    # all_pred and target_tensor_val contain lists of indices, we need to...
    # map them to words
    for index in all_pred[i]:
      if targ_lang.idx2word[index] == '<end>':
        break
      model_prediction.append(targ_lang.idx2word[index])

    # start from index 1 to avoid <start> token
    for index in target_tensor_val[i][1:]:
      if targ_lang.idx2word[index] == '<end>':
        break
      actual.append(targ_lang.idx2word[index])

   # compute_bleu is taken from here: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
    bleu_one_example = compute_bleu(actual, model_prediction, max_order=1,
                 smooth=True) 
    
    bleu_all_examples += bleu_one_example
    
  # average bleu score over all examples
  bleu_all_examples /= len(all_pred)
  return bleu_all_examples
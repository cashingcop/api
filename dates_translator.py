import tensorflow as tf


class Translator(tf.Module):
  def __init__(self,
               source_tokenizer, #input tokenizer
               target_tokenizer, #target tokenizer
               loaded_transformer, #model
               TX_SOURCE, TX_TARGET):
    
    self.source_tokenizer = source_tokenizer
    self.target_tokenizer = target_tokenizer
    self.loaded_transformer = loaded_transformer
    self.TX_SOURCE = TX_SOURCE
    self.TX_TARGET = TX_TARGET
    self.target_map = {v:k for k, v in target_tokenizer.word_index.items()}

  def __call__(self, sentence):

    tokenized_source = self.source_tokenizer.texts_to_sequences([sentence])
    tokenized_padded_source = tf.keras.preprocessing.sequence.pad_sequences(tokenized_source, maxlen=self.TX_SOURCE, padding='post', truncating='post')

    # As the output language is English, initialize the output with the `[START]` token.
    tokenized_target = self.target_tokenizer.texts_to_sequences(['#'])
    tokenized_target = tf.keras.preprocessing.sequence.pad_sequences(tokenized_target, maxlen=self.TX_TARGET, padding='post', truncating='post')

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.

    # Run from TX TARGET steps

    for t in range(self.TX_TARGET - 1):
      out = self.loaded_transformer(input_1 = tokenized_padded_source , input_2 = tokenized_target)
      predictions = out.get('output_1')
      attention_weights = out.get('output_2')
      # predictions = self.transformer([tokenized_padded_source, tokenized_target], training=False)
      predictions = predictions[:, t, :]  # Shape `(batch_size, 1, vocab_size)`.
      res = tf.argmax(predictions, axis=-1)
      tokenized_target[:, t +1 ] = res.numpy()[0]

    output_decoded = "".join([self.target_map.get(e) for e in tokenized_target[0]])
    print(f'Original Sentence: {sentence}')
    print('Translated Sentence:',output_decoded)

    #Tokens
    in_tokens = sentence.split()
    in_tokens = in_tokens[:self.TX_SOURCE]
    out_tokens = [x for x in output_decoded]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.

    # print('calculation attention weigths')
    # self.transformer([tokenized_padded_source, tokenized_target], training=False)
    # attention_weights = self.transformer.decoder.last_attn_scores

    return output_decoded, in_tokens, out_tokens, attention_weights
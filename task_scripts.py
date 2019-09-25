from tf2_models.lm_lstm import LmLSTM
from tfds_data.tal_agreement import SVAgreement

if __name__ == '__main__':


  class task_params(object):
    batch_size=32



  databuilder = SVAgreement(data_dir='data')
  dataset = databuilder.as_dataset(split="validation", batch_size=task_params.batch_size)

  class model_params(object):
    hidden_dim=32
    input_dim=databuilder.vocab_size()
    output_dim=databuilder.vocab_size()
    depth=2
    hidden_dropout_rate=0.1

  lm_lstm = LmLSTM(hparams=model_params)
  lm_lstm.build(input_shape=(None, None))
  lm_lstm.summary()

  for batch in dataset:
    lm_lstm(batch['sentence'])
    break
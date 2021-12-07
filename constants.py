# consume_grads.py: update weights when all layers have at least this many gradients
BATCH_SIZE = 2

# produce_grads.py: generate a prompt/prediction/lossz every n seconds
CREATE_SAMPLE_INTERVAL = 1

# lstm_model.py: hidden dimension of LSTM layer
# also used in consume_models.py, produce_grads.py
HIDDEN_SIZE = 50
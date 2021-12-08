# consume_grads.py: update weights when all layers have at least this many gradients
BATCH_SIZE = 32

# produce_grads.py: generate a prompt/prediction/loss every n seconds
CREATE_SAMPLE_INTERVAL = 0.01

# Path to text sources for grad producers (relative to produce_grads.py)
PATH_TO_DATA = 'text_sources/'

# file name for logging model predictions
PROMPT_LOG_FILENAME = 'prompts.txt'

# produce_grads.py: log every nth prompt/target/prediction
PROMPT_LOG_INTERVAL = 50

# lstm_model.py: hidden dimension of LSTM layer
# also used in consume_models.py, produce_grads.py
HIDDEN_SIZE = 50

# block on full buffer
WAIT_TIME_ON_BUFFER_ERROR = 10
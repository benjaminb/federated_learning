import lstm_model

model = lstm_model.LSTM_NWP(hidden_size=100, tokenizer=None)

prompt = "hello model"

y = model(prompt)
print(y)
print(y.shape)
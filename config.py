# config file so that weights for the ensembling is also frozen

ENSEMBLE_WEIGHTS = {
    "LSTM": 0.55,
    "BiLSTM": 0.25,
    "GRU": 0.20
}

SEQUENCE_WINDOW = 24

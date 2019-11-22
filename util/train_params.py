radam_slow = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 0
}

adam_slow = {
'learning_rate': 0.0001,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}


adam_mid = {
'learning_rate': 0.0005,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}

adam_midmid = {
'learning_rate': 0.0002,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}

radam_fast_long = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}

radam_slow_long = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}

radam_fast = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}

radam_mid = {
'learning_rate': 0.0005,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}


cosinerestart_fast = {
'learning_rate': 0.001,
'optimizer': 'adam',
'decay_steps': 1000,
'schedule': 'cosinerestart_fast'
}

cosinerestart_slow = {
'learning_rate': 0.001,
'optimizer': 'adam',
'decay_steps': 10000,
'schedule': 'cosinerestart_slow'
}

TRAIN_PARAMS = {'radam_slow': radam_slow,
                'radam_fast': radam_fast,
                'adam_slow':  adam_slow,
                'radam_fast_long': radam_fast_long,
                'radam_slow_long': radam_slow_long,
                'adam_mid': adam_mid,
                'adam_midmid': adam_midmid,
                'radam_mid': radam_mid,
                'cosinerestart_fast': cosinerestart_fast,
                'cosinerestart_slow': cosinerestart_slow}
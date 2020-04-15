radam_slw = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 0
}

adam_slw = {
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

radam_fst_long = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}

radam_slw2 = {
'learning_rate': 0.0005,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}

radam_slw_long = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}

radam_fst = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}

radam_mid = {
'learning_rate': 0.0005,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}


crs_fst = {
'learning_rate': 0.001,
'optimizer': 'adam',
'decay_steps': 1000,
'schedule': 'crs_fst'
}

crs_fst_v2 = {
'learning_rate': 0.0001,
'optimizer': 'adam',
'decay_steps': 1000,
'schedule': 'crs_fst'
}

crs_slw = {
'learning_rate': 0.001,
'optimizer': 'adam',
'decay_steps': 10000,
'schedule': 'crs_slw'
}

crs_slw_v2 = {
'learning_rate': 0.0001,
'optimizer': 'adam',
'decay_steps': 10000,
'schedule': 'crs_slw'
}

crs_slw_v3 = {
'learning_rate': 0.0005,
'optimizer': 'adam',
'decay_steps': 10000,
'schedule': 'crs_slw'
}


mnist_adam = {'optimizer': 'adam',
              'learning_rate': 0.001,
              'decay_steps': 10000,
              'num_train_epochs': 20
              }

svhn_adam_mid = {
'learning_rate': 0.0005,
'optimizer': 'adam',
'hold_base_rate_steps': 1000,
'num_train_epochs': 100,
}

svhn_radam_mid = {
'learning_rate': 0.0005,
'optimizer': 'radam',
'hold_base_rate_steps': 1000,
'num_train_epochs': 200
}

svhn_crs_slw = {
'learning_rate': 0.0005,
'optimizer': 'adam',
'hold_base_rate_steps': 0,
'num_train_epochs': 100,
'decay_steps': 10000,
'schedule': 'crs_slw',
'num_train_epochs': 200
}

TRAIN_PARAMS = {'radam_slw': radam_slw,
                'radam_fst': radam_fst,
                'adam_slw':  adam_slw,
                'radam_fst_long': radam_fst_long,
                'radam_slw_long': radam_slw_long,
                'adam_mid': adam_mid,
                'adam_midmid': adam_midmid,
                'radam_mid': radam_mid,
                'crs_fst': crs_fst,
                'crs_slw': crs_slw,
                'crs_slw_v2': crs_slw_v2,
                'crs_slw_v3': crs_slw_v3,
                'crs_fst_v2': crs_fst_v2,
                'mnist_adam': mnist_adam,
                'radam_slw2': radam_slw2,
                'svhn_adam_mid': svhn_adam_mid,
                'svhn_radam_mid': svhn_radam_mid,
                'svhn_crs_slw': svhn_crs_slw}
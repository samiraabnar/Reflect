pure_distill_1 = {
'distill_temp' : 5.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 1000,
'teacher_optimizer' : 'adam'
}


pure_distill_2 = {
'distill_temp' : 5.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam'
}

pure_distill_3 = {
'distill_temp' : 5.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam'
}

pure_distill_4 = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam'
}

pure_distill_4_radamfast = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'radam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'adam'
}

pure_distill_4_cosinerestart_fast = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 1000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'cosinerestart'
}


pure_distill_4_cosinerestart_slow = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'cosinerestart'
}

pure_distill_5 = {
'distill_temp' : 2.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam'
}

pure_distill_4_fastonline = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam'
}

DISTILL_PARAMS = {'pure_distill_1' :  pure_distill_1,
                  'pure_distill_2' :  pure_distill_2,
                  'pure_distill_3' :  pure_distill_3,
                  'pure_distill_4': pure_distill_4,
                  'pure_distill_4_radamfast': pure_distill_4_radamfast,
                  'pure_distill_5': pure_distill_5,
                  'pure_distill_4_fastonline': pure_distill_4_fastonline,
                  'pure_distill_4_cosinerestart_fast': pure_distill_4_cosinerestart_fast,
                  'pure_distill_4_cosinerestart_slow': pure_distill_4_cosinerestart_slow
                  }

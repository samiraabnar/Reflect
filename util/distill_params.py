pure_dstl_1 = {
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


pure_dstl_2 = {
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

pure_dstl_3 = {
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

pure_dstl_4 = {
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

pure_dstl_4_radamfst = {
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

pure_dstl_4_crs_fst = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 1000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs'
}


pure_dstl_4_crs_slw = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs'
}

pure_dstl_4_crs_slwfst = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 1000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'adam',
'schedule': 'crs_fst'
}

dstl_6_crs_slw = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs'
}

schdld_dstl_6_crs_slw = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs',
'dstl_decay_steps': 10000,
'dstl_warmup_steps': 0,
'hold_base_dstlrate_steps': 10000,
}

pure_dstl_4_crs_vslw = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 100000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.0001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs'
}

pure_dstl_5 = {
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

pure_dstl_4_fstonln = {
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

pure_dstl_mn_fstonln = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'student_gold_rate' : 0.0,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  1000,
'student_warmup_steps' : 10000,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
}

pure_rpdst_crs_slwfst = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.0,
'student_distill_rep_rate': 1.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 1000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'adam',
'schedule': 'crs_fst'
}

dstl_910_crs_slwfst_2 = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

dstl5_910_crs_slwfst_2 = {
'distill_temp' : 5.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

dstl5_910_crs_slwfst_3 = {
'distill_temp' : 5.0,
'student_distill_rate' : 0.9,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 5000,
'teacher_hold_base_rate_steps' : 1000000,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

schdexp_dstl_10_crs_slwfst_2 = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'distill_min_rate': 0.0,
'distill_schedule': 'exp',
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

schdexp_dstl_10_crs_slwfst_3 = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'distill_min_rate': 0.0,
'distill_schedule': 'exp',
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

schdcrs_dstl_10_crs_slwfst_2 = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'distill_min_rate': 0.0,
'distill_schedule': 'crs',
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

schdcrs_dstl_10_crs_slwfst_3 = {
'distill_temp' : 1.0,
'student_distill_rate' : 1.0,
'distill_min_rate': 0.0,
'distill_schedule': 'crs',
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0.0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

schdexp_dstl5_910_crs_slwfst_2 = {
'distill_temp' : 5.0,
'student_distill_rate' : 1.0,
'distill_min_rate': 0.0,
'distill_schedule': 'exp',
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst5_019_crs_slwfst_2 = {
'distill_temp' : 5.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 10000,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst5_019_crs_slwfst_3 = {
'distill_temp' : 5.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 1000000,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}


rpdst_0010_crs_slwfst_3 = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.0,
'student_distill_rep_rate': 1.0,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst_019_crs_slwfst = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 1000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'adam',
'schedule': 'crs_fst'
}

rpdst_019_crs_slwfst_2 = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0.0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst_019_crs_slwfst_3 = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst_0010_crs_slwfst_2 = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.0,
'student_distill_rep_rate': 1.0,
'student_learning_rate' : 0.0001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 0,
'teacher_hold_base_rate_steps' : 0,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst'
}

rpdst_019_crs_slwfst_sst = {
'distill_temp' : 1.0,
'student_distill_rate' : 0.0,
'student_gold_rate' : 0.1,
'student_distill_rep_rate': 0.9,
'student_learning_rate' : 0.001,
'student_decay_steps' : 10000,
'student_hold_base_rate_steps' :  0,
'student_warmup_steps' : 0,
'student_optimizer' : 'adam',
'teacher_learning_rate' : 0.001,
'teacher_decay_steps' : 10000,
'teacher_warmup_steps' : 5000,
'teacher_hold_base_rate_steps' : 1000000,
'teacher_optimizer' : 'radam',
'schedule': 'crs_fst',
}

DISTILL_PARAMS = {'pure_dstl_1' :  pure_dstl_1,
                  'pure_dstl_2' :  pure_dstl_2,
                  'pure_dstl_3' :  pure_dstl_3,
                  'pure_dstl_4': pure_dstl_4,
                  'pure_dstl_4_radamfst': pure_dstl_4_radamfst,
                  'pure_dstl_5': pure_dstl_5,
                  'pure_dstl_4_fstonln': pure_dstl_4_fstonln,
                  'pure_dstl_4_crs_fst': pure_dstl_4_crs_fst,
                  'pure_dstl_4_crs_slw': pure_dstl_4_crs_slw,
                  'pure_dstl_4_crs_vslw': pure_dstl_4_crs_vslw,
                  'dstl_6_crs_slw': dstl_6_crs_slw,
                  'pure_dstl_4_crs_slwfst': pure_dstl_4_crs_slwfst,
                  'pure_dstl_mn_fstonln': pure_dstl_mn_fstonln,
                  'pure_rpdst_crs_slwfst': pure_rpdst_crs_slwfst,
                  'dstl_910_crs_slwfst_2': dstl_910_crs_slwfst_2,
                  'dstl5_910_crs_slwfst_2': dstl5_910_crs_slwfst_2,
                  'dstl5_910_crs_slwfst_3': dstl5_910_crs_slwfst_3,
                  'rpdst_019_crs_slwfst': rpdst_019_crs_slwfst,
                  'rpdst5_019_crs_slwfst_2': rpdst5_019_crs_slwfst_2,
                  'rpdst_019_crs_slwfst_2': rpdst_019_crs_slwfst_2,
                  'schdexp_dstl5_910_crs_slwfst_2': schdexp_dstl5_910_crs_slwfst_2,
                  'schdexp_dstl_10_crs_slwfst_2': schdexp_dstl_10_crs_slwfst_2,
                  'schdexp_dstl_10_crs_slwfst_3': schdexp_dstl_10_crs_slwfst_3,
                  'schdcrs_dstl_10_crs_slwfst_2': schdcrs_dstl_10_crs_slwfst_2,
                  'schdcrs_dstl_10_crs_slwfst_3': schdcrs_dstl_10_crs_slwfst_3,
                  'rpdst5_019_crs_slwfst_3': rpdst5_019_crs_slwfst_3,
                  'rpdst_019_crs_slwfst_3': rpdst_019_crs_slwfst_3,
                  'rpdst_0010_crs_slwfst_2': rpdst_0010_crs_slwfst_2,
                  'rpdst_0010_crs_slwfst_3': rpdst_0010_crs_slwfst_3,
                  'rpdst_019_crs_slwfst_sst': rpdst_019_crs_slwfst_sst,
                  }

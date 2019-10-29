import tensorflow as tf
import re

def camel2snake(name):
  return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: '_' + x.group(0).lower(), name[1:])

def log_summary(log_value, log_name, summary_scope):
  tf.summary.scalar(name=log_name, tensor=log_value, family=summary_scope)



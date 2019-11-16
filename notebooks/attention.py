"""Module for postprocessing and displaying transformer attentions.
This module is designed to be called from an ipython notebook.
"""

import json
import os

import IPython.display as display

import numpy as np

vis_html = """
  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="all">All</option>
    </select>
  </span>
  <div id='vis'></div>
"""


vis_js = open('attention.js').read()


def show(inp_text, out_text, attentions):
  attentions = resize(attentions)
                                 
  attention = _get_attention(
      inp_text, out_text, attentions)
  att_json = json.dumps(attention)
  _show_attention(att_json)


def _show_attention(att_json):
  display.display(display.HTML(vis_html))
  display.display(display.Javascript('window.attention = %s' % att_json))
  display.display(display.Javascript(vis_js))


def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  for i, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      # Sum across different attention values for each token.
      att = att[:, :, :max_length, :max_length]
      row_sums = np.sum(att, axis=2)
      # Normalize
      att /= row_sums[:, :, np.newaxis]
    att_mat[i] = att
  return att_mat


def _get_attention(inp_text, out_text, attentions):
  """Compute representation of the attention ready for the d3 visualization.
  Args:
    inp_text: list of strings, words to be displayed on the left of the vis
    out_text: list of strings, words to be displayed on the right of the vis
    enc_atts: numpy array, encoder self-attentions
        [num_layers, batch_size, num_heads, enc_length, enc_length]
    dec_atts: numpy array, decoder self-attentions
        [num_layers, batch_size, num_heads, dec_length, dec_length]
    encdec_atts: numpy array, encoder-decoder attentions
        [num_layers, batch_size, num_heads, dec_length, enc_length]
  Returns:
    Dictionary of attention representations with the structure:
    {
      'all': Representations for showing all attentions at the same time.
      'inp_inp': Representations for showing encoder self-attentions
      'inp_out': Representations for showing encoder-decoder attentions
      'out_out': Representations for showing decoder self-attentions
    }
    and each sub-dictionary has structure:
    {
      'att': list of inter attentions matrices, one for each attention head
      'top_text': list of strings, words to be displayed on the left of the vis
      'bot_text': list of strings, words to be displayed on the right of the vis
    }
  """

  def get_inp_inp_attention(layer):
    att = np.transpose(attentions[layer], (0, 2, 1))
    return [ha.T.tolist() for ha in att]


  def get_attentions(get_attention_fn):
    num_layers = len(attentions)
    return [get_attention_fn(i) for i in range(num_layers)]

  attentions = {
      'all': {
          'att': get_attentions(get_inp_inp_attention),
          'top_text': inp_text,
          'bot_text': inp_text,
      },
  }

  return attentions
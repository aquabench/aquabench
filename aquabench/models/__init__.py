"""The trainable network pieces you swap on the CLI: the encoder and head axes.

`models.encoders` builds a frozen/trainable backbone that returns multi-scale
feature maps; `models.heads` builds a decoder that turns those maps into logits.
"""

# Understanding the model

## Default model (StarE_Transformer)

```python
StarE_Transformer(
  (bceloss): BCELoss()
  (conv1): StarEConvLayer(200, 200, num_rels=532)
  (conv2): StarEConvLayer(200, 200, num_rels=532)
  (hidden_drop): Dropout(p=0.3, inplace=False)
  (hidden_drop2): Dropout(p=0.1, inplace=False)
  (feature_drop): Dropout(p=0.3, inplace=False)
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=200, out_features=200, bias=True)
        )
        (linear1): Linear(in_features=200, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=512, out_features=200, bias=True)
        (norm1): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (position_embeddings): Embedding(14, 200)
  (layer_norm): LayerNorm((200,), eps=1e-05, elementwise_affine=True)
  (fc): Linear(in_features=200, out_features=200, bias=True)
)
```

```python
sum(p.numel() for p in model.parameters() if p.requires_grad) = 10,870,380
```

````json
{'BATCH_SIZE': 128, 'DATASET': 'wd50k', 'DEVICE': device(type='cpu'), 'EMBEDDING_DIM': 200, 'ENT_POS_FILTERED': True, 'EPOCHS': 401, 'EVAL_EVERY': 5, 'LEARNING_RATE': 0.0001, 'MAX_QPAIRS': 15, 'MODEL_NAME': 'stare_transformer', 'CORRUPTION_POSITIONS': [0, 2], 'SAVE': False, 'STATEMENT_LEN': -1, 'USE_TEST': True, 'WANDB': False, 'LABEL_SMOOTHING': 0.1, 'SAMPLER_W_QUALIFIERS': True, 'OPTIMIZER': 'adam', 'CLEANED_DATASET': True, 'GRAD_CLIPPING': True, 'LR_SCHEDULER': True, 'STAREARGS': {'LAYERS': 2, 'N_BASES': 0, 'GCN_DIM': 200, 'GCN_DROP': 0.1, 'HID_DROP': 0.3, 'BIAS': False, 'OPN': 'rotate', 'TRIPLE_QUAL_WEIGHT': 0.8, 'QUAL_AGGREGATE': 'sum', 'QUAL_OPN': 'rotate', 'QUAL_N': 'sum', 'SUBBATCH': 0, 'QUAL_REPR': 'sparse', 'ATTENTION': False, 'ATTENTION_HEADS': 4, 'ATTENTION_SLOPE': 0.2, 'ATTENTION_DROP': 0.1, 'HID_DROP2': 0.1, 'FEAT_DROP': 0.3, 'N_FILTERS': 200, 'KERNEL_SZ': 7, 'K_W': 10, 'K_H': 20, 'T_LAYERS': 2, 'T_N_HEADS': 4, 'T_HIDDEN': 512, 'POSITIONAL': True, 'POS_OPTION': 'default', 'TIME': False, 'POOLING': 'avg'}, 'NUM_ENTITIES': 47156, 'NUM_RELATIONS': 532}```
````

# Understanding the StarE codebase

## Initialization

`class StarE_Transformer(StarEEncoder)` is initialized with `kg_graph_repr: Dict[str,
np.ndarray]` and `config: dict`.

`config`:
```json
{'BATCH_SIZE': 128, 'DATASET': 'wd50k', 'DEVICE': device(type='cpu'),
'EMBEDDING_DIM': 200, 'ENT_POS_FILTERED': True, 'EPOCHS': 401, 'EVAL_EVERY': 5,
'LEARNING_RATE': 0.0001, 'MAX_QPAIRS': 15, 'MODEL_NAME': 'stare_transformer',
'CORRUPTION_POSITIONS': [0, 2], 'SAVE': False, 'STATEMENT_LEN': -1, 'USE_TEST': True,
'WANDB': False, 'LABEL_SMOOTHING': 0.1, 'SAMPLER_W_QUALIFIERS': True, 'OPTIMIZER':
'adam', 'CLEANED_DATASET': True, 'GRAD_CLIPPING': True, 'LR_SCHEDULER': True,
'STAREARGS': {'LAYERS': 2, 'N_BASES': 0, 'GCN_DIM': 200, 'GCN_DROP': 0.1, 'HID_DROP':
0.3, 'BIAS': False, 'OPN': 'rotate', 'TRIPLE_QUAL_WEIGHT': 0.8, 'QUAL_AGGREGATE': 'sum',
'QUAL_OPN': 'rotate', 'QUAL_N': 'sum', 'SUBBATCH': 0, 'QUAL_REPR': 'sparse',
'ATTENTION': False, 'ATTENTION_HEADS': 4, 'ATTENTION_SLOPE': 0.2, 'ATTENTION_DROP': 0.1,
'HID_DROP2': 0.1, 'FEAT_DROP': 0.3, 'N_FILTERS': 200, 'KERNEL_SZ': 7, 'K_W': 10, 'K_H':
20, 'T_LAYERS': 2, 'T_N_HEADS': 4, 'T_HIDDEN': 512, 'POSITIONAL': True, 'POS_OPTION':
'default', 'TIME': False, 'POOLING': 'avg'}, 'NUM_ENTITIES': 47156, 'NUM_RELATIONS':
532}
```

`kg_graph_repr` looks like
```text
-> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
-> edge_type (n) array with [relation] corresponding to sub, obj above
-> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edgethat has quals
```

```python
edge_index.shape
>>> torch.Size([2, 380696])
edge_type.shape
>>> torch.Size([380696])
rel_embed.shape
>>> torch.Size([1064, 200])
```


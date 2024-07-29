import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from .gnn_encoder import StarEEncoder, StarEBase
from utils.utils_gcn import get_param


class StarE_Transformer(StarEEncoder):
    model_name = 'StarE_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        """TAE:
        >>> kg_graph_repr.keys()
        dict_keys(['edge_index', 'edge_type', 'quals'])

        >>> kg_graph_repr["edge_index"].shape
        (2, 380696)

        >>> kg_graph_repr["edge_type"].shape
        (380696,)

        >>> kg_graph_repr["quals"].shape
        (3, 74866)

        >>> config
        {'BATCH_SIZE': 128, 'DATASET': 'wd50k', 'DEVICE': device(type='cpu'), 'EMBEDDING_DIM':
        200, 'ENT_POS_FILTERED': True, 'EPOCHS': 401, 'EVAL_EVERY': 5, 'LEARNING_RATE': 0.0001,
        'MAX_QPAIRS': 15, 'MODEL_NAME': 'stare_transformer', 'CORRUPTION_POSITIONS': [0, 2],
        'SAVE': False, 'STATEMENT_LEN': -1, 'USE_TEST': True, 'WANDB': False, 'LABEL_SMOOTHING':
        0.1, 'SAMPLER_W_QUALIFIERS': True, 'OPTIMIZER': 'adam', 'CLEANED_DATASET': True,
        'GRAD_CLIPPING': True, 'LR_SCHEDULER': True, 'STAREARGS': {'LAYERS': 2, 'N_BASES': 0,
        'GCN_DIM': 200, 'GCN_DROP': 0.1, 'HID_DROP': 0.3, 'BIAS': False, 'OPN': 'rotate',
        'TRIPLE_QUAL_WEIGHT': 0.8, 'QUAL_AGGREGATE': 'sum', 'QUAL_OPN': 'rotate', 'QUAL_N':
        'sum', 'SUBBATCH': 0, 'QUAL_REPR': 'sparse', 'ATTENTION': False, 'ATTENTION_HEADS': 4,
        'ATTENTION_SLOPE': 0.2, 'ATTENTION_DROP': 0.1, 'HID_DROP2': 0.1, 'FEAT_DROP': 0.3,
        'N_FILTERS': 200, 'KERNEL_SZ': 7, 'K_W': 10, 'K_H': 20, 'T_LAYERS': 2, 'T_N_HEADS': 4,
        'T_HIDDEN': 512, 'POSITIONAL': True, 'POS_OPTION': 'default', 'TIME': False, 'POOLING':
        'avg'}, 'NUM_ENTITIES': 47156, 'NUM_RELATIONS': 532}        

        >>> id2e
        None
        """
    
        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.p_option = config['STAREARGS']['POS_OPTION']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class StarE_ObjectMask_Transformer(StarEEncoder):
    model_name = 'StarE_ObjectMask_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']

        self.object_mask_emb = torch.nn.Parameter(torch.randn(1, self.emb_dim,dtype=torch.float32),True)
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'], self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
        self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, e1_embed, rel_embed, obj_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        obj_embed = obj_embed.view(-1,1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, obj_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp        # 14, 128, 200

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)


        # bs*emb_dim , ......, bs*6*emb_dim
        object_mask = self.object_mask_emb.repeat(sub.shape[0], 1)
        ins = torch.zeros((sub.shape), dtype=torch.bool, device=self.device)
        mask = torch.cat((mask[:, :2], ins.unsqueeze(1), mask[:, 2:]), axis=1)

        stk_inp = self.concat(sub_emb, rel_emb, object_mask, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)[2] # to get the object position

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class StarE_Transformer_TripleBaseline(StarEEncoder):
    model_name = 'StarE_Transformer_Triple_Baseline'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent, mask = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)
        mask = mask[:, :2]

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class Transformer_Statements(StarEBase):
    """Baseline for Transformer decoder only model w/o starE encoder"""

    def __init__(self, config: dict):
        super().__init__(config)

        #self.emb_dim = config['EMBEDDING_DIM']
        self.entities = get_param((self.num_ent, self.emb_dim))
        self.relations = get_param((2 * self.num_rel, self.emb_dim))

        self.model_name = 'Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']

        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat
        self.device = config['DEVICE']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden,
                                                 config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)

        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2 * qual_rel_embed.shape[1],
                                                                    qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):

        sub_emb = torch.index_select(self.entities, 0, sub)
        rel_emb = torch.index_select(self.relations, 0, rel)

        quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
        quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
        qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
        qual_rel_emb = torch.index_select(self.relations, 0, quals_rels)
        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])


        # so we first initialize with False
        mask = torch.zeros((sub.shape[0], quals.shape[1] + 2)).bool().to(self.device)
        # and put True where qual entities and relations are actually padding index 0
        mask[:, 2:] = quals == 0

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, self.entities.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


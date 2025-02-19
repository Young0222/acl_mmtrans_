import json
import os

import torch
from torch.utils.data import Dataset

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import RobertaModel, RobertaTokenizerFast

from config import edges
import ttm_utils


class MMQLDataset(Dataset):
    def __init__(self, fold: int, mode: str, opt):
        super(MMQLDataset).__init__()
        
        self.mode = mode

        self.question = []
        self.question_padded_schema = []
        self.schema_occurrence_mask = []
        self.schema_occurrence = []
        self.ql = []

        def remove_skeleton(q):
            if '|' not in q:
                return q
            return '|'.join(q.split('|')[1:]).strip()

        dataset = ttm_utils.dataset()
        indices = dataset['schema']['indices']
        n_schema_items = len(indices)
        
        if mode == 'train':
            samples = [sample for sample in dataset['samples'] if sample['fold'] != fold]
        else:
            samples = [sample for sample in dataset['samples'] if sample['fold'] == fold]

        schema_sequence = dataset['schema']['sequence']
        
        for sample in samples:
            gt = ttm_utils.gt_of(sample['question'])
            self.question_padded_schema.append(sample['question'] + ' | ' + schema_sequence)
            self.question.append(sample['question'])
            self.schema_occurrence_mask.append([1 if i in gt['schema_occurrence'] else 0 for i in range(n_schema_items)])
            self.schema_occurrence.append([indices[i]['item'] if i in gt['schema_occurrence'] else '' for i in range(n_schema_items)])

            queries = {}
            for ql, key in ttm_utils.QL_TYPENAMES.items():
                if key in gt:
                    if opt.use_mir and (key + '_mir') in gt:
                        q = gt[key + '_mir']
                    else:
                        q = gt[key]
                else:
                    q = ''
                if not opt.use_skeleton:
                    q = remove_skeleton(q)
                queries[ql] = q
            self.ql.append(queries)

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, i):
        return self.question[i], self.question_padded_schema[i], self.schema_occurrence[i], self.schema_occurrence_mask[i], \
            self.ql[i]['AQL'], self.ql[i]['mir'], '', '', self.ql[i]['ECQL'], self.ql[i]['SQL++']


class SADataset(Dataset):
    def __init__(self, sa_fp, mode, opt):
        super(SADataset).__init__()

        self.opt = opt
        self.edges = edges
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.question = []
        self.question_embedding = []
        self.question_token_mask = []
        self.mix_question_embedding = []
        self.table_name_embedding_list = []
        self.column_info_embedding_list = []
        self.schema_occurrence_mask = []
        self.table_occurrence_mask = []
        self.column_occurrence_mask = []
        self.schema_occurrence = []
        self.token_info = []

        self.n_tables = 0
        self.column_numbers = []
        self.table_position_indices = []
        self.column_position_indices = []

        col_info = ttm_utils.dataset()['schema']['cols']
        
        self.n_tables = len(col_info)
        offset = 0
        for table, cols in col_info.items():
            self.column_numbers.append(len(cols))
            self.table_position_indices.append(offset)
            offset += 1
            for c in range(len(cols)):
                self.column_position_indices.append(offset)
                offset += 1
        
        sa_dataset = torch.load(sa_fp)

        self.schema = sa_dataset['schema']
        self.n_schema_items = sa_dataset['n_schema_items']
        self.schema_embedding = sa_dataset['schema_embedding']
        self.schema_token_mask = sa_dataset['schema_token_mask']
        self.schema_embedding_onehot = torch.eye(self.n_schema_items).cuda()
        
        for data in sa_dataset['questions']:
            self.question.append(data['question'])
            self.question_embedding.append(data['question_embedding'])
            self.question_token_mask.append(data['question_token_mask'])
            self.mix_question_embedding.append(data['mix_question_embedding'])
            self.table_name_embedding_list.append(data['table_name_embedding_list'])
            self.column_info_embedding_list.append(data['column_info_embedding_list'])

            self.schema_occurrence.append(data['schema_occurrence_tokens'])
            som = data['schema_occurrence_mask']
            self.schema_occurrence_mask.append(som)
            self.table_occurrence_mask.append([som[i] for i in self.table_position_indices])
            self.column_occurrence_mask.append([som[i] for i in self.column_position_indices])

            self.token_info.append((
                data['mix_token_ids'],
                data['mix_token_mask'],
                data['aligned_question_ids'],
                data['aligned_table_name_ids'],
                data['aligned_column_info_ids']
            ))

    def gen_schema_items(self, prediction_mask):
        return [item for item, pred in zip(self.schema, prediction_mask) if pred == 1]

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, i):
        return self.question[i], self.question_embedding[i], self.token_info[i],\
            self.question_token_mask[i], self.mix_question_embedding[i], self.table_name_embedding_list[i], self.column_info_embedding_list[i], \
            self.schema_occurrence[i], self.schema_occurrence_mask[i], self.table_occurrence_mask[i], self.column_occurrence_mask[i]
    
    

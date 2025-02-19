import os
import re
import sys
import json
import time
import torch

import torch.optim as optim
import transformers

from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.load_dataset import AQLDataset, AMDataset

from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluator
import ttm_utils
import config


class Trainer:
    def __init__(self, model_cls, opt):
        self.model_cls = model_cls
        self.opt = opt
        
        self.writer = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.optimizer_fn = None
        self.scheduler_fn = None
        self.n_schema_items = None
        self.schema_token_ids = None
        self.schema_attention_mask = None
        self.sa_mode = SAMode(self.opt.sa_mode)
        self.sa_model = None
        self.current_fold = 0

    @staticmethod
    def read_batch_data(batch):
        items = None
        for r in batch:
            if items is None:
                items = [[] for i in range(len(r))]
            for v, d in zip(r, items):
                d.append(v)
        return items

    def setup(self, i=0):
        opt = self.opt
        self.current_fold = i

        set_seed(opt.seed)
        print(opt)

        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

        collate = lambda x: x

        ds_cls = AQLDataset

        train_dataset = ds_cls(self.current_fold, 'train', opt)
        self.train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate, drop_last=False)
        valid_dataset = ds_cls(self.current_fold, 'valid', opt)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate, drop_last=False)

        num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
        num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
        num_checkpoint_steps = num_training_steps - 1

        self.optimizer_fn = lambda params: Adafactor(
            params,
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold=1.0,
            warmup_init=False
        )

        self.scheduler_fn = lambda optimizer: transformers.get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_training_steps
        )
    
    def gradient_descent(self, train_step):
        if self.scheduler is not None:
            self.scheduler.step()

        if train_step % self.opt.gradient_descent_step == 0:
            torch.nn.utils.clip_grad_norm_(self.sa_model.parameters_requiring_grad(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_prediction_summary(self, prediction, epoch, period):
        n = len(prediction)
        n_matched = sum(r['matched'] for r in prediction)
        n_first_matched = sum(r['first_matched'] for r in prediction)
        accuracy = n_matched / n
        first_accuracy = n_first_matched / n

        summary = {
            'n_queries': n,
            'n_matched': n_matched,
            'n_first_matched': n_first_matched,
            'accuracy': accuracy,
            'first_accuracy': first_accuracy,
            'prediction': prediction
        }
        save_path = f'predictions/{self.opt.exp_name}/fold-{self.current_fold}'
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/{epoch}-{period}.json', 'w') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        return n, n_matched, n_first_matched, accuracy, first_accuracy

    def validate(self, epoch, period):
        if period == 'train':
            dataloader = self.train_dataloader
        elif period == 'valid':
            dataloader = self.valid_dataloader
        prediction = []
        for i, batch in enumerate(dataloader):
            batch = self.read_batch_data(batch)  # 分解各个member
            question, question_padded_schema, schema_occurrence, schema_occurrence_mask, aql, mir, _, _, ecql, sqlpp = batch
            b = len(question)

            with torch.no_grad():
                loss, model_outputs = self.sa_model.forward_translation(batch, 'valid')

            model_outputs = model_outputs.view(b, self.opt.num_return_sequences, model_outputs.shape[1])

            prediction += evaluator.evaluate_prediction(question, model_outputs, self.opt.ql, self.sa_model.tokenizer, mir=opt.use_mir)
        
        prediction = sorted(prediction, key=lambda p: p['id'])
        n, n_matched, n_first_matched, accuracy, first_accuracy = self.save_prediction_summary(prediction, epoch, period)

        print(f'Match {n_matched} of {n} queries, accuracy {accuracy:.5f}.')
        print(f'First match {n_first_matched} of {n} queries, first accuracy {first_accuracy:.5f}.')

        return first_accuracy, prediction

    def test_translation(self):
        self.sa_model = self.model_cls(self.opt)
        if self.sa_mode.use_sa:
            self.sa_model.load_best_sa_model()
        self.sa_model = self.sa_model.cuda()

        with self.sa_model.eval_mode():
            for period in ['train', 'valid']:
                self.validate(self.opt.eval_model_suffix, period)

    def train_schema_prob(self, fi=0, fk=0):
        self.sa_model = self.model_cls(self.opt).cuda()
        self.sa_model.train()

        def train_translation():
            self.sa_model.current_training_period = 'TRANSLATION'
            self.optimizer = self.optimizer_fn(self.sa_model.parameters_requiring_grad())
            self.scheduler = self.scheduler_fn(self.optimizer)
            best_train_accuracy = 0.0
            best_valid_accuracy = 0.0
            train_step = 0
            for epoch in range(opt.epochs):
                print(f'[Fold {fi} / {fk}] Train Translation epoch {epoch + 1}.', end='\t')

                for i, batch in enumerate(self.train_dataloader):
                    train_step += 1
                    batch = self.read_batch_data(batch)  # 分解各个member
                    
                    loss, *_ = self.sa_model.forward_translation(batch, 'train')
                    if i == 0:
                        print(f'Loss: {loss.item():.4f}', flush=True)

                    loss.backward()
                    self.gradient_descent(train_step)

                if (epoch + 1) % self.opt.validation_steps == 0:
                    print(f'Validate model after {epoch + 1} epochs.')
                    with self.sa_model.eval_mode():
                        print('On train dataset.')
                        train_accuracy, _ = self.validate(epoch + 1, 'train')
                        print('On valid dataset.')
                        valid_accuracy, _ = self.validate(epoch + 1, 'valid')
                        if valid_accuracy > best_valid_accuracy:
                            print(f'New highest accuracy, saving model.')
                            if valid_accuracy >= 0.4:
                                print(f'New highest accuracy, saving model.')
                                self.sa_model.save_model(self.current_fold, epoch + 1)
                            else:
                                print(f'New highest accuracy, not saving at early epochs')
                            best_valid_accuracy = valid_accuracy
                            best_train_accuracy = train_accuracy
            return best_train_accuracy, best_valid_accuracy

        start = time.time()
        best_train_accuracy, best_valid_accuracy = train_translation()
        translation_train_end = time.time()
        print(f'Translation training time: {(translation_train_end - start) / 60:.1f} min.')
        print(f'[Fold {fi} / {fk}] Best valid accuracy: {best_valid_accuracy:.5f}')

        return best_train_accuracy, best_valid_accuracy


    def train_k_fold(self):
        total_train_accuracy = 0
        total_valid_accuracy = 0
        n_train_samples = 0
        n_valid_samples = 0
        for i in range(1, self.opt.nfolds + 1):
            print('-' * 20 + f' Fold {i} ' + '-' * 20)
            self.setup(i)
            best_train_accuracy, best_valid_accuracy = self.train_schema_prob(i, self.opt.nfolds)
            total_train_accuracy += best_train_accuracy * len(self.train_dataloader)
            total_valid_accuracy += best_valid_accuracy * len(self.valid_dataloader)
            n_train_samples += len(self.train_dataloader)
            n_valid_samples += len(self.valid_dataloader)

        print('-' * 40)
        print(f'Average train accuracy: {total_train_accuracy / n_train_samples:.5f}')
        print(f'Average valid accuracy: {total_valid_accuracy / n_valid_samples:.5f}')

    def train(self):
        self.train_k_fold()


if __name__ == "__main__":
    opt = config.Option(
        model_cls='NoS',
        device='1',
        dataset='IMDBSQL++',
        ql='SQL++',
        epochs=1500,
        use_mir=False,
        use_skeleton=False,
        method='Bart-large',
        model_name_or_path='./pretrained_models/bart-large'
    )
    ttm_utils.total_opt = opt
    model_cls = {
        'AllS': PLMPlain,
        'SPG': PLMSAWithSchemaItemNames,
        'NoS': PLMPlainNoSchema
    }
    with open(opt.output_path, 'a') as f:
        sys.stdout = f
        trainer = Trainer(model_cls[opt.model_cls], opt)
        trainer.train()


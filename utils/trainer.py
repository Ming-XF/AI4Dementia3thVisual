import json
import os
from timeit import default_timer as timer
# import wandb
import logging
import torch
import numpy as np
from abc import abstractmethod
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

from config import init_model_config
from .optimizer import init_optimizer
from .schedule import init_schedule
from .accuracy import accuracy
from data import *

import pickle

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        self.task_id = task_id
        self.args = args
        self.local_rank = local_rank
        self.subject_id = subject_id
        self.data_config = DataConfig(args)
        self.data_loaders = self.load_datasets()

        model, self.model_config = init_model_config(args, self.data_config)
        if args.do_parallel:
            # self.model = torch.nn.DataParallel(self.model)
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
        # self.model = torch.compile(model, dynamic=True)

        self.optimizer = None
        self.scheduler = None

        self.best_result = None
        self.test_result = None

    @abstractmethod
    def prepare_inputs_kwargs(self, inputs):
        return {}

    def load_datasets(self):
        # datasets = eval(
        #     f"load_{self.args.dataset}_data")(self.data_config)
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def init_components(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        self.optimizer = init_optimizer(self.model, self.args)
        self.scheduler = init_schedule(self.optimizer, self.args, total)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss

            if self.data_config.dataset == "ZuCo":
                loss.backward()
                if step % self.data_config.batch_size == self.data_config.batch_size - 1:

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(), 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs*len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.5 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.5
            self.test_result = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_result is None or self.best_result['Accuracy'] <= self.test_result['Accuracy']:
                self.best_result = self.test_result
                self.save_model()
        # wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

    def evaluate(self):
        if self.data_config.num_class == 2:
            result = self.binary_evaluate()
        else:
            result = self.multiple_evaluate()
        return result

    def binary_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = []
        acc = []


        ts_datas = {
            0: [],
            1: [],
            2: [],
            3: [],
        }
        ori_datas = {
            0: [],
            1: [],
            2: [],
            3: [],
        }
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)

                ts_data = outputs.logits
                ori_data = outputs.loss

                for y in ts_datas.keys():
                    ts_datas[y].extend(ts_data[y])
                    ori_datas[y].extend(ori_data[y])

            for y in ts_datas.keys():
                if len(ts_datas[y]) > 0:
                    ts_datas[y] = np.concatenate(ts_datas[y])
                    ori_datas[y] = np.concatenate(ori_datas[y])

            # pdb.set_trace()
            data = [ts_datas, ori_datas]
            with open('data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("保存成功: 0样本{},1样本{},2样本{},3样本{}".format(len(ts_datas[0]), len(ts_datas[1]), len(ts_datas[2]), len(ts_datas[3])))
        return
                # loss = outputs.loss
                # losses += loss.item()
                # loss_list.append(loss.item())
                # # print(f"Evaluate loss: {loss.item():.5f}")

                # # top1 = accuracy(outputs.logits, input_kwargs['labels'][:, 1])[0]
                # # acc.append([top1*input_kwargs['labels'].shape[0], input_kwargs['labels'].shape[0]])
                # preds += F.softmax(outputs.logits, dim=1)[:, 1].tolist()
                # labels += input_kwargs['labels'][:, 1].tolist()
            
            # acc = np.array(acc).sum(axis=0)
            # result['Accuracy'] = acc[0] / acc[1]
        #     result['AUC'] = roc_auc_score(labels, preds)
        #     preds, labels = np.array(preds), np.array(labels)
        #     preds[preds > 0.5] = 1
        #     preds[preds <= 0.5] = 0
        #     result['Accuracy'] = (preds == labels).sum() / len(labels)
        #     metric = precision_recall_fscore_support(
        #         labels, preds, average="binary")
        #     result['Precision'] = metric[0]
        #     result['Recall'] = metric[1]
        #     result['F_score'] = metric[2]

        #     report = classification_report(
        #         labels, preds, output_dict=True, zero_division=0)

        #     recall = [0, 0]
        #     for k, v in report.items():
        #         if '.' in k:
        #             recall[int(float(k))] = v['recall']

        #     result['Specificity'] = recall[0]
        #     result['Sensitivity'] = recall[1]
        #     result['Loss'] = losses / len(loss_list)
        # if self.args.within_subject:
        #     print(f'\nTest{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
        #           f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
        #           f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        # else:
        #     print(f'\nTest{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
        #           f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
        #           f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        # for k, v in result.items():
        #     if v is not None:
        #         logger.info(f"{k}: {v:.5f}")
        # # wandb.log(result)
        # return result

    def multiple_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = None

        connect1s = []
        connect2s = []
        yss = []
        subids = []
        cnns = []

        mu1s = []
        mu2s = []
        mu3s = []
        logvar1s = []
        logvar2s = []
        logvar3s = []
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)

                con1s, con2s, ys, subject_id, cnn = outputs.logits
                connect1s.append(con1s)
                connect2s.append(con2s)
                yss.append(ys)
                subids.append(subject_id)
                cnns.append(cnn)

                #保存可视化数据
                mu1, mu2, mu3, logvar1, logvar2, logvar3 = outputs.loss
                mu1s.append(mu1)
                mu2s.append(mu2)
                mu3s.append(mu3)
                logvar1s.append(logvar1)
                logvar2s.append(logvar2)
                logvar3s.append(logvar3)

            # pdb.set_trace()
            connect1s = np.concatenate(connect1s, axis=0)
            connect2s = np.concatenate(connect2s, axis=0)
            yss = np.concatenate(yss, axis=0)
            subids = np.concatenate(subids, axis=0)
            cnns = np.concatenate(cnns, axis=0)

            mu1s = np.concatenate(mu1s, axis=0)
            mu2s = np.concatenate(mu2s, axis=0)
            mu3s = np.concatenate(mu3s, axis=0)
            logvar1s = np.concatenate(logvar1s, axis=0)
            logvar2s = np.concatenate(logvar2s, axis=0)
            logvar3s = np.concatenate(logvar3s, axis=0)
            weight = self.model.dense3.weight.detach().cpu().numpy()
            bias = self.model.dense3.bias.detach().cpu().numpy()

            # pdb.set_trace()
            data = [connect1s, connect2s, yss, subids, cnns]
            with open('data.pkl', 'wb') as f:
                pickle.dump(data, f)
            unique_labels, label_counts = np.unique(yss, return_counts=True)
            

            #保存可视化数据
            data2 = [mu1s, mu2s, mu3s, logvar1s, logvar2s, logvar3s, yss, weight, bias]
            with open('data2.pkl', 'wb') as f:
                pickle.dump(data2, f)

            print("标签统计结果：")
            print("=" * 40)
            for label, count in zip(unique_labels, label_counts):
                print(f"标签 {label}: {count} 个样本")
            
        return
                
            #     loss = outputs.loss
            #     losses += loss.item()
            #     loss_list.append(loss.item())
            #     # print(f"Evaluate loss: {loss.item():.5f}")
            #     if preds is None:
            #         preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
            #     else:
            #         preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
            #     labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            # try:
            #     result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            # except:
            #     result['AUC'] = 0.
            # preds = preds.argmax(axis=1).tolist()
            # result['Accuracy'] = accuracy_score(labels, preds)
            # preds, labels = np.array(preds), np.array(labels)
            # metric = precision_recall_fscore_support(
            #     labels, preds, average='macro')
            # result['Precision'] = metric[0]
            # result['Recall'] = metric[1]
            # result['F_score'] = metric[2]

            # report = classification_report(
            #     labels, preds, output_dict=True, zero_division=0)
            #
            # recall = [0, 0]
            # for k, v in report.items():
            #     if '.' in k:
            #         recall[int(float(k))] = v['recall']
            #
            # result['Specificity'] = recall[0]
            # result['Sensitivity'] = recall[1]
            # result['Loss'] = losses / len(loss_list)
        # if self.args.within_subject:
        #     print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        # else:
        #     print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        
        # if self.args.within_subject:
        #     print(f'\nTest{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
        #           f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
        #           , end=',')
        # else:
        #     print(f'\nTest{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, Precision:{result["Precision"]:.5f}, '
        #           f'AUC:{result["AUC"]:.5f}, Recall:{result["Recall"]:.5f}, F_score:{result["F_score"]:.5f}, '
        #           , end=',')
        
        # for k, v in result.items():
        #     if v is not None:
        #         logger.info(f"{k}: {v:.5f}")
        # # wandb.log(result)
        # return result

    def save_model(self):
        # Save model checkpoint (Overwrite)
        path = os.path.join(self.args.model_dir, self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = self.model.module if self.args.do_parallel else self.model
        torch.save(model_to_save, os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

        # Save training arguments together with the trained model
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", path)

    def load_model(self):
        path = os.path.join(self.args.model_dir, self.args.model + "_" + self.args.dataset)
        path = os.path.join(path, f'{self.args.model}-2.bin')
        if not os.path.exists(path):
            logger.info("Model doesn't exists! Train first!")
            print("Model doesn't exists! Train first!")
            return
            # raise Exception("Model doesn't exists! Train first!")

        if self.args.do_parallel:
            self.model = self.model.to(self.args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
        else:
            self.model = torch.load(os.path.join(".", path), weights_only=False)
            self.model.to(self.device)
        logger.info("***** Model Loaded *****")
        print("***** Model Loaded *****")

    def visualize(self):
        self.model.eval()
        inputs = (torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.time_series_size)),
                  torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.node_size)),
                  F.one_hot(torch.randint(0, self.model_config.num_classes, (self.data_config.batch_size,))))
        input_kwargs = self.prepare_inputs_kwargs(inputs)
        # save_path = os.path.join(self.args.model_dir, self.args.model, 'model.onnx')
        self.model.config.dict_output = False
        torch.onnx.export(self.model,
                          tuple([v for k, v in input_kwargs.items()]),
                          'model.onnx')
        # wandb.save('model.onnx')
        self.model.config.dict_output = True

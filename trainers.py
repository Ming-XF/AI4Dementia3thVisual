import numpy as np
import pywt
import torch
from einops import repeat, rearrange
from scipy import signal
import numpy as np

from utils import *

import pdb

class DFaSTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DFaSTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                time_series, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "labels": labels.float().to(self.device)}


class DFaSTOnlySpatialTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DFaSTOnlySpatialTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class FaSPTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FaSPTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class LMDATrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(LMDATrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class ShallowConvNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(ShallowConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class DeepConvNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DeepConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BNTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BNTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class FBNetGenTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FBNetGenTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        time_series_size = time_series.shape[-1] // self.model_config.window_size * self.model_config.window_size
        time_series = time_series[:, :, :time_series_size]
        node_feature = inputs['correlation']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class BrainNetCNNTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainNetCNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class TransformerTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TransformerTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class STAGINTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(STAGINTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        dyn_a, sampling_points = process_dynamic_fc(inputs['time_series'].transpose(1, 2),
                                                    self.model_config.window_size,
                                                    self.model_config.window_stride,
                                                    # self.model_config.dynamic_length,
                                                    self.model_config.sampling_init)
        sampling_endpoints = [p + self.model_config.window_size for p in sampling_points]
        dyn_v = repeat(torch.eye(self.model_config.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                       b=self.args.batch_size)
        if len(dyn_a) < self.args.batch_size:
            dyn_v = dyn_v[:len(dyn_a)]
        return {'v': dyn_v.to(self.device),
                'a': dyn_a.to(self.device),
                't': inputs['time_series'].permute(2, 0, 1).to(self.device),
                'sampling_endpoints': sampling_endpoints,
                'labels': inputs['labels'].float().to(self.device)}

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)


class EEGNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha,
                # beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class EEGChannelNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGChannelNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class RACNNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(RACNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        # time_series = self.get_complex_morlet_wavelets(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def get_complex_morlet_wavelets(time_series):
        time_series = time_series.numpy()
        Fa = np.arange(4, 31)
        new_time_series = []
        for i, ts in enumerate(time_series):
            cwt = abs(pywt.cwt(ts, Fa, 'cmor1-1', 1/200)[0])
            new_time_series.append(torch.from_numpy(cwt))
        time_series = torch.stack(new_time_series, dim=0)
        time_series -= time_series.mean()
        time_series /= time_series.std()
        return time_series


class SBLESTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SBLESTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        self.W = None
        self.Wh = None

    def load_datasets(self):
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        labels = ((labels[:, 1] == 1) * 2 - 1)
        idx0 = torch.argwhere(labels == -1)
        idx1 = torch.argwhere(labels == 1)
        idx0 = idx0[:len(idx1)]
        idx = torch.concat([idx0, idx1], dim=0).squeeze()
        time_series = time_series[idx]
        labels = labels[idx].unsqueeze(-1)
        time_series = time_series.permute(1, 2, 0)

        return {"time_series": time_series.double().to(self.device),
                "labels": labels.double().to(self.device)}

    def train(self):
        train_dataloader = self.data_loaders['train']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in train_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
            # break

        self.W, alpha, V, self.Wh = self.model(**inputs)
        self.best_result = self.test_result = self.evaluate()
        # wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

    def evaluate(self):
        test_dataloader = self.data_loaders['test']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in test_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
        R_test, _ = self.model.enhance_conv(inputs["time_series"], self.Wh)
        vec_W = self.W.T.flatten()  # vec operation (Torch)
        preds = R_test @ vec_W
        result = self.metrix(preds, inputs["labels"])
        # wandb.log(result)
        return result

    @staticmethod
    def metrix(predict_Y, Y_test):
        """Compute classification accuracy for test set"""

        predict_Y = predict_Y.cpu().numpy()
        Y_test = torch.squeeze(Y_test).cpu().numpy()
        total_num = len(predict_Y)
        error_num = 0
        auc = roc_auc_score(Y_test*0.5+0.5, predict_Y*0.5+0.5)
        # Compute classification accuracy for test set
        Y_predict = np.zeros(total_num)
        for i in range(total_num):
            if predict_Y[i] > 0:
                Y_predict[i] = 1
            else:
                Y_predict[i] = -1

        # Compute classification accuracy
        for i in range(total_num):
            if Y_predict[i] != Y_test[i]:
                error_num = error_num + 1

        acc = (total_num - error_num) / total_num

        report = classification_report(
            Y_test * 0.5 + 0.5, Y_predict * 0.5 + 0.5, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k, v in report.items():
            if '.' in k:
                recall[int(float(k))] = v['recall']
        specificity = recall[0]
        sensitivity = recall[1]

        result = {"Accuracy": acc,
                  "AUC": auc,
                  "Specificity": specificity,
                  "Sensitivity": sensitivity}
        return result


class TCANetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCANetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_local_and_top.item()
            loss_list.append(loss_local_and_top.item())
            # wandb.log({'Training loss': loss_local_and_top.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)


class TCACNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCACNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        wpser = self.wpser(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, wpser, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, wpser, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "node_feature": wpser.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def wpser(time_series):
        fs = 200
        lowcut = 8
        highcut = 30
        order = 4
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', output='ba')
        time_series = signal.filtfilt(b, a, time_series)
        _, n, _ = time_series.shape
        coeffs = pywt.wavedec(time_series, 'db4', level=5)
        energy = np.array([np.square(level).sum(-1) for level in coeffs])
        wpser = energy / np.repeat(np.expand_dims(energy.sum(-1), -1), n, -1)
        wpser = wpser.sum(0)
        wpser = torch.from_numpy(wpser).float()
        return wpser

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_global_model.item()
            loss_list.append(loss_global_model.item())
            # wandb.log({'Training loss': loss_global_model.item(),
                       # 'Training local loss': loss_local_and_top.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)


class SteadyNetTrainer(DFaSTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SteadyNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
    

class MTSTATrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(MTSTATrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        dyn_a, dyn_f_a, dyn_p_a, sampling_points = process_dynamic_muti_type_fc(inputs['time_series'].transpose(1, 2),
                                                    self.model_config.window_size,
                                                    self.model_config.window_stride,
                                                    # self.model_config.dynamic_length,
                                                    self.model_config.sampling_init)
        sampling_endpoints = [p + self.model_config.window_size for p in sampling_points]
        dyn_v = repeat(torch.eye(self.model_config.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                       b=self.args.batch_size)
        if len(dyn_a) < self.args.batch_size:
            dyn_v = dyn_v[:len(dyn_a)]
        return {'v': dyn_v.to(self.device),
                'a': dyn_a.to(self.device),
                'f_a': dyn_f_a.to(self.device),
                'p_a': dyn_p_a.to(self.device),
                't': inputs['time_series'].permute(2, 0, 1).to(self.device),
                'sampling_endpoints': sampling_endpoints,
                'labels': inputs['labels'].float().to(self.device)}

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            
            
class VAESTATrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(VAESTATrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        
    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)

            
class ALTERTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(ALTERTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}
        

        
class BrainVAETrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainVAETrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        
    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']
        return {"time_series": time_series.to(self.device),
                "node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device)}

    def train_epoch(self, epoch):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            # if self.best_result is None or self.best_result['AUC'] <= self.test_result['AUC']:
            #     self.best_result = self.test_result
            #     self.save_model()
            
            # self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            
class STWeightTrainer(BrainVAETrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(STWeightTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
            
class EESTWTrainer(BrainVAETrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EESTWTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

class SingleEncoderBVAETrainer(BrainVAETrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SingleEncoderBVAETrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

class CVIBTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(CVIBTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        
    def prepare_inputs_kwargs(self, inputs, r1=None, r2=None, r3=None, train=None):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        mu1 = []
        logvar1 = []
        mu2 = []
        logvar2 = []
        mu3 = []
        logvar3 = []
        if r1 is not None:
            class1_means = r1[0]
            class1_logvars = r1[1]

            class2_means = r2[0]
            class2_logvars = r2[1]

            class3_means = None
            class3_logvars = None
            if r3 is not None:
                class3_means = r3[0]
                class3_logvars = r3[1]
            
            for label in labels:
                mu1.append(torch.tensor(class1_means[int(np.argmax(label))]).unsqueeze(0))
                logvar1.append(torch.tensor(class1_logvars[int(np.argmax(label))]).unsqueeze(0))

                mu2.append(torch.tensor(class2_means[int(np.argmax(label))]).unsqueeze(0))
                logvar2.append(torch.tensor(class2_logvars[int(np.argmax(label))]).unsqueeze(0))

                if r3 is not None:
                    mu3.append(torch.tensor(class3_means[int(np.argmax(label))]).unsqueeze(0))
                    logvar3.append(torch.tensor(class3_logvars[int(np.argmax(label))]).unsqueeze(0))

                
            mu1 = torch.cat(mu1, dim=0)
            logvar1 = torch.cat(logvar1, dim=0)
            mu2 = torch.cat(mu2, dim=0)
            logvar2 = torch.cat(logvar2, dim=0)
            if r3 is not None:
                mu3 = torch.cat(mu3, dim=0)
                logvar3 = torch.cat(logvar3, dim=0)
            else:
                mu3 = None
                logvar3 = None

        
        return {"time_series": time_series.to(self.device),
                "node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device),
                "r1_mu": mu1.to(self.device) if len(mu1) > 0 else None,
                "r1_logvar": logvar1.to(self.device) if len(logvar1) > 0 else None,
                "r2_mu": mu2.to(self.device) if len(mu2) > 0 else None,
                "r2_logvar": logvar2.to(self.device) if len(logvar2) > 0 else None,
                "r3_mu": mu3.to(self.device) if mu3 is not None and len(mu3) > 0 else None,
                "r3_logvar": logvar3.to(self.device) if logvar3 is not None and len(logvar3) > 0 else None,
                "train": True if train else False}

    def train_epoch(self, epoch, r1=None, r2=None, r3=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        z1s = []
        z2s = []
        z3s = []
        ys = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, r1, r2, r3, True)
            outputs, z1, z2, z3 = self.model(**input_kwargs)
            y = input_kwargs['labels']
            z1s.append(z1.detach().cpu().numpy())
            z2s.append(z2.detach().cpu().numpy())
            if z3 is not None:
                z3s.append(z3.detach().cpu().numpy())
            ys.append(np.argmax(y.detach().cpu().numpy(), axis=1))
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            # if self.check_gradient_explosion(self.model, threshold=1e6):
            #     print("检测到梯度爆炸，中断训练")
            #     break
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})
        z1s = np.concatenate(z1s, axis=0)
        z2s = np.concatenate(z2s, axis=0)
        if len(z3s) > 0:
            z3s = np.concatenate(z3s, axis=0)
        else:
            z3s = None
        ys = np.concatenate(ys, axis=0)

        return losses / len(loss_list), z1s, z2s, z3s, ys
        
    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()

        z1s = None
        z2s = None
        z3s = None
        ys = None
        r1 = None
        r2 = None
        r3 = None
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            if z1s is not None:
                # if epoch % 5 == 0 or epoch == 2:
                #     r = self.compute_class_mean_logvar(zs, ys)
                r1 = self.compute_class_mean_logvar(z1s, ys)
                r2 = self.compute_class_mean_logvar(z2s, ys)
                if z3s is not None:
                    r3 = self.compute_class_mean_logvar(z3s, ys)
                else:
                    r3 = None
                train_loss, z1s, z2s, z3s, ys = self.train_epoch(epoch, r1, r2, r3)
            else:
                train_loss, z1s, z2s, z3s, ys = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            # if self.best_result is None or self.best_result['AUC'] <= self.test_result['AUC']:
            #     self.best_result = self.test_result
            #     self.save_model()
            
            # self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)

    
    def compute_class_mean_logvar(self, zs, ys):
        """
        计算每个类别的z的均值和方差
        
        参数:
        zs: shape (N, L, C, D)
        ys: shape (N,) 类别标签
        
        返回:
        class_means: 字典，键为类别，值为该类别z的均值数组 (L, C, D)
        class_vars: 字典，键为类别，值为该类别z的方差数组 (L, C, D)
        """
        # 获取所有唯一的类别标签
        unique_classes = np.unique(ys)
        
        # 初始化字典存储结果
        class_means = {}
        class_logvars = {}
        
        for cls in unique_classes:
            # 获取该类别对应的所有z
            mask = (ys == cls)
            class_z = zs[mask]  # shape (n_cls, L, C, D)，其中n_cls是该类别的样本数
            
            if len(class_z) == 0:
                continue
                
            # 计算均值
            mean_val = np.mean(class_z, axis=0)  # shape (L, C, D)
            
            # 计算方差，使用无偏估计（ddof=1）
            var_val = np.var(class_z, axis=0, ddof=1)  # shape (L, C, D)

            epsilon = 1e-8
            logvar = np.log(var_val + epsilon)
            
            class_means[cls] = mean_val
            class_logvars[cls] = logvar
        
        return class_means, class_logvars

    def check_gradient_explosion(self, model, threshold=1e6):
        """检查梯度是否爆炸"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > threshold:
            print(f"⚠️ 梯度爆炸! 总梯度范数: {total_norm:.2e} > {threshold:.0e}")
            return True
        return False

class SrCVIBTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SrCVIBTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        
    def prepare_inputs_kwargs(self, inputs, r=None, train=None):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']
        subject_id = inputs['subject_id']

        mu = []
        logvar = []
        if r is not None:
            class_means = r[0]
            class_logvars = r[1]

            for label in labels:
                mu.append(torch.tensor(class_means[int(np.argmax(label))]).unsqueeze(0))
                logvar.append(torch.tensor(class_logvars[int(np.argmax(label))]).unsqueeze(0))

                
            mu = torch.cat(mu, dim=0)
            logvar = torch.cat(logvar, dim=0)

        
        return {"time_series": time_series.to(self.device),
                "node_feature": node_feature.to(self.device),
                "labels": labels.float().to(self.device),
                "subject_id": subject_id.int().to(self.device),
                "r_mu": mu.to(self.device) if len(mu) > 0 else None,
                "r_logvar": logvar.to(self.device) if len(logvar) > 0 else None,
                "train": True if train else False}

    def train_epoch(self, epoch, r=None):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []
        zs = []
        ys = []
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, r, True)
            outputs, z = self.model(**input_kwargs)
            y = input_kwargs['labels']
            zs.append(z.detach().cpu().numpy())
            ys.append(np.argmax(y.detach().cpu().numpy(), axis=1))
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            # if self.check_gradient_explosion(self.model, threshold=1e6):
            #     print("检测到梯度爆炸，中断训练")
            #     break
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            # wandb.log({'Training loss': loss.item(),
                       # 'Learning rate': self.optimizer.param_groups[0]['lr']})
        zs = np.concatenate(zs, axis=0)
        ys = np.concatenate(ys, axis=0)

        return losses / len(loss_list), zs, ys
        
    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()

        zs = None
        ys = None
        r = None
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            if zs is not None:
                # if epoch % 5 == 0 or epoch == 2:
                #     r = self.compute_class_mean_logvar(zs, ys)
                r = self.compute_class_mean_logvar(zs, ys)
                train_loss, zs, ys = self.train_epoch(epoch, r)
            else:
                train_loss, zs, ys = self.train_epoch(epoch)
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            # pdb.set_trace()
            if self.best_result is None or self.best_result['F_score'] <= self.test_result['F_score']:
                self.best_result = self.test_result
                self.save_model()
            
            # self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)

    
    def compute_class_mean_logvar(self, zs, ys):
        """
        计算每个类别的z的均值和方差
        
        参数:
        zs: shape (N, L, C, D)
        ys: shape (N,) 类别标签
        
        返回:
        class_means: 字典，键为类别，值为该类别z的均值数组 (L, C, D)
        class_vars: 字典，键为类别，值为该类别z的方差数组 (L, C, D)
        """
        # 获取所有唯一的类别标签
        unique_classes = np.unique(ys)
        
        # 初始化字典存储结果
        class_means = {}
        class_logvars = {}
        
        for cls in unique_classes:
            # 获取该类别对应的所有z
            mask = (ys == cls)
            class_z = zs[mask]  # shape (n_cls, L, C, D)，其中n_cls是该类别的样本数
            
            if len(class_z) == 0:
                continue
                
            # 计算均值
            mean_val = np.mean(class_z, axis=0)  # shape (L, C, D)
            
            # 计算方差，使用无偏估计（ddof=1）
            var_val = np.var(class_z, axis=0, ddof=1)  # shape (L, C, D)

            epsilon = 1e-8
            logvar = np.log(var_val + epsilon)
            
            class_means[cls] = mean_val
            class_logvars[cls] = logvar
        
        return class_means, class_logvars

    # def check_gradient_explosion(self, model, threshold=1e6):
    #     """检查梯度是否爆炸"""
    #     total_norm = 0
    #     for p in model.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5
        
    #     if total_norm > threshold:
    #         print(f"⚠️ 梯度爆炸! 总梯度范数: {total_norm:.2e} > {threshold:.0e}")
    #         return True
    #     return False

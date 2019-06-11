"""
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
"""
import os
from argparse import ArgumentParser
import shutil
from pathlib import Path
from typing import List, Tuple

import mir_eval
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from tensorboardX import SummaryWriter
from torch import Tensor
from torchsummary import summary
from tqdm import tqdm

import data_manager
from adamwr import AdamW, CosineLRWithRestarts
from hparams import hparams
from models import UNet
from utils import draw_lineplot, print_to_file


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams, train_size: int, num_classes: int, class_weight: Tensor):
        # model, criterion, and prediction
        ch_out = num_classes if num_classes > 2 else 1
        self.model = UNet(ch_in=2, ch_out=ch_out, **hparams.model)
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss(reduction='none')
        self.class_weight = class_weight

        # for prediction
        self.frame2time = hparams.hop_size / hparams.sample_rate
        self.T_6s = round(6 / self.frame2time) - 1
        self.T_12s = round(12 / self.frame2time) - 1
        self.metrics = ('precision', 'recall', 'F1', 'mean', 'std')

        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(),
                               lr=hparams.learning_rate,
                               weight_decay=hparams.weight_decay,
                               )
        self.scheduler = CosineLRWithRestarts(self.optimizer,
                                              batch_size=hparams.batch_size,
                                              epoch_size=train_size,
                                              **hparams.scheduler
                                              )
        self.scheduler.step()

        # TODO: variables for stopping criterion
        # self.stop_thr = hparams.stop_thr
        # self.stop_thr_acc = hparams.stop_thr_acc
        # self.loss_last_restart = 1000
        # self.acc_last_restart = 0

        # device
        device_for_summary = self.init_device(hparams.device, hparams.out_device)

        # summary
        self.writer = SummaryWriter(logdir=hparams.logdir)
        path_summary = Path(self.writer.logdir, 'summary.txt')
        if not path_summary.exists():
            print_to_file(path_summary,
                          summary,
                          (self.model, (2, 128, 16 * hparams.model['stride'][1]**4)),
                          dict(device=device_for_summary)
                          )

        # save hyperparameters
        path_hparam = Path(self.writer.logdir, 'hparams.txt')
        if not path_hparam.exists():
            with path_hparam.open('w') as f:
                for var in vars(hparams):
                    value = getattr(hparams, var)
                    print(f'{var}: {value}', file=f)

    def init_device(self, device, out_device) -> str:
        if device == 'cpu':
            self.device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return 'cpu'

        # device type
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) == int:
                device = device
            else:
                device = [int(d[-1]) for d in device]

        # out_device type
        if type(out_device) == int:
            out_device = torch.device(f'cuda:{out_device}')
        else:
            out_device = torch.device(out_device)

        self.device = torch.device(f'cuda:{device[0]}')
        self.out_device = out_device

        if len(device) > 1:
            self.model = nn.DataParallel(self.model,
                                         device_ids=device,
                                         output_device=out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])
        else:
            self.str_device = str(self.device)

        self.model.cuda(device[0])
        self.criterion.cuda(out_device)
        if self.sigmoid:
            self.sigmoid.cuda(device[0])
        torch.cuda.set_device(device[0])
        return 'cuda'

    def calc_loss(self, y: Tensor, out: Tensor, Ts: List[int]):
        loss = torch.zeros(1, device=self.out_device)
        weight = (y > 0).float() * self.class_weight[1].item()
        weight += (y == 0).float() * self.class_weight[0].item()
        for ii, T in enumerate(Ts):
            loss_no_red = self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])
            loss += (loss_no_red * weight[ii:ii + 1, :T]).sum() / T

        return loss

    def predict(self, out: ndarray, Ts: List[int]) -> Tuple[List[ndarray], List]:
        """ peak-picking prediction

        :param out: (B, T) or (T,)
        :param Ts: length B list
        :return: length B list of boundary index ndarrays
        """
        if out.ndim == 1:
            out = [out]

        boundaries = []
        thresholds = []
        for item, T in zip(out, Ts):
            # candid_val = []
            candid_idx = []
            for idx in range(1, T - 1):
                i_first = max(idx - self.T_6s, 0)
                i_last = min(idx + self.T_6s + 1, T)
                if item[idx] >= np.amax(item[i_first:i_last]):
                    # candid_val.append(item[i])
                    candid_idx.append(idx)

            boundary_idx = []
            threshold = np.mean(item[candid_idx])
            for idx in candid_idx:
                if item[idx] - threshold > 0:
                    boundary_idx.append(idx)

            boundary_interval = np.array([[0] + boundary_idx,
                                          boundary_idx + [T]], dtype=np.float64).T
            boundary_interval *= self.frame2time

            boundaries.append(boundary_interval)
            thresholds.append(threshold)

        return boundaries, thresholds

    @staticmethod
    def evaluate(reference: List[ndarray], prediction: List[ndarray], out_np: ndarray):
        result = np.zeros(5)
        for item_truth, item_pred, item_out in zip(reference, prediction, out_np):
            mir_result = mir_eval.segment.detection(item_truth, item_pred, trim=True)
            result += np.array([*mir_result, item_out.mean(), item_out.std()])

        return result

    # Running model for train, test and validation.
    def run(self, dataloader, mode: str, epoch: int):
        self.model.train() if mode == 'train' else self.model.eval()
        if mode == 'test':
            self.model.load_state_dict(
                torch.load(
                    Path(self.writer.logdir, f'{epoch}.pt')
                )
            )
            path_test_result = Path(self.writer.logdir, f'test_{epoch}')
            os.makedirs(path_test_result, exist_ok=True)
        else:
            path_test_result = None

        avg_loss = 0.
        avg_eval = 0.
        all_thresholds = []
        # all_pred = []  # all predictions (for confusion matrix)
        print()
        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)

        for i_batch, (x, y, intervals, Ts, ids) in enumerate(pbar):
            # data
            # y_np = y.int().numpy()
            x = x.to(self.device)  # B, C, F, T
            x = dataloader.dataset.normalization.normalize_(x)
            y = y.to(self.out_device)  # B, T

            # forward
            out = self.model(x)  # B, C, 1, T
            out = self.sigmoid(out)[..., 0, 0, :]  # B, T

            # loss
            if mode != 'test':
                loss = self.calc_loss(y, out, Ts)
            else:
                loss = 0

            out_np = out.detach().cpu().numpy()
            prediction, thresholds = self.predict(out_np, Ts)

            eval_result = self.evaluate(intervals, prediction, out_np)

            if mode == 'train':
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.batch_step()
                loss = loss.item()
            elif mode == 'valid':
                # record
                loss = loss.item()
                if i_batch == 0:
                    id_0 = ids[0]
                    T_0 = Ts[0]
                    out_np_0 = out_np[0, :T_0]
                    t_axis = np.arange(T_0) * self.frame2time
                    pred_0 = prediction[0][1:, 0]
                    b_idx_0 = intervals[0][1:, 0]
                    fig = draw_lineplot(t_axis, out_np_0, pred_0, b_idx_0, id_0)
                    self.writer.add_figure(f'{mode}/out', fig, epoch)
                    np.save(Path(self.writer.logdir, f'{id_0}_{epoch}.npy'), out_np_0)
                    np.save(Path(self.writer.logdir, f'{id_0}_{epoch}_pred.npy'), pred_0)
                    if epoch == 0:
                        np.save(Path(self.writer.logdir, f'{id_0}_truth.npy'), b_idx_0)
            else:
                all_thresholds += thresholds
                for id_, item_truth, item_pred, item_out, T \
                        in zip(ids, intervals, prediction, out_np, Ts):
                    np.save(path_test_result / f'{id_}_truth.npy', item_truth)
                    np.save(path_test_result / f'{id_}.npy', item_out[:T])
                    np.save(path_test_result / f'{id_}_pred.npy', item_pred)

            str_eval = np.array2string(eval_result, precision=3)
            pbar.set_postfix_str(f'{loss:.3f}, {str_eval}')

            avg_loss += loss
            avg_eval += eval_result

        avg_loss = avg_loss / len(dataloader.dataset)
        avg_eval = avg_eval / len(dataloader.dataset)

        if mode == 'test':
            np.save(path_test_result / f'thresholds.npy', all_thresholds)

        return avg_loss, avg_eval

    # Early stopping function for given validation loss
    def step(self, epoch):
        self.scheduler.step()  # scheduler.last_restart can be updated

        if epoch == self.scheduler.last_restart:
            if epoch > 0:
                torch.save(self.model.state_dict(),
                           Path(self.writer.logdir, f'{epoch}.pt'))

        return self.scheduler.last_restart


def main(test_epoch: int):
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams,
                    len(train_loader.dataset),
                    train_loader.dataset.num_classes,
                    train_loader.dataset.class_weight)

    if test_epoch == -1:
        dict_custom_scalars = dict(
            loss=['Multiline', ['loss/train', 'loss/valid']],
        )
        for name in runner.metrics:
            dict_custom_scalars[name] = ['Multiline', [f'{name}/train', f'{name}/valid']]
        runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))

        epoch = 0
        epoch_last_restart = -1
        print(f'Training on {runner.str_device}')
        for epoch in range(hparams.num_epochs):
            # runner.writer.add_scalar('lr', runner.optimizer.param_groups[0]['lr'], epoch)

            train_loss, train_eval = runner.run(train_loader, 'train', epoch)
            runner.writer.add_scalar('loss/train', train_loss, epoch)
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/train', train_eval[idx], epoch)

            valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)
            runner.writer.add_scalar('loss/valid', valid_loss, epoch)
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/valid', valid_eval[idx], epoch)

            epoch_last_restart = runner.step(epoch)

        torch.save(runner.model.state_dict(), Path(runner.writer.logdir, f'{epoch}.pt'))
        print('Training Finished')
        test_epoch = epoch_last_restart

    _, test_eval = runner.run(test_loader, 'test', test_epoch)

    str_eval = np.array2string(test_eval, precision=4)
    print(f'Testset Evaluation: {str_eval}')
    runner.writer.add_text('Testset Evaluation', str_eval, test_epoch)

    runner.writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', type=int, default=-1)

    args = hparams.parse_argument(parser)
    test_epoch = args.test
    if test_epoch == -1:
        if list(Path(hparams.logdir).glob('events.out.tfevents.*')):
            while True:
                s = input(f'"{hparams.logdir}" already has tfevents. continue? (y/n)\n')
                if s.lower() == 'y':
                    shutil.rmtree(hparams.logdir)
                    os.makedirs(hparams.logdir)
                    break
                elif s.lower() == 'n':
                    exit()

    main(test_epoch)

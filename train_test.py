import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    def __init__(self, hparams, train_size: int, class_weight: Optional[Tensor] = None):
        # model, criterion, and prediction
        self.model = UNet(ch_in=2, ch_out=1, **hparams.model)
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.class_weight = class_weight

        # for prediction
        self.frame2time = hparams.hop_size / hparams.sample_rate
        self.T_6s = round(6 / self.frame2time) - 1
        self.T_12s = round(12 / self.frame2time) - 1
        self.metrics = ('precision', 'recall', 'F1')

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

        self.f1_last_restart = -1

        # device
        device_for_summary = self._init_device(hparams.device, hparams.out_device)

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

    def _init_device(self, device, out_device) -> str:
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

    def calc_loss(self, y: Tensor, out: Tensor, Ts: Union[List[int], int]) -> Tensor:
        """

        :param y: (B, T) or (T,)
        :param out: (B, T) or (T,)
        :param Ts: length B list or int
        :return:
        """
        assert self.class_weight is not None
        weight = (y > 0).float() * self.class_weight[1].item()
        weight += (y == 0).float() * self.class_weight[0].item()

        if y.dim() == 1:  # if batch_size == 1
            y = (y,)
            out = (out,)
            weight = (weight,)
            Ts = (Ts,)

        loss = torch.zeros(1, device=self.out_device)
        for ii, T in enumerate(Ts):
            loss_no_red = self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])
            loss += (loss_no_red * weight[ii:ii + 1, :T]).sum() / T

        return loss

    def predict(self, out_np: ndarray, Ts: Union[List[int], int]) \
            -> Tuple[List[ndarray], List]:
        """ peak-picking prediction

        :param out_np: (B, T) or (T,)
        :param Ts: length B list or int
        :return: boundaries, thresholds
            boundaries: length B list of boundary interval ndarrays
            thresholds: length B list of threshold values
        """
        if out_np.ndim == 1:  # if batch_size == 1
            out_np = (out_np,)
            Ts = (Ts,)

        boundaries = []
        thresholds = []
        for item, T in zip(out_np, Ts):
            candid_idx = []
            for idx in range(1, T - 1):
                i_first = max(idx - self.T_6s, 0)
                i_last = min(idx + self.T_6s + 1, T)
                if item[idx] >= np.amax(item[i_first:i_last]):
                    candid_idx.append(idx)

            boundary_idx = []
            threshold = np.mean(item[candid_idx])
            for idx in candid_idx:
                if item[idx] > threshold:
                    boundary_idx.append(idx)

            boundary_interval = np.array([[0] + boundary_idx,
                                          boundary_idx + [T]], dtype=np.float64).T
            boundary_interval *= self.frame2time

            boundaries.append(boundary_interval)
            thresholds.append(threshold)

        return boundaries, thresholds

    @staticmethod
    def evaluate(reference: Union[List[ndarray], ndarray],
                 prediction: Union[List[ndarray], ndarray]):
        """

        :param reference: length B list of ndarray or just ndarray
        :param prediction: length B list of ndarray or just ndarray
        :return: (3,) ndarray
        """
        if isinstance(reference, ndarray):  # if batch_size == 1
            reference = (reference,)

        result = np.zeros(3)
        for item_truth, item_pred in zip(reference, prediction):
            mir_result = mir_eval.segment.detection(item_truth, item_pred, trim=True)
            result += np.array(mir_result)

        return result

    # Running model for train, test and validation.
    def run(self, dataloader, mode: str, epoch: int):
        self.model.train() if mode == 'train' else self.model.eval()
        if mode == 'test':
            state_dict = torch.load(Path(self.writer.logdir, f'{epoch}.pt'))
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            path_test_result = Path(self.writer.logdir, f'test_{epoch}')
            os.makedirs(path_test_result, exist_ok=True)
        else:
            path_test_result = None

        avg_loss = 0.
        avg_eval = 0.
        all_thresholds = dict()
        print()
        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)

        for i_batch, (x, y, intervals, Ts, ids) in enumerate(pbar):
            # data
            n_batch = len(Ts) if hasattr(Ts, 'len') else 1
            x = x.to(self.device)  # B, C, F, T
            x = dataloader.dataset.normalization.normalize_(x)
            y = y.to(self.out_device)  # B, T

            # forward
            out = self.model(x)  # B, C, 1, T
            out = out[..., 0, 0, :]  # B, T

            # loss
            if mode != 'test':
                if mode == 'valid':
                    with torch.autograd.detect_anomaly():
                        loss = self.calc_loss(y, out, Ts)
                else:
                    loss = self.calc_loss(y, out, Ts)

            else:
                loss = 0

            out_np = self.sigmoid(out).detach().cpu().numpy()
            prediction, thresholds = self.predict(out_np, Ts)

            eval_result = self.evaluate(intervals, prediction)

            if mode == 'train':
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.batch_step()
                loss = loss.item()
            elif mode == 'valid':
                loss = loss.item()
                if i_batch == 0:  # save only the 0-th data
                    id_0, T_0 = ids[0], Ts[0]
                    out_np_0 = out_np[0, :T_0]
                    pred_0, truth_0 = prediction[0][1:, 0], intervals[0][1:, 0]
                    t_axis = np.arange(T_0) * self.frame2time
                    fig = draw_lineplot(t_axis, out_np_0, pred_0, truth_0, id_0)
                    self.writer.add_figure(f'{mode}/out', fig, epoch)
                    np.save(Path(self.writer.logdir, f'{id_0}_{epoch}.npy'), out_np_0)
                    np.save(Path(self.writer.logdir, f'{id_0}_{epoch}_pred.npy'), pred_0)
                    if epoch == 0:
                        np.save(Path(self.writer.logdir, f'{id_0}_truth.npy'), truth_0)
            else:
                # save all test data
                for id_, item_truth, item_pred, item_out, threshold, T \
                        in zip(ids, intervals, prediction, out_np, thresholds, Ts):
                    np.save(path_test_result / f'{id_}_truth.npy', item_truth)
                    np.save(path_test_result / f'{id_}.npy', item_out[:T])
                    np.save(path_test_result / f'{id_}_pred.npy', item_pred)
                    all_thresholds[str(id_)] = threshold

            str_eval = np.array2string(eval_result / n_batch, precision=3)
            pbar.set_postfix_str(f'{loss / n_batch:.3f}, {str_eval}')

            avg_loss += loss
            avg_eval += eval_result

        avg_loss = avg_loss / len(dataloader.dataset)
        avg_eval = avg_eval / len(dataloader.dataset)

        if mode == 'test':
            np.savez(path_test_result / f'thresholds.npz', **all_thresholds)

        return avg_loss, avg_eval

    def step(self, valid_f1: float, epoch: int):
        """

        :param valid_f1:
        :param epoch:
        :return: test epoch or 0
        """
        last_restart = self.scheduler.last_restart
        self.scheduler.step()  # scheduler.last_restart can be updated

        if epoch == self.scheduler.last_restart:
            if valid_f1 < self.f1_last_restart:
                return last_restart
            else:
                self.f1_last_restart = valid_f1
                torch.save(self.model.module.state_dict(),
                           Path(self.writer.logdir, f'{epoch}.pt'))

        return 0


def main(test_epoch: int):
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    if test_epoch == -1:
        runner = Runner(hparams,
                        len(train_loader.dataset),
                        train_loader.dataset.class_weight)
        dict_custom_scalars = dict(loss=['Multiline', ['loss/train', 'loss/valid']])
        for name in runner.metrics:
            dict_custom_scalars[name] = ['Multiline', [f'{name}/train', f'{name}/valid']]
        runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))

        epoch = 0
        test_epoch_or_zero = 0
        print(f'Training on {runner.str_device}')
        for epoch in range(hparams.num_epochs):
            # training
            train_loss, train_eval = runner.run(train_loader, 'train', epoch)
            runner.writer.add_scalar('loss/train', train_loss, epoch)
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/train', train_eval[idx], epoch)

            # validation
            valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)
            runner.writer.add_scalar('loss/valid', valid_loss, epoch)
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/valid', valid_eval[idx], epoch)

            # check stopping criterion
            test_epoch_or_zero = runner.step(valid_eval[2], epoch)
            if test_epoch_or_zero > 0:
                break

        torch.save(runner.model.module.state_dict(), Path(runner.writer.logdir, f'{epoch}.pt'))
        print('Training Finished')
        test_epoch = test_epoch_or_zero if test_epoch_or_zero > 0 else epoch
    else:
        runner = Runner(hparams, len(test_loader.dataset))

    # test
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
        # check overwrite or not
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

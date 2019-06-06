"""
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
"""
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
import mir_eval
# from torch.utils.tensorboard.writer import SummaryWriter
from tensorboardX import SummaryWriter
from torchsummary import summary
from torch import Tensor
from tqdm import tqdm

import data_manager
from adamwr import AdamW, CosineLRWithRestarts
from hparams import hparams
from models import UNet
from utils import draw_segmap, print_to_file, draw_lineplot


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams, train_size: int, num_classes: int, class_weight: Tensor):
        # model, criterion, and prediction
        self.ch_out = num_classes if num_classes > 2 else 1
        self.model = UNet(ch_in=2, ch_out=self.ch_out, **hparams.model)
        if num_classes == 2:
            self.sigmoid = torch.nn.Sigmoid()
            if class_weight is not None:
                self.criterion = torch.nn.BCELoss(reduction='none')
            else:
                self.criterion = torch.nn.BCELoss()
            self.class_weight = class_weight

            # for prediction
            self.T_6s = round(6 * hparams.sample_rate / hparams.hop_size) - 1
            self.T_12s = round(12 * hparams.sample_rate / hparams.hop_size) - 1
            self.thrs_pred = hparams.thrs_pred
            self.metrics = ('precision', 'recall', 'F1', 'mean', 'std')
        else:
            self.sigmoid = None
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
            self.class_weight = None

            # for prediction
            self.T_6s = 0
            self.T_12s = 0
            self.thrs_pred = 0

        self.frame2time = hparams.hop_size / hparams.sample_rate

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
        print_to_file(Path(self.writer.logdir, 'summary.txt'),
                      summary,
                      (self.model, (2, 128, 256)),
                      dict(device=device_for_summary)
                      )

        # save hyperparameters
        with Path(self.writer.logdir, 'hparams.txt').open('w') as f:
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

    def calc_loss(self, y: Tensor, out: Tensor, len_x: List[int]):
        loss = torch.zeros(1, device=self.out_device)
        if self.class_weight is not None:
            weight = (y > 0).float() * self.class_weight[1].item()
            weight += (y == 0).float() * self.class_weight[0].item()
            for ii, T in enumerate(len_x):
                loss_no_red = self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])
                loss += (loss_no_red * weight[ii:ii + 1, :T]).sum() / T
        else:
            for ii, T in enumerate(len_x):
                loss += self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])

        return loss

    def predict(self, out: ndarray, len_x: List[int]) -> List[ndarray]:
        """ peak-picking prediction

        :param out: (B, T) or (T,)
        :param len_x: length B list
        :return: length B list of boundary index ndarrays
        """
        if out.ndim == 1:
            out = [out]

        boundaries = []
        for item, T in zip(out, len_x):
            # candid_val = []
            candid_idx = []
            for idx in range(1, T - 1):
                i_first = max(idx - self.T_6s, 0)
                i_last = min(idx + self.T_6s + 1, T)
                if item[idx] >= np.amax(item[i_first:i_last]):
                    # candid_val.append(item[i])
                    candid_idx.append(idx)

            boundary_idx = []
            for idx in candid_idx:
                i_first = max(idx - self.T_12s, 0)
                i_last = min(idx + self.T_6s + 1, T)
                if item[idx] - self.thrs_pred * np.mean(item[i_first:i_last]) > 0:
                    boundary_idx.append(idx)

            boundary_interval = np.array([[0] + boundary_idx, boundary_idx + [T]],
                                         dtype=np.float64).T
            boundary_interval *= self.frame2time

            boundaries.append(boundary_interval)

        return boundaries

    def evaluate(self, y_np: ndarray, boundaries: List[ndarray],
                 out_np: ndarray, prediction: List[ndarray], len_x: List[int]):
        if self.ch_out == 1:
            acc = np.array([0., 0., 0., 0., 0.])
            for item_truth, item_pred, item_out in zip(boundaries, prediction, out_np):
                eval_result = mir_eval.segment.detection(item_truth, item_pred, trim=True)
                acc += np.array([*eval_result, item_out.mean(), item_out.std()])
        else:
            acc = 0.
            for ii, T in enumerate(len_x):
                corrected = (prediction[ii, :T] == y_np[ii, :T]).sum().item()
                acc += corrected / T / self.ch_out

        return acc

    # Running model for train, test and validation.
    def run(self, dataloader, mode: str, epoch: int):
        self.model.train() if mode == 'train' else self.model.eval()

        avg_loss = 0.
        avg_acc = 0.
        # all_pred = []  # all predictions (for confusion matrix)
        print()
        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)

        for i_batch, (x, y, boundaries, len_x, ids) in enumerate(pbar):
            # data
            y_np = y.int().numpy()
            x = x.to(self.device)  # B, C, F, T
            x = dataloader.dataset.normalization.normalize_(x)
            y = y.to(self.out_device)  # B, T
            # len_x = len_x.to(self.device)

            # forward
            out = self.model(x)  # B, C, 1, T
            if self.ch_out == 1:
                out = self.sigmoid(out)[..., 0, 0, :]  # B, T
            else:
                out = out.squeeze_(-2)  # B, C, T

            # loss
            if mode != 'test':
                loss = self.calc_loss(y, out, len_x)
            else:
                loss = 0

            if self.ch_out == 1:
                out_np = out.detach().cpu().numpy()
                # prediction = (out > 0.5).cpu().int()
                prediction = self.predict(out_np, len_x)
            else:
                out_np = None
                prediction = out.argmax(1).cpu().int().numpy()

            acc = self.evaluate(y_np, boundaries, out_np, prediction, len_x)

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
                    T_0 = len_x[0]
                    if self.ch_out == 1:
                        out_np_0 = out_np[0, :T_0]
                        t_axis = np.arange(T_0) * self.frame2time
                        pred_0 = prediction[0][1:, 0]
                        b_idx_0 = boundaries[0][1:, 0]
                        fig = draw_lineplot(t_axis, out_np_0, pred_0, b_idx_0, id_0)
                        self.writer.add_figure(f'{mode}/out', fig, epoch)
                        np.save(Path(self.writer.logdir, f'{id_0}_{epoch}.npy'), out_np_0)
                        np.save(Path(self.writer.logdir, f'{id_0}_{epoch}_pred.npy'), pred_0)
                        if epoch == 0:
                            np.save(Path(self.writer.logdir, f'{id_0}_truth.npy'), b_idx_0)
                    else:
                        pred_0 = prediction[0, :T_0]
                        fig = draw_segmap(id_0, pred_0)
                        self.writer.add_figure(f'{mode}/prediction', fig, epoch)
                        if epoch == 0:
                            y_np_0 = y_np[0, :T_0]
                            fig = draw_segmap(id_0, y_np_0, dataloader.dataset.sect_names)
                            self.writer.add_figure(f'{mode}/truth', fig, epoch)
            else:
                for id_, item_truth, item_pred, item_out \
                        in zip(ids, boundaries, prediction, out_np):
                    np.save(Path(self.writer.logdir, 'test', f'{id_}_truth.npy'), item_truth)
                    np.save(Path(self.writer.logdir, 'test', f'{id_}.npy'), item_out)
                    np.save(Path(self.writer.logdir, 'test', f'{id_}_pred.npy'), item_pred)

            s_acc = np.array2string(acc, precision=3) if type(acc) == ndarray else f'{acc:.3f}'
            pbar.set_postfix_str(f'{loss:.3f}, {s_acc}')

            avg_loss += loss
            avg_acc += acc

        avg_loss = avg_loss / len(dataloader.dataset)
        avg_acc = avg_acc / len(dataloader.dataset)

        # draw confusion matrix
        # if mode != 'train':
        #     all_pred = np.concatenate(all_pred)
        #     fig = plot_confusion_matrix(dataloader.dataset.y, all_pred, hparams.genres)
        #     self.writer.add_figure(f'confmat/{mode}', fig, epoch)

        return avg_loss, avg_acc

    # Early stopping function for given validation loss
    def early_stop(self, epoch, train_acc, valid_acc, valid_loss):
        # TODO: stopping criterion
        # last_restart = self.scheduler.last_restart
        #
        self.scheduler.step()  # scheduler.last_restart can be updated
        #
        # if last_restart > 0 and epoch == self.scheduler.last_restart or train_acc >= 0.999:
        #     if (self.loss_last_restart * self.stop_thr < valid_loss
        #             or self.acc_last_restart + self.stop_thr_acc > valid_acc):
        #         # stop at the last restart epoch
        #         self.model.load_state_dict(
        #             torch.load(
        #                 Path(self.writer.logdir, f'{last_restart}.pt')
        #             )
        #         )
        #         return last_restart
        #     elif train_acc >= 0.999:  # stop here
        #         return epoch
        #
        # # save state at restart
        # if epoch == self.scheduler.last_restart:
        #     if epoch > 0:
        #         torch.save(self.model.state_dict(),
        #                    Path(self.writer.logdir, f'{epoch}.pt'))
        #     self.loss_last_restart = valid_loss
        #     self.acc_last_restart = valid_acc

        return 0


def main():
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams,
                    len(train_loader.dataset),
                    train_loader.dataset.num_classes,
                    train_loader.dataset.class_weight)

    dict_custom_scalars = dict(
        loss=['Multiline', ['loss/train', 'loss/valid']],
        accuracy=['Multiline', ['accuracy/train', 'accuracy/valid']],
    )
    for name in runner.metrics:
        dict_custom_scalars[name] = ['Multiline', [f'{name}/train', f'{name}/valid']]
    runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))

    epoch = 0
    print(f'Training on {runner.str_device}')
    for epoch in range(hparams.num_epochs):
        # runner.writer.add_scalar('lr', runner.optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_acc = runner.run(train_loader, 'train', epoch)
        runner.writer.add_scalar('loss/train', train_loss, epoch)
        if type(train_acc) == ndarray:
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/train', train_acc[idx], epoch)
        else:
            runner.writer.add_scalar('accuracy/train', train_acc, epoch)

        valid_loss, valid_acc = runner.run(valid_loader, 'valid', epoch)
        runner.writer.add_scalar('loss/valid', valid_loss, epoch)
        if type(valid_acc) == ndarray:
            for idx, name in enumerate(runner.metrics):
                runner.writer.add_scalar(f'{name}/valid', valid_acc[idx], epoch)
        else:
            runner.writer.add_scalar('accuracy/valid', valid_acc, epoch)

        # print(f'[Epoch {epoch:2d}/{hparams.num_epochs:3d}] '
        #       f'[Train Loss: {train_loss:.4f}] '
        #       f'[Train Acc: {train_acc:.4f}] '
        #       f'[Valid Loss: {valid_loss:.4f}] '
        #       f'[Valid Acc: {valid_acc:.4f}]')

        epoch_or_zero = runner.early_stop(epoch, train_acc, valid_acc, valid_loss)
        if epoch_or_zero != 0:
            epoch = epoch_or_zero
            print(f'Early stopped at {epoch}')
            break

    os.makedirs(Path(hparams.logdir, 'test'))
    # _, test_acc = runner.run(test_loader, 'test', epoch)
    print('Training Finished')

    # print(f'Test Accuracy: {100 * test_acc:.2f} %')
    # runner.writer.add_text('Test Accuracy', f'{100 * test_acc:.2f} %', epoch)
    torch.save(runner.model.state_dict(), Path(runner.writer.logdir, 'state_dict.pt'))
    runner.writer.close()


if __name__ == '__main__':
    hparams.parse_argument()
    if list(Path(hparams.logdir).glob('events.out.tfevents.*')):
        while True:
            s = input(f'"{hparams.logdir}" already has tfevents. continue? (y/n)\n')
            if s.lower() == 'y':
                shutil.rmtree(hparams.logdir)
                os.makedirs(hparams.logdir)
                break
            elif s.lower() == 'n':
                exit()

    main()

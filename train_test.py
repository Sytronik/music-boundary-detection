"""
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

import data_manager
from adamwr import AdamW, CosineLRWithRestarts
from hparams import hparams


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams, train_size):
        # TODO: model initialization
        # self.model = models.ResNet(num_classes=len(hparams.genres), **hparams.ResNet)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = AdamW(self.model.parameters(),
                               lr=hparams.learning_rate,
                               weight_decay=hparams.weight_decay,
                               )
        self.scheduler = CosineLRWithRestarts(self.optimizer,
                                              batch_size=hparams.batch_size,
                                              epoch_size=train_size,
                                              **hparams.CosineLRWithRestarts
                                              )
        self.scheduler.step()

        # TODO: variables for stopping criterion
        # self.stop_thr = hparams.stop_thr
        # self.stop_thr_acc = hparams.stop_thr_acc
        # self.loss_last_restart = 1000
        # self.acc_last_restart = 0

        if hparams.device == 'cpu':
            self.device = torch.device('cpu')
            self.out_device = torch.device('cpu')
        else:
            if type(hparams.device) == int:
                device = [hparams.device]
            elif type(hparams.device) == str:
                device = [int(hparams.device[-1])]
            else:  # sequence of devices
                if type(hparams.device[0]) == int:
                    device = hparams.device
                else:
                    device = [int(d[-1]) for d in hparams.device]
            if len(device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=device)

            self.model.cuda(device[0])

            self.criterion.cuda(device[0])
            self.device = torch.device(f'cuda:{device[0]}')
            if type(hparams.out_device) == int:
                self.out_device = torch.device(f'cuda:{hparams.out_device}')
            else:
                self.out_device = torch.device(hparams.out_device)

        # summary(self.model, [(1, 128, 128), (1, 1, 1)])
        self.writer = SummaryWriter(log_dir=hparams.log_dir)

        # save hyperparameters
        with Path(self.writer.log_dir, 'hparams.txt').open('w') as f:
            for var in vars(hparams):
                value = getattr(hparams, var)
                print(f'{var}: {value}', file=f)

    # Accuracy function works like loss function in PyTorch
    @staticmethod
    def accuracy(source, target):
        # TODO: evaluation using mir_eval
        # correct = (source == target).sum().item()

        # return correct / source.size(0)
        return 0

    # Running model for train, test and validation.
    # mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode, epoch):
        self.model.train() if mode == 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        all_pred = []  # all predictions (for confusion matrix)
        print()
        pbar = tqdm(dataloader, desc=f'epoch {epoch:3d}', postfix='-', dynamic_ncols=True)
        for idx, (x, y, len_x) in enumerate(pbar):
            y_cpu = y
            x = x.to(self.device)
            x = dataloader.dataset.normalization.normalize_(x)
            y = y.to(self.out_device)
            len_x = len_x.to(self.device)

            out = self.model(x, len_x)

            if mode != 'test':
                loss = torch.zeros(1, device=self.out_device)
                for T, item_y, item_out in zip(len_x, y, out):
                    T = T.item()
                    loss += self.criterion(item_out[..., :T], item_y[..., :T]) / int(T)
            else:
                loss = 0
            prediction = out.max(1)[1].int().cpu()

            acc = self.accuracy(prediction, y_cpu)

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.batch_step()
            else:
                if idx == 0:
                    # y_cpu 랑 prediction을 matplotlib 으로 visualize하는 함수를 호출
                    color_pred = plt.cm.Set3(prediction[idx, :])
                    color_y_cpu = plt.cm.Set3(y_cpu[idx, :])
                    pos_pred = np.arange(prediction.shape[1])
                    pos_y_cpu = np.arange(y_cpu.shape[1])
                    fig = plt.figure()
                    fig.subplot(2, 1, 1, title='prediction')
                    fig.bar(pos_pred, height=1, color=color_pred)
                    fig.subplot(2, 1, 2, title='y_cpu')
                    fig.bar(pos_y_cpu, height=1, color=color_y_cpu)
                    self.writer.add_figure(mode, fig, epoch)
                all_pred.append(prediction.numpy())

            pbar.set_postfix_str(str(loss.item()))
            epoch_loss += out.size(0) * loss.item() if mode != 'test' else 0
            epoch_acc += out.size(0) * acc

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_acc = epoch_acc / len(dataloader.dataset)

        # draw confusion matrix
        # if mode != 'train':
        #     all_pred = np.concatenate(all_pred)
        #     fig = plot_confusion_matrix(dataloader.dataset.y, all_pred, hparams.genres)
        #     self.writer.add_figure(f'confmat/{mode}', fig, epoch)

        return epoch_loss, epoch_acc

    # Early stopping function for given validation loss
    def early_stop(self, epoch, train_acc, valid_acc, valid_loss):
        # TODO: stopping criterion
        # last_restart = self.scheduler.last_restart
        #
        # self.scheduler.step()  # scheduler.last_restart can be updated
        #
        # if last_restart > 0 and epoch == self.scheduler.last_restart or train_acc >= 0.999:
        #     if (self.loss_last_restart * self.stop_thr < valid_loss
        #             or self.acc_last_restart + self.stop_thr_acc > valid_acc):
        #         self.model.load_state_dict(
        #             torch.load(
        #                 Path(self.writer.log_dir, f'{last_restart}.pt')
        #             )
        #         )
        #         return last_restart
        #     elif train_acc >= 0.999:
        #         return epoch
        #
        # # if epoch == self.scheduler.last_restart:
        #     if epoch > 0:
        #         torch.save(self.model.state_dict(),
        #                    Path(self.writer.log_dir, f'{epoch}.pt'))
        #     self.loss_last_restart = valid_loss
        #     self.acc_last_restart = valid_acc

        return 0


def main():
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams, len(train_loader.dataset))

    epoch = 0
    print(f'Training on {hparams.device}')
    for epoch in range(hparams.num_epochs):
        # runner.writer.add_scalar('lr', runner.optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_acc = runner.run(train_loader, 'train', epoch)
        valid_loss, valid_acc = runner.run(valid_loader, 'valid', epoch)

        runner.writer.add_scalars('loss', dict(train=train_loss, valid=valid_loss), epoch)
        runner.writer.add_scalars('accuracy', dict(train=train_acc, valid=valid_acc), epoch)

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

    _, test_acc = runner.run(test_loader, 'test', epoch)
    print('Training Finished')
    print(f'Test Accuracy: {100 * test_acc:.2f} %')
    runner.writer.add_text('Test Accuracy', f'{100 * test_acc:.2f} %', epoch)
    torch.save(runner.model.state_dict(), Path(runner.writer.log_dir, 'state_dict.pt'))
    runner.writer.close()


if __name__ == '__main__':
    main()

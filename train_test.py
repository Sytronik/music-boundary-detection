"""
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
"""
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
# from torch.utils.tensorboard.writer import SummaryWriter
from tensorboardX import SummaryWriter
from torchsummary import summary
from torch import Tensor
from tqdm import tqdm

import data_manager
from adamwr import AdamW, CosineLRWithRestarts
from hparams import hparams
from models import UNet
from utils import draw_segmap, print_to_file


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams, train_size: int, num_classes: int, class_weight: Tensor):
        # TODO: model initialization
        self.num_classes = num_classes
        ch_out = num_classes if num_classes > 2 else 1
        self.model = UNet(ch_in=2, ch_out=ch_out, **hparams.model)
        if num_classes == 2:
            self.sigmoid = torch.nn.Sigmoid()
            self.criterion = torch.nn.BCELoss(reduction='none')
            self.class_weight = class_weight
        else:
            self.sigmoid = False
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
            self.class_weight = None

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

        if hparams.device == 'cpu':
            self.device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            summary_device = 'cpu'
            self.str_device = 'cpu'
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

            if type(hparams.out_device) == int:
                out_device = torch.device(f'cuda:{hparams.out_device}')
            else:
                out_device = torch.device(hparams.out_device)

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
            summary_device = 'cuda'

        self.writer = SummaryWriter(logdir=hparams.logdir)
        print_to_file(Path(self.writer.logdir, 'summary.txt'),
                      summary,
                      (self.model, (2, 128, 128)),
                      dict(device=summary_device)
                      )
        # summary(self.model, (2, 128, 256))

        # save hyperparameters
        with Path(self.writer.logdir, 'hparams.txt').open('w') as f:
            for var in vars(hparams):
                value = getattr(hparams, var)
                print(f'{var}: {value}', file=f)

    # Accuracy function works like loss function in PyTorch
    @staticmethod
    def eval(source, target, Ts):
        # TODO: evaluation using mir_eval
        pass

    # Running model for train, test and validation.
    # mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode, epoch):
        self.model.train() if mode == 'train' else self.model.eval()

        avg_loss = 0.
        avg_acc = 0.
        # all_pred = []  # all predictions (for confusion matrix)
        print()
        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', postfix='-', dynamic_ncols=True)

        for i_batch, (x, y, len_x, ids) in enumerate(pbar):
            y_cpu = y.int()
            x = x.to(self.device)  # B, C, F, T
            x = dataloader.dataset.normalization.normalize_(x)
            y = y.to(self.out_device)  # B, T
            # len_x = len_x.to(self.device)

            out = self.model(x)  # B, C, 1, T
            if self.sigmoid:
                out = self.sigmoid(out)[..., 0, 0, :]
                prediction = (out > 0.5).cpu().int()
            else:
                out = out.squeeze_(-2)  # B, C, T
                prediction = out.argmax(1).cpu().int()

            if mode != 'test':
                loss = torch.zeros(1, device=self.out_device)
                if self.class_weight is not None:
                    weight = (y==1)*self.class_weight[1].item() + (y==0) * self.class_weight[0].item()
                    weight = weight.float()
                    for ii, T in enumerate(len_x):
                        no_reduct = self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])
                        loss += (no_reduct * weight[ii:ii+1, :T]).sum() / T
                else:
                    for ii, T in enumerate(len_x):
                        loss += self.criterion(out[ii:ii + 1, ..., :T], y[ii:ii + 1, :T])

                # for T, item_y, item_out in zip(len_x, y, out):
                #     loss += self.criterion(item_out[..., :T], item_y[..., :T]) / int(T)
            else:
                loss = 0

            acc = 0.
            for ii, T in enumerate(len_x):
                corrected = (prediction[ii, :T] == y_cpu[ii, :T]).sum().item()
                acc += corrected / T / self.num_classes

            # acc = self.eval(prediction, y_cpu, len_x)

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.batch_step()
            else:
                if i_batch == 0:
                    # y_cpu 랑 prediction을 matplotlib 으로 visualize하는 함수를 호출
                    pred_0_np = prediction[0, :len_x[0]].numpy()
                    fig = draw_segmap(ids[0], pred_0_np)
                    self.writer.add_figure(f'{mode}/prediction', fig, epoch)
                    if epoch == 0:
                        y_0_np = y_cpu[0, :len_x[0]].numpy()
                        fig = draw_segmap(ids[0], y_0_np, dataloader.dataset.sect_names)
                        self.writer.add_figure(f'{mode}/truth', fig, epoch)
                # all_pred.append(prediction.numpy())

            if mode != 'test':
                loss = loss.item()
            pbar.set_postfix_str(f'{loss:.3f}, {acc * 100:.2f} %')
            avg_loss += loss if mode != 'test' else 0
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
        #         self.model.load_state_dict(
        #             torch.load(
        #                 Path(self.writer.logdir, f'{last_restart}.pt')
        #             )
        #         )
        #         return last_restart
        #     elif train_acc >= 0.999:
        #         return epoch
        #
        # # if epoch == self.scheduler.last_restart:
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
    runner.writer.add_custom_scalars(dict(
        training=dict(
            loss=['Multiline', ['loss/train', 'loss/valid']],
            accuracy=['Multiline', ['accuracy/train', 'accuracy/valid']],
        ),
    ))

    epoch = 0
    print(f'Training on {runner.str_device}')
    for epoch in range(hparams.num_epochs):
        # runner.writer.add_scalar('lr', runner.optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_acc = runner.run(train_loader, 'train', epoch)
        runner.writer.add_scalar('loss/train', train_loss, epoch)
        runner.writer.add_scalar('accuracy/train', train_acc, epoch)

        valid_loss, valid_acc = runner.run(valid_loader, 'valid', epoch)
        runner.writer.add_scalar('loss/valid', valid_loss, epoch)
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

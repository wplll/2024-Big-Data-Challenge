from fastprogress import progress_bar
from tqdm import tqdm
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
    
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        if self.args.val_ratio > 0.0:
            train_set, train_loader, val_set, val_loader = data_provider(self.args)
            return train_set, train_loader, val_set, val_loader
        else:
            train_set, train_loader = data_provider(self.args)
            return train_set, train_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # def train(self, setting):
    #     train_set, train_loader, val_set, val_loader = self._get_data()

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)

    #     model_optim = self._select_optimizer()
    #     criterion = self._select_criterion()

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()

    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []

    #         self.model.train()
    #         epoch_time = time.time()
    #         progress = tqdm(enumerate(train_loader),total=len(train_loader))
    #         for i, (batch_x, batch_y) in progress:
    #             iter_count += 1
    #             model_optim.zero_grad()
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x)[0]
    #                     else:
    #                         outputs = self.model(batch_x)

    #                     f_dim = -1 if self.args.features == 'MS' else 0
    #                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                     loss = criterion(outputs, batch_y)
    #                     train_loss.append(loss.item())
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x)[0]
    #                 else:
    #                     outputs = self.model(batch_x)

    #                 f_dim = -1 if self.args.features == 'MS' else 0
    #                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #                 loss = criterion(outputs, batch_y)
    #                 train_loss.append(loss.item())

                
    #             progress.set_description("loss: {:.7f}".format(np.average(train_loss)))
    #             # speed = (time.time() - time_now) / iter_count
    #             # left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #             # progress.set_description('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #             # iter_count = 0
    #             # time_now = time.time()

    #             if self.args.use_amp:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(model_optim)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 model_optim.step()

    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)

    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
    #             epoch + 1, train_steps, train_loss))
    #         if not self.args.debug:
    #             torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}_{train_loss}.pth')
    #         adjust_learning_rate(model_optim, epoch + 1, self.args)

    #     return self.model

    def train(self, setting):
        if self.args.val_ratio > 0.0:
            train_set, train_loader, val_set, val_loader = self._get_data()
        else:
            train_set, train_loader = self._get_data()
            val_set = None
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')

        for epoch in range(self.args.train_epochs):
            print('>>>>>>> epoch : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(epoch+1))
            iter_count = 0
            train_loss = []
            train_loss_last = []
            self.model.train()
            epoch_time = time.time()
            progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for i, (batch_x, batch_y) in progress:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.mode:
                            if self.args.output_attention:
                                outputs = self.model(batch_x)[0]
                            else:
                                if self.args.mode :
                                    outputs,last_output = self.model(batch_x)
                                else:
                                    outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x)[0]
                            else:
                                if self.args.mode :
                                    outputs,last_output = self.model(batch_x)
                                else:
                                    outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) + criterion(last_output, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        if self.args.mode :
                            outputs,last_output = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.mode:
                        loss = (criterion(outputs, batch_y) + criterion(last_output, batch_y)) / 2
                    else:
                        loss = criterion(outputs, batch_y)

                    loss_last = criterion(outputs, batch_y)
                    train_loss.append(loss_last.item())
                    if self.args.mode:
                        loss_last = criterion(last_output, batch_y)
                        train_loss_last.append(loss_last.item())
                if self.args.mode:
                    progress.set_description("loss : 1/37: {:.7f} 1/1: {:.7f} ".format(np.average(train_loss), np.average(train_loss_last)))
                else:
                    progress.set_description("loss : 1/37: {:.7f} ".format(np.average(train_loss)))

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_last = np.average(train_loss_last)
            if self.args.mode:
                print("Epoch: {0}, Steps: {1} | Train Loss: 1/37 : {2:.7f} 1/1 : {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, train_loss_last))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
            
            
            if val_set is not None:
                val_loss = self.validate(val_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Validation Loss: {2:.7f}".format(
                    epoch + 1, train_steps, val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), path + '/' + 'checkpoint_best.pth')
                    print("Best model saved with loss: {:.7f}".format(best_val_loss))

            if not self.args.debug:
                try:
                    torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}_val_{val_loss:.3f}_train_{train_loss:.3f}.pth')
                except:
                    torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}_train_{train_loss:.3f}.pth')
                    
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = []
        progress = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        with torch.no_grad():
            for i, (batch_x, batch_y) in progress:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    if self.args.mode:
                        outputs,last_output = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.mode:
                    loss = (criterion(outputs, batch_y) + criterion(last_output, batch_y)) / 2
                else:
                    loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())
                progress.set_description("loss: {:.7f}".format(np.average(val_loss)))

        val_loss = np.average(val_loss)
        self.model.train()
        return val_loss
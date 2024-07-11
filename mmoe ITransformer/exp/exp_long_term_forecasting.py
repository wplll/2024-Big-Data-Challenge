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
        data_set, data_loader = data_provider(self.args)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            temp_loss = []
            wind_loss = []
            self.model.train()
            epoch_time = time.time()
            progress = tqdm(train_loader, total=len(train_loader), leave=False)
            for batch_x, batch_y in progress:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs_temp,outputs_wind = self.model(batch_x)[0]
                        else:
                            outputs_temp,outputs_wind = self.model(batch_x)

                        batch_y_temp = batch_y[:, -self.args.pred_len:, -2:-1].to(self.device)
                        batch_y_wind = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                        loss = (criterion(outputs_temp, batch_y_temp) + criterion(outputs_wind, batch_y_wind)) / 2
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs_temp,outputs_wind = self.model(batch_x)[0]
                    else:
                        outputs_temp,outputs_wind = self.model(batch_x)

                    batch_y_temp = batch_y[:, -self.args.pred_len:, -2:-1].to(self.device)
                    batch_y_wind = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                    loss = (criterion(outputs_temp, batch_y_temp) + criterion(outputs_wind, batch_y_wind)) / 2
                    train_loss.append(loss.item())
                    temp_loss.append(criterion(outputs_temp, batch_y_temp).item())
                    wind_loss.append(criterion(outputs_wind, batch_y_wind).item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()
                progress.set_description("temp loss : {:.7f} , wind loss : {:.7f} ".format(np.average(temp_loss),np.average(wind_loss)))
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            print(f"Temp Loss : {np.average(temp_loss)} , Wind Loss : {np.average(wind_loss)}")
            
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}_train_{train_loss:.3f}.pth')
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model
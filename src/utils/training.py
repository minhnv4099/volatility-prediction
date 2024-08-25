import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            loss_fn,
            optimizer,
            writer_path: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.writer_path = writer_path

    def run(
            self,
            epoch: int,
    ):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('{}_{}'.format(self.writer_path,timestamp))
        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            writer.add_scalars(
                main_tag='Training vs. Validation Loss',
                tag_scalar_dict={'Training': avg_loss, 'Validation': avg_vloss},
                global_step=epoch_number + 1
            )
            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def train_one_epoch(
            self,
            epoch_index,
            tb_writer
    ):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(self.train_loader):
            inputs, targets = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
                tb_writer.add_scalar(
                    main_tag='Loss/train',
                    tag_scalar_dict=last_loss,
                    global_step=tb_x
                )
                running_loss = 0.

        return last_loss
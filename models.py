
""" 
Model definitions
"""

import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models
import wandb
import common 
import data_utils
import train_utils
import sklearn.metrics as metrics


BACKBONES = {
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'densenet121': models.densenet121,
    'densenet201': models.densenet201
}

class Trainer:

    def __init__(self, args):
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Initiate model, optimizer and scheduler
        assert self.config['model']['name'] in BACKBONES.keys(), f"Unrecognized model name {self.config['model']['name']}"
        self.model = BACKBONES[self.config['model']['name']](pretrained=self.config['model']['pretrained']).to(self.device)
        self.optim = train_utils.get_optimizer(self.config['optimizer'], self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(self.config['scheduler'], self.optim)

        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config['optim']['lr'] - 1e-12) / self.warmup_epochs

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = data_utils.get_dataloaders(
            train_root = self.cofig['data']['train_root'], 
            test_root = self.config['data']['test_root'], 
            transforms = self.config['data']['transforms'], 
            val_split = self.config['data']['val_split'], 
            batch_size = self.config['data']['batch_size'])

        # Logging and model saving
        self.criterion = nn.NLLLoss()
        self.best_val_acc = 0
        self.done_epochs = 0

        # Wandb
        run = wandb.init(project='deepfake-dl-hack')
        self.logger.write(f"Wandb: {run.get_url()}", mode='info')

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.ckpt"))

    def load_model(self, path):
        if os.path.exists(os.path.join(path, "best_model.ckpt")):
            state = torch.load(os.path.join(path, "best_model.ckpt"))
            self.model.load_state_dict(state)
        else:
            raise NotImplementedError(f"No saved model found in {path}")

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def get_metrics(self, output, targets):
        preds = output.argmax(dim=-1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        acc = metrics.accuracy_score(targets, preds)
        f1 = metrics.f1_score(targets, preds)
        return {"accuracy": acc, "f1": f1}
    
    def train_one_step(self, batch):
        img, trg = batch 
        img, trg = img.to(self.device), trg.to(self.device)
        out = self.model(img)
        loss = self.criterion(out, trg)
        train_metrics = self.get_metrics(out, trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item(), **train_metrics}

    def validate_one_step(self, batch):
        img, trg = batch 
        img = img.to(self.device)
        with torch.no_grad():
            out = self.model(img).detach().cpu()
        loss = self.criterion(out, trg)
        val_metrics = self.get_metrics(out, trg)
        return {"loss": loss.item(), **val_metrics}

""" 
Model definitions
"""

import os
import torch 
import torch.nn as nn  
from torchvision import models
import torch.nn.functional as F
import torch.distributions.beta as beta
import wandb
import common
import losses
import resnets
import networks 
import data_utils
import train_utils
import numpy as np 
import pandas as pd
import sklearn.metrics as metrics

NETWORKS = {
    'resnet18': dict(net=models.resnet18, dim=512),
    'resnet34': dict(net=networks.Resnet34, dim=512),
    'resnet50': dict(net=networks.Resnet50, dim=2048),
    'resnet101': dict(net=networks.Resnet101, dim=2048),
    'resnet152': dict(net=networks.Resnet152, dim=2048),
    'efficientnet': dict(net=networks.EfficientNet, dim=1536)
}


class DeepfakeClassifier:

    def __init__(self, args):
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Initiate model, optimizer and scheduler
        assert self.config['model']['name'] in NETWORKS.keys(), f"Unrecognized model name {self.config['model']['name']}"
        self.model = NETWORKS[self.config['model']['name']]['net'](pretrained=self.config['model']['pretrained']).to(self.device)
        self.optim = train_utils.get_optimizer(self.config['optimizer'], self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            {**self.config['scheduler'], "epochs": self.config["epochs"]}, self.optim)

        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config['optimizer']['lr'] - 1e-12) / self.warmup_epochs

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = data_utils.get_dataloaders(
            train_root = self.config['data']['train_root'], 
            test_root = self.config['data']['test_root'], 
            transforms = self.config['data']['transforms'], 
            val_split = self.config['data']['val_split'], 
            batch_size = self.config['data']['batch_size'])
        self.beta_dist = beta.Beta(self.config['data'].get("alpha", 0.3), self.config['data'].get("alpha", 0.3))
        self.batch_size = self.config['data']['batch_size']

        # Logging and model saving
        self.criterion = losses.LogLoss()
        self.best_val_loss = np.inf
        self.done_epochs = 0

        # Wandb
        run = wandb.init(project='deepfake-dl-hack')
        self.logger.write(f"Wandb: {run.get_url()}", mode='info')

        # Load model
        if args['load'] is not None:
            self.load_model(args['load'])

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.ckpt"))

    def load_model(self, path):
        if os.path.exists(os.path.join(path, "best_model.ckpt")):
            state = torch.load(os.path.join(path, "best_model.ckpt"))
            self.model.load_state_dict(state)
            self.logger.print(f"Successfully loaded model from {path}", mode='info')
        else:
            raise NotImplementedError(f"No saved model found in {path}")

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def get_metrics(self, output, targets):
        preds = output.argmax(dim=-1).detach().cpu().numpy()[:self.batch_size]
        targets = targets.detach().cpu().numpy()[:self.batch_size]
        acc = metrics.accuracy_score(targets, preds)
        f1 = metrics.f1_score(targets, preds, zero_division=0)
        return {"accuracy": acc, "f1": f1}

    def get_mixed_batch(self, img, trg):
        idx = torch.randperm(img.size(0))
        img_shuffle, trg_shuffle = img[idx], trg[idx]
        
        lbd = self.beta_dist.sample()
        img_mixed = img * lbd + img_shuffle * (1.0 - lbd)
        trg_mixed = trg * lbd + trg_shuffle * (1.0 - lbd)
        
        img_batch = torch.cat([img, img_mixed], dim=0)
        trg_batch = torch.cat([trg, trg_mixed], dim=0)
        return img_batch.to(self.device), trg_batch.to(self.device)
    
    def train_one_step(self, batch):
        self.model.train()
        img, trg = batch 
        img, trg = self.get_mixed_batch(img, trg)
        out = self.model(img)
        loss = self.criterion(out, trg)
        train_metrics = self.get_metrics(out, trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item(), **train_metrics}

    def validate_one_step(self, batch):
        self.model.eval()
        img, trg = batch 
        img = img.to(self.device)
        with torch.no_grad():
            out = self.model(img).detach().cpu()
        loss = self.criterion(out, trg)
        val_metrics = self.get_metrics(out, trg)
        return {"loss": loss.item(), **val_metrics}

    @staticmethod
    def polarize_predictions(s):
        if s > 0.65:
            return 0.95
        elif s < 0.35:
            return 0.05
        else:
            return s

    def get_test_predictions(self, polarize=True):
        self.model.eval()
        if self.args["load"] is None:
            self.load_model(self.output_dir)
            
        test_ids, test_preds = [], []
        for idx, batch in enumerate(self.test_loader):
            img, ids = batch
            img = img.to(self.device)
            with torch.no_grad():
                out = self.model(img).detach().cpu().numpy()
            test_ids.extend(ids.cpu().numpy().tolist())
            test_preds.extend(out[:, 1].tolist())
            common.progress_bar(progress=(idx+1)/len(self.test_loader), status="")
            
        common.progress_bar(progress=1.0, status="")
        sub_df = pd.DataFrame({"id": test_ids, "p_real": test_preds})
        sub_df["id"] = sub_df["id"].astype("int")
        sub_df["p_real"] = sub_df["p_real"].astype("float").apply(lambda x: min(max(0.05, x), 0.95))     
        sub_df = sub_df.sort_values(by="id", ascending=True)
        if polarize:
            sub_df["p_real"] = sub_df["p_real"].apply(DeepfakeClassifier.polarize_predictions)
            sub_df.to_csv(os.path.join(self.output_dir, "polarized_test_predictions.csv"), index=False)
        else:
            sub_df.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)

    def custom_test_predictions(self, polarize=False):
        model = resnets.resnet18(pretrained=False).to(self.device)
        state = torch.load("./models/resnet-18.pt")
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        test_ids, test_preds = [], []
        for idx, batch in enumerate(self.test_loader):
            img, ids = batch
            img = img.to(self.device)
            with torch.no_grad():
                out = F.softmax(model(img), dim=1).detach().cpu().numpy()
            test_ids.extend(ids.cpu().numpy().tolist())
            test_preds.extend(out[:, 0].tolist())
            common.progress_bar(progress=(idx+1)/len(self.test_loader), status="")
            
        common.progress_bar(progress=1.0, status="")
        sub_df = pd.DataFrame({"id": test_ids, "p_real": test_preds})
        sub_df["id"] = sub_df["id"].astype("int")
        sub_df["p_real"] = sub_df["p_real"].astype("float").apply(lambda x: min(max(0.05, x), 0.95))     
        sub_df = sub_df.sort_values(by="id", ascending=True)
        if polarize:
            sub_df["p_real"] = sub_df["p_real"].apply(DeepfakeClassifier.polarize_predictions)
            sub_df.to_csv(os.path.join(self.output_dir, "polarized_test_predictions.csv"), index=False)
        else:
            sub_df.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)

    def train(self):
        for epoch in range(self.config["epochs"] - self.done_epochs):
            self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='train')
            train_meter = common.AverageMeter()

            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train loss": train_meter.return_metrics()["loss"]})
                common.progress_bar(progress=(idx+1)/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1.0, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            self.adjust_learning_rate(epoch+1)
            wandb.log({
                "Train accuracy": train_meter.return_metrics()["accuracy"],
                "Train F1 score": train_meter.return_metrics()["f1"],
                "Epoch": epoch+1})

            if (epoch+1) % self.config["eval_every"] == 0:
                self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='val')
                val_meter = common.AverageMeter()
                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=(idx+1)/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1.0, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["loss"],
                    "Validation accuracy": val_meter.return_metrics()["accuracy"],
                    "Validation F1 score": val_meter.return_metrics()["f1"],
                    "Epoch": epoch+1})

                if val_meter.return_metrics()["loss"] < self.best_val_loss:
                    self.best_val_loss = val_meter.return_metrics()["loss"]
                    self.save_model()

        print("\n\n")
        self.logger.record("Finished training! Generating test predictions...", mode='info')
        self.get_test_predictions(self.config.get("polarize_predictions", True))
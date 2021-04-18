
""" 
Model definitions
"""

import os
import torch 
import torch.nn as nn  
import torch.nn.functional as F
import wandb
import common
import networks 
import data_utils
import train_utils
import numpy as np 
import pandas as pd
import sklearn.metrics as metrics

NETWORKS = {
    'resnet34': networks.Resnet34,
    'resnet50': networks.Resnet50,
    'resnet101': networks.Resnet101,
    'resnet152': networks.Resnet152,
    'densenet121': networks.Densenet121,
    'densenet161': networks.Densenet161,
    'densenet169': networks.Densenet169,
    'densenet201': networks.Densenet201
}


class DeepfakeClassifier:

    def __init__(self, args):
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Initiate model, optimizer and scheduler
        assert self.config['model']['name'] in NETWORKS.keys(), f"Unrecognized model name {self.config['model']['name']}"
        self.model = NETWORKS[self.config['model']['name']](pretrained=self.config['model']['pretrained']).to(self.device)
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

        # Logging and model saving
        self.criterion = nn.NLLLoss()
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
        preds = output.argmax(dim=-1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        acc = metrics.accuracy_score(targets, preds)
        f1 = metrics.f1_score(targets, preds, zero_division=0)
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

    def get_test_predictions(self):
        if self.args["load"] is None:
            self.load_model(self.output_dir)
            
        test_ids, test_preds = [], []
        for idx, batch in enumerate(self.test_loader):
            img, ids = batch
            img = img.to(self.device)
            with torch.no_grad():
                out = F.softmax(self.model.base(img), dim=-1).detach().cpu().numpy()
            test_ids.extend(ids.cpu().numpy().tolist())
            test_preds.extend(out[:, 0].tolist())
            common.progress_bar(progress=(idx+1)/len(self.test_loader), status="")
            
        common.progress_bar(progress=1.0, status="")
        sub_df = pd.DataFrame({"id": test_ids, "p_real": test_preds})
        sub_df["id"] = sub_df["id"].astype("int")
        sub_df["p_real"] = sub_df["p_real"].astype("float").apply(lambda x: min(max(0.05, x), 0.95))     
        sub_df = sub_df.sort_values(by="id", ascending=True)
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
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["loss"],
                    "Validation accuracy": val_meter.return_metrics()["accuracy"],
                    "Validation F1 score": val_meter.return_metrics()["f1"],
                    "Epoch": epoch+1})

                if val_meter.return_metrics()["loss"] < self.best_val_loss:
                    self.best_val_loss = val_meter.return_metrics()["loss"]
                    self.save_model()

        print(f"\n\n{common.COLORS['yellow']}[INFO] Finished training! Generating test predictions...{common.COLORS['end']}")
        self.get_test_predictions()
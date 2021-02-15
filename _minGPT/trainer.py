import math 

import torch

import logging

from tqdm import tqdm
import numpy as np

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger('my logger')

class Trainer:
    
    def __init__(self, model, train_dataset, test_dataset, config, lr):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.lr = lr
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
            
        # parallelize batch, not specifying ids??    
        self.model = torch.nn.DataParallel(self.model).to(self.device) 

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model 
        logger.info(f'Saved to: {self.config.ckpt_path}')
        torch.save(raw_model.state_dict(), sefl.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config gg
        device = self.device
        raw_model = model.module if hasattr(self.model, 'module') else model # must we get module?
        # optimizer = raw_model.configure_
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=self.lr)

        def run_epoch(training_stage=True):
            model.train(training_stage)
            data = self.train_dataset if training_stage else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memeory=True, batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []

            pbar = tqdm(enumerate(loader), total=len(loader)) if training_stage else enumerage(loader)
            for it, x, y in pbar:

                x,y = x.to(device), y.to(device)

                #model forward pass
                with torch.set_grad_enabled(training_stage):
                    logits, loss = model(x, y)
                    loss = loss.mean() #multiple gpu avg
                    losses.append(loss.item())
                
                if training_stage:
                    # back prop
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_clip_norm)
                    optimizer.step()

                    # lr decay
                    if config.lr_decay:
                        # >>> y
                        # tensor([ 1.,  2.,  3., -1.])
                        # >>> (y >=0)
                        # tensor([ True,  True,  True, False])
                        # >>> (y >=0).sum()
                        # tensor(3)
                        self.tokens += (y>=0).sum()
                        if self.tokens < config.warmup_tokens:
                            #linear warmup increase
                            lr_mult = float(self.tokens)/float(max(1, config.warmup_tokens))
                        else:
                            #cosine l r decay
                            progress = float(self.tokens-config.warmup_tokens)/float(max(1, config.final_tokens-config.warmup_tokens))
                            lr_mult = max(0.1, 0.5*(1.0+math.cos(math.pi*progress))) 
                        lr = config.learning_rate*lr_mult 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr 


                    #progress report 
                    pbar.set_description(f'epoch {epoch+1} iteration {it}: train loss{loss.item()} lr {lr}')

                if not training_stage:
                    #testing stage
                    test_loss = float(np.mean(losses))
                    logger.info(f'test loss: {test_loss}')
                    return test_loss

        #run through epochs
        best_loss = float('inf')
        self.tokens = 0 #lr decay counter
        for epoch in range(config.max_epochs):

            run_epoch(training_stage=True)

            if self.test_dataset is not None:
                test_loss = run_epoch(training_stage=False)

            is_curr_best_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and is_curr_best_model:
                best_loss = test_loss 
                self.save_checkpoint()
        





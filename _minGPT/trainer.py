import math 

import torch


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
        model, config = self.model, self.config 
        raw_model = model.module if hasattr(self.model, 'module') else model # must we get module?
        # optimizer = raw_model.configure_
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=self.lr)
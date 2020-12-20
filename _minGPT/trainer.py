import math 

import torch


class Trainer:
    
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
            
        # parallelize batch, not specifying ids??    
        self.model = torch.nn.DataParallel(self.model).to(self.device) 
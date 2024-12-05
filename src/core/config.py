"""by lyuwenyu
"""

from pprint import pprint
import mindspore #rxz
import mindspore.nn as nn #rxz
import mindspore.dataset as ds
from mindspore.train.callback import Callback
from mindspore.amp import auto_mixed_precision, DynamicLossScaleManager

# from torch.utils.data import Dataset, DataLoader
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LRScheduler
# from torch.cuda.amp.grad_scaler import GradScaler

from typing import Callable, List, Dict


__all__ = ['BaseConfig', ]



class BaseConfig(object):
    # TODO property


    def __init__(self) -> None:
        super().__init__()

        self.task :str = None 
        
        self._model :nn.Cell = None 
        self._postprocessor :nn.Cell = None 
        self._criterion :nn.Cell = None 
        self._optimizer :nn.Optimizer = None 
        self._lr_scheduler :Callable = None 
        self._train_dataloader :ds.Dataset = None 
        self._val_dataloader :ds.Dataset = None 
        self._ema :nn.Cell = None 

        self.train_dataset :ds.Dataset = None
        self.val_dataset :ds.Dataset = None
        self.num_workers :int = 0
        self.collate_fn :Callable = None

        self.batch_size :int = None
        self._train_batch_size :int = None
        self._val_batch_size :int = None
        self._train_shuffle: bool = None  
        self._val_shuffle: bool = None 

        self.evaluator :Callable[[nn.Cell, ds.Dataset, str], ] = None

        # runtime
        self.resume :str = None
        self.tuning :str = None

        self.epoches :int = None
        self.last_epoch :int = -1
        self.end_epoch :int = None

        self.use_amp :bool = False 
        self.use_ema :bool = False 
        self.sync_bn :bool = False 
        self.clip_max_norm : float = None
        self.find_unused_parameters :bool = None
        # self.ema_decay: float = 0.9999
        # self.grad_clip_: Callable = None

        self.log_dir :str = './logs/'
        self.log_step :int = 10
        self._output_dir :str = None
        self._print_freq :int = None 
        self.checkpoint_step :int = 1

        self.loss_scale_manager = None  # zxp
        self.train_one_step_cell = None  # 用于混合精度训练的Cell

        # self.device :str = torch.device('cpu')
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = torch.device(device)


    @property
    def model(self, ) -> nn.Cell:
        return self._model 
    
    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Cell), f'{type(m)} != nn.Cell, please check your model class'
        if self.use_amp:
            m = auto_mixed_precision(m, 'O1')  # Apply AMP to the model #zxp

            self.loss_scale_manager = DynamicLossScaleManager()
                
            # 创建 TrainOneStepWithLossScaleCell 实例
            self.train_one_step_cell = nn.TrainOneStepWithLossScaleCell(
                network=m,
                optimizer=self.optimizer,
                scale_sense=self.loss_scale_manager
            )
        self._model = m 

    @property
    def postprocessor(self, ) -> nn.Cell:
        return self._postprocessor
    
    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Cell), f'{type(m)} != nn.Cell, please check your model class'
        self._postprocessor = m 

    @property
    def criterion(self, ) -> nn.Cell:
        return self._criterion
    
    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Cell), f'{type(m)} != nn.Cell, please check your model class'
        self._criterion = m 

    @property
    def optimizer(self, ) -> nn.Optimizer:
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(m, nn.Optimizer), f'{type(m)} != optim.Optimizer, please check your model class'
        self._optimizer = m 

    @property
    def lr_scheduler(self, ) -> Callable:
        return self._lr_scheduler
    
    @lr_scheduler.setter
    def lr_scheduler(self, m):
        self._lr_scheduler = m 

    @property
    def train_dataloader(self):
        if self._train_dataloader is None and self.train_dataset is not None:
            self._train_dataloader = self.create_dataloader(self.train_dataset, self.train_batch_size, self.train_shuffle)
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader 

    @property
    def val_dataloader(self):
        if self._val_dataloader is None and self.val_dataset is not None:
            self._val_dataloader = self.create_dataloader(self.val_dataset, self.val_batch_size, self.val_shuffle, drop_last=False)
        return self._val_dataloader
    
    @val_dataloader.setter
    def val_dataloader(self, loader):
        self._val_dataloader = loader 


    # TODO method
    # @property
    # def ema(self, ) -> nn.Cell:
    #     if self._ema is None and self.use_ema and self.model is not None:
    #         self._ema = ModelEMA(self.model, self.ema_decay)
    #     return self._ema

    @property
    def ema(self, ) -> nn.Cell:
        return self._ema 

    @ema.setter
    def ema(self, obj):
        self._ema = obj

    @property
    def val_shuffle(self):
        if self._val_shuffle is None:
            print('warning: set default val_shuffle=False')
            return False
        return self._val_shuffle

    @val_shuffle.setter
    def val_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._val_shuffle = shuffle

    @property
    def train_shuffle(self):
        if self._train_shuffle is None:
            print('warning: set default train_shuffle=True')
            return True
        return self._train_shuffle

    @train_shuffle.setter
    def train_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._train_shuffle = shuffle


    @property
    def train_batch_size(self):
        if self._train_batch_size is None and isinstance(self.batch_size, int):
            print(f'warning: set train_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._train_batch_size = batch_size

    @property
    def val_batch_size(self):
        if self._val_batch_size is None:
            print(f'warning: set val_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._val_batch_size = batch_size


    @property
    def output_dir(self):
        if self._output_dir is None:
            return self.log_dir
        return self._output_dir

    @output_dir.setter
    def output_dir(self, root):
        self._output_dir = root

    @property
    def print_freq(self):
        if self._print_freq is None:
            # self._print_freq = self.log_step
            return self.log_step
        return self._print_freq

    @print_freq.setter
    def print_freq(self, n):
        assert isinstance(n, int), 'print_freq must be int'
        self._print_freq = n

    def create_dataloader(self, dataset, batch_size, shuffle, num_parallel_workers=None, drop_last=True, collate_fn=None):
        return ds.GeneratorDataset(
            source=dataset,
            column_names=["data", "label"],
            shuffle=shuffle,
            num_parallel_workers=num_parallel_workers or self.num_workers,
        ).batch(batch_size=batch_size, drop_remainder=drop_last)


    # def __repr__(self) -> str:
    #     pass 




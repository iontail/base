from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, ConstantLR
import math


def warmup_fn(step: int, warmup_epochs: int, warmup_start_lr: float = 0.0):
    if warmup_epochs <= 0:
        return 1.0
    
    elif step < warmup_epochs:
        return warmup_start_lr + (1.0 - warmup_start_lr) * (step / warmup_epochs)

    else:
        return 1.0
        


class WarmupConstantScheduler(LambdaLR):
    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 warmup_start_lr: float
                 ):
        """
        Even if ther is a LinearLR class, it only start at factored lr
        To make it starts from absolute lr, I cutomized it
        """

        def lr_lambda(step):
            return warmup_fn(step, warmup_epochs, warmup_start_lr)
        
        super().__init__(optimizer, lr_lambda=lr_lambda)


class WarmupCosineScheduler(LambdaLR):
    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 warmup_start_lr: float,
                 total_epochs: int,
                 min_lr: float = 1e-6
                 ):
        
        # http://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L701-707
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_ratio = min_lr / base_lr
        # as we multiply ratio base_lr

        def lr_lambda(step):
            if step < warmup_epochs:
                return warmup_fn(step, warmup_epochs, warmup_start_lr)
            
            else:
                after_warmup_step = step - warmup_epochs
                after_warmup_total = total_epochs - warmup_epochs

                cosine_element = math.pi * after_warmup_step / after_warmup_total
                lr = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(cosine_element))
                return lr
            
        
        super().__init__(optimizer, lr_lambda=lr_lambda)


class WarmupStepScheduler(LambdaLR):
    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 milestones: list[int] = None,
                 warmup_start_lr: float = 0.0,
                 gamma: float = 0.1
                 ):
        
        milestones = sorted(milestones)

        def lr_lambda(step):
            if step < warmup_epochs:
                return warmup_fn(step, warmup_epochs, warmup_start_lr)
            else:
                # https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L477
                # check _get_closed_form_lr method in StepLR class
                count = sum(1 for m in milestones if step >= m)
                return gamma**count

            
        super().__init__(optimizer, lr_lambda=lr_lambda)


def get_scheduler(optimizer, 
                  scheduler_name: str = 'constant',
                  warmup_epochs: int = 0,
                  warmup_start_lr: int = 0.0,
                  total_epochs: int = -1,
                  min_lr: float = 1e-6,
                  milestones: list[int] = None 
                  ):
    
    scheduler_name = scheduler_name.lower()

    warmup = True if warmup_epochs > 0 else False

    if scheduler_name == 'constant':
        if warmup:
            return WarmupConstantScheduler(optimizer, warmup_epochs, warmup_start_lr)
        else:
            return None


    elif scheduler_name == 'cosine':
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive integer. Got {total_epochs}")

        if warmup:
            return WarmupCosineScheduler(optimizer, warmup_epochs, warmup_start_lr, total_epochs, min_lr)
        else:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
        
    elif scheduler_name == 'step':
        if milestones is None:
            if warmup:
                after_warmup_total = total_epochs - warmup_epochs
                milestones = [warmup_epochs + after_warmup_total // 3, warmup_epochs + 2 * after_warmup_total // 3]
            else:
                milestones = [total_epochs // 3, 2 * total_epochs // 3]

        # fixing factor 0.1
        if warmup:
            return WarmupStepScheduler(optimizer, warmup_epochs, milestones, warmup_start_lr) 
        else:
            return WarmupStepScheduler(optimizer, 0, milestones, warmup_start_lr)
        




    

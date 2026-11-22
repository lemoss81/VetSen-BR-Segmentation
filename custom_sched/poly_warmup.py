# -*- coding: utf-8 -*-
"""Warmup linear + decaimento polinomial, aplicado POR ITERACAO (receita do SegFormer).

Hydra nao instancia uma lambda, entao LambdaLR precisa de um alvo importavel: esta classe.

    lr(t) = base * (1 - t/total_iters)^power * w(t)
    w(t)  = 1 - (1 - start_ratio) * (1 - t/warmup_iters)   se t < warmup_iters, senao 1

Com power=1.0 o LR cai linearmente ate 0 exatamente no passo total_iters.
Em t=0 o LR vale base*start_ratio; no fim do warmup vale base*(1 - warmup_iters/total_iters)
-- ou seja, ~0,99*base quando o warmup e 1% de T. Isso e esperado (a forma e um produto).

USO (configs/base_vetsenbr.yaml):
    scheduler_list:
      - scheduler:
          _target_: custom_sched.poly_warmup.PolyWarmupLR
          total_iters: ${total_iters}
          warmup_iters: ${warmup_iters}
          power: 1.0
        interval: step        # <- por iteracao, nao por epoca
        frequency: 1
        name: poly_warmup

ATENCAO: nao defina `hyperparameters.warmup_epochs`. O PSMT envelopa qualquer scheduler
que nao seja OneCycleLR num LinearLR(start_factor=1e-2)+SequentialLR quando esse campo e > 0,
o que produziria um warmup duplo e com milestones contados em passos.
"""
import torch


class PolyWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_iters: int, warmup_iters: int,
                 power: float = 1.0, start_ratio: float = 1e-6, last_epoch: int = -1):
        total_iters = int(total_iters)
        warmup_iters = int(warmup_iters)
        if warmup_iters < 1:
            raise ValueError(f"warmup_iters deve ser >= 1 (recebido {warmup_iters})")
        if total_iters <= warmup_iters:
            raise ValueError(f"total_iters ({total_iters}) deve ser > warmup_iters ({warmup_iters})")
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.power = float(power)
        self.start_ratio = float(start_ratio)

        def fn(t: int) -> float:
            if t < warmup_iters:
                w = 1.0 - (1.0 - start_ratio) * (1.0 - t / warmup_iters)
            else:
                w = 1.0
            decay = max(0.0, 1.0 - t / total_iters) ** self.power
            return decay * w

        super().__init__(optimizer, fn, last_epoch=last_epoch)

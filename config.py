#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :
@InitTime    :2024/05/10 11:07:20
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn
'''


NET = {
}
STRATEGY = {
    'train': {
        "batch_size": 64,
        "epoch": 1000,
        "patience": 1000,
        'train_size': 0.8,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-4},
    },
}

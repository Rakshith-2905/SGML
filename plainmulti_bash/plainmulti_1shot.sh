#!/usr/bin/env bash
python main.py --datasource=plainmulti --datadir=data --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/ARML/1shot/15_node --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01 --num_vertex=16
python main.py --datasource=plainmulti --datadir=data --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/ARML/1shot/15_node --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01 --test_dataset=0 --train=False --test_epoch=49999 --num_vertex=16

#!/usr/bin/env bash
python main.py --datasource=plainmulti --datadir=data --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/TreeGraph/graph_level_att/1shot/one2many/same_meta_edge/31 --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01
python main.py --datasource=plainmulti --datadir=data --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/TreeGraph/graph_level_att/1shot/one2many/same_meta_edge/31 --num_filters=32 --hidden_dim=128 --emb_loss_weight=0.01 --test_dataset=0 --train=False --test_epoch=49999
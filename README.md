# Inveritble Networks for Compression-aware JPEG pre-editing



## v 0.1

- [ ] 搭建训练框架和测试框架，基于可学习类小波变换和可微分JPEG simulator
- [ ] 多卡训练和推理


提前下载数据集，这里以DIV2K为例


训练命令：
cd到`/codes`下面，执行
```
python train.py -opt options/train/train_INCA_JPEG_demo.yml
```

测试命令：
cd到`/codes`下面，执行
```
python test.py -opt options/test/test_INCA_JPEG_demo.yml
```

## Acknowledgment

训练和测试框架来自SAIN [[arxiv](https://arxiv.org/abs/2303.02353)].

# Chinese Hip-pop Generation
2018 DeeCamp 人工智能夏令营项目，使用 GAN 进行中文嘻哈歌词生成。

Project of DeeCamp 2018, generating Chinese hip-pop lyrics using GAN.

## Note

因时间仓促，代码写的有些混乱（诸如变量命名 etc.)，也可能存在一些错误，欢迎提 PR 共同改进。由于精力有限，不再更新，有问题欢迎提 Issue，我会尽最大努力地提供帮助。

The code is a mess right now, please be careful when using. It is welcome to make a PR if you find some mistakes and raise issues for asking help.

## Dataset 

训练数据以及押韵表可以从 Google 云[下载](https://drive.google.com/drive/folders/1QrO0JAti3A3vlZlUemouOW7jC3K5dFZr?usp=sharing)，也可以可以根据我的[博客说明](https://tobiaslee.top/2018/08/23/Generate-hip-pop-lyrcis-using-GAN/)更改数据集

The processed training data and rhyme table can be downloaded from [Goolge Drive](https://drive.google.com/drive/folders/1QrO0JAti3A3vlZlUemouOW7jC3K5dFZr?usp=sharing), the origin lyrics (10, 000 sentence) is also [provided](https://drive.google.com/open?id=1JupHuqoDXa_cDCVi8KtwDv1rrJShWoQ9).
## Requirements

- Python3
- TensorFlow >= 1.7.0
- Jieba 

## Model

模型基于 SeqGAN，进行一些修改：

![Model](/img/AI-hippop.jpg)

## Training and Evaluate

下载数据集之后将 data 放在项目目录下，运行

> python3 seq_gan.py

完成训练之后，使用运行 `translate.py` 查看生成的测试集结果，或者使用 `generate_paragraph.py` 测试生成多句效果。

**Note**: CPU version tensorflow may meet problem when doing inference due to `tf.multinomial`. You can change the sampling function in `generator.py`

## Results

下面是一些比较好的结果，Input 为模型输入（**均为测试集中的句子，未参与训练**），Result 为模型生成结果，Origin 为原歌词。

Example 1:

> Input: 
>
> 不是乐理专修
>
> Result:
>
> 做点儿曲式研究
> 我们的力量来自宇宙 
> 自己的节奏
>
> Origin:
>
> 所有听的观众打破他们传统
> 进到环球 
> 继续让你感受

Example 2:

> Input:
>
> 音响里换首咏叹听
>
> Result:
>
> 冷酷如透明的冰
> 反抗的标语文艺复兴
> 你唱的听不清
>
> Origin:
>
> 音响里换首咏叹听
> 技巧  强你
> 从来不需要靠垄断赢

Example 3:

> Input:
>
> 自己就带上了有色眼镜
>
> Result:
>
> 金钱摧毁多少事情
> 瓦解你的中枢神经
>
> Origin:
> 我只想把世界分的更清
> 却发现自己却模糊了心

更多结果可以在 `good_cases.txt` 中查看

## Acknowledgement

This project is a collaboration fruit with the following great team members:

[lihao2333](https://github.com/lihao2333)

[llluckygirlrhy](https://github.com/llluckygirlrhy)

[Linfeng Zhang](https://github.com/zhanglinfeng1997)

[FrankLiu](https://github.com/FrankLiu2018)

[liuaiting](https://github.com/liuaiting)




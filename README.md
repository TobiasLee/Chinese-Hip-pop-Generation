# Chinese Hip-pop Generation
2018 DeeCamp 人工智能夏令营项目，使用 GAN 进行中文嘻哈歌词生成。

## Dataset 

训练数据以及押韵表可以从 Google 云[下载]()

可以根据我的[博客说明]()更改数据集

## Requirements

- Python3
- TensorFlow >= 1.7.0
- Jieba 

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

Thanks to our team: [lihao2333](https://github.com/lihao2333)、[llluckygirlrhy](https://github.com/llluckygirlrhy)、[张林峰](https://github.com/zhanglinfeng1997)、[FrankLiu](https://github.com/FrankLiu2018)、[liuaiting](https://github.com/liuaiting)




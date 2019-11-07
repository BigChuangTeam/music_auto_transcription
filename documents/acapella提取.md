人声提取

1 项目分析

 我们希望能够将含人声的音乐中的人声提取出来，免除了伴奏干扰，方便了单独识别人声部分的音高。由于音频可以与它的频谱图相互转化，故可以构建卷积神经网络，以含人声音乐的频谱图作为网络的输入层，以去除了伴奏的人声音频的频谱图作为网络的输出层，从而达到人声提取的目的。

2 卷积层与卷积计算

 卷积层由若干个滤波器（filter）构成，通过滤波器对输入层进行卷积计算，得到卷积层的输出。如：

输入层（5*5*1）

                    

        Input

          filter 1

          filter 2

滤波器（3*3*2）

![图片](https://assets-cdn.shimo.im/docs/assets/shape_table_error.png)                     
















                      输出（3*3*2）


![图片](https://assets-cdn.shimo.im/docs/assets/shape_table_error.png)








 其中（0，1，0）是由输入层标出部分与滤波器1对应位置相乘相加得到，即0*1 + 0*2 + 0*1 + 1*0 + 2*0 + 2*1 + 2*1 + (-1)*(-1) + 0*0=5。需要注意的是，我们在这里没有加入偏差bias，并且卷积计算的步长strides为1，而一般在数据量较大或滤波器边长较大时，会采用大于等于2的步长。在本例中，采用了深度为1的输入层，而一般针对图像处理的输入层深度为3，分别代表了图像的RGB值。

 对于滤波器的输出，我们还要采用激活函数将输出值激活。在卷积神经网络中，我们采用relu激活函数，如下图所示：




使用relu函数的原因：

 relu函数相比传统的sigmoid函数，计算量相对小，并且sigmoid函数在反向传播时，容易出现梯度消失的情况，面对大量数据时比relu函数结果更差。

 同时，Relu函数会使一部分神经元的输出为0，这样造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。这个特点也配合了池化层的设计。


3池化层

 即对上一层输出做降维处理，同时也是特征提取处理。一般将 2*2的矩阵转化为单个元素，取区域平均值（max pooling）或区域最大值（mean pooling）即可。例如：



通过max pooling得到2，通过mean pooling得到0.5


 故若池化层的输入为64*64的矩阵，输出则为32*32的矩阵。在我们建立的神经网络中，采用取区域最大值的池化层。

4 构建卷积神经网络

 用keras.layers 中的函数构建神经网络。

 用conv2D( ) 生成的卷积层，会对输入层进行滑动窗卷积。我们的模型会用到五种参数不同的卷积层，如下所示：

    1. filters = 64, kernel_size = 3, activation = ‘relu’                          记作conv1
    2. filters = 64, kernel_size = 4, strides = 2, activation = ‘relu’       记作conv2
    3. filters = 128, kernel_size = 3, activation = ‘relu’                        记作conv3
    4. filters = 32, kernel_size = 3, activation = ‘relu’                          记作conv4
    5. filters = 1, kernel_size = 3, activation = ‘relu’                            记作conv5

 用BatchNormalization( ) 将激活值规范化，使得输出的数据的期望趋向于0，而标准差趋向于1，记作batch_normalize；用UpSampling2D( ) 得到池化层，提取上一层数据的特征，记作up_sample；用Concatenate( ) 进行数据的合并，将池化层输出的数据与较早得出的输出数据进行合并，增加了深度学习的稳定性，记作concatenate。

Input

conv1

conv2

convA

batch_normalize

conv1

convB

conv2

batch_normalize

up_sample

concatenate

+convB

conv3

conv3

batch_normalize

conv1

conv1

batch_normalize

up_sample

concatenate

+convA

conv1

conv1

conv4

conv5

Output

![图片](https://assets-cdn.shimo.im/docs/assets/shape_table_error.png) 整个神经网络如下图所示：

5 神经网络的训练

 与传统神经网络的训练方法相同，采用反向传播算法。损失函数为 ，考虑梯度下降算法，只需要求出  对每个滤波器参数的偏微分，为每个参数使得  最速下降的方向，用  来更新参数。经过迭代，在  比较小时，能够获得效果比较好的神经网络，即完成了神经网络的构建，使其能够投入应用。

6 结果分析

 在神经网络训练完成之后，进行测试。输入音频，得到的频谱图如下：

![图片](https://uploader.shimo.im/f/vgZORjx99h4u9cLo!thumbnail)

 输入神经网络，得到的输出如下：

![图片](https://uploader.shimo.im/f/RedjLE4dkEU1W4SV!thumbnail)

 可以在生成的频谱图上看到，一部分声音被“抹去”了，剩下的部分即为人声部分。对比来看，原频谱图更加连续、含有更多的信息，而输出的频谱图更加离散、粗糙。将频谱图转化为音频，听感上也有相同的感觉，音质会变得更加粗糙，不像原音频那样悦耳。总体来说，人声提取效果并未达到预期中的水准，不过伴奏部分被明显的削弱了，人声部分凸显了出来，这符合了我们希望“单独提取人声音高”的目标。

 此外，考虑我们的“音高识别”模型，由于模型建立的数据集主要基于钢琴曲，故其对于人声的识别还不太成熟，若未来有较好的人声音高识别的数据集，相信我们能得到比较好的识别效果。


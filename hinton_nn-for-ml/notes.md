### 感知器

https://www.zybuluo.com/hanbingtao/note/433855

注意权重（包括偏置）的更新公式：

![1504775510(1).jpg](https://github.com/ChaoZeyi/machine-learning/blob/master/hinton_nn-for-ml/images/1504775510(1).jpg?raw=true)

$w_i$是与输$x_i$入对应的权重项，$b$是偏置项。事实上，可以把$b$看作是值永远为1的输入所对应的权重。$t$是训练样本的**实际值**，一般称之为**label**。而$y$是感知器的输出值，$η$是一个称为**学习速率**的常数，其作用是控制每一步调整权的幅度。
#### 一般步骤

1. 得到数据集，一般有两种形式：

   - DataSets ：使用a.data a.target等形式，使用a.DESCR查看描述信息。
   - DataFrame ：使用a[列名]的形式，包含feature和target，target一般在最后一列，使用a.describe() a.info()查看描述信息

2. 对数据集进行划分，得到训练集和测试集

   from sklearn.cross_validation import train_test_split

3. 数据处理：归一化

   from sklearn.preprocessing import StandardScaler

4. 模型选择及训练

   from sklearn.linear_model import LinearRegression

   from sklearn.tree import DecisionTreeClassifier

   from sklearn.ensemble import RandomForestClassifier

5. 训练结果

   ​






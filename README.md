## TODO：
1. **改进文本预处理**
   - 移除URL等等
   - 去除@、#、：准确率都下降
   - emoji替换效果不好
   - 疑似将@具体人名改为@user有微弱效果提升  `text = re.sub(r'@(\w+)', '@user', text)`
2. **事件特征增强**
   - 只加入事件元数据
   ```python
   # 在数据集中添加事件元数据
   def __getitem__(self, idx):
       ...
       event_metadata = torch.tensor([
           self.event_freq[self.events[idx]],  # 事件频率
           self.event_rumor_ratio[self.events[idx]]  # 事件中谣言比例
       ], dtype=torch.float)
       return x_text, x_event, event_metadata, y
   ```
   - 考虑tag特征和事件元数据，效果不稳定
   ```python
   # 在数据集中添加hashtag(形如#tag的统计特征)和事件元数据
   #尝试添加了text中hashtag(形如#tag的统计特征)、事件频率、谣言比例
   def __getitem__(self, idx):
      x_text = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
      x_event = torch.tensor(self.events[idx], dtype=torch.long)
      _, hashtags = preprocess_text(self.texts[idx])
      # hashtag特征: 平均长度
      hashtag_feats = [
         len(hashtags),
         sum(len(tag) for tag in hashtags)/(len(hashtags)+1e-6)
      ]
      # 事件发生频率和谣言比例
      event_meta = torch.tensor([
         self.event_freq[self.events[idx]],#事件频率
         self.event_rumor_ratio[self.events[idx]],#谣言比例
         *hashtag_feats
      ], dtype=torch.float)
      y = torch.tensor(self.labels[idx], dtype=torch.float)
      return x_text, x_event, event_meta, y
   ```
3. **不同的分词器**
   - 用spaCy结果差别不大
   - 自训练BPE效果不佳
4. **不同的vocab.txt**
5. **数据增强**
   - 同义词替换、随机插入、删除、交换等方法生成新的训练样本
   - 用t5改写效果很差
   - data后缀的1、2、4分别对应同义词替换、用Bert随机插入、用t5改写，前两者效果略有下降
6. **调整Transformer参数**：Transformer编码器的层数、注意力头数、前馈神经网络的维度等等
   - 适当提高max_len有用
   - 增强特征交互，给event和text之间用transformer编码，效果不明显
     - 本来想用`nn.MultiHeadAttention`对拼接后的特征进行注意力交互，捕捉文本和事件之间的相关性，但是我的`pytorch`版本不知道怎么回事好像不兼容
   - 加入预训练词向量`glove`初始化嵌入层，效果很差
   - 学习率均衡调整，效果下降


## 代码结构说明
1. 集成学习文件夹中代码在非集成学习代码基础上加入独立训练加合并评估步骤，在随机种子选取合适的情况下，集成学习效果更好
2. 已经拆分为`model.py`、`train.py`和`classify.py`
   
   如果希望自己修改训练参数/训练数据，请修改`train.py`后重新运行`train.py`存储最新模型，再运行`classify.py`进行效果测试。
   
   `classify.py`中测试了单条`text`和`csv`文件，可以自行修改想要测试的内容，如果能找到多元数据集并用于测试训练就更好了

## 数据集扩展相关
1. data文件夹下新增了在twitter15、twitter16两个新文件夹，是找到的新数据，只含有id、text和正确性标签。新数据中的标签除true、false外，还有non-rumor和unverified两种，不能完全利用
2. augment_twitter_data.py把新增的twitter数据中所有.train,.test,.dev文件中标签为true、false的条目添加到train_new.csv中，event栏默认填0。如果找到其他相同格式的数据集，也可以用此程序添加
3. 现在的train_new.csv中包括原train.csv的所有数据和twitter数据扩展，共3900条
4. testing.csv中式来源于twitter数据扩展的最后126条，不包含在train_new.csv中，可用作测试
5. 删除ai_generate.csv，准确性不可靠

## 去除event依赖
- 现在无集成学习/train.py中以去除对event的使用，测试时需通过classify_noevent.py


## 实验结果总结：
1. 没有加入新数据集：
   
   - 在集成学习和非集成学习中均测试了是否加入event进行训练，不加入event时val acc轻微下降，testing因为无event，测试时也不能输入event
   - 集成学习：
     - 加入event训练，集成模型val acc:87.65，最佳单一模型val acc:86.17，test acc:64.00

         此处因为模型参数需要一致，测试时默认了所有event均为0，实际上不应该这么做  
     - 不加入event训练，集成模型val acc:87.16，最佳单一模型val acc:86.67，test acc:66.40
   - 非集成学习：
     - 加入event训练，val acc:不记得了
     - 不加入event训练，val acc:85.19, test acc:72.00
2. 加入新数据集：不能加入event进行训练，，因为新数据集无event选择全部用0填充
   - 非集成学习：
     - val acc:84.69, test acc:87.20
   - 集成学习：
     - 集成模型val acc:88.40，单一模型val acc:86.67, test acc:91.20

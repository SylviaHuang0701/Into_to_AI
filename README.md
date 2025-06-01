### TODO：
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

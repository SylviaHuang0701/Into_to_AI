### TODO：
1. **改进文本预处理**
- 移除URL等等
2. **事件特征增强**
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
3. **不同的分词器**
4. **不同的vocab.txt**
5. **数据增强**
   - 同义词替换、随机插入、删除、交换等方法生成新的训练样本
7. **调整Transformer参数**：Transformer编码器的层数、注意力头数、前馈神经网络的维度等等

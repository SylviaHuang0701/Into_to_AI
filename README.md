## `classify.py`调用说明


### **1. 单条文本谣言检测**
```python
# 初始化检测器（使用默认带事件信息的模型）
detector = RumourDetectClass(
    use_event=False,          # 默认使用不带事件信息的模型
    vocab_path='vocab.json'  # 默认使用当前目录的vocab.json
)
text = "要测试的文本内容"
result = detector.classify(text)
```

---

### **2. 批量测试CSV文件命令行调用示例**
```bash
# 使用带事件信息的模型测试数据
python classify.py --use-event --test-file ../data/test.csv

# 使用不带事件信息的模型测试数据
python classify.py --test-file ../data/test.csv
```
   因为目前效果最好的模型是在加入新数据的情况下训练而成，新数据不带有event信息，训练模型时未加入，默认不加参数时直接调用该模型。

   如果使用`use_event`参数，使用`transformer_rumor_detector_withevent`模型，但由于文件夹中的`vocab.json`是训练`transformer_rumor_detector`时生成，使用此参数时，需要重新执行`train.py --use-event`,使用`train.csv`进行训练再调用接口·。

---

### **关键参数说明**
| **参数**      | **说明**                                                                 |
|---------------|-------------------------------------------------------------------------|
| `model_paths` | 模型路径列表，默认使用带/不带事件信息的预训练模型                          |
| `vocab_path`  | 词表文件路径（默认`vocab.json`）                                         |
| `use_event`   | 是否使用事件信息模型（默认`True`）                                       |
| `text`        | 待检测文本（字符串）                                                    |
| `event_id`    | 事件ID（仅`use_event=True`时生效，未提供则默认为0）                      |
| `csv_path`    | 测试数据路径（CSV需包含`text`和`label`列，带事件模型需含`event`列）       |

---

### **注意事项**
1. **文件路径**：确保词表`vocab.json`、模型文件及测试CSV存在于指定路径
2. **事件ID**：使用带事件模型时，事件ID范围需在训练时的`num_events`内（默认0~6）
3. **输出结果**：
   - `classify()` 返回 `0`（非谣言） 或 `1`（谣言）
   - `test_csv()` 返回准确率


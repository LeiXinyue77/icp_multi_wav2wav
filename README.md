# icp_multi_wav2wav

## 1. 预处理
### 1.1 筛选信号
#### 选出含有ICP、ABP等信号，且fs=125Hz的记录
```python 
# preprocess/select.py
if __name__ == '__main__':
    select()  # 运行写入筛选结果 result/pre 00.txt, 01.txt, ..., 09.txt
    
# test/test_select.py
if __name__ == '__main__':
    test_select()  # 测试筛选结果生成文件 result/pre file_content.txt

```

### 1.2 统计信号和病人数目
```python
# preprocess/statistic.py
if __name__ == '__main__':
    # 统计信号总类（38）all_signal.dat
    # file_content.dat ---> file_content_list.dat
    # 统计记录总数（462）file_signal_map.dat
    statistic_signal() 

    # 统计包含 ICP, ABP, PLETH 信号的记录 和 病人个数
    target_signals = {"ICP", "ABP", "PLETH"}
    statistics_target(target_signals, "file_signal_map_icp_abp_pleth")
```
  - 包含 ICP ABP PLETH 信号的记录 (345) 和 病人个数 (210)  
  - 包含 ICP ABP PLETH  I II III 信号的记录 (116) 和 病人个数 (89)   
  - 包含 ICP ABP PLETH  I II III  RESP信号的记录 (108)和 病人个数 (81) 

### 1.3 下载数据
```python
# preprocess/download.py
if __name__ == '__main__':
    channel_list = ["ICP", "ABP", "PLETH", "II"]
    save_path = "data/pXX"
    download_target_signals(channel_list, save_path)
    print("=================================  save finished ===============================")
```

### 1.4 清理数据

```python

if __name__ == '__main__':
  '''
  # 统计信号总类（38）all_signal.dat
  # file_content.dat ---> file_content_list.dat
  # 统计记录总数（462）file_signal_map.dat
  '''
  statistic_all_signals()

  # 统计包含 ICP, ABP, PLETH 信号的记录 和 病人个数
  target_signals = {"ICP", "ABP", "PLETH"}
  statistic_target(target_signals, "file_signal_map_icp_abp_pleth")

  statistic_Large5min("data/pXX", "csv_datLarge5min.csv", 125 * 60 * 5)
  print(
    f" ========================= save Large5min finished: csv_datLarge5min !!! ========================")

  statistic_noNaN(save_path="preprocess/result/pre", save_file="noNaN.csv")
  print("=================================  save noNaN finished: noNaN.csv ===============================")

  statistic_validICP(read_file="preprocess/result/noNaN.csv", save_path="preprocess/result/pre",
                     save_file="validICP.csv")
  print("=================================  save validABP finished: validABP.csv===============================")

  statistic_validABP(read_file="preprocess/result/validICP.csv", save_path="preprocess/result/pre",
                     save_file="validABP.csv")
  print("=================================  save validABP finished: validABP.csv===============================")
```
- 包含 ICP , ABP, PLETH, II 的病人总数: 161
- len(signals) >= min_length 的病人总数: 124
- No NaN values present in any signal 的病人总数: 122 

以下操作先滤波：
- ICP > -10 and ICP < 200 的病人总数：122  (滤波 numtaps = 65
- ABP  > 20 and ABP < 300 的病人总数：122 （滤波 numtaps = 9

以下需要先分段，逐个片段计算，符合要求的片段累计5min，则保留
- No NaN in mean HR 的病人总数：
- 排除周期< 0.56 s和> 1.04 s的ICP/ABP/PLETH波形 的病人总数 ? 
  - 自相关函数确定ABP ICP PLETH的周期 
  - 确定心跳周期的范围
- ...


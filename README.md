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
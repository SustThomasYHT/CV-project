# 利用TrackNet和XGBoost实现羽毛球跟踪和对打判断

先配置好环境
```bash
pip install -r requirements.txt
```

创建文件夹ckpts，将下载好的模型放入其中
模型请联系作者获取

创建文件夹video，放入想要分析的羽毛球比赛视频

在thomas_predict.py中修改视频名称和选择xgboost模型

运行
```bash
python thomas_predict.py
```

即可在project文件夹得到最终的剪辑完成后的视频
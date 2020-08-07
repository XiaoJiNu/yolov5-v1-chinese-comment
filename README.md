# yolov5-v1-chinese-comment

yolov5 v1版本中文注释

1 yolov5的整体训练流程如下图所示，本仓库主要对模型构建以及损失函数计算部分进行了详细地注释，希望对新手有帮助。

https://github.com/XiaoJiNu/yolov5-v1-chinese-comment/blob/master/%E4%BB%A3%E7%A0%81%E8%A7%A3%E8%AF%BB/yolov5%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%9B%BE.png

2 所有的注释文档都放在代码解读文件夹，最核心的文件是“yolo.py解析”和“loss计算”这两个文件夹，分别对应模型构建和loss计算部分。

3 我在注释的时候md文档是在typro缩小视图下进行编辑的，上传到github后排版发生了变化，请在缩小视图下阅读。


建议：这些注释是在yolov5中提供的coco128数据上进行的调试，建议一边调试一边对照着看文档。
     注释主要以文档为主，代码中的注释我没有进行整理，有些可能与文档对应不上。
     
注：有一些细节目前依然不是很清楚，肯定会有一些错误，希望大家批评指正，让文档更加完善。

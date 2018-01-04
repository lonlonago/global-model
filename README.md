global model 复现

最近复现了ICCV 2015的论文 Context-aware CNNs for person head detection 的 global model 部分，论文虽然有点老了，但是复现起来还是有点工程价值的，说下各个部分的细节。

论文的网址如下：
http://www.di.ens.fr/willow/research/headdetection/


我写的说明文章地址：
https://zhuanlan.zhihu.com/p/32625093

主要思想
Global model： 主要是给出一个四个尺度人头热力图，越热的地方含有人头的概率越大。使用整幅图像的信息来定位物体，论文使用 RCNN模型 作为 local model，然后 combine global model 的结果来提升模型表现，相当于一个 ensemble 的过程。

不过这篇论文更有价值的部分其实是在它的 pairwise model 部分，通过考虑 RCNN 网络提取的 box 之间的关联关系来抑制错误的 box，但这部分由其他人完成，所以这里不再叙述。



结果对比
论文的Global model dets结果:
mean recall_1: 0.5448           mean recall_0: 0.9943
mean FP: 0.00563                mean FN: 0.4551

我的训练结果：
mean recall_1: 0.5976           mean recall_0: 0.9964
mean FP: 0.00352                mean FN: 0.4023

最终 combine 后提升 faster-rcnn 的检测结果0.32%（论文提升rcnn 的 local model 结果0.7% ）


Global model 训练流程
1.生成标签
通过global_utils.py 文件里面的 prepare_global_label（） 函数为每张图片生成四个尺度上的284个人头标签，图片数据和xml数据格式需要为VOC 格式，返回经过 pading 之后的图片和标签列表。

2. 转化图片和标签为tfrecord
通过 convert_tfrecord.py 文件里面的 convert_original_iamge（） 函数转换，生成 tfreocrd 文件。

3. 训练和测试
通过my_train.py , my_test.py 两个文件（需要使用两个服务器分开运行），使用上 tfrecord 文件进行模型训练，使用VGG19 作为基本网络，开始训练最后四层，学习率0.001进行训练，然后调整学习率为0.00001 训练所有层，重点关注模型的recall_1 指标。

4. combine 结果
得到训练好的模型后，在faster-rcnn的test模式中，在它计算mAP之前，将它生成的框和得分score 送入 global_utils.py 中的 combine_global（） 函数中，得到经过global model重新调整后的得分score，然后计算mAP。（调整得分的权重在val 数据集上调试得到一个最优值，然后应用在test 数据集上）

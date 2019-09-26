# 代码解释
## data数据
    train-images训练集
    train-labels训练集标签
    t10k-images预测集
    t10k-labels预测集标签
## tmp存放生成模型
    ckpt：存放saver生成文件
    test：存放builder生成（预测必须）
## my_mnist_train.py
    训练：保存模型
## my_mnist_predict.py
    预测：加载模型
# 使用方法
    先执行my_mnist_train.py(每次执行需把tmp/test下的文件删除)
    再执行my_mnist_predict.py

    


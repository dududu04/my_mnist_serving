import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
pb_file_path = os.getcwd()


def full_connected():
    data = input_data.read_data_sets("./data", one_hot=True)

    # 定义数据占位符 x[None,784] y_true[None,10]

    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 建立一个全连接层的神经网络 w[784,10] bias[10]

    weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0))
    bias = tf.Variable(tf.constant(0.0, shape=[10]))
    y = tf.add(tf.matmul(x, weight), bias, name="y")

    # 计算损失

    # 求平均交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # 梯度下降算法

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 计算准确率

    equal_list = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            # 取出真实存在的特征值和目标值
            data_x, data_y = data.train.next_batch(50)

            # 运行train_op的参数训练
            sess.run(train_op, feed_dict={x: data_x, y_: data_y})

            print("训练第%d步，准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: data_x, y_: data_y})))

        # 创建模型导出的目录
        export_path_base = os.path.join('./tmp', 'test')
        export_path = os.path.join(tf.compat.as_bytes(export_path_base),
                                   tf.compat.as_bytes(str(FLAGS.model_version)))
        print('导出的模型存放在', export_path)

        """
        构建SavedModel协议缓冲区并保存变量和资产。
        SavedModelBuilder类提供了构建SavedModel协议缓冲区的功能。 
        具体来说，这允许将多个元图保存为单个语言中性SavedModel的一部分，同时共享变量和资产。
        """
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # x 为输入tensor
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x)}

        # inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x),
        #           'input_w': tf.saved_model.utils.build_tensor_info(weight),
        #           'input_b': tf.saved_model.utils.build_tensor_info(bias)}

        # y 为最终需要的输出结果tensor
        outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'predict_images': signature, tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})

        builder.save()
    return None


if __name__ == '__main__':
    full_connected()

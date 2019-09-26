import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


pb_file_path = os.getcwd()
signature_key = 'predict_images'
input_key = 'input_x'
output_key = 'output'


def full_connected_predict():
    data = input_data.read_data_sets("./data/", one_hot=True)
    with tf.Session() as sess:
        # 加载操作需要用于恢复图形定义和变量的会话，用于标识要加载的元图形def的标记以及SavedModel的位置。
        # 加载后，作为特定元图def的一部分提供的变量和资产子集将恢复到提供的会话中。
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], pb_file_path+'/tmp/test/1')

        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        for i in range(1):

            x_test, y_test = data.test.next_batch(1)
            print(x_test)
            print("第%d张图片,手写数字目标是:%s,预测结果是:%s"%(
                i,
                tf.argmax(y_test, 1).eval()
                ,
                tf.argmax(sess.run(y, feed_dict={x: x_test, y: y_test}), 1).eval()
            ))
    return None


if __name__ == '__main__':
    full_connected_predict()

import tensorflow as tf

from model import Unet
from utils import *

flags = tf.app.flags

flags.DEFINE_integer('width', 256,
                     """ The size of image to use.""")
flags.DEFINE_integer('height', None,
                     """ The size of image to use. If None, same value as width""")
flags.DEFINE_float('learning_rate', 0.0001,
                     """ Learning rate of for RMSProp""")

flags.DEFINE_string('data_set', 'data_set',
                    """ The name of dataset matlab file.""")
flags.DEFINE_string('test_set', 'test_set',
                    """ The name of testset matlab file.""")
flags.DEFINE_string('result_name', None,
                    """ The name of matlab file to save the result.""")

flags.DEFINE_string('ckpt_dir', None,
                    """ Directory name to save the checkpoints.""")
flags.DEFINE_integer('logs_step', None,
                     """ logs_step. If none, epoch_num/5. """)
flags.DEFINE_integer('restore_step', None,
                     """ Index of restore ckpt file.""")

flags.DEFINE_integer('hidden_num', 64,
                     """ Number of channels at first hidden layer.""")
flags.DEFINE_integer('epoch_num', 2000,
                     """ Epoch to train.""")
flags.DEFINE_integer('batch_size', 32,
                     """ The size of batch images.""")

flags.DEFINE_integer('num_gpu', 1,
                     """How many GPUs.""")
flags.DEFINE_boolean('is_train', True,
                     """ True for training, False for testing.""")
flags.DEFINE_boolean('w_bn', False,
                     """ Use batch-normalization.""")


FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    if FLAGS.height is None:
        FLAGS.height = FLAGS.width

    unet = Unet(width=FLAGS.width, height=FLAGS.height, learning_rate=FLAGS.learning_rate,
                data_set=FLAGS.data_set, test_set=FLAGS.test_set, result_name=FLAGS.result_name,
                ckpt_dir=FLAGS.ckpt_dir, logs_step=FLAGS.logs_step, restore_step=FLAGS.restore_step,
                hidden_num=FLAGS.hidden_num, epoch_num=FLAGS.epoch_num, batch_size=FLAGS.batch_size,
                num_gpu=FLAGS.num_gpu, is_train=FLAGS.is_train, w_bn=FLAGS.w_bn)

    show_all_variables()

    if FLAGS.is_train:
        unet.train()
    else:
        unet.test()


if __name__ == '__main__':
    tf.app.run()

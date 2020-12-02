from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

#通过Callback实现动态的学习率控制。
class WarmupDecayNoDown(Callback):
    def __init__(self, lr_base=0.0002, warmup_epochs=2, norm_epochs = 4, steps_per_epoch = 0, epoches = 10):
        self.num_passed_batchs = 0   #一个计数器
        self.warmup_epochs = warmup_epochs
        self.norm_epochs = norm_epochs
        self.lr = lr_base # learning_rate_base
        self.steps_per_epoch = steps_per_epoch # 也是一个计数器
        self.epoches = epoches
        self.batch_loc = 0

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        warmup_count = self.warmup_epochs * self.steps_per_epoch
        norm_count = self.norm_epochs * self.steps_per_epoch
        all_count = self.epoches * self.steps_per_epoch
        self.batch_loc += 1
        if self.batch_loc <= warmup_count:
            tlr = self.batch_loc / warmup_count * self.lr
            K.set_value(self.model.optimizer.lr, tlr)
        elif self.batch_loc <= norm_count:
            K.set_value(self.model.optimizer.lr, self.lr)
        else:
            tlr = self.lr + (0.00 - self.lr) * (self.batch_loc - norm_count) / (all_count - norm_count)
            K.set_value(self.model.optimizer.lr, max(tlr, 0))
        return
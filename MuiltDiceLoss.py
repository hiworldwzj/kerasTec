import tensorflow.keras.backend as K

"""
多分类使用的带权重衰减的dice loss函数
"""

def get_multi_dice_loss_fun(class_count, smooth=1, alpha = 2):
    def dice_loss(y_true, y_pred):
        weight_delay = K.pow(1 - y_pred, alpha)
        weight_delay_pred = weight_delay * y_pred
        loss_ans = 2 * (weight_delay_pred * y_true + smooth) / (weight_delay_pred + y_true + smooth)
        return -K.mean(loss_ans)
    return dice_loss

def get_multi_focal_loss(class_count, alpha = 2):
    def focal_loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype="float32")
        return K.mean(-K.sum(K.pow(1 - y_pred, 2) * K.log(y_pred) * y_true, axis=-1, keepdims=False))
    return focal_loss

def get_direct_multi_dice_loss_fun(class_count, smooth=0.1):
    """
    直接建模f值的函数
    :param class_count: 分类个数
    :param smooth: 平滑常数
    :return: dice_loss函数
    """
    def dice_loss(y_true, y_pred):
        ans_list = []
        for i in range(class_count):
            tmp_y_true = y_true[:, i]
            tmp_y_pred = y_pred[:, i]
            tmp_y_max_pred = K.max(y_pred, axis=-1, keepdims=False)
            tmp_pred_one_zero = K.cast(K.equal(tmp_y_pred, tmp_y_max_pred), dtype="float32")
            tmp_true_pred = tmp_y_true * tmp_pred_one_zero
            p = (K.sum(K.stop_gradient(tmp_true_pred - tmp_y_pred) + tmp_y_pred) + smooth) / (K.sum(K.stop_gradient(tmp_pred_one_zero - tmp_y_pred) + tmp_y_pred) + smooth)
            r = (K.sum(K.stop_gradient(tmp_true_pred - tmp_y_pred) + tmp_y_pred) + smooth) / (K.sum(tmp_y_true) + smooth)
            ans_list.append(2*p*r /(p + r))
        return -K.mean(K.stack(ans_list))
    return dice_loss

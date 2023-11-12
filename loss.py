import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coefficient(self, y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice

    def call(self, y_true, y_pred):
        # Compute mean dice loss across all classes
        dice_loss = 0
        for class_idx in range(y_true.shape[-1]):
            dice_loss += (1 - self.dice_coefficient(y_true[..., class_idx], y_pred[..., class_idx]))
        return dice_loss / tf.cast(y_true.shape[-1], tf.float32)



class DiceCELoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def dice_coefficient(self, y_true, y_pred):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice

    def call(self, y_true, y_pred):
        # Compute mean dice loss across all classes
        dice_loss = 0
        for class_idx in range(y_true.shape[-1]):
            dice_loss += (1 - self.dice_coefficient(y_true[..., class_idx], y_pred[..., class_idx]))

        # Compute the mean cross-entropy loss
        ce_loss = self.cross_entropy(y_true, y_pred)

        # Combine Dice and cross-entropy loss
        dice_ce_loss = dice_loss / tf.cast(y_true.shape[-1], tf.float32) + ce_loss
        return dice_ce_loss
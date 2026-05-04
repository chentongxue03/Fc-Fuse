import tensorflow as tf
from loss import cross_modal_alignment_loss


def train_step(model_parts, optimizer, sample1, sample2, label1, label2,
               alpha=1.0, beta=1.0, margin=1.0):
    encode, fusion_dense, classifier, *_ = model_parts

    with tf.GradientTape() as tape:
        # 编码两组样本，得到模态特征
        features1 = encode(*sample1)  # 样本组1
        features2 = encode(*sample2)  # 样本组2

        # 融合并分类
        fused1 = fusion_dense(tf.concat(features1, axis=-1))
        pred1 = classifier(fused1)
        fused2 = fusion_dense(tf.concat(features2, axis=-1))
        pred2 = classifier(fused2)

        # 计算跨模态对齐损失
        total_loss, loss_dict = cross_modal_alignment_loss(
            features1, features2,
            pred1, pred2,
            label1, label2,
            margin=margin,
            alpha=alpha,
            beta=beta
        )

    # 收集所有可训练变量
    trainable_vars = []
    for model in model_parts[3:]:  # stats_nn, pkt_nn, pyl_nn
        trainable_vars.extend(model.trainable_variables)
    trainable_vars.extend(fusion_dense.trainable_variables)
    trainable_vars.extend(classifier.trainable_variables)

    # 应用梯度
    grads = tape.gradient(total_loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    # 返回损失字典
    return {k: v.numpy() for k, v in loss_dict.items()}
import glob
import tensorflow as tf
from tensorflow.keras import optimizers
from data_pressor import load_data
import numpy as np
from model import build_aligned_model
from train import train_step
from evaluate import evaluate_model
import os
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sample_batch(data, batch_size):
    """采样函数，返回两个样本组"""
    n = len(data['y_train'])
    idx1 = np.random.choice(n, batch_size, replace=False)
    idx2 = np.random.choice(n, batch_size, replace=False)

    s1 = (
        data['X_train_stats'][idx1],
        data['X_train_pkt_mag'][idx1],
        data['X_train_pyl'][idx1]
    )
    s2 = (
        data['X_train_stats'][idx2],
        data['X_train_pkt_mag'][idx2],
        data['X_train_pyl'][idx2]
    )
    return s1, s2, data['y_train'][idx1], data['y_train'][idx2]


def evaluate_accuracy(model_parts, X_stats, X_pkt, X_pyl, y_true, batch_size=512):
    encode, fusion_dense, classifier, *_ = model_parts
    preds_labels = []
    all_labels = []

    # 分批处理所有样本
    for i in range(0, len(y_true), batch_size):
        # 获取当前批次
        end_idx = min(i + batch_size, len(y_true))
        xs = X_stats[i:end_idx]
        xp = X_pkt[i:end_idx]
        xl = X_pyl[i:end_idx]

        # 特征提取和预测
        f1, f2, f3 = encode(xs, xp, xl)
        fused = fusion_dense(tf.concat([f1, f2, f3], axis=-1))
        preds = classifier(fused)

        # 收集预测结果和真实标签
        batch_preds = tf.argmax(preds, axis=1).numpy()
        preds_labels.extend(batch_preds)
        all_labels.extend(y_true[i:end_idx])

    # 计算整体准确率
    return accuracy_score(all_labels, preds_labels)


def main():
    # 配置参数
    csv_path = "../file/botiot_last_datasets.csv"
    epochs = 50
    batch_size = 256
    alpha = 1.0  # 组内对齐权重
    beta = 0.5  # 组间排斥权重
    margin= 1.0
    learning_rate = 1e-3
    patience = 5
    best_val_acc = 0.0
    wait = 0
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{timestamp}"

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 载入数据
    data = load_data(csv_path)
    print("数据加载完成。")
    print(f"训练集大小: {len(data['y_train'])}")
    print(f"测试集大小: {len(data['y_test'])}")

    # 构建模型
    encode, fusion_dense, classifier, stats_nn, pkt_nn, pyl_nn = build_aligned_model(
        stats_dim=data['X_train_stats'].shape[1],
        pkt_len=data['max_pkt_len'],
        pyl_len=data['max_payload_len_combined'],
        mag_input_dim=data['mag_input_dim'],
        num_classes=data['num_classes']
    )

    model_parts = (encode, fusion_dense, classifier, stats_nn, pkt_nn, pyl_nn)
    optimizer = optimizers.Adam(learning_rate)

    # 训练循环
    for epoch in range(epochs):
        epoch_losses = {"CL1": [], "CL2": [], "DN1": [], "DN2": [], "DJ": [], "Total": []}
        start_time = time.time()

        # 计算批次数量
        n_batches = len(data['y_train']) // batch_size
        for _ in range(n_batches):
            s1, s2, y1, y2 = sample_batch(data, batch_size)
            losses = train_step(
                model_parts, optimizer, s1, s2, y1, y2,
                # 第二种
                alpha=alpha, beta=beta, margin=margin
            # #     第三种/四
            #     alpha=alpha, beta=beta,temperature=0.3
            )

            # 收集损失
            for k in epoch_losses:
                epoch_losses[k].append(losses[k])

        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        # 评估
        train_acc = evaluate_accuracy(model_parts,
                                      data['X_train_stats'],
                                      data['X_train_pkt_mag'],
                                      data['X_train_pyl'],
                                      data['y_train'])

        val_acc = evaluate_accuracy(model_parts,
                                    data['X_test_stats'],
                                    data['X_test_pkt_mag'],
                                    data['X_test_pyl'],
                                    data['y_test'])

        # 计算时间
        epoch_time = time.time() - start_time

        # 打印结果
        print(f"Epoch {epoch + 1}/{epochs} [{epoch_time:.1f}s] - "
              f"Loss: {avg_losses['Total']:.4f} | "
              f"CL: {avg_losses['CL1']:.4f}/{avg_losses['CL2']:.4f} | "
              f"DN: {avg_losses['DN1']:.4f}/{avg_losses['DN2']:.4f} | "
              f"DJ: {avg_losses['DJ']:.4f} | "
              f"Acc: Train {train_acc:.4f} Val {val_acc:.4f}")

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            # 保存最佳模型
            save_path = os.path.join(log_dir, f"best_model_epoch{epoch + 1}_acc{val_acc:.4f}")
            os.makedirs(save_path, exist_ok=True)
            stats_nn.save(os.path.join(save_path, "stats_nn"))
            pkt_nn.save(os.path.join(save_path, "pkt_nn"))
            pyl_nn.save(os.path.join(save_path, "pyl_nn"))
            fusion_dense.save(os.path.join(save_path, "fusion_dense"))
            classifier.save(os.path.join(save_path, "classifier"))
            print(f"保存最佳模型: Val Acc = {val_acc:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"早停: {patience} 轮无改进")
                break

    print(f"训练完成。最佳验证准确率: {best_val_acc:.4f}")

    # 加载最佳模型进行评估
    if best_val_acc > 0:
        print("\n加载最佳模型进行详细评估...")
        best_save_path = os.path.join(log_dir, f"best_model_epoch*_acc{best_val_acc:.4f}")
        best_dirs = sorted(glob.glob(best_save_path))
        if best_dirs:
            best_path = best_dirs[0]
            print(f"加载最佳模型: {best_path}")

            # 重新构建模型
            encode, fusion_dense, classifier, _, _, _ = build_aligned_model(
                stats_dim=data['X_train_stats'].shape[1],
                pkt_len=data['max_pkt_len'],
                pyl_len=data['max_payload_len_combined'],
                mag_input_dim=data['mag_input_dim'],
                num_classes=data['num_classes']
            )

            # 加载权重
            stats_nn = tf.keras.models.load_model(os.path.join(best_path, "stats_nn"))
            pkt_nn = tf.keras.models.load_model(os.path.join(best_path, "pkt_nn"))
            pyl_nn = tf.keras.models.load_model(os.path.join(best_path, "pyl_nn"))
            fusion_dense = tf.keras.models.load_model(os.path.join(best_path, "fusion_dense"))
            classifier = tf.keras.models.load_model(os.path.join(best_path, "classifier"))

            # 重新定义encode函数
            def encode(x_stats, x_pkt, x_pyl):
                f1 = stats_nn(x_stats)
                f2 = pkt_nn(x_pkt)
                f3 = pyl_nn(x_pyl)
                return f1, f2, f3

            best_model_parts = (encode, fusion_dense, classifier, stats_nn, pkt_nn, pyl_nn)

            # 进行详细评估
            print("\n在测试集上进行详细评估:")
            evaluate_model(
                best_model_parts,
                data['X_test_stats'],
                data['X_test_pkt_mag'],
                data['X_test_pyl'],
                data['y_test'],
                data['label_encoder']
            )

            # 可视化训练过程中的损失变化
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_losses['Total'], label='Total Loss')
            plt.plot(epoch_losses['CL1'], label='CL1')
            plt.plot(epoch_losses['CL2'], label='CL2')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Classification Losses')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epoch_losses['DN1'], label='DN1')
            plt.plot(epoch_losses['DN2'], label='DN2')
            plt.plot(epoch_losses['DJ'], label='DJ')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Alignment Losses')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "training_losses.png"))
            print(f"训练损失图已保存到: {os.path.join(log_dir, 'training_losses.png')}")
        else:
            print("未找到最佳模型路径")
    else:
        print("没有可用的最佳模型进行评估")


if __name__ == "__main__":
    main()
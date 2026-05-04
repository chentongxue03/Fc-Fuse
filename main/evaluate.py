import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_parts, X_stats, X_pkt, X_pyl, y_true, label_encoder, batch_size=128):
    encode, fusion_dense, classifier, *_ = model_parts

    y_preds = []

    for i in range(0, len(X_stats), batch_size):
        batch_stats = X_stats[i:i + batch_size]
        batch_pkt = X_pkt[i:i + batch_size]
        batch_pyl = X_pyl[i:i + batch_size]

        # 逐批提取特征并分类
        f1, f2, f3 = encode(batch_stats, batch_pkt, batch_pyl)
        fused = fusion_dense(tf.concat([f1, f2, f3], axis=-1))
        logits = classifier(fused)
        y_pred = tf.argmax(logits, axis=1).numpy()
        y_preds.append(y_pred)

    y_pred_all = np.concatenate(y_preds)

    acc = np.mean(y_pred_all == y_true)
    print(f"\n 测试集准确率: {acc:.4f}")

    # === 分类报告 ===
    report = classification_report(y_true, y_pred_all, target_names=label_encoder.classes_, digits=4)
    print(" 分类报告:\n", report)

    # 保存报告为 txt 文件
    with open("classification_report.txt", "w") as f:
        f.write(f"测试集准确率: {acc:.4f}\n\n")
        f.write(report)

# === 混淆矩阵 ===
    cm = confusion_matrix(y_true, y_pred_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()




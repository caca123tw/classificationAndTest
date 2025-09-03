import os
import csv
import json
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# ======================
# 0) 基本設定
# ======================
DATA_DIR   = r"C:\Users\CalvinPC\Desktop\database\flower_photos\flower_photos"
OUT_DIR    = os.path.join(os.getcwd(), f"run_{time.strftime('%Y%m%d_%H%M%S')}")
MODEL_BASENAME = "resnet50_flower"

SEED      = 123
IMG_SIZE  = (448, 448)      # 可調 448 以追更高精度
NUM_EPOCHS_STAGE1 = 35      # 凍結
NUM_EPOCHS_STAGE2 = 45      # 微調
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-6
DROPOUT   = 0.5
LABEL_SMOOTHING = 0.05

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.7, 0.2, 0.1  # 70/20/10

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# ======================
# 損失函式相容封裝（舊版 TF 沒有 label_smoothing 會自動退回）
# ======================
def make_sparse_ce(label_smoothing: float = 0.1):
    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        print("⚠️ 目前的 TensorFlow 不支援 label_smoothing，改用一般版 SparseCategoricalCrossentropy。")
        return tf.keras.losses.SparseCategoricalCrossentropy()

# ======================
# 1) GPU VRAM 決定 batch_size
# ======================
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip().split("\n")[0]
        return int(result)
    except Exception:
        return None

gpu_mem = get_gpu_memory()
if gpu_mem:
    if gpu_mem > 16000:      BATCH_SIZE = 128
    elif gpu_mem > 8000:     BATCH_SIZE = 64
    elif gpu_mem > 4000:     BATCH_SIZE = 32
    else:                    BATCH_SIZE = 16
else:
    BATCH_SIZE = 16
print(f"✅ Detected GPU Memory: {gpu_mem} MB" if gpu_mem else "⚠️ No GPU detected")
print(f"👉 Using batch_size = {BATCH_SIZE}")

# ======================
# 2) 掃描所有檔案與標籤 → 分層切分 70/20/10
# ======================
# 只用來掃描（不 split）
from tensorflow.keras.preprocessing.image import ImageDataGenerator
scanner = ImageDataGenerator()
scan_flow = scanner.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=1,
    shuffle=False,              # 重要：保持順序
    class_mode="sparse",        # 取得整數標籤
    interpolation="bilinear"
)

idx_to_class = {v: k for k, v in scan_flow.class_indices.items()}
CLASS_NAMES  = [idx_to_class[i] for i in range(len(idx_to_class))]
NUM_CLASSES  = len(CLASS_NAMES)
rel_paths    = scan_flow.filenames
ALL_PATHS    = [os.path.join(DATA_DIR, p) for p in rel_paths]
ALL_LABELS   = scan_flow.classes.astype(np.int32)

print("Classes:", CLASS_NAMES, "| Total images:", len(ALL_PATHS))

def stratified_split(paths, labels, seed=SEED, r_train=0.7, r_val=0.2, r_test=0.1):
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    paths, labels = np.array(paths), np.array(labels)
    tr_idx, va_idx, te_idx = [], [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * r_train))
        n_va = int(round(n * r_val))
        n_te = n - n_tr - n_va  # 補齊
        tr_idx.extend(idx[:n_tr])
        va_idx.extend(idx[n_tr:n_tr+n_va])
        te_idx.extend(idx[n_tr+n_va:])

    tr_idx = rng.permutation(tr_idx)
    va_idx = rng.permutation(va_idx)
    te_idx = rng.permutation(te_idx)

    return (paths[tr_idx], labels[tr_idx],
            paths[va_idx], labels[va_idx],
            paths[te_idx], labels[te_idx])

train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
    ALL_PATHS, ALL_LABELS, r_train=SPLIT_TRAIN, r_val=SPLIT_VAL, r_test=SPLIT_TEST
)

print(f"Split -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")

# ======================
# 3) 建 tf.data 資料集（外部不做 preprocess/rescale；增強在模型內）
# ======================
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)  # 保持 0~255；preprocess 在模型內
    return img, label

def make_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_paths, train_labels, training=True)
val_ds   = make_ds(val_paths,   val_labels,   training=False)
test_ds  = make_ds(test_paths,  test_labels,  training=False)

# ======================
# 4) 建立模型（ResNet50；內建 preprocess_input）
# ======================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.08, 0.08),  # 新增：最多 ±8% 平移
], name="data_aug")

def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, dropout=DROPOUT):
    base = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)                             # 只在訓練時生效
    x = tf.keras.applications.resnet50.preprocess_input(x)    # 放在模型內，避免雙重化
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="resnet50_flower"), base

model, base_model = build_model()

# ======================
# 5) Callbacks
# ======================
ckpt_path = os.path.join(OUT_DIR, "best_model.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",  # 監控驗證準確率
        patience=20,  # 容忍 10 個 epoch 沒提升
        min_delta=1e-4,  # 至少要提升 0.0001
        mode="max",  # 準確率越大越好
        restore_best_weights=True,
        verbose=1
    )
]

# ======================
# 6) Stage 1（凍結）
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_STAGE1),
    loss=make_sparse_ce(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)
print("\n===== Stage 1: Frozen backbone =====")
hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS_STAGE1,
    callbacks=callbacks,
    verbose=1
)

# ======================
# 7) Stage 2（微調最後 150 層）
# ======================

base_model.trainable = True
for layer in base_model.layers[:-150]:
    layer.trainable = False

BATCH_SIZE_STAGE2 = 8
BATCH_SIZE_VAL    = 16
train_ds_ft = train_ds.unbatch().batch(BATCH_SIZE_STAGE2).prefetch(AUTOTUNE)
val_ds_ft   = val_ds.unbatch().batch(BATCH_SIZE_VAL).prefetch(AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_STAGE2),
    loss=make_sparse_ce(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"]
)
print("\n===== Stage 2: Fine-tuning last 150 layers =====")
hist2 = model.fit(
    train_ds_ft,
    validation_data=val_ds_ft,
    epochs=NUM_EPOCHS_STAGE2,
    callbacks=callbacks,
    verbose=1
)

# 儲存模型
final_model_path = os.path.join(OUT_DIR, "final_model.h5")
model.save(final_model_path)
print(f"\n💾 Saved final model to: {final_model_path}")
print(f"💾 Best checkpoint at:  {ckpt_path}")

# 讀最佳模型
best_model = load_model(ckpt_path) if os.path.exists(ckpt_path) else model

# ======================
# 8) 在 val/test 上做評估 & 輸出成果
# ======================
def evaluate_and_export(ds, paths, labels, split_name: str):
    print(f"\n📊 Evaluate on {split_name}")
    loss, acc = best_model.evaluate(ds, verbose=1)
    print(f"{split_name} Accuracy: {acc:.4f}")

    # 預測（順序與 paths/labels 對齊）
    preds = best_model.predict(ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = labels

    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix ({split_name})')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=range(NUM_CLASSES), labels=CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(ticks=range(NUM_CLASSES), labels=CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, f"{split_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=160)
    plt.close()
    print(f"🖼️ Confusion matrix saved: {cm_path}")

    # 誤判清單
    mis_rows = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            conf_pred = float(preds[i, y_pred[i]])
            mis_rows.append({
                "filepath": paths[i],
                "true_label": CLASS_NAMES[y_true[i]],
                "pred_label": CLASS_NAMES[y_pred[i]],
                "pred_confidence": f"{conf_pred:.4f}"
            })
    mis_csv = os.path.join(OUT_DIR, f"{split_name.lower()}_misclassified.csv")
    with open(mis_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "true_label", "pred_label", "pred_confidence"])
        w.writeheader()
        w.writerows(mis_rows)
    print(f"📄 Misclassified saved: {mis_csv} (count={len(mis_rows)})")

    # Top-5
    topk = min(5, NUM_CLASSES)
    topk_csv = os.path.join(OUT_DIR, f"{split_name.lower()}_pred_top5.csv")
    with open(topk_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["filepath", "true_label"] + [f"top{i+1}_label" for i in range(topk)] + [f"top{i+1}_prob" for i in range(topk)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(len(y_true)):
            probs = preds[i]
            top_idx = np.argsort(probs)[::-1][:topk]
            row = {"filepath": paths[i], "true_label": CLASS_NAMES[y_true[i]]}
            for rk, cls_idx in enumerate(top_idx, start=1):
                row[f"top{rk}_label"] = CLASS_NAMES[cls_idx]
                row[f"top{rk}_prob"]  = f"{float(probs[cls_idx]):.4f}"
            w.writerow(row)
    print(f"📄 Top-5 saved: {topk_csv}")

    # 分類報告
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    with open(os.path.join(OUT_DIR, f"{split_name.lower()}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"{split_name} Accuracy: {acc:.4f}\n\n")
        f.write(report_str)
    print(f"📄 Classification report saved for {split_name}.")

    return acc

val_acc = evaluate_and_export(val_ds,  val_paths,  val_labels,  "Validation")
test_acc= evaluate_and_export(test_ds, test_paths, test_labels, "Test")

# ======================
# 9) 訓練曲線
# ======================
def _merge_history(h1, h2):
    hist = {}
    for k in set(list(h1.history.keys()) + list(h2.history.keys())):
        hist[k] = h1.history.get(k, []) + h2.history.get(k, [])
    return hist

hist = _merge_history(hist1, hist2)
epochs_all = list(range(1, len(hist.get("loss", [])) + 1))

plt.figure(figsize=(8, 6))
plt.plot(epochs_all, hist.get("loss", []), label="train_loss")
plt.plot(epochs_all, hist.get("val_loss", []), label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "training_curves_loss.png"), dpi=160); plt.close()

plt.figure(figsize=(8, 6))
plt.plot(epochs_all, hist.get("accuracy", []), label="train_acc")
plt.plot(epochs_all, hist.get("val_accuracy", []), label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training/Validation Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "training_curves_acc.png"), dpi=160); plt.close()
print("🖼️ Training curves saved.")

# ======================
# 10) 紀錄設定
# ======================
config_path = os.path.join(OUT_DIR, "run_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump({
        "data_dir": DATA_DIR,
        "out_dir": OUT_DIR,
        "seed": SEED,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_stage1": NUM_EPOCHS_STAGE1,
        "epochs_stage2": NUM_EPOCHS_STAGE2,
        "learning_rate_stage1": LEARNING_RATE_STAGE1,
        "learning_rate_stage2": LEARNING_RATE_STAGE2,
        "label_smoothing": LABEL_SMOOTHING,
        "dropout": DROPOUT,
        "classes": CLASS_NAMES,
        "gpu_mem_mb": gpu_mem,
        "split": {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST},
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
    }, f, ensure_ascii=False, indent=2)

print("\n✅ All done!")
print(f"📂 Outputs saved in: {OUT_DIR}")
print("👉 交付時：附上 Validation/Test 的混淆矩陣、分類報告、訓練曲線截圖。")
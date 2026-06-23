import os
import shutil
import random

source_dir = "dataset/all_data"
train_dir = "dataset/train"
test_dir = "dataset/test"

classes = ["ambulance", "no_ambulance"]

split_ratio = 0.8  # 80% train, 20% test

for cls in classes:
    files = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(files)

    train_count = int(len(files) * split_ratio)

    train_files = files[:train_count]
    test_files = files[train_count:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for f in train_files:
        shutil.copy(
            os.path.join(source_dir, cls, f),
            os.path.join(train_dir, cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(source_dir, cls, f),
            os.path.join(test_dir, cls, f)
        )

print("Dataset split completed!")
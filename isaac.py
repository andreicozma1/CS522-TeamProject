from PIL import Image
from kmeans import kmeans_image_compress, kmeans_classify
from data_functions import get_dataset_fn, load_dataset, resize_dataset, to_categorical
import numpy as np
import os

dataset_dir = os.path.join("datasets", "face_mask")
size = (128, 128, 3)

X_train_fn, y_train = get_dataset_fn(dataset_dir, "train")
X_train = load_dataset(X_train_fn)

y_train = y_train.astype(int)
X_train_resized = resize_dataset(X_train, size)
X_train_subset = X_train_resized[np.random.randint(0, len(X_train_resized), 1000),:]
print(X_train_subset.shape)

# mean_0 = np.mean(X_train_resized[y_train == 0], axis=0).astype(np.uint8)
# mean_1 = np.mean(X_train_resized[y_train == 1], axis=0).astype(np.uint8)

# im = Image.fromarray(mean_0)
# im.save("img/avg_no_mask.png")

# im = Image.fromarray(mean_1)
# im.save("img/avg_mask.png")

means = kmeans_classify(X_train_subset, 2, 0.0).astype(np.uint8)

im = Image.fromarray(means[0])
im.save("img/kmeans_A.png")

im = Image.fromarray(means[1])
im.save("img/kmeans_B.png")

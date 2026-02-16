

# 🚀 Running the Project in Google Colab

## ✅ Step 1: Open Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**

---

## ✅ Step 2: Install Required Libraries

If using **TensorFlow/Keras**:

```python
!pip install tensorflow torchvision
```

If using **PyTorch**:

```python
!pip install torch torchvision
```

---

## ✅ Step 3: Load EMNIST Dataset

### 🔹 Option A: Using PyTorch (Recommended – Easiest)

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='balanced',   # options: byclass, bymerge, balanced, letters, digits
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='balanced',
    train=False,
    download=True,
    transform=transform
)
```

It will automatically download EMNIST.

---

### 🔹 Option B: Using TensorFlow

TensorFlow does not directly provide EMNIST.

Download manually:

```python
!wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
!unzip gzip.zip
```

Or use `tensorflow_datasets`:

```python
!pip install tensorflow-datasets

import tensorflow_datasets as tfds

dataset, info = tfds.load('emnist/balanced', as_supervised=True, with_info=True)
```

---

## ✅ Step 4: Train Model

Run your CNN / RNN / GRU / EfficientNet / etc. model cells.

---

## ✅ Step 5 (Optional): Save Model

```python
model.save("cnn_emnist_model.h5")
```

To save to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

# 💻 Running the Project in VS Code (Local System)

---

## ✅ Step 1: Install Python

Download from: [https://www.python.org](https://www.python.org)
Make sure Python is added to PATH.

Check version:

```bash
python --version
```

---

## ✅ Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv emnist_env
```

Activate:

### Windows:

```bash
emnist_env\Scripts\activate
```

### Mac/Linux:

```bash
source emnist_env/bin/activate
```

---

## ✅ Step 3: Install Dependencies

If using PyTorch:

```bash
pip install torch torchvision matplotlib numpy
```

If using TensorFlow:

```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```

---

## ✅ Step 4: Download EMNIST Dataset

### 🔹 Using PyTorch (Auto Download)

Same code as Colab — it will download automatically.

### 🔹 Manual Download

1. Visit:
   [https://www.nist.gov/itl/products-and-services/emnist-dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

2. Download the dataset

3. Extract into your project folder

4. Load using your preprocessing script

---

## ✅ Step 5: Run the Project

If using Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

If using Python script:

```bash
python train.py
```

---

# 📁 Recommended Project Structure

```
EMNIST_Project/
│── data/
│── models/
│── train.py
│── requirements.txt
│── README.md
```

---

# 📌 Best Practice Recommendation

For EMNIST:
✔ Use `torchvision.datasets.EMNIST`
✔ Use `split='balanced'` for characters + digits
✔ Normalize images
✔ Use CNN for best accuracy (as per your result)

---


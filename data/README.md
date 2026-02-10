## 📊 Dataset Access & Setup

The data used in this project is the **Turbofan Engine Degradation Simulation Dataset (C-MAPSS)** provided by the NASA Ames Prognostics Data Repository.

### **How to Download**

1. Visit the [NASA C-MAPSS Kaggle Page](https://www.kaggle.com/datasets/behrad37/nasa-cmaps) or the official [NASA Data Portal](https://www.nasa.gov/intelligent-systems-division-prognostics-center-of-excellence-data-repository/).
2. Download the zip file containing the `.txt` files.
3. Extract the files and place the **raw text files** into the `data/raw/` directory.

### **Required Directory Structure**

For the `main.py` pipeline to run successfully, your folder must look like this:

```text
data/
└── raw/
    ├── train_FD001.txt
    ├── train_FD002.txt
    ├── train_FD003.txt
    ├── train_FD004.txt
    ├── test_FD001.txt
    ├── ...
    └── RUL_FD001.txt
    └── ...
```

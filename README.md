# Iris Flower Classification with K-Nearest Neighbors (KNN

This Jupyter notebook demonstrates the classification of iris flower species using the K-Nearest Neighbors (KNN) algorithm. The project includes data preprocessing, model training, evaluation, and visualization of results.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a machine learning pipeline to classify iris flowers into three species:
- **Setosa**
- **Versicolor** 
- **Virginica**

The classification is based on four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The notebook uses the K-Nearest Neighbors algorithm and compares performance across different values of k (number of neighbors).

## 📊 Dataset

The project uses the classic Iris flower dataset, which should be located at:
```
data/iris-flower-dataset.csv
```

**Dataset Structure:**
- **Features:** 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Target:** Species (categorical: setosa, versicolor, virginica)
- **Samples:** Typically 150 samples (50 per species)

## 🛠️ Requirements

The following Python packages are required:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## 🚀 Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
   
   Or using conda:
   ```bash
   conda install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Ensure dataset is in correct location:**
   - Place `iris-flower-dataset.csv` in the `data/` directory
   - Or modify the file path in the code accordingly

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## 💻 Usage

1. **Open the notebook** in Jupyter
2. **Run all cells** sequentially from top to bottom
3. **View results** including:
   - Accuracy scores for different k values
   - Confusion matrices
   - Classification reports
   - Visualization plots

### Key Parameters
- **Test size:** 20% of data (0.2)
- **Random state:** 42 (for reproducibility)
- **K values tested:** 5 and 50
- **Feature scaling:** StandardScaler normalization

## 📁 Code Structure

### 1. **Data Loading and Preprocessing**
```python
# Load dataset
df = pd.read_csv("data/iris-flower-dataset.csv")

# Prepare features and target
X = df.drop("species", axis=1)
y = df["species"]
```

### 2. **Data Splitting**
```python
# Split into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. **Feature Scaling**
```python
# Standardize features for better KNN performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. **Model Training and Evaluation**
The `evaluate_knn()` function:
- Trains KNN classifier with specified k value
- Makes predictions on test set
- Calculates accuracy score
- Generates confusion matrix
- Provides detailed classification report
- Creates visualization heatmap

### 5. **Performance Comparison**
Tests KNN performance with k=5 and k=50 to demonstrate the impact of the number of neighbors.

## 📈 Results

The notebook outputs the following metrics for each k value:

- **Accuracy Score:** Overall classification accuracy
- **Confusion Matrix:** Detailed breakdown of predictions vs actual
- **Classification Report:** Precision, recall, and F1-score per class
- **Heatmap Visualization:** Visual representation of confusion matrix

### Expected Performance
- KNN typically achieves high accuracy (>90%) on the Iris dataset
- Lower k values (like k=5) often perform better than higher k values (like k=50)
- The dataset is well-separated, making it ideal for demonstrating classification algorithms

## 🔧 Customization

You can modify the code to:

1. **Test different k values:**
   ```python
   for k in [1, 3, 5, 7, 10, 15, 20]:
       evaluate_knn(k, X_train, X_test, y_train, y_test)
   ```

2. **Change train/test split ratio:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

3. **Try different preprocessing methods:**
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   ```

4. **Add cross-validation:**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(classifier, X_train, y_train, cv=5)
   ```

## 🤝 Contributing

Feel free to fork this project and submit pull requests for:
- Additional algorithms comparison
- Enhanced visualizations
- Performance optimization
- Code documentation improvements

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📚 References

- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [K-Nearest Neighbors Algorithm](https://scikit-learn.org/stable/modules/neighbors.html)

---

**Note:** Make sure your dataset file path matches the one specified in the code (`data/iris-flower-dataset.csv`) or update the path accordingly.

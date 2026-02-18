📘 Implementing and Evaluating Logistic Regression from Scratch
📌 Project Overview

This project demonstrates a complete implementation of Logistic Regression from scratch using NumPy, without relying on high-level machine learning libraries for model training.

The objective is to understand the core mathematical mechanics behind Logistic Regression, including:

Sigmoid activation function

Binary Cross-Entropy (Log Loss)

Gradient Descent optimization

Model evaluation using classification metrics

For validation and comparison, the custom implementation is evaluated against Scikit-Learn's LogisticRegression model.

🎯 Objectives

Generate synthetic binary classification data

Implement Logistic Regression manually using NumPy

Train the model using Gradient Descent

Evaluate model performance using:

Accuracy

Precision

Recall

Compare results with Scikit-Learn implementation

Analyze differences in learned weights and performance

🧠 Theoretical Background
What is Logistic Regression?

Logistic Regression is a supervised learning classification algorithm used for binary classification problems.

Instead of predicting continuous values like Linear Regression, it predicts probabilities using the Sigmoid function.

Sigmoid Function
𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​


Where:

𝑧
=
𝑤
𝑇
𝑋
+
𝑏
z=w
T
X+b

The output is a probability between 0 and 1.

Cost Function (Binary Cross-Entropy)
𝐿
𝑜
𝑠
𝑠
=
−
1
𝑚
∑
[
𝑦
log
⁡
(
𝑦
^
)
+
(
1
−
𝑦
)
log
⁡
(
1
−
𝑦
^
)
]
Loss=−
m
1
	​

∑[ylog(
y
^
	​

)+(1−y)log(1−
y
^
	​

)]

This measures how well predicted probabilities match actual labels.

Optimization

Gradient Descent is used to update weights iteratively:

𝑤
=
𝑤
−
𝛼
⋅
∂
𝐿
∂
𝑤
w=w−α⋅
∂w
∂L
	​

𝑏
=
𝑏
−
𝛼
⋅
∂
𝐿
∂
𝑏
b=b−α⋅
∂b
∂L
	​


Where:

α = learning rate

L = loss function

🏗️ Project Structure
logistic-regression-from-scratch/
│
├── custom_logistic_regression.py
├── main.py
├── README.md
└── requirements.txt

⚙️ Implementation Details
Dataset

Generated using sklearn.datasets.make_classification

200 samples

5 features

Binary classification (0/1)

80-20 train-test split

Custom Model Features

The custom implementation includes:

Sigmoid function

Binary Cross-Entropy loss

Gradient computation

Gradient Descent optimization

Prediction function

Probability prediction function

📊 Model Evaluation Metrics

The following metrics are used to evaluate performance:

Accuracy

Precision

Recall

Both the custom model and Scikit-Learn model are evaluated on the same test dataset.

🔍 Comparative Analysis
Metric	Custom Model	Sklearn Model
Accuracy	~0.80–0.90	~0.85–0.92
Precision	High	Slightly Higher
Recall	Good	Slightly Better
Observations

Both models produce similar results.

Minor differences arise because:

Scikit-Learn uses advanced solvers (lbfgs, liblinear).

Scikit-Learn includes regularization by default.

Scikit-Learn has optimized convergence handling.

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/yourusername/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch

2️⃣ Install Dependencies
pip install -r requirements.txt


Or manually:

pip install numpy scikit-learn

3️⃣ Run the Script
python main.py

📈 Sample Output
---- Custom Logistic Regression ----
Accuracy: 0.87
Precision: 0.89
Recall: 0.85

---- Sklearn Logistic Regression ----
Accuracy: 0.90
Precision: 0.91
Recall: 0.88


(Note: Results may vary slightly due to randomness.)

🧩 Key Learnings

Understood the mathematical foundation of Logistic Regression

Implemented Gradient Descent manually

Learned how loss functions guide optimization

Observed impact of solver and regularization

Compared custom ML implementation with production-level library

📌 Conclusion

This project demonstrates that Logistic Regression can be fully implemented using only NumPy and mathematical concepts.

While high-level libraries simplify usage, building the algorithm from scratch provides:

Strong understanding of ML fundamentals

Better debugging skills

Deeper insight into optimization

Improved interview preparation

🛠️ Technologies Used

Python

NumPy

Scikit-Learn (only for validation and comparison)

📎 Future Improvements

Add regularization (L1/L2)

Implement early stopping

Add confusion matrix visualization

Plot loss curve during training

Extend to multi-class classification

👨‍💻 Author

Gowtham

If you want, I can now:

✅ Make a more advanced GitHub-style README with badges

✅ Add project diagrams

✅ Add mathematical derivation section

✅ Add screenshots section

✅ Generate requirements.txt file

# Toxic-content-classification-using-machine-learning-and-deep-learning
Built a Toxic Content Classification project using ML &amp; DL. Applied preprocessing, EDA, and trained Random Forest, SVM, KNN, Naive Bayes &amp; CNN models. Compared performance via accuracy, F1 &amp; ROC curves. Deployed an interactive predictor to detect toxic vs non-toxic comments.
🎯 Objectives

Clean and preprocess raw text data (remove URLs, special characters, mentions, etc.).

Perform Exploratory Data Analysis (EDA) to understand class distribution and comment characteristics.

Train and evaluate multiple ML models:

✅ Random Forest

✅ Support Vector Machine (SVM)

✅ K-Nearest Neighbors (KNN)

✅ Naive Bayes

Train and evaluate a Convolutional Neural Network (CNN) for deep text classification.

Compare all models using metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Visualize results with confusion matrices, ROC curves, and performance comparison graphs.

Build an interactive toxic comment predictor where users can input text and see predictions.

📂 Dataset

Source: A collection of comments labeled as toxic (1) or non-toxic (0).

Columns:

Comment → Original text

Label → Target class (0 = Non-toxic, 1 = Toxic)

Clean → Preprocessed text (after cleaning & normalization)

🔧 Methodology (Step by Step)

Data Loading & Cleaning

Removed URLs, mentions, hashtags, punctuation, and special characters.

Converted text to lowercase and normalized spaces.

Exploratory Data Analysis (EDA)

Distribution of toxic vs non-toxic comments.

Comment length analysis (histogram, boxplots).

Word frequency & WordCloud visualization for each class.

Feature Engineering

For ML models → used Bag-of-Words (CountVectorizer with n-grams).

For CNN → used Tokenizer + Word Embeddings + Padding.

Model Training & Evaluation

Trained Random Forest, SVM, KNN, Naive Bayes, CNN.

Evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC.

Visualized with Confusion Matrices, ROC curves, Performance Bar Charts.

Model Comparison

Tabular & graphical comparison of all models.

CNN often performs better in capturing contextual meaning.

Interactive Prediction System

User can enter custom sentences.

Model predicts whether the input is Toxic or Non-Toxic.

📊 Key Visualizations

Class Distribution Plot (toxic vs non-toxic).

Comment Length Histogram & Boxplot.

WordClouds for toxic vs non-toxic comments.

Confusion Matrices for each model.

ROC Curves to compare sensitivity & specificity.

Bar Chart comparing Accuracy, Precision, Recall, and F1-score across models.

🚀 Results & Insights

Naive Bayes & SVM work well for simple text classification.

Random Forest provides robustness but may overfit.

KNN struggles with high-dimensional sparse data.

CNN outperforms ML models by capturing semantic context better.

🔮 Future Enhancements

Use pre-trained embeddings (GloVe, FastText, BERT).

Handle multi-label classification (toxic, insult, threat, hate speech, etc.).

Deploy as a web app / API for real-time toxic comment detection.

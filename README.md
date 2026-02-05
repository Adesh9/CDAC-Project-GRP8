# CDAC-DBDAProject-KNOWITGRP8

Project Report On

Amazon Review Sentiment Analysis




Submitted in the partial fulfillment for the award of Post Graduate Diploma in Big Data Analytics (PG-DBDA)
from Know-IT ATC, CDAC ACTS, Pune


Guided by:

                                                       Mr. Milind Kapse 




Submitted By:

Vedant Chavda (250843025016)
Pruthviraj Desai (250843025042)
Shreyansh Jadhav (250843025050)
Adesh Kawre (250843025002)
 
CERTIFICATE



TO WHOMSOEVER IT MAY CONCERN




This is to certify that
Vedant Chavda (250843025016)
Pruthviraj Desai (250843025042)
Shreyansh Jadhav (250843025050)
Adesh Kawre (250843025002)




have successfully completed their project on


			          Amazon Review Sentiment Analysis


Under the guidance of Mr. Milind Kapse
 
ACKNOWLEDGEMENT


The Amazon Review Sentiment Analysis System is designed to analyze large volumes of customer reviews and classify them into positive, neutral, and negative sentiments. This project provided valuable hands-on experience in applying big data processing techniques and machine learning algorithms to extract meaningful insights from unstructured textual data.

We are all grateful to Milind Kapase Sir and for their invaluable help while working on this project. His advice and support enabled us to overcome a variety of challenges and complexities during the course of the project.

We are grateful to Mr. Shrinivas Jadhav (Vice-President, Know-it, Pune) for his assistance and support throughout the Post Graduate Diploma in Big Data Analytics (PGDBDA) course at CDAC ACTS, Pune.

Our heartfelt gratitude goes to Mrs. Dhanashree (Course Coordinator, PG-DBDA), who provided all of the necessary support and kind coordination to provide all of the necessities, such as required hardware, internet access, and extra lab hours, to complete the project and throughout the course up to the last day here at CDAC KnowIT, Pune.








From:

								    Vedant Chavda (250843025016)
                                                                          Pruthviraj Desai (250843025042)
                                                                             Shreyansh Jadhav (250843025050)
                                                                      Adesh Kawre (250843025002)     
TABLE OF CONTENTS


ABSTRACT

1.	INTRODUCTION

2.	SYSTEM REQUIREMENTS

2.1	Software Requirements

2.2	Hardware Requirements

3.	FUNCTIONAL REQUIREMENTS

4.	SYSTEM ARCHITECTURE

5.	METHODOLOGY

6.	MACHINE LEARNING ALGORITHMS

7.	MODEL TRAINING

8.	DATA VISUALIZATION AND REPRESENTATION

9.	CONCLUSION AND FUTURE SCOPE

10.	DEPLOYMENT IMAGES 
ABSTRACT

With the rapid growth of e-commerce platforms, organizations receive massive volumes of unstructured customer feedback in the form of textual reviews. Manually analyzing this data is impractical and inefficient. This project presents an end-to-end Customer Sentiment Analysis Platform that leverages Natural Language Processing (NLP) and Machine Learning techniques to automatically classify customer reviews into Positive, Neutral, or Negative sentiments.
The system integrates large-scale data preprocessing using Hive and PySpark, feature extraction using TF-IDF and transformer-based BERT embeddings as independent approaches, and supervised machine learning models for sentiment classification, where experimental results showed that BERT embeddings outperform TF-IDF due to superior contextual understanding. A user-friendly web application enables real-time single review analysis as well as batch processing of large text files. The solution is deployed on Amazon Web Services (AWS) using EC2 and RDS, ensuring scalability and persistence. This project demonstrates a production-oriented approach aligned with real-world industry practices.  
INTRODUCTION

In the digital era, customer feedback plays a critical role in shaping products and services. E-commerce platforms such as Amazon receive millions of reviews daily, making it difficult to manually analyze customer sentiment. Sentiment analysis, a subfield of Natural Language Processing (NLP), helps organizations automatically interpret opinions expressed in text data.
This project aims to design and implement a scalable sentiment analysis system that can efficiently process large volumes of customer reviews and provide meaningful insights. By leveraging both traditional NLP techniques and advanced transformer-based models, the system achieves high accuracy while remaining suitable for real-world deployment.

Objectives
•	To acquire and process large volumes of customer review data
•	To compare traditional text vectorization techniques with transformer-based embeddings
•	To build reliable sentiment classification models
•	To develop a user-friendly application for single and batch sentiment prediction
•	To deploy the solution on a scalable cloud infrastructure
•	To acquire and process large volumes of customer review data
•	To compare traditional text vectorization techniques with transformer-based embeddings
•	To build reliable sentiment classification models
•	To develop a user-friendly application for single and batch sentiment prediction
•	To deploy the solution on a scalable cloud infrastructure




















                           
DATA COLLECTION AND FEATURES

DATA COLLECTION AND FEATURES
Data Source:
The primary data source for this project is the Amazon Fine Food Reviews dataset obtained from the Kaggle platform. This dataset contains a large collection of product reviews submitted by customers on Amazon’s e-commerce platform. The dataset provides detailed information related to customer feedback, including review text, ratings, and timestamps, making it suitable for sentiment analysis and opinion mining tasks.
The dataset reflects real-world customer opinions and purchasing experiences, offering a reliable foundation for analyzing sentiment trends and consumer behavior in the e-commerce domain.

Dataset Size:

The dataset comprises approximately 568,454 review records, collected over several years. Each record represents an individual customer review and includes both structured and unstructured data. The large size and diversity of the dataset enable robust training and evaluation of machine learning models and provide sufficient variability to capture different sentiment patterns.
This extensive dataset forms a strong basis for performing large-scale sentiment analysis and deriving meaningful insights from customer reviews.

Features /Attributes: 

Below is an overview of the key features (attributes) available in the dataset:

1. Id
o Unique identifier assigned to each review record.
o Helps in distinguishing and referencing individual reviews within the dataset.

2. ProductId
o Unique identifier for the product being reviewed.
o Multiple reviews may exist for the same product, allowing analysis of aggregated sentiment.

3. UserId
o Unique identifier for the customer who submitted the review.
o Used to track user-level review behavior and contribution patterns.

4. ProfileName
o Display name of the reviewer as provided on the platform.
o Primarily used for identification and contextual reference.

5. HelpfulnessNumerator
o Number of users who found the review helpful.
o Indicates the perceived usefulness of the review content.

6. HelpfulnessDenominator
o Total number of users who voted on the helpfulness of the review.
o Used in combination with the numerator to assess review credibility.

7. Score
o Rating assigned by the user to the product, typically ranging from 1 to 5.
o Used as a key indicator for deriving sentiment labels.

8. Time
o Timestamp representing when the review was submitted.
o Enables temporal analysis of sentiment trends over time.

9. Summary
o A short summary or title of the review provided by the user.
o Offers a concise overview of the reviewer’s opinion.

10. Text
o The full review text written by the customer.
o This unstructured textual data is the primary input for sentiment analysis and machine learning models.

11. Sentiment (Derived Feature)
o A derived attribute created during preprocessing based on review scores.
o Reviews are classified into Positive, Neutral, or Negative sentiment categories.
o This classification forms the target variable for the sentiment analysis models	                          
 
SYSTEM REQUIREMENTS

Hardware Requirements:
•	Platform: Windows 10 or above, Linux
•	RAM: Recommended 16 GB of RAM
•	Peripheral Devices: Keyboard, Monitor, Mouse
•	WiFi connection with minimum 20 Mbps speed

Software Requirements:
•	Language: Python 3
•	Machine Learning
•	Tableau
•	OS: Windows
•	Pyspark
•	Hive

























	                        
 
FUNCTIONAL REQUIREMENTS

1.	Python 3:
•	Python was chosen as the primary programming language for this project due to its simplicity, readability, and ease of implementation, making it well-suited for building end-to-end machine learning applications.
•	Being an interpreted language, Python allows rapid development and testing, enabling seamless experimentation with text preprocessing, feature extraction, and model training without the need for prior compilation.
•	Python is open-source and freely available, supported by a large and active developer community, which ensures continuous improvements, extensive documentation, and reliable long-term support for machine learning and data engineering tasks.
•	The language provides access to a rich ecosystem of libraries and frameworks such as NumPy and Pandas for data handling, Matplotlib for visualization, and Scikit-learn for implementing TF-IDF–based machine learning models, all of which played a crucial role in developing the sentiment analysis pipeline.
2.	Tableau:
•	Tableau was utilized as a data visualization and business intelligence tool to analyze and present sentiment trends and insights derived from the processed review data in an interactive and visual format.
•	The platform provides an intuitive drag-and-drop interface, allowing dashboards, charts, and reports to be created efficiently without extensive coding, which helped in quickly interpreting model outputs and sentiment distributions.
•	Tableau supports multiple data sources, including structured datasets, relational databases, and big data platforms. In this project, it was used to visualize data generated from large-scale processing and model outputs, enabling better understanding of sentiment patterns across the dataset.
3.	Data Cleaning:
•	Data cleaning plays a critical role in this project as it directly affects the performance and reliability of the sentiment classification models. Proper preprocessing ensures that the textual data is consistent, relevant, and suitable for feature extraction and model training.
•	In the absence of effective data cleaning, sentiment analysis models may learn from noisy or misleading inputs, resulting in biased predictions and inaccurate sentiment classification. Therefore, data cleaning is a fundamental stage of data preparation that significantly influences model accuracy and overall system performance.
•	Enhancing data quality through systematic cleaning and preprocessing enables deeper insights into customer opinions and sentiment patterns, allowing organizations to make more reliable, data-driven decisions based on the analysis outcomes.
 
SYSTEM ARCHITECTURE

 
Fig. System Architecture of Amazon Sentiment Analysis
 
METHODOLOGY


 
Fig. Methodology of Amazon Review Sentiment Analysis
 
	                  MACHINE LEARNING ALGORITHMS

Machine learning is a subfield of artificial intelligence that involves developing algorithms and models that enable computers to learn from data and make predictions or decisions without being explicitly programmed. The goal of machine learning is to enable computers to improve their performance over time by learning from experience and feedback. In our project, we applied various Classification Algorithms such as Neural Network, Random Forest, SVM, XGBoost and Logistic Regression. After the implementation, we analyzed the accuracy of all the algorithms on our data.
1.	Neural Network 
Neural Network is a machine learning model inspired by the structure and functioning of the human brain. It consists of interconnected layers of artificial neurons that learn complex patterns by adjusting weights during training. Neural networks are widely used for tasks such as text classification, image recognition, and sentiment analysis due to their ability to capture non-linear relationships in data.
Pros: 
•	Neural Networks are capable of learning complex and non-linear patterns, making them highly effective for tasks like sentiment analysis on large and unstructured text data.
•	They can automatically extract high-level features from raw input data, reducing the need for extensive manual feature engineering.
•	Neural Networks perform well when trained on large datasets, as they improve with more data and training iterations.
2.	Random Forest 
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their predictions for classification tasks or the mean prediction for regression tasks. It enhances predictive accuracy and controls overfitting.
Pros:
•	Random Forest is robust to overfitting, especially with large datasets, due to its ensemble 
nature.
 
•	It can handle a mix of numerical and categorical features and is effective for high-dimensional data.
•	It provides feature importance scores, helping to identify the most influential variables.
Cons:
•	The model can be complex and less interpretable compared to simpler models like decision 
trees.
•	It may require more computational resources and time for training, especially with large 
datasets.
•	Random Forest can struggle with imbalanced datasets, leading to biased predictions.

3.	XGBoost
XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting framework designed for 
speed and performance. It builds models in a sequential manner, where each new model corrects the errors of the previous ones.
Pros:
•	XGBoost is highly efficient and can handle large datasets with high dimensionality.
•	It often provides superior predictive performance compared to other algorithms due to its 
•	regularization techniques.
•	It includes built-in cross-validation, making it easier to tune hyperparameters.
Cons:
•	The model can be complex and may require careful tuning of hyperparameters to achieve optimal performance.
•	It can be sensitive to noisy data and outliers, which may affect the model's accuracy.
•	XGBoost may require more computational resources compared to simpler models.
 
4.	SVM
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding an optimal hyperplane that best separates data points of different classes with the maximum margin. SVM is particularly effective in high-dimensional spaces and text classification problems such as sentiment analysis.
Pros:
•	SVM is highly effective for high-dimensional data, making it well-suited for text and NLP-based tasks.
•	It performs well even with limited training data by focusing on critical data points known as support vectors.
•	The use of kernel functions allows SVM to model complex, non-linear decision boundaries.

Cons:
•	SVM can be computationally expensive, especially when working with large datasets.
•	Choosing the right kernel and tuning hyperparameters can be challenging and time-consuming.
•	The model lacks interpretability compared to simpler algorithms like decision trees or logistic regression.
 
Model Training
Multiple classification models were trained and evaluated on both TF-IDF vectors and BERT embeddings, including:
•	Random Forest Classifier
•	XGBoost Classifier
•	Neural Networks (Feedforward Neural Network)
•	Support Vector Machine (SVM)
During experimentation, it was observed that Support Vector Machine (SVM) consistently outperformed other models in terms of classification stability and balanced sentiment prediction.
Reason for SVM Superior Performance
•	High-dimensional feature handling: BERT embeddings produce high-dimensional dense vectors. SVMs are well-suited for such spaces and are effective at finding optimal decision boundaries.
•	Better class separation: SVM was able to clearly separate Positive, Neutral, and Negative classes, whereas tree-based models struggled with overlapping sentiment boundaries.
•	XGBoost limitations: XGBoost showed a strong bias toward extreme classes and frequently failed to correctly predict the Neutral class, effectively reducing the task to binary classification (Positive vs Negative).
•	Neural Network constraints: The neural network model required significantly more epochs and hyperparameter tuning to converge. Due to limited training iterations and computational constraints, it did not achieve optimal generalization.
•	Regularization advantage: SVM’s margin maximization helped reduce overfitting and improved generalization on unseen reviews.
The final production model selected was: DistilBERT embeddings + SVM classifier
Application Development: A full-featured interactive application was developed with the following capabilities:
Single Review Prediction: Users can input a single sentence or review. The system instantly predicts sentiment (Positive / Neutral / Negative)
Batch Processing: Users can upload a .txt file containing multiple reviews. Reviews are processed in batch mode
The application generates a downloadable CSV file with review and predicted_sentiment.
Database & Storage: AWS RDS (MySQL) was integrated for cloud-based persistent storage. All predictions and uploaded files can be stored for future analysis

 
System Architecture Overview

The system follows a modular and layered architecture:
1.	Data Layer – Customer review data sourced from Kaggle and stored in distributed storage
2.	Processing Layer – Data preprocessing using Hive and PySpark in a Linux environment
3.	Feature Engineering Layer – Text transformed using TF-IDF and BERT embeddings
4.	Model Layer – Machine learning models trained for sentiment classification
5.	Application Layer – Streamlit-based web interface for user interaction
6.	Cloud Layer – AWS EC2 and RDS for deployment and persistence. This architecture ensures scalability, maintainability, and efficient processing.

The project was deployed on Amazon Web Services (AWS) using the following services:
EC2: Hosts the sentiment analysis application
RDS (MySQL): Stores prediction results and uploaded files
Linux-based cloud environment for scalability and reliability
This deployment enables real-time access, scalability, and enterprise-level availability.

Results and Evaluation
The BERT + SVC model demonstrated superior performance compared to TF-IDF based models
Context-aware embeddings improved sentiment prediction accuracy
Batch processing efficiently handled large volumes of reviews
Rule-based enhancement improved robustness for edge cases
 
        DATA VISUALIZATION AND REPRESENTATION

1.	Tableau Dashboard

 


2.	Confusion Matrices:
a.	SVM:

 

 
b.	Random Forest Classifiers: 
 
c.	Neural Network:

 

 
d.	XGBoost: 
 

Table: Performance comparison of sentiment classification models

Model	Accuracy	Macro F1	Weighted F1	Negative Recall	Neutral Recall	Positive Recall
Support Vector Classifier (SVC)	0.836	0.66	0.84	0.74	0.42	0.90
Random Forest	0.848	0.61	0.82	0.42	0.21	0.99
Neural Network (NLP)	0.860	0.62	0.84	0.73	0.14	0.96
XGBoost	0.870	0.67	0.86	0.68	0.25	0.97


 
Model Evaluation Metrics – Visual Comparison
Accuracy Comparison
 
Macro F1-Score Comparison
 

 
CONCLUSION 


This project demonstrates a complete, production-ready machine learning pipeline for sentiment analysis. By combining distributed data processing, advanced NLP techniques, interactive application design, and cloud deployment, the system provides an efficient and scalable solution for extracting actionable insights from customer reviews.

 
Future Scope
•	Integration of multilingual sentiment analysis
•	Fine-tuning domain-specific transformer models
•	Real-time streaming sentiment analysis
•	Expansion to emotion and aspect-based sentiment analysis
•	Incorporation of a Transformer architecture–based Machine Learning Translator to enable automatic translation of reviews across multiple languages before sentiment analysis, allowing the system to support global, multilingual datasets
 
Deployment Images

 
 
 
 
 
 


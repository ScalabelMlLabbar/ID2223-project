
# ID2223-project
## Project Description
The purpose of this project is to predict whether a URL is a phishing URL or a secure URL based on different URL features. We use two different data sources, one for phishing URLs and one for secure URLs. The confirmed active phishing URLs are sourced from a phishing database on github. The secure URLs are sourced from parsing domain sitemaps for URLs from the Tranco list top 1 million domains. In this project we make the assumption that the URLs we find in the sitemaps are secure and that URLs in the phishing database are malicious, without additional checking. All of the processed data is stored in our feature store in Hopsworks, we also have our model in the model registry in Hopsworks. The chosen model was a MLP, and the UI for user inference was built with a HuggingFace Space.
- Here is the link to the HuggingFace space: https://huggingface.co/spaces/BimonML/Project_phising_detection

Here is the test performance of the MLP model. This can also be found on the HuggingFace space: 
```
📊 Test Performance
Accuracy: 0.9420
Precision: 0.9256
Recall: 0.9614
F1 Score: 0.9431
ROC-AUC: 0.9741
```

## Dynamic Datasources
The Phishing databse: https://github.com/Phishing-Database/
- A regularly updated repository that helps identify phishing threats. This is the specific folder with the updated .txt files that we use in the project: https://github.com/Phishing-Database/Phishing.Database/tree/master/phishing-links-ACTIVE. Currently, in the beginning of January 2026, there are more that 750 000 active links. 

The Tranco list: https://tranco-list.eu/
- A list of current top 1 million domains on the internet. It is updated every day and is calculated by averaging different popularity ranks over a period of 30 days. 

## Feature Extraction
These are the features that are extracted from the data and used to train the model.
- _domain_age_days_: How many days a domain has been registered.
- _secure_percentage_: Percentage of HTTPS requests
- _has_umbrella_rank_: A check if there exists a umbrella rank
- _umbrella_rank_: Ranking of most queried domains based on global passive DNS usage by Cisco Umbrella
- _has_tls_: A check if there exists TLS
- _tls_valid_days_: The number of days a TLS/SSL certificate is considered valid
- _url_length_: The length of the url
- _subdomain_count_: How many subdomains there are to the domain

## Model Selection
The chosen model was a neural network, more specifically an MLP (Multi-Layer Perceptron). The model selection process involved:
1. **Initial Comparison**: Trained multiple candidate models (Random Forest, Gradient Boosting, Logistic Regression, MLP, SVM, Naive Bayes, KNN) and compared their performance on a validation set
2. **Hyperparameter Tuning**: Selected the top 3 performing models and tuned their hyperparameters using grid search with 5-fold cross-validation
3. **Final Selection**: Compared the tuned models and selected the MLP as it achieved the best accuracy while also not overfitting, achieving general good generalization.
4. **Final Training**: Retrained the MLP on the combined train+validation set with extensive hyperparameter search (100 iterations) to produce the final model with 94.2% test accuracy 

## Repository Structure
```
├── .github/
│   └── workflows/
│       ├── scheduled_data_pipeline.yml   # Scheduled data ingestion & processing
│       ├── test.yml                      # CI tests
│       └── train_model.yml               # Model training workflow
│
├── src/
│   └── phishing_detection/
│       ├── data/                         # Data ingestion and preprocessing
│       │   ├── extract_urls.py           # Script to extract urls from domains
│       │   ├── load_legit_domains.py     # Script to get the Tranco domains
│       │   ├── load_phishing_urls.py     # Script to get the phishing urls from the database
│       │   ├── seperate_domain_urls.py   # Script to seperate domains from urls
│       │   └── sitemap_parser.py         # Functions to collect domain sitemaps
│       │
│       ├── features/                     # Feature engineering
│       │   ├── batch_url_scanner.py      # Scan the collect urls and extracts features in batches
│       │   ├── feature_pipeline.py       # calling all scripts from src/data
│       │   └── urlscan_features.py
│       │
│       ├── inference/                    # Inference pipeline
│       │   └── pipeline.py
│       │
│       ├── models/                       # Model configs, model-dependent transformations, training and evaluation
│       │   ├── model_utils/
│       │   │   ├── data_prep.py          
│       │   │   ├── evaluation.py          
│       │   │   ├── model_configs.py
│       │   │   ├── train_pipeline.py
│       │   │   └── visualization.py
│       │   │
│       │   ├── model_selection.py        # Compares models and returns the best one
│       │   └── train_final_model.py      
│       │
│       └── utils/                        # Shared utilities
│           ├── hopsworks_utils.py        # functions to simplify interactions with Hopsworks
│           └── urlscan.py                # functions to simplify interactions with URLScan.io
```
## ML Pipeline Structure

### 1. Feature Pipeline
- **Data Collection**: Fetches phishing URLs from the Phishing Database and secure URLs from Tranco list domains
- **URL Scanning**: Uses URLScan.io API to analyze URLs and extract security features
- **Feature Engineering**: Extracts 8 key features (domain age, TLS status, URL length, subdomain count, etc.)
- **Storage**: Processed features are stored in Hopsworks feature store for training and inference

### 2. Training Pipeline
- **Data Preparation**: Retrieves balanced datasets from Hopsworks feature store
- **Model Selection**: Compares multiple models (Random Forest, Gradient Boosting, Logistic Regression, MLP)
- **Hyperparameter Tuning**: Performs randomized search with cross-validation to optimize model performance
- **Model Registry**: Best performing model (MLP) is saved to Hopsworks model registry

### 3. Inference Pipeline
- **User Input**: Accepts URL from user via HuggingFace Space UI
- **Feature Extraction**: Extracts same features from input URL using URLScan.io
- **Prediction**: Loads trained model from Hopsworks and predicts phishing probability
- **Result Display**: Shows prediction with confidence score to user





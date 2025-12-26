# YouTube Comment Sentiment Analysis – End-to-End MLOps Project
## 📌 Introduction
Influencers with large audiences receive thousands of comments on their YouTube videos. Manually analyzing these comments to understand audience sentiment, feedback, and engagement trends is impractical and inefficient.

This project provides a **complete end-to-end MLOps solution** for analyzing YouTube comments using machine learning and deploying the system in production. The solution is exposed through a **Chrome Extension** that delivers real-time sentiment insights to users.

The system classifies comments into **Positive, Neutral, and Negative** sentiments and provides rich analytical insights such as sentiment distribution, trends, word clouds, and downloadable reports.
## 🎯 Project Objectives
- Automate sentiment analysis of YouTube comments
- Build a scalable and reproducible ML pipeline
- Track and compare multiple experiments using MLflow
- Version data and models using DVC
- Deploy the model as a production-ready API
- Integrate predictions into a Chrome Extension
- Implement CI/CD for testing, validation, and deployment


## 🧠 Key Features
### Sentiment Analysis
- Classifies comments into **Positive / Neutral / Negative**
- ML-based supervised classification
- Handles real-world noisy text data

### Analytics & Insights
- Sentiment distribution (Pie Chart)
- Sentiment trend tracking over time
- Word cloud for frequent keywords
- Average comment length
- Average sentiment score
- Total comments & unique comments

### Reporting
- Download detailed analysis as **CSV**
- Export visual insights as **PDF**

### Chrome Extension
- Seamless integration with YouTube
- Real-time comment analysis
- Clean and interactive UI

## Project Architecture (High Level)
- Chrome Extension (JS)
- Flask API
- ML Model (Production)
- MLflow Model Registry
- Model training and experimentation are handled offline
- The best model is promoted to production
- API serves real-time predictions
- CI/CD automates testing and deployment
## 📂 Project Structure

├── data \
│ ├── raw # Original data\
│ ├── interim # Intermediate transformed data\
│ ├── processed # Final datasets for modeling\
├── models # Trained and serialized models\
├── notebooks # EDA and experimentation notebooks \
├── reports \
│ └── figures # Generated plots and charts \
├── flask_app\
│ ├── app.py   # backend code \
│ ├── requirements.txt # for dockerization\
├── src \
│ ├── data \
│ │ ├── data_ingestion.py \
│ │ └── data_preprocessing.py \
│ ├── models \
│ │ ├── model_building.py \
│ │ ├── model_evaluation.py \
│ │ └── register_model.py \
│ ├── features \
│ │ └── build_features.py \
│ ├── visualization \
│ │ └── visualize.py \
├── scripts # CI/CD & MLflow test scripts \
├── Dockerfile # Docker configuration \
├── dvc.yaml # DVC pipeline \
├── params.yaml # Model & pipeline parameters \
├── requirements.txt \
├── README.md
## 🔬 Machine Learning & Experimentation
- Multiple experiments conducted during the initial phase
- All experiments tracked using **MLflow**
- Logged:
  - Parameters
  - Metrics
  - Artifacts
- Best model selected based on evaluation metrics
- Model registered in **MLflow Model Registry**
- Promotion to production handled via CI/CD

---


## 🔁 Data & Model Versioning (DVC)
- DVC used for:
  - Dataset versioning
  - Pipeline reproducibility
- Remote storage configured (cloud-ready)
- Ensures:
  - Reproducible experiments
  - Traceable data lineage

---

## 🚀 CI/CD Pipeline (GitHub Actions)
The CI/CD pipeline automates:
- Code quality checks
- Unit tests (model, API, MLflow)
- Model validation & signature testing
- Promotion of the best model
- Docker image build
- Deployment readiness checks

All ML lifecycle steps are validated before deployment.

---

## 🐳 Docker & Deployment
- Flask API containerized using Docker
- Docker image built and pushed automatically
- Deployment-ready configuration

## 🧪 Testing
### Python Tests
- Model loading tests
- MLflow tracking tests
- API endpoint tests
- Model signature tests
---

## ⚠️ Challenges Addressed
- Noisy, unstructured real-world text
- Class imbalance
- Multi-experiment tracking
- Reproducibility of ML pipelines
- Low-latency inference
- Production-grade deployment

---

## 🙌 Acknowledgements

- Open-source ML and MLOps communities
- MLflow & DVC documentation
- YouTube creators for real-world inspiration
- Tools and frameworks that made production ML possible
## ⚠️ Important Note on Deployment & Service Availability

This project was **fully deployed on AWS** during development and testing. However, the **AWS Free Tier account used for deployment has expired**, so the live production services are **currently not accessible**.

Despite this, the **entire MLOps pipeline, cloud infrastructure setup, and deployment workflow are fully implemented and reproducible**. Anyone can redeploy the project by following the instructions below.

---

## 🚀 Redeployment Guide (Reproduce the Full Project)

To redeploy and run this project end-to-end, the following setup is required.

---

### 📦 Required Repositories

You must clone **two repositories**:

1. **Backend / MLOps Repository**  
   - Contains:
     - Data ingestion & preprocessing  
     - Model training, evaluation, and registration  
     - MLflow experiment tracking  
     - DVC pipelines  
     - CI/CD using GitHub Actions  
     - Dockerized Flask inference API  

2. 2. **Chrome Extension (Frontend) Repository**  
   - Repository Link:  
     👉 https://github.com/jatin-agrawal17/Youtube-chrome-plugin-frontend  
   - Contains:
     - Chrome plugin source code  
     - UI for visualizing sentiment insights  
     - Integration with the backend inference API   

---

### 🔐 Required Accounts & Credentials

Before running the project, ensure you have the following:

- ✅ **AWS Account**
- ✅ **S3 Bucket**
  - Used for DVC data storage and MLflow artifacts
- ✅ **IAM Role / AWS Credentials**
  - Permissions for S3, EC2, and ECR
- ✅ **EC2 Instance**
  - Hosts:
    - MLflow Tracking Server  
    - Flask inference API  
- ✅ **YouTube Data API Key**
  - Used to fetch YouTube video comments
- ✅ **Gemini API Key**
  - Used in the Chrome Extension frontend
- ✅ **Hosted Backend API URL**
  - Consumed by the Chrome plugin for predictions

> ⚠️ All sensitive credentials are managed using environment variables and **are intentionally not pushed to GitHub**.

---

### 🐳 Docker Image (Pre-built Backend)

A production-ready backend Docker image is already available:

```bash
docker pull jatinag12/plugin:latest
```
## 📊 Dataset Used

[Data](https://github.com/jatin-agrawal17/Youtube-Chrome_plugin/blob/main/data/reddit.csv)

## Screenshots

[App Screenshot](https://github.com/jatin-agrawal17/youtube-comment-sentiment-analysis/tree/master/reports/figures)

## 👤 Author

Jatin Agrawal  
📬 [LinkedIn](https://www.linkedin.com/in/jatin-agrawal-b80092367/)

## 📎 License

This project is open-source and available under the MIT License.

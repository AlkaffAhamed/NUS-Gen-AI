# NUS Gen AI
This repository contains my work for the NUS Generative AI Certification course. 

ref: [https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

## To Run Locally

**Requirements:** 

1. Anaconda (Install from [https://www.anaconda.com/download](https://www.anaconda.com/download))
2. Jupyter Notebook (can be installed from Anaconda Navigator or by running the install command in Anaconda Prompt)

**Steps:**

1. Clone this repository
2. Open Jupyter Notebook
3. Open the Notebooks from this repository using Jupyter Notebook 
3. Run the code 

## To Run in Google Colab

**Requirements:**

1. Google Account

**Steps:**

1. Clone this repository

2. Upload into Google Drive

3. Open the Notebook using Google Colab

4. Run the code

5. If required, you may need to modify the code to use the data sets from your google drive:
   ```python
   # Add before loading the dataset
   
   from google.colab import drive
   drive.mount('/content/drive')
   file_path = "/content/drive/My Drive/path_to_data_set"
   
   df = pd.read_excel(file_path)
   ```

## Course Contents

### Week 1

- Introduction to Python
- Introduction to libraries: Pandas and Matplotlib
- Object Oriented Programming
- Quiz and Assignment

### Week 2

- Introduction to Supervised, Unsupervised and Reinforcement Learning
- K-Means
  - Feature Selection
  - Cleaning Data
  - Normalization - Min-Max Scaler, Normal Scaler
  - Elbow and Silhouette methods to find optimum K
  - Training and plotting results
  - Interpreting results
- Quiz and Assignment








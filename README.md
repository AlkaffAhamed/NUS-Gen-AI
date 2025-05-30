# NUS Gen AI
This repository contains my work for the NUS Generative AI Certification course. 

ref: [https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

## To Run Locally

**Requirements:** 

1. Anaconda (Install from [https://www.anaconda.com/download](https://www.anaconda.com/download))
2. Jupyter Notebook (can be installed from Anaconda Navigator or by running the install command in Anaconda Prompt)
3. Separate environment for TensorFlow setup and Keras-Jax setup

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

### Week 3

- Introduction to Deep Learning - Artificial Neural Networks (ANN)
- Introduction to Deep Learning - Deep Convolutional Neural Networks (DCNN)
- Building ANN and DCNN models for Regression, Classification and Image Classification problems
  - Build the model architecture
  - Split Data to Train/Val/Test
  - Train Model
  - Evaluation Metrics
- Hyperparameter Tuning Technique - Manual Grid Search
- Training Optimization Techniques
  - Regularization Techniques: Dropout, L2 Regularization
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Quiz and Assignment

### Week 4

- Introducing NLP 
  - Text to Vector: Bag of Words
  - Text to Vector: Word Embeddings
  - RNN - LSTM
  - Limitations of LSTM - Sequential and cannot be processed in parallel
- Introducing Attention Mechanism 
  - Query (Q), Key (K), Value (V)
- Multi-Head Attentions vs Single Attention
- Encoder and Decoder Architectures
- Decoder Techniques for Text Generation: 
  - Greedy Search
  - Top-K Sampling
  - Top-p (Nucleus) Sampling
  - Beam Search
  - Sampling with Temperature
- Quiz and Assignment

### Week 5

- Introducing BERT and GPT Architectures
- 3 Main Types
  - Encoder only: BERT
  - Decoder only (Autoregressive): GPT
  - Encoder and Decoder: T5
- BERT
  - Encoder only model
  - Excels at contextual understanding
  - Mainly trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) Techniques (and other variants)
  - Applications: Sentiment Analysis, Named Entity Recognition, Text Classification
  - Limitations: Low Context Window, No Generation Capabilities
- GPT
  - Decoder only model (Autoregression Model)
  - Generates Texts by masking future words
  - Mainly trained with Causal Language Modelling (CLM)
- GPT and Reinforcement Learning
- GPT with RAG
- Encoder-Decoder Models: BARD, T5, ...
- Quiz and Assignment

### Week 6

- Introducing Reinforcement Learning
- Agentic AI Task Workflow: Environment, State, Action, Reward, Policy, Q-Value
- Unique features of RL: 
  - Delayed Consequences
  - Exploration
- Markov Decision Processes (MDPs), 4 Bellman Equations Optimal  Policy
- Model-Based Learning: Dynamic Programming
- Model-Free Learning 
  - Monte Carlo 
  - Temporal Difference Learning
- Q-Learning and Deep Q-Networks (DQN)
- Policy Gradient Techniques
  - Baseline b
  - Temporal Structure
  - KL Divergence Trust Region
- Actor Critic and Proximal Policy Optimisation (PPO)
- Alignment Problem
  - Reinforcement Learning from Human Feedback (RLHF)
  - Direct Preference Optimisation (DPO)
- Quiz and Assignment

### Week 7

- Autoencoders and VAEs
  - Encoder
  - Decoder
  - Latent Representation
  - Sampling and Reparameterization Trick (VAE)
  - Other variants of VAEs: 
    - Disentangled Variational Autoencoders 
    - Adversarial Autoencoders
    - Variational Recurrent Autoencoders
    - Hierarchical Variational Autoencoders
- GAN's
  - Generator
  - Discriminator
  - Problem of highly unstable training, vanishing gradients and model can collapse
  - Wasserstein GAN
  - Other variants of GANs: 
    - CycleGAN
    - StyleGAN
    - DCGAN
    - BigGAN
    - SRGAN
    - VAE-GAN
- Training VAEs and GANs
- Real Life Usage of GANs
- Quiz and Assignment








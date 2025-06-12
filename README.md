# Unsupervised Log Anomaly Detection with Few Unique Tokens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

This repository contains the code for the paper "Unsupervised Log Anomaly Detection with Few Unique Tokens." [cite_start]The project introduces a novel unsupervised method for detecting anomalies in log data, specifically tailored for the control system logs from the European XFEL (EuXFEL) accelerator. 

[cite_start]Effective anomaly detection is crucial for providing operators with a clear understanding of each node's availability, status, and potential problems, thereby ensuring smooth accelerator operation.  [cite_start]Traditional machine learning approaches often fail in this environment due to the sequential nature of the logs and the lack of a rich, node-specific text corpus.  [cite_start]Our approach addresses these challenges by using Word2Vec embeddings to represent log entries and a Hidden Markov Model (HMM) to learn the typical sequential patterns for each node.  [cite_start]Anomalies are then identified by scoring how well a new log entry fits into the learned sequence. 

### Methodology

[cite_start]The anomaly detection process consists of four main stages as described in the manuscript:

1.  **Log Pre-processing and Tokenization**: Raw log entries are cleaned and standardized. [cite_start]This involves removing special characters, masking server/device names, abstracting numerical values, converting to lowercase, and removing common stopwords. 
2.  **Log Entry Embedding with Word2Vec**: Each pre-processed log entry is converted into a dense numerical vector using the Word2Vec (CBOW) model. [cite_start]An entire log entry is represented by the mean of the vectors of its constituent tokens. 
3.  **Sequential Pattern Learning via HMM**: An HMM is trained on sequences of the log entry vectors from historical data. [cite_start]The HMM learns the probability distribution over typical sequences, effectively modeling the normal temporal behavior of a given log source. 
4.  **Anomaly Scoring**: A new log entry is scored based on a probability ratio that compares the likelihood of the log sequence with and without the new entry. [cite_start]A high score indicates that the new entry disrupts the learned pattern and is therefore a potential anomaly.  The anomaly score *s* for a new entry *o<sub>i</sub>* is calculated as:
    `s = log P(o1,...,oi-1) - log P(o1,...,oi)` 

### Dataset Information

The data used for this study consists of internal operational logs from the control system at the **European XFEL (EuXFEL) facility**.  Due to operational sensitivity, this dataset is not publicly available.

The key characteristics of the dataset are:
* **Sparse and Non-Verbose**: Log entries are diverse but not rich in text. 
* **Limited Vocabulary**: After pre-processing, the corpus contains only 475 unique tokens, resulting in fewer than 1,000 unique log messages. 

No external or third-party datasets were used in this work.

### Requirements

The code was developed in Python 3.9 and relies on the following libraries:

* `numpy`
* `gensim` 
* `hmmlearn==0.3.3` 
* `matplotlib`
* `scikit-learn` (dependency of hmmlearn)
* `scipy` (dependency of hmmlearn)

You can install the primary dependencies using pip:
```bash
pip install numpy gensim==4.3.2 hmmlearn==0.3.3 matplotlib
```

### Usage Instructions

#### Minimal Example

This minimal example demonstrates the core logic of scoring a sequence using a Gaussian HMM.

```python
from hmmlearn import hmm
import numpy as np

# A sequence of 2D observations
x = np.stack([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

# Initialize and fit the HMM on all but the last observation
model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
model.fit(x[:-1,:])

# Calculate the log probability of progressively longer sequences
logp = []
for i in range(1,x.shape[0]+1):
    logp.append(model.score(x[:i]))

logp = np.array(logp)

# The anomaly score is the difference in log probability
# A larger negative score indicates a higher anomaly likelihood
score = logp[:-1] - logp[1:]

print("Anomaly Scores:", -score)
```

#### Reproducing Paper Figures

The following code demonstrates how to reproduce the synthetic data experiments shown in **Figures 1 & 3** of the papers.

```python
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

state_dict = {
    '0' : np.array([0,1]),
    '1' : np.array([1,0]),
    'a' : np.array([1,1]) # Represents an anomalous event
}

def plot_seq_offline(seq, draw_dot_at=False):
    x = []
    for i in range(len(seq)):
        x.append(state_dict[seq[i]])
    x = np.stack(x)
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
    # Fit the HMM on the entire sequence (for Figure 3)
    model.fit(x)
    score = []
    for i in range(1,x.shape[0]+1):
        score.append(model.score(x[:i]))
    score = np.array(score)
    s = score[1:] - score[:-1]
    s = np.insert(s,0, s.max())
    plt.plot(np.array(range(len(s)))+1, -s)
    if draw_dot_at:
        plt.plot(len(s),-s[-1],'ro')
    plt.xlabel('Event number', fontsize=16)
    plt.ylabel('Score <span class="math-inline">s</span>', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

# --- Generate Figure ---
plt.figure(figsize=(16, 4), dpi=300)

# 1. No anomaly
seq = ['0','1','0','1','0','1','0','1']
plt.subplot(131)
plot_seq_offline(seq,True)
plt.title('No anomaly', fontsize=16)

# 2. Sequential anomaly
seq = ['0','1','0','1','0','1','0','0']
plt.subplot(132)
plot_seq_offline(seq,True)
plt.title('Last event has unusual position',  fontsize=16)

# 3. Content anomaly
seq = ['0','1','0','1','0','1','0','a']
plt.subplot(133)
plot_seq_offline(seq,True)
plt.title('Anomalous event',  fontsize=16)

plt.tight_layout()
plt.savefig('Figure_HMM_Anomaly_Detection.pdf', format='pdf')
plt.show()

```

### Evaluation

#### Selection of Techniques
The techniques were chosen to address the specific challenges of the EuXFEL log data:
* **Word2Vec**: Chosen to create meaningful, dense vector representations of log entries, which helps mitigate the data scarcity and high-dimensionality problems common in text data. 
* **Hidden Markov Models (HMMs)**: Chosen over more complex deep learning models (like LSTMs or Transformers) because our dataset is small, unlabeled, and sparse.  HMMs are well-suited for modeling sequential data, require fewer parameters, and can be trained effectively in an unsupervised manner on limited data.  They were chosen over simpler clustering methods to explicitly capture the sequential relationships between log entries. 

#### Assessment Metrics
The primary assessment metric used is the **anomaly score (s)**.  This is not a standard classification metric like Precision or Recall because the method is fully unsupervised and operates without labeled data.

The score `s` for a new log entry is the negative log probability ratio, which measures how much the new entry violates the learned sequential patterns of the HMM. A higher score signifies a greater deviation from the norm and a higher likelihood of being an anomaly.  The evaluation is therefore qualitative, based on observing sharp spikes in this score that correspond to known error messages or disruptions in real-world log data. 

#### Evaluation Method
The proposed technique was evaluated through a **qualitative analysis of representative instances** from both real-world and synthetic data:
1.  **Synthetic Data**: A minimal 8-event synthetic sequence was used to demonstrate the model's ability to detect different anomaly types: content anomalies (a novel event) and sequential anomalies (a known event in an unexpected position). This is shown in Figures 1 and 3 of the manuscript. 
2.  **Real-World Data**: The method was applied to four distinct, anonymized log instances from the EuXFEL system. The evaluation consisted of observing the computed anomaly scores and confirming that significant score spikes corresponded to the appearance of error messages or other pattern disruptions in the logs, as detailed in Figures 4, 5, 6, and 7 of the manuscript. 

### Computing Infrastructure
The code is platform-independent and was developed and tested in a Linux-based environment (Google Colaboratory). It does not require specialized hardware and can be run on a standard personal computer.

### Citation
If you use this code or dataset in your research, please cite our work:

*Sulc, A., Eichler, A., & Wilksen, T. (2023). Unsupervised Log Anomaly Detection with Few Unique Tokens.*

```
@article{sulc2023unsupervised,
  title={Unsupervised Log Anomaly Detection with Few Unique Tokens},
  author={Sulc, Antonin and Eichler, Annika and Wilksen, Tim},
  journal={arXiv preprint arXiv:2310.08951},
  year={2023}
}
```

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

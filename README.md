# CS522 - Machine Learning - Team Project
- Team Members: Isaac Sikkema, Nan Tang, Sabrullah Deniz, Hunter Price, Andrei Cozma
- Class Website: https://web.eecs.utk.edu/~hqi/cosc522/
- Project Google Drive: https://drive.google.com/drive/folders/1Cl_dF2ptlYzd_hIid7rArQXoGCKIbNTS
- GitHub Repo: https://github.com/andreicozma1/CS522-TeamProject

## Objective
The objective of the final project is to integrate various machine learning techniques to achieve the best performance. Final project is a group effort. Each group can have 4-5 members. You are required to apply ALL techniques learned in this semester.

## Schedule
1. (5) **Milestone 1** (Due 11/4): 
	- Group Formation and Topic selection. The same topic cannot be chosen by more than 1 group. 
	- Submit through Canvas. Approval and comments will be returned in one day. 
	- The topic follows the first-come first-served rule. So pick a topic as soon as possible.
3. (5) **Milestone 2** - Literature Survey (Due 11/11): 
	- Background study including references and state-of-the-art performance on the dataset.
	- (2-page report need to be submitted)
5. (5) **Milestone 3** - Prototype 1 (Due 11/18):
	- Prototype, preliminary results and task allocation among group members.
	- Apply at least one learned technique successfully for each component in the pipeline on the chosen dataset and submit a 1-page report.
7. (5) **Milestone 4** - Prototype 2 (Due 12/02):
	- Implement at least two solutions to each component of the pipeline. 
	- Determine what metrics to use. 
	- Provide performance evaluation results.
9. (100) **Final presentation** (Due 12/08):
	- Presentation slides due the midnight before the presentation on 12/9. 
	- Submit through Canvas.
11. (80) **Final report** (Due 12/10):
	- Submit through Canvas.

## Potential Topics
Each group can choose one topic from the following sources. All selection needs to be approved by instructor.
- [KDD-Cup 1997-2009](http://www.kdnuggets.com/datasets/kddcup.html)
- [Kaggle Competitions](http://www.kaggle.com/competitions)
- Other topics: You can select a topic yourself from other resources.

## Requirement
General steps involved in a machine learning problem include

- Data collection (raw data)
- Feature extraction (how to extract features from the raw data)
- Feature selection (dimensionality reduction - Fisher's linear discriminant or PCA)
- Classification/Regression methods need to be included
  - Supervised learning and Unsupervised learning
  - Baysian approaches and non-Baysian approaches
  - Parametric and Non-parametric density estimation in supervised learning
  - Fusion
- Performance evaluation
- Feedback system

You are required to evaluate the effect of various aspects of the classification/regression process, including but not limited to
- the effect of assuming the data is Gaussian-distributed
- the effect of assuming parametric pdf vs. non-parametric pdf
- the effect of using different prior probability ratio
- the effect of using different distance
- the effect of knowing the class label
- the effect of dimension of the feature space (e.g., changed through dimensionality reduction)
- the effect of fusion

To be more specific, you need to at least go through the following steps:
- Data normalization
- Dimensionality reduction
- Classification/Regression with the following
	- MPP (case 1, 2, and 3)
	- kNN with different k's
	- BPNN
	- Decision tree
	- SVM
	- Clustering (kmeans, wta)
- Classifier fusion
- Evaluation (use n-fold cross validation to generate confusion matrix and ROC curve if applicable).

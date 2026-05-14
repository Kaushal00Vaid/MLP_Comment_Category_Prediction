# Comment Category Prediction Challenge

Predict category assigned to each comment.
(Ranked **54** out of 2744 total participants)

## Overview

Welcome to the Comment Category Prediction Challenge! This dataset provides insight into how an online platform processes and categorizes user-generated comments. Each record represents a single comment and includes metadata such as interaction feedback, symbolic expressions, topic reference indicators, and internal system signals.
Your task is to explore this dataset and build predictive models that can accurately determine the final category assigned to each comment. Apply your analytical and modeling skills to uncover patterns across textual, numerical, and categorical features.

## Description

In this competition, you'll analyze a dataset containing information about short textual entries submitted to an online discussion system. The dataset includes the content of each entry, engagement signals from other users, and outputs from automated analysis components.
Your goal is to use this information to predict how each entry was ultimately categorized by the system.

## Dataset Description

The dataset consists of the following files:

- **train.csv**: The training set, including the target variable label along with all feature columns.
- **test.csv**: The test set, containing the same feature columns but excluding the target variable label.
- **sample_submission.csv**: A sample submission file provided in the correct format for competition submissions.

### Column Description

- **comment** : The raw text content of the comment.
- **created_date** : The date and time when the comment was posted.
- **post_id** : A unique identifier linking the comment to a discussion thread or parent post.
- **emoticon_1** : Indicator for the presence of symbols from the first internal emoticon group.
- **emoticon_2** : Indicator for the presence of symbols from the second internal emoticon group.
- **emoticon_3** : Indicator for the presence of symbols from the third internal emoticon group.
- **upvote** : Number of positive reactions received by the comment.
- **downvote** : Number of negative reactions received by the comment.
- **if_1** : Internal feature one that is hidden by the platform.
- **if_2** : Internal feature two that is hidden by the platform.
- **race** : Indicator showing whether the system detected references to a specific group identity.
- **religion** : Indicator showing whether the system detected references to a belief-related topic.
- **gender** : Indicator showing whether the system detected references to a gender-related topic.
- **disability** : Indicator showing whether the system detected references to an ability-related topic.
- **label** : The final category assigned to the comment by the platform (Target Variable). This variable can take four distinct values, each corresponding to a different internal handling category.

## Restrictions on Libraries

`NumPy`
`Pandas`
`Matplotlib`
`Scikit-learn`
`XGBoost`
`Seaborn`
`Imblearn`
`SciPy`
`Pickle`
`regex`
`Lightgbm`
`Plotly`
`statsmodel`
`wordcloud`

---

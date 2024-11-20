# __Project Overview__

In this project, I developed an AI-powered career recommendation system using GPT-2. The goal was to provide personalized career path suggestions based on a user’s skills, interests, and strengths. 

Initially, the baseline GPT-2 model was tested without fine-tuning to evaluate its general performance. 

Subsequently, the model was fine-tuned on a specialized dataset of job descriptions to enhance its recommendation capabilities, highlighting my expertise in natural language processing (NLP), machine learning, and model fine-tuning.

## __Modeling Approach__

Baseline GPT-2
The pre-trained GPT-2 model was used to generate career recommendations from user input. Although it provided coherent text, the results were generic and repetitive due to the lack of domain-specific knowledge.

Fine-Tuned GPT-2
The fine-tuning process involved:

  1. Dataset Preparation:
   - Used a job descriptions dataset consisting of fields like company_name, position_title, and job_description.
   - Prompts were crafted using company_name and position_title, while job_description was treated as the target output.
    
  2. Fine-Tuning:
   - Fine-tuned GPT-2 on the job descriptions dataset using Hugging Face Transformers.
   - Employed techniques such as padding, truncation, and attention masking for efficient training on a GPU.
    
3. Comparison:
  - Compared the baseline GPT-2 results with the fine-tuned model, observing significant improvement in relevance and diversity of career suggestions.


## __Model Performance__

__Baseline Results__\
The baseline model struggled with repetitive output and lacked domain specificity. For example:

<ins>Input:</ins>\
  Skills: Python, Data Analysis, Machine Learning\
  Interests: Technology, Education\
  Strengths: Problem-solving, Fast Learner


<ins>Output:</ins>\
  Python, Data Analysis, Machine Learning\
  Interests: Technology, Education\
  Strengths: Problem-solving, Fast Learner

(The output is duplicated from the pre-trained model)


## __Fine-Tuned Results__
After fine-tuning, the model produced more coherent and relevant recommendations:

<ins>Input:</ins>\
  Skills: Python, Data Analysis, Machine Learning\
  Interests: Technology, Education\
  Strengths: Problem-solving, Fast Learner

<ins>Output:</ins>\
Data Scientist\
Machine Learning Engineer\
EdTech Consultant


## __How to Use__

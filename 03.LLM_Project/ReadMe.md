# __Project Overview__

In this project, I developed an AI-powered career recommendation system using GPT-2. The target of the model was to provide recommendations on career paths and skills based on a user’s skills or position. 

Initially, the baseline GPT-2 model was tested without fine-tuning to evaluate its general performance. 

Subsequently, the model was fine-tuned on a specialized dataset of job descriptions to enhance its recommendation capabilities.

I include this project to highlight my expertise in natural language processing (NLP), machine learning, and model fine-tuning.

## __Modeling Approach__

Baseline GPT-2

The pre-trained GPT-2 model was used to generate career recommendations from user input. Although it provided coherent text, the results were generic and repetitive due to the lack of domain-specific knowledge.

Fine-Tuned GPT-2

The fine-tuning process involved:

  1. Dataset Preparation:
   - Used a job descriptions dataset consisting of fields like company_name, position_title, and job_description.
   - Prompts were crafted using position_title, while job_description was treated as the target output.
    
  2. Fine-Tuning:
   - Fine-tuned GPT-2 on the job descriptions dataset using Hugging Face Transformers.
   - Employed techniques such as padding, truncation, and attention masking for efficient training on a GPU.
    
3. Comparison:
  - Compared the baseline GPT-2 results with the fine-tuned model, observing significant improvement in relevance of career suggestions.


## __Model Performance__

### __Baseline Results__
The baseline model struggled with repetitive output and lacked domain specificity. For example:

&nbsp;<ins>Input:</ins>

"Tell me about the qualifications required for a Data scientist."


&nbsp;<ins>Output:</ins>

Tell me about the qualifications required for a Data scientist.

A: I am an engineer, and my background is in data science at Microsoft Research (MSR). My main job involves analyzing large
datasets to understand how they are used by organizations or individuals within their organization;
this includes using machine learning techniques such as deep neural networks that can be applied
across multiple domains of analysis including human-computer interaction research with social media
platforms like Facebook Messenger etc., but also other types where you need more than just one
person working on your project - it's much easier if there aren't many people involved! In addition
we have our own team who work closely together so when someone comes along looking into something
new then everyone will know what was done before them... So please

:memo: **Note:** The untrained GTP2 model gives a slightly generic reply, which doesnt directly address the question in the prompt. Although still related to the matter, it lacks logical flow and fails to communicate requirements for the position. 


### __Fine-Tuned Results__
After fine-tuning, the model produced more coherent and relevant recommendations:

&nbsp;<ins>Input:</ins>

"Tell me about the qualifications required for a Data scientist."

&nbsp;<ins>Output:</ins>

Tell me about the qualifications required for a Data scientist. 

What you'll do: Work with data scientists to understand how our customers use our products and services, identify opportunities
that could improve their business processes or drive revenue growth by leveraging AI models in order
create insights into customer behavior through analytics tools such as Salesforce Analytics,
SalesForce Cloud Platforms (SaaN), etc., Develop best practices across all of your projects Build
scalable solutions using Python/Python-based technologies like Django & Flask Create automated tests
on existing systems Write clean code based upon user feedback Participate actively within teams
where necessary Provide technical support via email Confirm work is completed successfully
Qualifications Required Skills You will be working at a startup environment Experience building
software applications from scratch Strong understanding


:memo: **Note:** The trained network response is more targeted to the prompt as it directly outlines tasks that a Data Scientist might perform. The response highlights specific tools and technologies (e.g., Python, Django, Flask, Salesforce Analytics) and is more structured.


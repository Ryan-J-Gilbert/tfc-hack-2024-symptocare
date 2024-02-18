# SymptoCare

SymptoCare is a platform designed to provide healthcare accessibility and delivery through a disease search tool. This project was created at the Tech For Change 2024 Hackathon.

## Features

- **Symptom Checker**: Use our tool to check your symptoms against a database, providing insights into potential medical conditions.
- **GPT Integration**: Get a second opinion and information through our GPT integration. 

## How It Works

1. **Check Symptoms**: Enter your symptoms into our Symptom Checker tool.
2. **Receive Insights**: Receive insights into potential medical conditions based on your symptoms.
3. **Access Resources**: Access health information, educational resources, and more.

## How This Benefits The Community

- **Digital Health Access**: Access to healthcare services is a fundamental right. SymptoCare provides health information as a platform ANYBODY can use FREE of charge.
- **Community Health Empowerment**: SymptoCare promotes knowledge and action-taking. Visiting the doctor is expensive and time-consuming. SymptoCare takes the pain out of the diagnosis.

## Data Collection

The data for the Symptom Checker was collected from discharge summaries of patients at the New York Presbyterian Hospital in 2004. The data was then processed and cleaned so that each disease is associated with a group of symptoms. The link to the database can be found [here](https://impact.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html).

## Uses for the Symptom Checker

This tool is intended for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. It can be used to:

- Identify potential medical conditions based on symptoms.
- Understand more about a symptom or condition.
- Prepare questions for your healthcare provider.

## Tech Stack

- **Python**: Programming language used for backend development.
- **Flask**: Micro web framework used for backend development.
- **Pandas**: Library used for data handling and manipulation.
- **OpenAI**: Access to GPT (Generative Pre-trained Transformer) for providing insights and additional information.
- **scikit-learn**: Library used for cosine similarity calculations.

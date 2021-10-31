# **Midterm Project**

## **The problem:**
I wanted initially to look at and analyze a dataset that models performance and/or prospective performance of some population of students. I wasn't able to find a dataset I liked for students specifically, so I shifted my attention to classification of enrollment programs. The problem this dataset presents is training a model, based on the performance of an individual in employee training, whether or not the said employee is likely to stay with the company or look for a new job. The dataset is imbalanced with about a 3:1 majority of 0 (Not looking for new job) to 1 (Looking for new job) entries.

Data: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists?select=aug_test.csv

## **Virtual Environment**
This project is built into a virtual environment using WSL 2.0 on Ubuntu 20.04.3. To build the virtual environment, the provided Pipfile contains the relevant dependencies. To build from scratch, do  

> pip install pipenv
> pipenv install numpy pandas scikit-learn xgb flask gunicorn

To activate the virtual environment afterwards, run 

> pipenv shell

This project can also be deployed to a Docker container running on a Ubuntu 20.04.3 image using WSL 2.0. In order to build it, navigate to the directory that contains the Dockerfile and run

> docker build -t enrollee-predict .

To deploy the Docker container after it has been build, run

> docker run -it --rm -p 9696:9696 enrollee-predict

The project will run on **localhost:9696** once deployed.
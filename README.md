# Textbook Chatbot 


CSE 6550: Software Engineer Concepts, Fall 24

California State University, San Bernardino
## Description
The Textbook Chatbot project for CSE 6550 is designed to assist with queries related to the textbook."Software Engineering: A Practitioner's Approach." The chatbot serves as an educational tool, helping users by providing information, answering questions, and possibly retrieving content from the textbook.

## Prerequisites
Before you begin, make sure you have the following installed on your machine:
- **Git**
- **Docker**

## Setup
To get started, first clone the repository to your local machine:
```
git clone https://github.com/DrAlzahraniProjects/csusb_fall2024_cse6550_team3.git
```


select while running the docker
After cloning the repository, navigate to the project directory:
```
cd csusb_fall2024_cse6550_team3
```


Once you are in correct folder, build the Docker image:
```
docker build -t team3-app .
```

Now, run the Docker container:
```
docker run -p 5003:5003 team3-app
```
The application will be available at:  http://localhost:5003/team3

<!-- Accessing Jupyter Notebook http://localhost:6003/ -->

---
## Project Structure

- `.github/workflows/docker-publish.yml`: Defines a GitHub Action workflow to automate Docker publishing
- `.gitignore`: Specifies which files and directories should be ignored by Git
- `Dockerfile`: Contains instructions to build the Docker image for the project
- `README.md`: Project documentation containing setup instructions and information about the project
- `app.py`: Main entry point for the application
- `nginx.conf`: Nginx congifuration file
- `requirements.txt`: Lists Python package dependencies required for the project

---

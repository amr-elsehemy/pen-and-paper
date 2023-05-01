from word_detector import prepare_img, detect, sort_line
import matplotlib.pyplot as plt
import cv2
import sys, os
import main
import openai
from jira import JIRA
import json
import argparse

DEBUG=True

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]
# Set up authentication details
jira_url = '[JIRA_URL]'
jira_username = '[JIRA_USERNAME_OR_EMAIL]'
jira_api_token = '[JIRA_API_TOKEN]'

# Authenticate with JIRA
jira = JIRA(options={'server': jira_url}, basic_auth=(jira_username, jira_api_token))

def get_response(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0,
      max_tokens=1750,
      n=1,
      top_p=1.0,
      stop=None,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response.choices[0].text.strip()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['multi_linear', 'one_line'], default='one_line')
    parser.add_argument('--image', default='test.png')

    return parser.parse_args()

def pipeline_runner():
    args = parse_args()
    image = args.image
    text_type = args.mode

    raw_ocr_text = main.read(image)
    print("raw_ocr_text : ", raw_ocr_text[0])
    print("------------------------------------------------------------------------------")

    prompt_engineered_aug =f"""
        Hi ChatGPT, you will be given a text that contains a data science related instruction such as : ( "fix test data", or "correct training data", or "enhance", or "debug" .. etc) which user wants to autmoate
        (example : "enhance chat widget appearance"), the instruction will have misspelled words, your task is to try to correct the misspelled words in the text between the curly brackets, 
        and provide the most likely instruction without deleting any words, then extract the entities and intents and find the releationships between the entities and intents, 
        return your response in as a json with the following keys : correction, intent, entity, relationships
        """
    hand_crafted_prompt = prompt_engineered_aug + "{" + raw_ocr_text[0] + "}"
    print("hand_crafted_prompt", hand_crafted_prompt)

    response_engineered = get_completion(hand_crafted_prompt)
    print("response_engineered", response_engineered)
    
    data_dict = json.loads(response_engineered)
    print(data_dict)

    # Define the task details
    task_name = data_dict["correction"]
    task_description = json.dumps(data_dict)
    jira_project_name = 'TA'

    # Create the task in JIRA
    new_issue = jira.create_issue(project=jira_project_name, summary=task_name, description=task_description, issuetype={'name': 'Task'})
    print(f'Task created with ID: {new_issue.key}')

if __name__ == '__main__':
    pipeline_runner()

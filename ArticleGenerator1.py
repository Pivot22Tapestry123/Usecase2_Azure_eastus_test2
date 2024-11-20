import traceback
import requests
import streamlit as st
import os
import json
import warnings
from crewai import Agent, Task, Crew, Process
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Azure authentication
def get_azure_credentials():
    try:
        # First try to get credentials from environment variables
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_api_key:
            return azure_api_key
        
        # If not in environment variables, try Azure Key Vault
        key_vault_name = os.getenv("AZURE_KEY_VAULT_NAME")
        if key_vault_name:
            key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
            credential = DefaultAzureCredential()
            secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
            azure_api_key = secret_client.get_secret("AZURE-OPENAI-API-KEY").value
            return azure_api_key
        
        # If neither is available, return None
        return None
    except Exception as e:
        st.error(f"Error getting Azure credentials: {str(e)}")
        return None

# Helper function to load and save configurations
def load_config():
    try:
        with open("agent_task_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_config(config):
    with open("agent_task_config.json", "w") as f:
        json.dump(config, f)

# Load persisted configurations at startup
config = load_config()

# Streamlit UI
st.title("Research Article Generator")

# File uploader
uploaded_file = st.file_uploader("Upload your transcript file", type="txt")
st.write(uploaded_file)

# Try to get Azure credentials
azure_api_key = get_azure_credentials()

# Only show API key input if not available from Azure authentication
if not azure_api_key:
    azure_api_key = st.text_input("Enter your Azure OpenAI API Key", type="password")
    azure_api_key = azure_api_key.strip() if azure_api_key else ""

# Get other Azure configurations from environment variables or UI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://rstapestryopenai2.openai.azure.com/")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Configure OpenAI settings
def setup_azure_openai():
    try:
        # Set OpenAI API configurations
        openai.api_type = "azure"
        openai.api_key = azure_api_key
        openai.api_base = azure_endpoint
        openai.api_version = azure_api_version

        # Test the connection
        response = requests.post(
            f"{azure_endpoint}openai/deployments/{azure_deployment}/chat/completions?api-version={azure_api_version}",
            headers={
                "api-key": azure_api_key,
                "Content-Type": "application/json"
            },
            json={
                "messages": [{"role": "system", "content": "Test connection."}],
                "max_tokens": 5
            }
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Azure OpenAI Setup Error: {str(e)}")
        return False

# Temperature slider
temperature = st.slider("Set the temperature for the output (0 = deterministic, 1 = creative)", 
                       min_value=0.0, max_value=1.0, value=0.7)

# Define prompts for agents and tasks
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = config or {
        "planner": {
            "role": "Content Planner",
            "goal": "Plan engaging and factually accurate content on the given topic",
            "backstory": "You're working on planning a research report about a given topic."
        },
        "writer": {
            "role": "Content Writer",
            "goal": "Write insightful and factually accurate research report",
            "backstory": "You're working on writing a new opinion piece about a given topic."
        },
        "editor": {
            "role": "Editor",
            "goal": "Edit a given blog post",
            "backstory": "You are an editor who receives a research article from the Content Writer."
        },
        "tasks": {
            "plan": "Plan content for the topic",
            "write": "Write a research article based on the content plan",
            "edit": "Edit and finalize the research article"
        }
    }

# User inputs for each prompt
st.header("Agent Prompts")

for agent, prompts in st.session_state['prompts'].items():
    if agent != "tasks":
        st.subheader(f"{agent.capitalize()} Agent")
        prompts["role"] = st.text_input(f"{agent.capitalize()} Role", 
                                      value=prompts["role"], 
                                      key=f"{agent}_role")
        prompts["goal"] = st.text_area(f"{agent.capitalize()} Goal", 
                                     value=prompts["goal"], 
                                     key=f"{agent}_goal")
        prompts["backstory"] = st.text_area(f"{agent.capitalize()} Backstory", 
                                          value=prompts["backstory"], 
                                          key=f"{agent}_backstory")

st.header("Task Descriptions")
for task, description in st.session_state['prompts']["tasks"].items():
    st.session_state['prompts']["tasks"][task] = st.text_area(
        f"{task.capitalize()} Task Description",
        value=description,
        key=f"{task}_description"
    )

# Button to save user modifications
if st.button("Save Configuration"):
    save_config(st.session_state['prompts'])
    st.success("Configuration saved successfully!")

# Button to start processing
if st.button("Generate Research Article"):
    if not uploaded_file:
        st.error("Please upload a transcript file.")
    elif not azure_api_key:
        st.error("Please enter your Azure OpenAI API Key.")
    else:
        transcripts = uploaded_file.read().decode("utf-8")

        try:
            # Setup and test Azure OpenAI connection
            if not setup_azure_openai():
                st.error("Failed to setup Azure OpenAI connection. Please check your credentials.")
                st.stop()

            st.success("API connection successful!")

            # Define agents with user-defined prompts and proper error handling
            try:
                planner = Agent(
                    role=st.session_state['prompts']['planner']['role'],
                    goal=st.session_state['prompts']['planner']['goal'],
                    backstory=st.session_state['prompts']['planner']['backstory'],
                    allow_delegation=False,
                    verbose=True,
                    temperature=temperature
                )

                writer = Agent(
                    role=st.session_state['prompts']['writer']['role'],
                    goal=st.session_state['prompts']['writer']['goal'],
                    backstory=st.session_state['prompts']['writer']['backstory'],
                    allow_delegation=False,
                    verbose=True,
                    temperature=temperature
                )

                editor = Agent(
                    role=st.session_state['prompts']['editor']['role'],
                    goal=st.session_state['prompts']['editor']['goal'],
                    backstory=st.session_state['prompts']['editor']['backstory'],
                    allow_delegation=False,
                    verbose=True,
                    temperature=temperature
                )

                # Define tasks with error handling
                plan = Task(
                    description=f"{st.session_state['prompts']['tasks']['plan']}: {transcripts}",
                    agent=planner,
                )

                write = Task(
                    description=st.session_state['prompts']['tasks']['write'],
                    agent=writer,
                )

                edit = Task(
                    description=st.session_state['prompts']['tasks']['edit'],
                    agent=editor
                )

                # Create and execute crew
                crew = Crew(
                    agents=[planner, writer, editor],
                    tasks=[plan, write, edit],
                    verbose=True
                )

                # Process the transcript with progress indication
                with st.spinner("Generating research article... This may take a few minutes."):
                    result = crew.kickoff()

                # Display the result
                st.success("Research article generated successfully!")
                st.markdown(result)

            except Exception as agent_error:
                st.error(f"Error in agent/task setup: {str(agent_error)}")
                st.error(f"Detailed error: {traceback.format_exc()}")

        except requests.exceptions.RequestException as api_error:
            st.error("API Error occurred:")
            st.error(f"Error details: {str(api_error)}")
            if hasattr(api_error, 'response'):
                st.error(f"Response Status Code: {api_error.response.status_code}")
                st.error(f"Response Content: {api_error.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

st.markdown("---")
st.markdown("Tapestry Networks")

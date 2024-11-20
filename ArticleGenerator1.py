import traceback
import requests
import streamlit as st
import os
import json
import warnings
from crewai import Agent, Task, Crew, Process
from crewai import LLM
import openai

# Suppress warnings
warnings.filterwarnings('ignore')

# Helper function to load and save configurations
llm = LLM(model="gpt-4", temperature=0.7)
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

# API Key and endpoint inputs for Azure OpenAI
AZURE_API_KEY = st.text_input("Enter your Azure OpenAI API Key", type="password")
AZURE_API_KEY = AZURE_API_KEY.strip()
AZURE_API_BASE = "https://rstapestryopenai2.openai.azure.com/"
azure_deployment = "gpt-4"  # Deployment name as per your Azure configuration
AZURE_API_VERSION = "2024-08-01-preview"

# Set OpenAI API base and key for Azure
openai.api_key = AZURE_API_KEY
openai.api_base = AZURE_API_BASE

# Set the OPENAI_API_KEY environment variable for libraries expecting it this way
os.environ["OPENAI_API_KEY"] = AZURE_API_KEY

# Temperature slider
temperature = st.slider("Set the temperature for the output (0 = deterministic, 1 = creative)", min_value=0.0, max_value=1.0, value=0.7)

# Define prompts for agents and tasks
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = config or {
        "planner": {"role": "Content Planner", "goal": "Plan engaging and factually accurate content on the given topic",
                    "backstory": "You're working on planning a research report about a given topic."},
        "writer": {"role": "Content Writer", "goal": "Write insightful and factually accurate research report",
                   "backstory": "You're working on writing a new opinion piece about a given topic."},
        "editor": {"role": "Editor", "goal": "Edit a given blog post",
                   "backstory": "You are an editor who receives a research article from the Content Writer."},
        "tasks": {"plan": "Plan content for the topic", "write": "Write a research article based on the content plan",
                  "edit": "Edit and finalize the research article"}
    }

# User inputs for each prompt
st.header("Agent Prompts")

for agent, prompts in st.session_state['prompts'].items():
    if agent != "tasks":
        st.subheader(f"{agent.capitalize()} Agent")
        prompts["role"] = st.text_input(f"{agent.capitalize()} Role", value=prompts["role"], key=f"{agent}_role")
        prompts["goal"] = st.text_area(f"{agent.capitalize()} Goal", value=prompts["goal"], key=f"{agent}_goal")
        prompts["backstory"] = st.text_area(f"{agent.capitalize()} Backstory", value=prompts["backstory"], key=f"{agent}_backstory")

st.header("Task Descriptions")
for task, description in st.session_state['prompts']["tasks"].items():
    st.session_state['prompts']["tasks"][task] = st.text_area(f"{task.capitalize()} Task Description", value=description, key=f"{task}_description")

# Button to save user modifications
if st.button("Save Configuration"):
    save_config(st.session_state['prompts'])
    st.success("Configuration saved successfully!")

# Button to start processing
if st.button("Generate Research Article"):
    if not uploaded_file:
        st.error("Please upload a transcript file.")
    else:
        transcripts = uploaded_file.read().decode("utf-8")

        try:
            # Test Azure OpenAI API connection with a small completion request
            response = requests.post(
                f"{AZURE_API_BASE}openai/deployments/{azure_deployment}/chat/completions?api-version={AZURE_API_VERSION}",
                headers={
                    "api-key": AZURE_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "messages": [{"role": "system", "content": "Test connection."}],
                    "max_tokens": 5
                }
            )
            response.raise_for_status()
            st.success("API connection successful!")

            # Define agents and tasks with user-defined prompts
            planner = Agent(
                role=st.session_state['prompts']['planner']['role'],
                goal=st.session_state['prompts']['planner']['goal'],
                backstory=st.session_state['prompts']['planner']['backstory'],
                allow_delegation=False,
                verbose=True,
                temperature=temperature,
                llm=llm 
            )

            writer = Agent(
                role=st.session_state['prompts']['writer']['role'],
                goal=st.session_state['prompts']['writer']['goal'],
                backstory=st.session_state['prompts']['writer']['backstory'],
                allow_delegation=False,
                verbose=True,
                temperature=temperature,
                llm=llm 
            )

            editor = Agent(
                role=st.session_state['prompts']['editor']['role'],
                goal=st.session_state['prompts']['editor']['goal'],
                backstory=st.session_state['prompts']['editor']['backstory'],
                allow_delegation=False,
                verbose=True,
                temperature=temperature,
                llm=llm 
            )

            # Define tasks
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

            # Create crew
            crew = Crew(
                agents=[planner, writer, editor],
                tasks=[plan, write, edit],
                verbose=True
            )

            # Process the transcript
            with st.spinner("Generating research article... This may take a few minutes."):
                result = crew.kickoff()

            # Display the result
            st.success("Research article generated successfully!")
            st.markdown(result)

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            if hasattr(e, 'response'):
                st.error(f"Response Status Code: {e.response.status_code}")
                st.error(f"Response Content: {e.response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

st.markdown("---")
st.markdown("Tapestry Networks")

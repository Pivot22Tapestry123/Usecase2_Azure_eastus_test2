# Tapestry Demo

### Steps

Upload the code files to Github. The demo will run of Github Spaces

Install the pre-requisites and dependecies such as:

openai==0.27.9,

langchain,

docx2txt,

pypdf,

streamlit,

chromadb==0.3.29 # version because of Sqlite error on Streamlit Community Sharing,

tiktoken,

unstructured[local-inference],

layoutparser[layoutmodels,tesseract],

To run the demo use the following commands:

1>source ./venv/bin/activate,

2>streamlit run chat_with_documents.py --server.enableCORS false --server.enableXsrfProtection false

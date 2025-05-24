# ai-chatbot-for-pdf

Tutorial found here: https://medium.com/@param775/build-an-ai-chatbot-for-custom-pdf-documents-with-python-and-langchain-4089fe30b30f

Fails to install requirements.txt:
WARNING: Failed to activate VS environment: Could not find C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe


To run in Windows CMD shell:

* python -m venv venv
* venv\Scripts\activate.bat
* pip install -r requirements.txt
* python vectorize_pdf_files.py
* python query_llm.py

To run in a Codespace (Linux):

* python -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt
* python vectorize_pdf_files.py
* python query_llm.py

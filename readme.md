# SOAP Note Generation - Generative AI

We've devised a novel approach for automating SOAP note generation through patient-doctor conversations, utilizing state-of-the-art language models (LLMs). 
Our methodology employs two key techniques:
1.	RAG
2.	Advanced prompting.

## What is in this Repo?
1.	soap_notes_development (Jupyter Notebook) : This Notebook consists of testing multiple LLM, retrieval techniques, prompting and evaluation on Test Data.
2.	soap_notes.py (Python File) : This python code file consists of backend - helper functions, vectorstore, LLM for Chatbot.
3.	app_rag.py & app_prompt.py (Python Files) : These python codes consist of streamlit codes to be utilized as chatbot, app_rag consist of RAG based chatbot and app_prompt consist of advanced prompting based chatbot.
4.	test_evaluate.csv (Worksheet): The worksheet consists of test conversation and their reference call_notes, extracted call_conversation and call_notes pair from RAG, RAG based generation, advanced prompting based generation and their respective BLEU and ROUGE Scores.

**Note:**Before running codes in your environment please make sure to install required libraries using requirements.txt (pip install -r requirements.txt).
**Note:**If you want to launch the chatbot, please use these commands in your command prompt in the same directory of the project folder. streamlit run app_rag.py for RAG based chatbot and  streamlit run app_prompt.py for advanced prompting based chatbot




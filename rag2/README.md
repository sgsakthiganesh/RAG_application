# RAG_application for fetching hints abouts World Historical events between 1900 to present

requirements

1. fastapi
2. uvicorn[standard]
3. pydantic
4. requests
5. streamlit
6. langchain
7. langchain-community
8. pypdf
9. transformers
10. sentence-transformers
11. faiss-cpu

# Front-End

Frontend was built using streamlit

------Streamlit - A lightweight UI framework to create interface for the application.

# Back-end

Backend was built using FastAPI

------FastAPI was used to create Backend integration via API to connect with the Frontend to create request and response.

Commands to run the application:

To run API file: uvicorn backend:app --reload --{port}
To run Streamlit app file: streamlit run app.py
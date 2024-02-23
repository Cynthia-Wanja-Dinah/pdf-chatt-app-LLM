import streamlit as st
from dotenv import load_dotenv 
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import FAISS

#sidebar contents
with st.sidebar:
    st.title('LLM chat app')
    st.markdown('''
        ##About
        This app is an LLM powerd chatbot
        -[Streamlit](https://streamlit.io/)
        -[Langchain](https://python.langchain.com/)
        -[OpenAi](https://platform.openAi.com/docs/models) LLM model     
                
                ''')
    
    
    add_vertical_space(5)
    st.write('Made with Love by Cynthia Wanja' )
    
    load_dotenv()
    
    
def main():
     st.write("chat with the PDF")
     #upload your PDF
     pdf=st.file_uploader ("upload your file ", type=['pdf'])
     #st.write(pdf)
     
     if pdf is not None:
      pdf_reader=PdfReader(pdf)

     

    
     
     text =""
     for page in pdf_reader.pages:
         text += page.extract_text()
         
         
         text_splitter= RecursiveCharacterTextSplitter(
             
              chunk_size=1000,
              chunk_overlap=200,
              length_function=len 
             
         )
         chunks = text_splitter.split_text(text=text)
         st.write(chunks)
         embeddings=OpenAIEmbeddings()
         Vectorestore=FAISS.from_texts(chunks, embedding=embeddings)
         store_name=pdf.name[:-4]
         with open(f"{store_name}.pkl", "wb") as f:
             pickle.dump
         
         
        # embeddings=OpenAIEmbeddings()
         
         #vectorstore=FAISS.from_texts(chunks, embedding=embeddings)
     #st.write(text)
if __name__=='__main__':
        main()
    
    
    
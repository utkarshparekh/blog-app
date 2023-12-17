import logging
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getLLamaResponse(input_text, no_words, blog_style, temprature):
    """
    Generate a blog response using the LLama model.

    Args:
        input_text (str): The input text or topic for the blog.
        no_words (int): The desired number of words for the blog.
        blog_style (str): The style or category of the blog.
        temprature (float): The temperature parameter for controlling the randomness of the generated text.

    Returns:
        str: The generated blog response.

    """
    
    ### Load the model
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q2_K.bin",
                        model_type="llama",
                        config = {"max_new_tokens": 256, 'temperature': temprature})
    
    ### Load the prompt template

    template = """
       Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "no_words", "input_text"],
                            template=template)
    logging.info("Prompt: %s", prompt)
    ### Generate the response
    response = llm(prompt.format(blog_style=blog_style, no_words=no_words, input_text=input_text))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs", page_icon=":llama:",
                   layout="centered", initial_sidebar_state="collapsed")
st.header("Generate Blogs")

input_text = st.text_input("\nEnter the Blog Topic")

## Create two more columns for additional two fields
col1, col2, col3 = st.columns([5,5,5])

with col1:
    no_words = st.text_input("Enter number of words", value=100)
with col2:
    temprature = st.slider("Set the temparature", 0.1, 2.0, 0.09)
with col3:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
submit = st.button("Generate Blog")

#Final response
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style, temprature))
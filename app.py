import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers  # Correct class import

def getLLamaresponse(input_text, no_words, blog_style):
    ### LLama2 model
    llm = CTransformers(  # Correct instantiation using CTransformers class
        model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", 'no_words'],
        template=template
    )
    
    ## Format the prompt
    final_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    )

    ## Generate the response from the Llama 2 model
    response = llm(final_prompt)
    
    return response


# Streamlit app settings
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

# Input field for blog topic
input_text = st.text_input('Blog Topic')  # Fixed incorrect `st.text`

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'), index=0
    )

# Button to submit and generate the blog
submit = st.button("Generate")

# Final response
if submit:
    try:
        # Convert `no_words` to an integer
        no_words = int(no_words)
        
        # Ensure input text is provided
        if input_text.strip() == "":
            st.error("Please enter a valid blog topic.")
        else:
            # Generate the response and display it
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.write(response)
        
    except ValueError:
        st.error("Please enter a valid number for 'No of Words'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

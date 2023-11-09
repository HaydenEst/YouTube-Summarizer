import streamlit as st
from langchain.llms import OpenAI
import os
from utils import summarize_video, search_query

# Initialize session_state if not already defined
if 'url' not in st.session_state:
    st.session_state.url = ""
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False
if 'generated_summary' not in st.session_state:
    st.session_state.generated_summary = ""

# Define a function for generating summary
def generate_summary(url, api_key):
    return summarize_video(yt_url=url, apikey=api_key)

# Define a function for generating more details
def generate_more_details(url, query, api_key):
    return search_query(yt_url=url, query=query, apikey=api_key)

# Main Streamlit app
# UI Starts here
# Set background color using custom CSS
st.set_page_config(page_title="YouTube Summarizer",
                   page_icon='✅',
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("Let me provide you with summaries of YouTube videos!")
# Custom CSS to set the background color
background_color = "rgb(300, 240, 330)"  # Adjust the color as needed
css = f"""
    <style>
        .stApp {{
            background-color: {background_color} !important;
        }}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)


if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''

st.sidebar.title("😎")
st.session_state['API_Key'] = st.sidebar.text_input("What's your API key?", type="password")

with st.form("user_input_form"):
    url = st.text_area('Enter the YouTube URL here:', height=25, value=st.session_state.url)
    submit = st.form_submit_button("Generate Summary")

if submit:
    st.session_state.url = url  # Save the url to session_state
    st.session_state.summary_generated = True
    st.session_state.generated_summary = generate_summary(url, st.session_state['API_Key'])
    summary_container = st.empty()
    summary_container.write(st.session_state.generated_summary)

if st.session_state.summary_generated:
    with st.form("more_details_form"):
        query = st.text_area('Is there something from the summary that you would like more details about?', 
                            height=25, 
                            value=st.session_state.query)
        more_details_submit = st.form_submit_button("Generate More Details")

    if more_details_submit:
        st.session_state.query = query  # Save the query to session_state
        details_container = st.empty()
        # details_container.write(generate_more_details(url, query, st.session_state['API_Key']))
        details = str(generate_more_details(url, query, st.session_state['API_Key']))
        details_container.markdown(f"""
                    <p style='font-size: 18px; color: red;'>
                        {details}</p>""", 
                    unsafe_allow_html=True)

        st.markdown(f"""
                    <p style='font-size: 15px; color: green;'>
                        VIDEO SUMMARY: {st.session_state.generated_summary}</p>""", 
                    unsafe_allow_html=True)

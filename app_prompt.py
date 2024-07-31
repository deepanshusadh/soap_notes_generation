import streamlit as st
import soap_notes


st.title("Prompting SOAP Note Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "soap_note" not in st.session_state:
    st.session_state.soap_note = 1

if "config_id" not in st.session_state:
    st.session_state.config_id = "ab"


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_prompt= st.chat_input("Enter your input here....")

if st.button('Start a new Conversation'):
    # if Start a new conversation get clicked, clear all the message history make the soap_note flat as 1 and create new config id
    st.session_state.messages = []
    st.session_state.soap_note = 1
    st.session_state.config_id+=str(1)


if st.session_state.soap_note==1:
    # if soap_note flat is 1, then take in conversation as input and generate soap notes
    if input_prompt:
        # Display user message in chat message container
        st.chat_message("user").markdown(input_prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": input_prompt})

        # generate soap notes taking in conversation
        response_gen= f"{soap_notes.generate_response_direct(input_prompt,st.session_state.config_id)}"
        # Display assistant response_retr"ieve in chat message container
        with st.chat_message("assistant"):
            st.write("**Generated SOAP Notes**")
            st.write(response_gen)
        # Add assistant response_retrieve to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_gen})
        st.session_state.soap_note=0 #soap note flag gets 0, after one generation and moves to chat

# elif soap note flag is 0, start the chat version keeping generated soap note as context.
elif st.session_state.soap_note==0:
    if input_prompt :
        st.chat_message("user").markdown(input_prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": input_prompt})

        #chat
        response_chat= f"{soap_notes.chat_direct(input_prompt,st.session_state.config_id)}"
        with st.chat_message("assistant"):
            st.write(response_chat)
        # Add assistant response_retrieve to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_chat})



            
        
       
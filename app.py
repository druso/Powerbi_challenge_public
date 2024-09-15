import streamlit as st
import os
import uuid
import json
import zipfile
import yaml
import re
from datetime import datetime
from io import BytesIO
from openai import OpenAI
from groq import Groq

################################################ CONFIGS LOADER&SETUP

def substitute_env_vars(yaml_content):
    pattern = re.compile(r'\$([A-Za-z0-9_]+)')
    
    def replace(match):
        env_var = match.group(1)
        return os.environ.get(env_var, f'${env_var}')
    
    return pattern.sub(replace, yaml_content)

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# Load prompts
with open('prompt.yaml', 'r') as file:
    prompt_content = file.read()
    prompt_content = substitute_env_vars(prompt_content)
    prompts = yaml.safe_load(prompt_content)

# Load llm_models
with open('llm_models.yaml', 'r') as file:
    llm_models = yaml.safe_load(file)

# Setup web-page configurations  
st.set_page_config(
    page_title=f"{config['page_title']} - {config['page_subtitle']}",
    page_icon=config['assistant_avatar'],
    layout="wide",
)

# Assign starting values

st.session_state.discussions_folder = "discussions"
st.session_state.sys_prompt = prompts['sys_prompt_short']
st.session_state.llm_model = llm_models['openai_active_model']
st.session_state.dumb_assistant = False
st.session_state.llm_config={
    'client': OpenAI(),
    'punish_model': llm_models['openai_punish_model'],
    'active_model': llm_models['openai_active_model'],
    'selector_model': llm_models['openai_selector_model']
}

# Cazzate
st.session_state.baloons = False
entered_keyword = ""
powerbi_image_path = 'img.png'

# Create the discussions folder if not existant
if not os.path.exists(st.session_state.discussions_folder):
    os.makedirs(st.session_state.discussions_folder)

# Generate a session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.start_date = datetime.now().strftime("%Y-%m-%d--%H:%M")
    print(f"Welcome to session {st.session_state.start_date} - {st.session_state.session_id}")



################################################ FUNCTIONS & LLM CALLS

# Define llm function with streaming
def openai_request_stream(chat_messages, sys_msg, history_lenght=8, model=st.session_state.llm_model, client= st.session_state.llm_config['client']): 
    """Call Openai and return a streaming of token. Takes as input chat_message, sys_message. Lenght of the history to consider."""
    # Cut conversation history to history_length (domanda + risposta = 2)
    final_chat_messages = chat_messages[-history_lenght:]
    # Add system message to the top
    final_chat_messages.insert (0,{"role": "system", "content": sys_msg})
    stream = client.chat.completions.create(
        model=model,
        messages=final_chat_messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Define llm function without streaming
def openai_request(chat_messages, sys_msg, history_lenght=8, model=st.session_state.llm_model, client= st.session_state.llm_config['client']): 
    """Call Openai and return the full openai response. Takes as input chat_message, sys_message. Lenght of the history to consider."""
    # Cut conversation history to history_length (domanda + risposta = 2)
    final_chat_messages = chat_messages[-history_lenght:]
    # Add system message to the top
    final_chat_messages.insert (0,{"role": "system", "content": sys_msg})

    response = client.chat.completions.create(
        model=model,
        messages=final_chat_messages,
        stream=False,
    )
    return response

# Function to update the LLM_config, the list of available model is here for practical use
available_models=[
    'GPT4o',
    'GPT3_turbo',
    'llama8',
    'llama70',
    'mixtral8x7b',
    'gemma7',]

def get_llm_config(mode):

    if mode == 'llama8':
        st.session_state.llm_config.update({
            'client': Groq(),
            'punish_model': llm_models['llama8_model'],
            'active_model': llm_models['llama8_model'],
            'selector_model': llm_models['llama8_model']
        })
    elif mode == 'llama70':
        st.session_state.llm_config.update({
            'client': Groq(),
            'punish_model': llm_models['llama70_model'],
            'active_model': llm_models['llama70_model'],
            'selector_model': llm_models['llama70_model']
        })
    elif mode == 'mixtral8x7b':
        st.session_state.llm_config.update({
            'client': Groq(),
            'punish_model': llm_models['mixtral8x7b_model'],
            'active_model': llm_models['mixtral8x7b_model'],
            'selector_model': llm_models['mixtral8x7b_model']
        })
    elif mode == 'gemma7':
        st.session_state.llm_config.update({
            'client': Groq(),
            'punish_model': llm_models['gemma7_model'],
            'active_model': llm_models['gemma7_model'],
            'selector_model': llm_models['gemma7_model']
        })
    elif mode == 'GPT3_turbo':
        st.session_state.llm_config.update({
            'client': OpenAI(),
            'punish_model': llm_models['openai_punish_model'],
            'active_model': llm_models['openai_punish_model'],
            'selector_model': llm_models['openai_punish_model']
        })
    elif mode == 'GPT4o':
        st.session_state.llm_config.update({
            'client': OpenAI(),
            'punish_model': llm_models['openai_punish_model'],
            'active_model': llm_models['openai_active_model'],
            'selector_model': llm_models['openai_selector_model']
        })


# Function to update the chat history
def save_chat_history():
    """will save the chat history version saved in memory in the discussion folder"""
    filename = f"{st.session_state.discussions_folder}/{st.session_state.start_date}_{st.session_state.session_id}.json"
    with open(filename, 'w') as f:
        json.dump(st.session_state.chat_messages, f, indent=4)

# Function to all conversations saved in the discussions folder
def list_files_in_directory(directory):
    """will read all files in a directory and return a list of them"""
    # List all files in the given directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        # Return an empty list if the directory does not exist
        return []

# Function to zip the discussions directory for download
def zip_discussions(directory='discussions'):
    # Create a byte stream to hold the zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory and add each file to the zip file
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                # Create the full path to the file
                file_path = os.path.join(foldername, filename)
                # Add file to zip, naming it by its filename
                zip_file.write(file_path, arcname=filename)
    # Important: reset buffer's position to the beginning after writing
    zip_buffer.seek(0)
    return zip_buffer

# Function to unzip the uploaded discussions package
def unzip_discussions(zip_file, target_dir):
    """
    Unzip the specified ZIP file into the target directory.
    Only extracts JSON files to ensure that only expected file types are handled.
    """
    with zipfile.ZipFile(zip_file, 'r') as z:
        # Filter and extract only JSON files
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        z.extractall(path=target_dir, members=json_files)



################################################ SIDEBAR

# LLM Model selector
st.sidebar.write("### Curioso di provare LLM alternativi?")
selected_model = st.sidebar.selectbox('Seleziona un LLM e attivalo',available_models, placeholder='GPT4_turbo')
if st.sidebar.button('Activate LLM', use_container_width=True):
    llm_config = get_llm_config(selected_model)
    st.toast(f"{st.session_state.llm_config['active_model']} attivato!", icon='🧠')
    print (f"{st.session_state.start_date} - {st.session_state.session_id}: Model '{st.session_state.llm_config['active_model']}' has been activated.")

st.sidebar.divider()

# Create the dropdown menu to all discussions in the folder
st.sidebar.write("### Curioso di vedere cosa chiedono gli altri?")
files = list_files_in_directory(st.session_state.discussions_folder)
sorted_files = sorted(files, reverse=True)
file_to_view = st.sidebar.selectbox("Seleziona un ID e carica la chat*", sorted_files, placeholder="Scegli una discussione e caricala")

# Load selected discussions content at the press of the button
if st.sidebar.button('Carica la chat', use_container_width=True):
    if file_to_view:
        # Construct the full file path
        file_path = os.path.join(st.session_state.discussions_folder, file_to_view)
        # Read and display the file content
        with open(file_path, 'r') as file:
            st.session_state.chat_messages = json.load(file)
            parts = file_to_view.split('_')
            st.session_state.session_id = parts[1].split('.')[0]
            st.session_state.start_date = parts[0]
    else:
        st.error('No file selected or available to display.')

# Return the user the conversation ID 
st.sidebar.write(f"**Attiva:** {st.session_state.start_date}_{st.session_state.session_id}")

# Advanced Options activator
st.sidebar.divider()
show_secret = st.sidebar.checkbox("Opzioni Sviluppo")

# If Advanced options are ticked the password prompt is displayed
if show_secret:
    st.sidebar.write('Giù le mani! 🔗[*maggiori informazioni*](https://www.youtube.com/watch?v=otCpCn0l4Wo)')
    entered_keyword = st.sidebar.text_input("Inserisci la parola segreta:", placeholder="Non ci provare...",help="Davvero, concentrati sulla PowerBI Challenge")

# If password is entered correctly baloons and options are displayed
if entered_keyword == config['download_keyword']:

    if st.session_state.baloons == False:
        st.balloons()
        st.session_state.baloons = True
    
    print (f"{st.session_state.start_date} - {st.session_state.session_id}: Dev Options accessed")
    st.sidebar.write('### Downloader')
    # Selector for which stuff the user wants to download
    options = ["discussions", "configs", "prompt", "llm_models",]
    choice = st.sidebar.selectbox("Select file to download:", options)

    # Prepeare the download button to be either the discussions zip or the yaml files
    if st.sidebar.button('Prepara il file', use_container_width=True):
        # Based on the selection, set up the appropriate file for download
        if choice == "discussions":
            # Call the function to zip files
            zip_buffer = zip_discussions()
            st.sidebar.download_button(
                label="Download discussions.zip",
                data=zip_buffer,
                file_name="discussions.zip",
                mime="application/zip",
                use_container_width=True
            )
        elif choice == "configs":
            # Read the configs YAML file's content
            with open('config.yaml', 'rb') as f:
                yaml_file = f.read()
            st.sidebar.download_button(
                label="Download Config.yaml",
                data=yaml_file,
                file_name="config.yaml",
                mime="application/x-yaml",
                use_container_width=True
            )
        elif choice == "prompt":
            # Read the prompt YAML file's content
            with open('prompt.yaml', 'rb') as f:
                yaml_file = f.read()
            st.sidebar.download_button(
                label="Download prompt.yaml",
                data=yaml_file,
                file_name="prompt.yaml",
                mime="application/x-yaml",
                use_container_width=True
            )
        elif choice == "llm_models":
            # Read the prompt YAML file's content
            with open('llm_models.yaml', 'rb') as f:
                yaml_file = f.read()
            st.sidebar.download_button(
                label="Download llm_models.yaml",
                data=yaml_file,
                file_name="llm_models.yaml",
                mime="application/x-yaml",
                use_container_width=True
            )

    st.sidebar.divider()
    st.sidebar.write('### Uploader')
    # Accept the upload of a yaml or zip file by the user, then put the info where needed
    uploaded_file = st.sidebar.file_uploader("Carica un file ZIP o YAML", type=['zip', 'yaml', 'yml'])

    if uploaded_file is not None:
        # Check the uploaded file's extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'zip':
            # Process the ZIP file
            unzip_discussions(uploaded_file, st.session_state.discussions_folder)
            st.sidebar.success("Discussioni caricate con successo!")
            
        elif file_extension in ['yaml', 'yml']:
            # Save the YAML file in the app folder, overwriting if necessary
            file_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"File {uploaded_file.name} salvato con successo!")

# If no password is entered nothing will happen
elif entered_keyword == "":
    pass

# If "scusa" is entered as a password it will restore chatbot functionalities
elif entered_keyword == "scusa":     
    if st.session_state.baloons == False:
        st.balloons()
        st.session_state.baloons = True
        
    if entered_keyword:
        st.sidebar.success("Concentriamoci sulla PowerBI Challenge")
        st.sidebar.success(f"{st.session_state.llm_config['active_model']} Attivato")
        st.session_state.chat_messages.append({"role": "assistant", "content": "Va bene, torniamo seri"})
        st.session_state.llm_model={st.session_state.llm_config['active_model']} 
        st.session_state.dumb_assistant = False
        print (f"{st.session_state.start_date} - {st.session_state.session_id}: Un curiosone si è scusato")


# If wrong password is entered the chatbot will be rendered dumb
else:
    if entered_keyword:
        st.sidebar.error("We pirletti! Questa non è la parola segreta")
        st.sidebar.error(f"Downgraded to {st.session_state.llm_config['punish_model']}")
        st.session_state.chat_messages.append({"role": "assistant", "content": "Visto che ti diverti a toccare cose che non devi toccare adesso anche io mi diverto"})
        st.session_state.llm_model=st.session_state.llm_config['punish_model']
        st.sidebar.write('Scusati e non farlo più')
        st.session_state.dumb_assistant = True
        print (f"{st.session_state.start_date} - {st.session_state.session_id}: Un curiosone sta toccando ciò che non dovrebbe toccare.")


################################################ CHATBOX

st.title(config['page_title'])
st.write(f"##### {config['page_subtitle']}")
# PowerBI Challenge meme for good luck and url to the challenges brief
st.write(f"Trovi opzioni nella sidebar, chatta sotto, recupera il brief qui: 🔗[**Challenge Brief**]({os.environ.get('brief')})")
#st.image(powerbi_image_path)

message_box = st.container()

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "E così sei venuto a supplicare aiuto eh?"}]

# Display chat messages from history on app rerun
for message in st.session_state.chat_messages:
    if message["role"] == "assistant":
        avatar = config['assistant_avatar']
    else:
        avatar = config['user_avatar']
    with message_box.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Aiutami non sto capendo PowerBI"):
    # Display user message in chat message container
    with message_box.chat_message("user", avatar=config['user_avatar']):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    print (f"{st.session_state.start_date} - {st.session_state.session_id}: Chat messages up to {len(st.session_state.chat_messages)} messages")
    
    if st.session_state.dumb_assistant:
        st.session_state.sys_prompt = prompts['sys_prompt_dumb']

    else:
        prompt_selection = openai_request(st.session_state.chat_messages, sys_msg=prompts['sys_prompt_select'], history_lenght=4, model=st.session_state.llm_config['selector_model'])

        if prompt_selection.choices[0].message.content == "True":
            print(f"{st.session_state.start_date} - {st.session_state.session_id}: Long Prompt selected for question: {prompt}")
            st.session_state.sys_prompt = prompts['sys_prompt_full']
        else:
            print(f"{st.session_state.start_date} - {st.session_state.session_id}: Short Prompt selected for question: {prompt}")
            st.session_state.sys_prompt = prompts['sys_prompt_short']

    if config['stream']:
        # Display assistant responses as they are streamed
        with message_box.chat_message("assistant", avatar=config['assistant_avatar']):
            # Use st.write_stream to handle and display the generator output
            api_responses = st.write_stream(openai_request_stream(st.session_state.chat_messages, sys_msg=st.session_state.sys_prompt))
            full_response = ''.join(api_responses)
    else:
        api_response = openai_request(st.session_state.chat_messages, sys_msg=st.session_state.sys_prompt)
        # Display assistant response in chat message container
        with message_box.chat_message("assistant", avatar = config['assistant_avatar']):
            st.write_stream()
            st.markdown(api_response.choices[0].message.content)
        full_response = api_response.choices[0].message.content
        # Add assistant response to chat history

    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
    save_chat_history()
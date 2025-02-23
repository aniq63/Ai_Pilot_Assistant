import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import tool
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
import speech_recognition as sr
import pyttsx3

# Set up the avionics theme
st.set_page_config(
    page_title="AI Pilot Assistant",
    page_icon="✈️",
    layout="wide",
)

# Custom CSS for avionics theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load environment variables
checkwx_api = os.getenv("CHECKWX_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
aviation_stack_key = os.getenv("AVIATION_STACK_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="mixtral-8x7b-32768",
    api_key=groq_api_key,
)

# Load emergency procedures from PDF
loader = PyPDFLoader("Flight Emergency Procedure.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)

# Emergency Procedure Tool
@tool
def emergency_procedures_tool(query: str) -> str:
    """
    Retrieves emergency procedures for aviation scenarios.
    """
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

# METAR Tool with Enhanced Error Handling
@tool
def metar_tool(icao_code: str) -> str:
    """
    Fetches METAR data for a given airport.
    """
    try:
        url = f"https://api.checkwx.com/metar/{icao_code}"
        headers = {"X-API-Key": checkwx_api}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["data"][0] if data.get("results") > 0 else "No METAR data found."
    except Exception as e:
        return f"⚠️ Error retrieving METAR: {str(e)}"

# Flight Data Tool
@tool
def flight_data_tool(icao24: str) -> str:
    """
    Fetches live flight data for a given aircraft.
    """
    try:
        url = f"https://opensky-network.org/api/states/all?icao24={icao24}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("states"):
            flight_data = data["states"][0]
            return f"""
            Callsign: {flight_data[1].strip()}
            Altitude: {flight_data[7]} meters
            Speed: {flight_data[9]} m/s
            Heading: {flight_data[10]}°
            Latitude: {flight_data[6]}
            Longitude: {flight_data[5]}
            """
        return "No flight data found."
    except Exception as e:
        return f"⚠️ Error retrieving flight data: {str(e)}"

# Weather Forecast Tool
@tool
def weather_forecast_tool(city_name: str) -> str:
    """
    Fetches weather forecast for a given city.
    """
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_name, "count": 1, "format": "json"}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            latitude, longitude = data["results"][0]["latitude"], data["results"][0]["longitude"]
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,weathercode",
                "forecast_days": 1,
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "hourly" in data:
                forecast = []
                for i in range(len(data["hourly"]["time"])):
                    forecast.append(
                        f"Time: {data['hourly']['time'][i]}, "
                        f"Temperature: {data['hourly']['temperature_2m'][i]}°C, "
                        f"Weather Code: {data['hourly']['weathercode'][i]}"
                    )
                return "\n".join(forecast)
        return "No weather forecast data found."
    except Exception as e:
        return f"⚠️ Error retrieving weather forecast: {str(e)}"

# Flight Status Tool (Using Aviation Stack API)
@tool
def flight_status_tool(flight_number: str) -> str:
    """
    Get real-time flight status with delay and gate information.
    """
    try:
        url = "http://api.aviationstack.com/v1/flights"
        params = {"access_key": aviation_stack_key, "flight_iata": flight_number}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            flight = data["data"][0]
            return f"""Flight {flight['flight']['iata']} ({flight['airline']['name']}):
Departure: {flight['departure']['airport']} at {flight['departure']['estimated']}
Arrival: {flight['arrival']['airport']} at {flight['arrival']['estimated']}
Status: {flight['flight_status']}
Delay: {flight['departure']['delay']} minutes"""
        return "Flight not found."
    except Exception as e:
        return f"⚠️ Error retrieving flight status: {str(e)}"

# Define the tools
tools = [
    Tool(
        name="Emergency Procedures",
        func=emergency_procedures_tool,
        description="Useful for retrieving emergency procedures for aviation scenarios.",
    ),
    Tool(
        name="METAR Data",
        func=metar_tool,
        description="Useful for fetching METAR data for airports.",
    ),
    Tool(
        name="Flight Data",
        func=flight_data_tool,
        description="Useful for retrieving live flight data.",
    ),
    Tool(
        name="Weather Forecast",
        func=weather_forecast_tool,
        description="Useful for fetching weather forecasts for destinations.",
    ),
    Tool(
        name="Flight Status",
        func=flight_status_tool,
        description="Useful for checking real-time flight status and delays.",
    ),
]

# Initialize memory for the chatbot
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a more conversational prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI Pilot Assistant specializing in aviation support. Your capabilities include:
- Retrieving emergency procedures from official flight manuals
- Providing real-time METAR data for airports
- Accessing live flight tracking information
- Analyzing weather patterns and forecasts
- Checking real-time flight status and delays"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Modify the agent initialization
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    memory=st.session_state.memory,
    verbose=True,
    prompt=prompt,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": "REMEMBER: You are a pilot assistant, not a general AI. "
                  "If asked non-aviation questions, respond with: "
                  "'I specialize in aviation support. How can I assist with your flight operations?'"
    }
)

# Initialize the recognizer and TTS engine (for local use only)
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def listen_to_user():
    """
    Listens to the user's voice input and converts it to text.
    """
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.write("Sorry, the speech service is down.")
            return None

def speak_response(response: str):
    """
    Converts the text response to speech.
    """
    tts_engine.say(response)
    tts_engine.runAndWait()

# Streamlit App
st.title("✈️ AI Pilot Assistant")
st.markdown("Welcome to the AI Pilot Assistant!")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Chat", "Example Queries"])

with tab1:
    st.markdown("""
    ### Features in this Assistant:
    - Real-time METAR data of any airport (ICAO code)
    - Emergency Guidance
    - Real-time Flight data & Status
    - Real-time Weather data of any location
    - Voice Integration (Local Only)
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add a button for voice input (only works locally)
    if st.button("🎤 Speak (Local Only)"):
        st.warning("Voice input is only available when running locally. Please type your query below.")

    # User input at the bottom
    if prompt := st.chat_input("Ask me anything about aviation..."):
        # Get response from agent
        response = agent.run(prompt)
        
        # Add messages to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display immediately
        st.rerun()

with tab2:
    st.markdown("""
    ### Example Queries
    Here are some example queries you can try:
    #### METAR Data
    - "What is the METAR for KJFK?"
    - "Get the METAR for EGLL."
    #### Emergency Procedures
    - "What is the procedure for engine failure?"
    - "How do I handle a cabin depressurization?"
    #### Flight Data
    - "What is the status of flight BA123?"
    - "Get live data for aircraft with ICAO24 code ABC123."
    #### Weather Forecast
    - "What is the weather forecast for New York?"
    - "Get the weather for London."
    #### Flight Status
    - "What is the status of flight AA456?"
    - "Is flight DL789 delayed?"
    """)

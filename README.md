# ✈️ AI Pilot Assistant

**An Intelligent Aviation Support System with Real-Time Data Integration**

A next-generation aviation assistant powered by Groq's lightning-fast LLM and Ai Agents designed to support pilots with real-time operational data and emergency procedures.

## 🌐 Live Demo  
Access the hosted version: https://huggingface.co/spaces/Aniq-63/AI_Pilot_Assistant

Live Video Demo : https://youtu.be/QE9iGox6elE

*(Note: Voice features disabled in cloud deployment)*

## Screenshots

![image alt](https://github.com/aniq63/Ai_Pilot_Assistant/blob/0e9bcf6abad5533554f4bef24661441dc9cd77de/Capture1.PNG)
![image alt](https://github.com/aniq63/Ai_Pilot_Assistant/blob/4d3fb467f24215f16ac6841b008af06c3b5fb49a/Capture2.PNG)

## 🚀 Features

- **Emergency Procedure Database**  
  Instant access to PDF flight manuals with semantic search
- **Real-Time Aviation Data**  
  METAR/TAF reports • Live flight tracking • Flight status/delay info
- **Smart Weather Integration**  
  Detailed forecasts • Weather pattern analysis
- **Voice Interface** (Local)  
  Speech-to-text • Text-to-speech responses
- **Aviation-Specific AI**  
  Strict terminology adherence • Context-aware responses

## Workflow Description

**User Interaction:**

The user interacts with the Streamlit App (UI) by typing or speaking (voice input, local only).

**Input Handling:**

The Input Handler processes the user's input (text or voice) and prepares it for the LangChain Agent.

**LangChain Agent:**

The LangChain Agent uses the Groq LLM to understand the query and selects the appropriate tool based on the input.

### Tool Execution:

The agent routes the query to one of the following tools:

**Emergency Procedures Tool:** Searches a vector store (PDF embeddings) for emergency procedures.

**METAR Tool:** Fetches real-time weather data for airports using the CheckWX API.

**Flight Data Tool:** Retrieves live flight data using the OpenSky API.

**Weather Forecast Tool:** Fetches weather forecasts using the Open-Meteo API.

**Flight Status Tool:** Retrieves real-time flight status using the Aviation Stack API.

### Data Retrieval:

Each tool retrieves the required data (e.g., emergency procedures, METAR data, flight info, or weather forecasts).

### Response Generation:

The retrieved data is formatted into a response by the respective tool.

### Output Handling:

The Output Handler displays the response in the chat interface and, if enabled, speaks the response using text-to-speech (local only).

### User Feedback:

The final output is delivered to the User (pilot/operator) for decision-making or further action.

## Future Enhancements

Aircraft Performance Analytics

Real time best Route Provider

Real time NOTAM data

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-pilot-assistant.git

# Navigate to project directory
cd ai-pilot-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to the .env file
```

## 🔑 API Configuration

Register for required services:

1. **Groq Cloud**: [Get API Key](https://console.groq.com/)
2. **CheckWX**: [METAR API](https://www.checkwx.com/)
3. **AviationStack**: [Flight Data](https://aviationstack.com/)
4. **OpenSky Network** (Optional): [Register](https://opensky-network.org/)

Add keys to `.env`:
```env
GROQ_API_KEY=your_key_here
CHECKWX_API_KEY=your_key_here
AVIATION_STACK_API_KEY=your_key_here
```


**Interface Guide:**
- **Chat Tab**: Primary interaction interface
- **Examples Tab**: Pre-built aviation queries
- **Voice Control**: Microphone button (local execution only)

## 💬 Example Queries

```text
"What's the emergency procedure for hydraulic failure?"
"Show METAR for EHAM"
"Get live data for ICAO24 code a0f3b1"
"What's the weather forecast for Dubai?"
"Explain rapid decompression protocol"
"Check status for Delta flight DL143"
```

## 📦 Dependencies

- `streamlit==1.33.0`
- `langchain-groq==0.1.3`
- `python-dotenv==1.0.0`
- `pyttsx3==2.90`
- `SpeechRecognition==3.10.0`
- `requests==2.31.0`

Full list in [requirements.txt](requirements.txt)


**Project Lead**: Mohammad Aniq Ramzazn - aniqramzan5758@gmail.com  


**Disclaimer**: This project is for educational/demonstration purposes only. Never use in real flight operations.
```

1. Clear technology badges for quick scanning
2. Visual placeholder for interface preview
3. Step-by-step setup instructions
4. API configuration details with direct links
5. Usage examples with realistic aviation queries
6. Voice interface limitations noted
7. Safety disclaimer

The structure emphasizes aviation professionalism while maintaining developer-friendly documentation.

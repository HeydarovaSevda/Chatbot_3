## Key Differences Between Two Chatbots  

### 1. Functionality Change  
**First bot:** Calculator and weather forecast  
**Second bot:** Customer feedback sentiment analysis and storage  

### 2. Number and Type of Tools  
**First bot:** 2 tools (calculator, forecaster)  
**Second bot:** 2 tools (sentiment_api, label_writer)    
  
### 3. External API Integration  
**First bot:** Simple wttr.in weather API  
**Second bot:** HuggingFace Inference API (ML model integration)  

### 4. Data Persistence  
**First bot:** Only log file  
**Second bot:**  
•Log file  
• Structured data in CSV file (labeled_feedbacks.csv)  
• SQLite database (ChatHistory.db)  

### 5. Chat Memory  
**First bot:** No memory  
**Second bot:** Session memory with RunnableWithMessageHistory and SQLChatMessageHistory  

### 6. Error Handling  
**First bot:** Basic try-except  
**Second bot:** Retry mechanism, timeout, multiple exception types  

### 7. Data Processing Pipeline  
**First bot:** Returns answer directly  
**Second bot:** 2-stage pipeline (API call → CSV write)  

### 8. Code Structure  
**First bot:** ~80 lines, simple  
**Second bot:** ~200 lines, modular, type hints  

### 9. Configuration  
**First bot:** 1 API key  
**Second bot:** 2 API keys, path management, constants  

### 10. System Prompt  
**First bot:** Simple instructions  
**Second bot:** Detailed 6-point strict rules  

#### Conclusion:
The second bot added enterprise-level features: ML integration, data persistence, memory, robust error handling, and structured workflow.

## ERRORS  

### 1.AgentExecutor  

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
agent = create_tool_calling_agent(llm, tools, prompt)  
agent_with_memory = RunnableWithMessageHistory(  
    agent,  
    get_session_history,  
    input_messages_key="input",  
    history_messages_key="chat_history")  

DEFAULT_SESSION = "feedback_session_hf"  
agent_with_memory = agent_with_memory.with_config(configurable={"session_id": DEFAULT_SESSION})    
agent_exe = AgentExecutor(agent=agent_with_memory, tools=tools, verbose=False, callbacks=loggers)  
```   
--------------------------------------------------------------------------------------  

**Reason:** While the LangChain agent was running, it was writing its intermediate step logs (“AgentActionMessageLog”) to the SQLChatMessageHistory.  

**Trying Solving way:** Write-level filter (SafeSQLChatMessageHistory) was implemented using the LangChain core message types to prevent unsupported message types such as AgentActionMessageLog from being written to the database. Later, an enhanced version with a read-level (@property) filter was added to handle any legacy or corrupted records that might still exist in the database, ensuring both safe writing and safe reading of chat history like:  

--------------------------------------------------------------------------------------  
```python  
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage   
ALLOWED_MESSAGE_TYPES = {  
    HumanMessage().type, AIMessage().type, SystemMessage().type, ToolMessage().type  
}  
class SafeSQLChatMessageHistory(SQLChatMessageHistory):  
    def add_message(self, message):  
        mtype = getattr(message, "type", None) or getattr(message, "to_dict", lambda: {})().get("type")    
        if mtype not in ALLOWED_MESSAGE_TYPES:  
            return  
        return super().add_message(message)  
```  
--------------------------------------------------------------------------------------  


--------------------------------------------------------------------------------------  
```python  
class SafeSQLChatMessageHistory(SQLChatMessageHistory):  
    @property  
    def messages(self):  
        try:  
            raw = super().messages  
        except Exception:  
            return []  
        safe = []  
        for m in raw:  
            try:  
                t = getattr(m, "type", None) or (m.to_dict().get("type") if hasattr(m,"to_dict") else None)  
                if t in ALLOWED_MESSAGE_TYPES:  
                    safe.append(m)  
            except Exception:  
                continue  
        return safe  
```  
--------------------------------------------------------------------------------------  

**Solving way:** I applied RunnableWithMessageHistory to the AgentExecutor, not to the agent.  

--------------------------------------------------------------------------------------  
```python  
DEFAULT_SESSION = "feedback_session_hf"  
agent = create_tool_calling_agent(llm, tools, prompt)  
agent_exe = AgentExecutor(agent=agent, tools=tools, verbose=False, callbacks=loggers)  
agent_with_memory = RunnableWithMessageHistory(  
    agent_exe,  
    get_session_history,  
    input_messages_key="input",  
    history_messages_key="chat_history",  
    output_messages_key="output"  
)  
agent_with_memory = agent_with_memory.with_config(configurable={"session_id": DEFAULT_SESSION})  
```  
--------------------------------------------------------------------------------------   


### 2.404 Client Error: Not Found for url: https://api-inference.huggingface.co/models/LocalDoc/sentiment_analysis_azerbaijani  

**Previous version:**   

--------------------------------------------------------------------------------------  
```python  
F_MODEL_ID = "LocalDoc/sentiment_analysis_azerbaijani"  
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"  
HF_HEADERS = {"Authorization": f"Bearer {API_KEY2}"}  
```  
--------------------------------------------------------------------------------------    

**Solving way:** I decided changed llm model  

--------------------------------------------------------------------------------------  
```python  
HF_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"  
HF_HEADERS = {"Authorization": f"Bearer {API_KEY2}"}  
```  
--------------------------------------------------------------------------------------  


### 3.Logger file not created    

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
result = agent_exe.invoke({"input": f"Analyze and save this JSON: {json.dumps(payload, ensure_ascii=False)}"},)  
```  
--------------------------------------------------------------------------------------  

**Solving way:**  

--------------------------------------------------------------------------------------  
```python  
result = agent_with_memory.invoke(  
                {"input": f"Analyze and save this JSON: {json.dumps(payload, ensure_ascii=False)}"},  
                config={"callbacks": loggers})  
```  
--------------------------------------------------------------------------------------  


### 4.Model loading issue after first question  

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
if __name__ == "__main__":  
    print("Customer feedback analyzer. Press q to exit.")   
    while True:  
        user_text = input("Feedback text: ").strip()  
        if user_text.lower() == "q":  
            print("See you!!")  
            break  

        fb_id = f"fb_{int(time.time())}"  
        payload = {  
            "feedback_id": fb_id,  
            "text": user_text,  
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S")   
        }  

        result = agent_with_memory.invoke(  
                {"input": f"Analyze and save this JSON: {json.dumps(payload, ensure_ascii=False)}"},  
                config={"callbacks": loggers}  
            )  

        print("\n--- Result (Summary) ---")  
        print(result.get("output", ""))  
        print("------------------------\n")  
```  
--------------------------------------------------------------------------------------  

**Solving way:** Added small pause between queries  

--------------------------------------------------------------------------------------  
```python  
if __name__ == "__main__":  
    print("Customer feedback analyzer. Press q to exit.")  
    
    while True:  
        user_text = input("Feedback text: ").strip()  
        if user_text.lower() == "q":  
            print("See you!!")  
            break  

        fb_id = f"fb_{int(time.time())}"  
        payload = {  
            "feedback_id": fb_id,  
            "text": user_text,  
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S")  
        }  

        try:  
            result = agent_with_memory.invoke(  
                {"input": f"Analyze and save this JSON: {json.dumps(payload, ensure_ascii=False)}"},  
                config={"callbacks": loggers}  
            )  

            print("\n--- Result (Summary) ---")  
            print(result.get("output", ""))  
            print("------------------------\n")  
            
            time.sleep(2)  
            
        except Exception as e:  
            print(f"❌ Error: {e}\n")  
```  
--------------------------------------------------------------------------------------  


### 5.CSV file issue( Just first question is in)  

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
prompt = ChatPromptTemplate.from_messages([  
    ("system",  
    "You are an Azerbaijani-language CX analytics agent.\n"  
    "Rules:\n"  
    "1) When a feedback text is provided, call the 'sentiment_api' tool.\n"  
    "2) Pass the returned JSON to the 'label_writer' tool to append it to the CSV file.\n"  
    "3) Give the user a short summary (sentiment + confidence).\n"  
    "4) If the input is not a feedback, do not call any tool and respond with an informative message."  
    ),  
    ("placeholder", "{chat_history}"),  
    ("human", "{input}")  
])  
```  
--------------------------------------------------------------------------------------  

**Solving way:** Added ("placeholder", "{agent_scratchpad}") and change a bit prompt

--------------------------------------------------------------------------------------    
```python  
prompt = ChatPromptTemplate.from_messages([  
    ("system",  
    "You are an Azerbaijani-language CX analytics agent.\n"  
    "Rules:\n"  
    "1) When a feedback text is provided, ALWAYS call the 'sentiment_api' tool.\n"  
    "2) After getting the result from sentiment_api, ALWAYS call the 'label_writer' tool to save it.\n"  
    "3) You MUST call both tools in sequence for every feedback.\n"  
    "4) After saving, give the user a short summary (sentiment + confidence).\n"  
    "5) If the input is not a feedback, do not call any tool and respond with an informative message.\n"  
    "IMPORTANT: Every feedback must be analyzed AND saved, no exceptions!"  
    ),  
    ("placeholder", "{chat_history}"),  
    ("human", "{input}"),  
    ("placeholder", "{agent_scratchpad}")  
])  
```  
--------------------------------------------------------------------------------------  


### 6.label_writer tool worked twice for one feedback sometimes  

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
def call_hf_sentiment(text: str, timeout: float = 30.0, retries: int = 2) -> Dict[str, Any]:  
    payload = {"inputs": text, "options": {"wait_for_model": True}}  
    last_err = None  
    for _ in range(retries + 1):  
        try:  
            r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=timeout)  
            if r.status_code == 503:  
                time.sleep(1.5)  
                last_err = {"error": f"service_503: {r.text[:200]}"}  
                continue  
            r.raise_for_status()  
            data = r.json()  
            if isinstance(data, list) and data and isinstance(data[0], dict) and "label" in data[0]:  
                best = max(data, key=lambda x: x.get("score", 0.0))  
                raw = str(best["label"])  
                label = LABEL_MAP.get(raw, raw.lower())  
                return {"label": label, "score": float(best.get("score", 0.0)), "raw": data}  
            return {"error": "unexpected_response", "raw": data}  
        except requests.RequestException as e:  
            last_err = {"error": "http_error", "detail": str(e)}  
            time.sleep(1.2)  
        except Exception as e:  
            last_err = {"error": "runtime_error", "detail": str(e)}  
            time.sleep(0.8)  
    return last_err or {"error": "unknown"}    
```  
--------------------------------------------------------------------------------------  

**Solving way:**  

--------------------------------------------------------------------------------------  
```python  
def call_hf_sentiment(text: str, timeout: float = 30.0, retries: int = 2) -> Dict[str, Any]:  
    payload = {"inputs": text, "options": {"wait_for_model": True}}  
    last_err = None  
    for _ in range(retries + 1):  
        try:  
            r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=timeout)  
            if r.status_code == 503:  
                time.sleep(1.5)  
                last_err = {"error": f"service_503: {r.text[:200]}"}  
                continue  
            r.raise_for_status()  
            data = r.json()  
            
            if isinstance(data, list) and len(data) > 0:  
                if isinstance(data[0], list):  
                    data = data[0]  
                    
                if isinstance(data[0], dict) and "label" in data[0]:  
                    best = max(data, key=lambda x: x.get("score", 0.0))  
                    raw = str(best["label"])  
                    label = LABEL_MAP.get(raw, raw.lower())  
                    return {"label": label, "score": float(best.get("score", 0.0)), "raw": data}  
            
            if isinstance(data, dict) and "label" in data:  
                raw = str(data["label"])  
                label = LABEL_MAP.get(raw, raw.lower())  
                return {"label": label, "score": float(data.get("score", 0.0)), "raw": data}  
                
            return {"error": "unexpected_response", "raw": data}  
            
        except requests.RequestException as e:  
            last_err = {"error": "http_error", "detail": str(e)}  
            time.sleep(1.2)  
        except Exception as e:  
            last_err = {"error": "runtime_error", "detail": str(e)}  
            time.sleep(0.8)  

    return last_err or {"error": "unknown"}  
```  
--------------------------------------------------------------------------------------  


### 7.Agent rewrite JSON between the two tool calls.  

**Previous version:**  

--------------------------------------------------------------------------------------  
```python  
prompt = ChatPromptTemplate.from_messages([  
    ("system",  
    "You are an Azerbaijani-language CX analytics agent.\n"    
    "Rules:\n"  
    "1) When a feedback text is provided, ALWAYS call the 'sentiment_api' tool.\n"  
    "2) After getting the result from sentiment_api, ALWAYS call the 'label_writer' tool to save it.\n"    
    "3) You MUST call both tools in sequence for every feedback.\n"  
    "4) After saving, give the user a short summary (sentiment + confidence).\n"  
    "5) If the input is not a feedback, do not call any tool and respond with an informative message.\n"   
    "IMPORTANT: Every feedback must be analyzed AND saved, no exceptions!"  
    ),  
    ("placeholder", "{chat_history}"),  
    ("human", "{input}"),  
    ("placeholder", "{agent_scratchpad}")  
])  
```  
--------------------------------------------------------------------------------------  

**Solving way:** Make prompt more clearly and professionally

--------------------------------------------------------------------------------------  
```python  
prompt = ChatPromptTemplate.from_messages([  
    ("system",  
    "You are an Azerbaijani-language CX analytics agent.\n"  
    "Rules:\n"  
    "1) When a feedback text is provided, ALWAYS call the 'sentiment_api' tool.\n"  
    "2) After calling 'sentiment_api', you MUST pass its JSON output UNCHANGED as 'record_json' to 'label_writer'.\n"  
    "3) Do NOT modify, reformat, round, translate, or add/remove any field and 'confidence' MUST stay as a decimal between 0 and 1 (no percentages).\n"  
    "4) After saving, you MUST include the EXACT output message from 'label_writer' in your final answer, and THEN add a short summary (sentiment + confidence).\n"  
    "5)If you don’t have the exact JSON, call 'sentiment_api' again. Never invent values.\n"  
    "6) If the input is not a feedback, do not call any tool and respond with an informative message.\n"  
    "IMPORTANT: Every feedback must be analyzed AND saved, no exceptions!"  
    ),  
    ("placeholder", "{chat_history}"),  
    ("human", "{input}"),  
    ("placeholder", "{agent_scratchpad}")  
])  
```  
--------------------------------------------------------------------------------------  



















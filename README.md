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
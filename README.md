A Python-based sentiment analysis agent that classifies customer feedback as positive, neutral, or negative using Hugging Face and OpenAI models.
The system automatically analyzes each feedback message, stores the labeled results in a CSV file, and logs all tool activity for full transparency.

It uses LangChain’s tool-calling and memory framework to manage conversations efficiently:

🧩 Logging system: Every tool call (start, end, input, output) is recorded in tool_usage.log, allowing you to trace how each feedback was processed.

💾 Memory system: Conversation and analysis history are stored in a local SQLite database (ChatHistory.db) using LangChain’s SQLChatMessageHistory, enabling persistent sessions across runs.
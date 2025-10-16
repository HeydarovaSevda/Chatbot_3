
import os, json, csv, time
from pathlib import Path
from typing import Dict, Any, List
import requests
from sqlalchemy import create_engine
from config import API_KEY1, API_KEY2 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "labeled_feedbacks.csv"
LOG_PATH = BASE_DIR / "tool_usage.log"


HF_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_HEADERS = {"Authorization": f"Bearer {API_KEY2}"}


engine = create_engine("sqlite:///ChatHistory.db", echo=False)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection=engine)


class ToolLogger(BaseCallbackHandler):
    def __init__(self, filename: str = "tool_usage.log"):
        self.path = str(BASE_DIR / filename)

    def _w(self, obj: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._w({
            "event": "tool_start",
            "tool": serialized.get("name"),
            "input": str(input_str)[:500],
            "t": time.strftime("%F %T")
        })

    def on_tool_end(self, output, **kwargs):
        self._w({
            "event": "tool_end",
            "output": str(output)[:800],
            "t": time.strftime("%F %T")
        })


LABEL_MAP = {
    "LABEL_0": "negative", 
    "LABEL_1": "neutral", 
    "LABEL_2": "positive",
    "negative": "negative", 
    "neutral": "neutral", 
    "positive": "positive",
    "Negative": "negative",
    "Neutral": "neutral", 
    "Positive": "positive"
}


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


@tool
def sentiment_api(payload_json: str) -> str:
    """
    Input JSON: {"feedback_id": "...", "text": "...", "processed_at": "..."}
    Output JSON: {"feedback_id", "text", "sentiment", "confidence", "processed_at"} or {"error": ...}
    """
    try:
        payload = json.loads(payload_json)
        text = payload.get("text", "").strip()
        if not text:
            return json.dumps({"error": "empty_text"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"json_parse_error: {e}"}, ensure_ascii=False)

    if not API_KEY2:
        return json.dumps({"error": "missing_hf_api_key"}, ensure_ascii=False)

    res = call_hf_sentiment(text)
    if "error" in res:
        return json.dumps({"error": "hf_error", "detail": res}, ensure_ascii=False)

    out = {
        "feedback_id": payload.get("feedback_id"),
        "text": text,
        "sentiment": res["label"],
        "confidence": round(float(res["score"]), 4),
        "processed_at": payload.get("processed_at")
    }
    return json.dumps(out, ensure_ascii=False)


@tool
def label_writer(record_json: str) -> str:
    """
    Input: The result of sentiment_api (JSON string).
    Action: Appends the result to the labeled_feedbacks.csv file.
    """
    try:
        record = json.loads(record_json)
    except Exception as e:
        return f"JSON parse error: {e}"

    header: List[str] = ["feedback_id", "text", "sentiment", "confidence", "processed_at"]
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in header})

    return f"OK: record appended to {CSV_PATH.name}"


llm = ChatOpenAI(api_key=API_KEY1, model="gpt-4o-mini", temperature=0.1)
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

tools = [sentiment_api, label_writer]
loggers = [ToolLogger()]



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
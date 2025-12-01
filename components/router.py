import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Configure Gemini
# Make sure GEMINI_API_KEY is in your .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class RouteDecision(BaseModel):
    decision: str
    reasoning: str

def route_query(user_query: str, web_search_allowed: bool = True) -> RouteDecision:
    """
    Decides the best strategy using Google Gemini (Free Tier).
    Respects user web search preference.
    """
    # Use 'gemini-1.5-flash' for speed
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Updated prompt with web restriction info
    web_restriction = "ALLOWED" if web_search_allowed else "DISABLED"
    prompt = f"""
    You are a Router Agent. 
    
    WEB SEARCH RESTRICTION: {web_restriction}
    
    Analyze the user query and classify it into one of these categories:
    
    1. 'RAG': The user is asking about a document, file, or specific context provided previously.
    {f"2. 'WEB': The user is asking for real-time news, stock prices, or facts about current events (post-2023)." if web_search_allowed else "2. Skip WEB route (disabled by user preference)"}
    3. 'DIRECT': The user is asking a general knowledge question, coding task, logic puzzle, or greeting.
    
    Return ONLY a raw JSON object. Do not output markdown blocks.
    Format: {{ "decision": "...", "reasoning": "..." }}
    
    User Query: "{user_query}"
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up markdown if Gemini adds it (e.g. ``````)
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "")
        text = text.strip()

            
        data = json.loads(text)
        
        return RouteDecision(
            decision=data.get("decision", "DIRECT"),
            reasoning=data.get("reasoning", "No reasoning provided")
        )

    except Exception as e:
        print(f"[Router Error] Gemini failed: {e}")
        return RouteDecision(decision="DIRECT", reasoning="Router failed")

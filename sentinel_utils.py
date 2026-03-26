"""
SentinelAI - Utility helpers
"""

import httpx


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # change to "mistral" if preferred


# -----------------------------------------------------------------------------
# DECISION ENGINE
# -----------------------------------------------------------------------------
def risk_decision(risk_score: float) -> dict:
    """
    risk_score: 0-100

    < 40   -> approve
    40-75  -> otp
    > 75   -> block
    """
    if risk_score < 40:
        return {
            "decision": "APPROVE",
            "label": "Low Risk",
            "color": "green",
            "message": "Transaction cleared. No suspicious activity detected.",
        }
    elif risk_score <= 75:
        return {
            "decision": "OTP",
            "label": "Medium Risk",
            "color": "orange",
            "message": "Unusual patterns detected. Step-up authentication required.",
        }
    else:
        return {
            "decision": "BLOCK",
            "label": "High Risk",
            "color": "red",
            "message": "High fraud probability. Transaction blocked for review.",
        }


# -----------------------------------------------------------------------------
# OLLAMA LOCAL LLM
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a fraud detection AI assistant for SentinelAI.

Transaction Details:
- Amount: ${amount}
- Time (seconds since epoch start): {time}
- Device / Channel: {device}
- Location: {location}
- Risk Score: {risk_score}/100
- Model Decision: {decision}

Provide a structured fraud analysis with exactly these four sections:
1. Suspicion Indicators: Why this transaction looks suspicious
2. Risk Assessment: Detailed risk level explanation
3. Recommended Action: What the system and customer should do
4. Possible Attacker Intent: What a fraudster might be attempting

Keep it concise, professional, and actionable. Do NOT add extra headers."""


async def get_llm_explanation(
    amount: float,
    time: float,
    device: str,
    location: str,
    risk_score: float,
    decision: str,
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        amount=f"{amount:.2f}",
        time=f"{time:.0f}",
        device=device,
        location=location,
        risk_score=f"{risk_score:.1f}",
        decision=decision,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "LLM response unavailable.")
    except httpx.ConnectError:
        return (
            "Ollama is not running. Start it with `ollama serve` and pull a model "
            "with `ollama pull llama3`.\n\n"
            "Fallback analysis: Transaction flagged by ML model based on statistical "
            f"anomaly detection. Risk score: {risk_score:.1f}/100. Decision: {decision}."
        )
    except Exception as exc:
        return f"LLM error: {exc}"
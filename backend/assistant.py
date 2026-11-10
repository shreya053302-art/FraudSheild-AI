"""ShieldBot: Gemini-powered conversational assistant with a useful offline fallback."""
from __future__ import annotations

import re
import requests

from . import database as db
from .config import GEMINI_MODEL, gemini_key

SYSTEM_PROMPT = """
You are ShieldBot, the friendly AI assistant built into FraudShield AI, a student project for
credit-card fraud detection and investigation.

PERSONALITY:
- Be warm, patient, encouraging and conversational, like a helpful tutor.
- Understand the user's exact question. Never reuse an unrelated answer.
- Explain in simple English first; define technical terms and give a practical example.
- Give enough detail to be genuinely useful, normally 2-6 short paragraphs or concise steps.
- If the user asks "how", give actionable step-by-step instructions.
- If the user asks about this application, explain how to use its relevant screens:
  Analyze Transaction, Bulk CSV Screening, Investigation Center, Analytics & Reports.
- For fraud detection, explain warning signs such as unusual amount, unfamiliar merchant/device,
  unusual location or timing, repeated attempts, account changes, and customer confirmation.
- Explain that a model score is a signal for review, not proof. Never claim a transaction is
  definitely fraudulent based only on a model prediction.
- Do not invent transaction details, model results, dataset columns, or app capabilities.
- For Kaggle's common credit-card dataset, Time and V1-V28 are anonymized/numeric features,
  Amount is the transaction amount, and Class is the target (0 legitimate, 1 fraud).
- Never ask for card numbers, CVV, OTP, passwords, or other sensitive credentials.
- If a question is outside fraud/finance/security, answer briefly if appropriate, then relate
  back to the app when useful. If uncertain, say so clearly.
"""

SUGGESTED = [
    "What is financial fraud?",
    "How do I read a risk score?",
    "How can I detect a suspicious transaction?",
    "What should I do when a transaction is flagged?",
    "Explain precision, recall and F1 score",
    "How does the credit-card dataset work?",
]

def _transaction_answer(question: str) -> str | None:
    m = re.search(r"\b(?:transaction|txn|case)\s*#?\s*(\d+)\b|#\s*(\d+)", question.lower())
    if not m:
        return None
    tx_id = int(next(g for g in m.groups() if g))
    tx = db.get_transaction(tx_id)
    if tx is None:
        return f"I couldn't find transaction or case #{tx_id} in the saved records. First analyze a transaction or upload a CSV, then try its saved ID."
    return (
        f"Here is what FraudShield has saved for transaction #{tx['id']}:\n\n"
        f"- **Prediction:** {tx['prediction']}\n"
        f"- **Risk score:** {float(tx['risk_score']):.1f}/100 ({tx['risk_level']})\n"
        f"- **Status:** {tx['status']}\n"
        f"- **Model explanation:** {tx['explanation']}\n\n"
        "What to do next: compare the activity with the customer's usual spending, verify the transaction "
        "through a trusted contact method, and record the evidence in Investigation Center. This score is "
        "a warning signal, not proof of fraud."
    )

def fallback_answer(question: str) -> str:
    q = re.sub(r"[^a-z0-9%+.-]+", " ", question.lower()).strip()
    tx = _transaction_answer(question)
    if tx:
        return tx

    # Specific intents first. Avoid broad matches such as the word "why" by itself.
    rules = [
        (("precision", "recall", "f1", "pr auc", "accuracy", "confusion matrix"),
         "**Model metrics, simply explained:**\n\n- **Precision:** Of the transactions flagged as fraud, how many really were fraud?\n- **Recall:** Of all actual fraud cases, how many did the model catch?\n- **F1-score:** A balance between precision and recall.\n- **PR-AUC:** Measures precision/recall performance across thresholds.\n- **Accuracy:** Overall correct predictions; it can be misleading when fraud is rare.\n\nFor fraud detection, look closely at precision, recall, F1 and PR-AUC—not accuracy alone."),
        (("random forest", "logistic regression", "which model", "model comparison"),
         "**Why use Random Forest?** It combines many decision trees and can learn non-linear relationships between transaction features. It is often effective on tabular data and can capture combinations of unusual signals. Logistic Regression is a useful simpler baseline. Compare both on the same untouched test set using precision, recall, F1 and PR-AUC; don't choose a model using accuracy alone."),
        (("dataset", "creditcard csv", "v1", "v28", "class column", "time column"),
         "**How the credit-card dataset works:** Each row represents one transaction. `Time` records elapsed seconds from the first transaction, `Amount` is the transaction amount, `V1` to `V28` are anonymized numerical features, and `Class` is the label: `0` means legitimate and `1` means fraud. The V-columns are not ordinary readable details such as customer name or merchant. In FraudShield, use the dataset to train the model, then use Analyze Transaction or Bulk CSV Screening to score new rows."),
        (("risk score", "risk level", "score mean", "risk percentage", "high risk"),
         "**How to read the FraudShield risk score:** It is a 0–100 score used to rank how unusual a transaction looks to the model. In this project, scores below 50 are not flagged and scores of 50 or above are flagged. The displayed bands are Low (0–24), Medium (25–49), High (50–74) and Critical (75–100). It is **not** a guaranteed probability that fraud occurred. A high score means investigate; it does not prove wrongdoing."),
        (("flagged", "suspicious transaction", "detect fraud", "identify fraud", "fraud pattern", "spot fraud", "warning sign"),
         "**How to spot a potentially fraudulent transaction:** Look for several signals together:\n\n1. An amount that is unusual for the customer.\n2. An unfamiliar merchant, device or location.\n3. Transactions at unusual times or many attempts in a short period.\n4. Sudden account or contact-detail changes.\n5. The customer says they did not make the payment.\n\nIn FraudShield, run the transaction through Analyze Transaction or upload a CSV in Bulk CSV Screening. Review its risk score and explanation, then verify through a trusted channel. One unusual feature alone is not proof."),
        (("investigat", "what should i do", "next step", "review a case", "case status"),
         "**A safe investigation workflow:**\n\n1. Open **Investigation Center** and select the flagged case.\n2. Read the risk score and model explanation.\n3. Compare the amount, timing and merchant/activity with the customer's normal behaviour.\n4. Verify with the customer using a phone number or channel already on file—never ask for an OTP or PIN.\n5. Record objective evidence in Investigation Notes.\n6. Update the status to In Review, Escalated, Resolved or False Positive as appropriate.\n\nTreat the model as decision support and follow your organisation's review policy."),
        (("false positive",),
         "A **false positive** happens when the model flags a legitimate transaction as suspicious. For example, a customer may make an unusually large but genuine purchase. Investigators verify the activity and mark the case False Positive when evidence supports that conclusion. Tracking false positives helps teams tune thresholds and improve the model."),
        (("false negative",),
         "A **false negative** is a fraudulent transaction that the model incorrectly classifies as legitimate. This is important because the fraud may go undetected. Recall measures how many actual fraud cases the model catches. Lowering the decision threshold may catch more fraud, but it can also create more false alarms."),
        (("financial fraud", "what is fraud", "define fraud", "fraud mean"),
         "**Financial fraud** is when someone uses deception or stolen information to take money or gain a financial benefit without permission. Examples include stolen-card purchases, phishing, identity theft and account takeover. A simple example: someone uses another person's card details to buy something without the cardholder's approval."),
        (("phishing", "otp", "account takeover", "card security", "protect my account", "secure my account"),
         "**Protect your account:** Never share your OTP, PIN, CVV or password—even with someone claiming to be from your bank. Use a unique password and two-factor authentication, turn on transaction alerts, avoid suspicious links, review statements, and contact your bank using its official app or number if you see an unknown payment."),
        (("hello", "hi", "hey", "good morning", "good evening"),
         "Hi there! 👋 I'm ShieldBot, your FraudShield AI assistant. I can explain suspicious transaction signals, risk scores, model metrics, dataset fields and investigation steps. What would you like to understand?"),
    ]
        # Handle greetings only when the message is just a greeting.
    if re.fullmatch(
        r"(hi|hello|hey|good morning|good afternoon|good evening)( shieldbot)?[!. ]*",
        q
    ):
        return "Hi there! 👋 I'm ShieldBot. How can I help you today?"

    # Help users understand how to use the application.
    if any(x in q for x in [
    "how to use the app",
    "how do i use the app",
    "how can i use the app",
    "use this app"
]):
        return (
            "Of course! 😊 Here is a quick guide to FraudShield AI:\n\n"
            "1. **Dashboard:** View fraud detection statistics and summaries.\n"
            "2. **Analyze Transaction:** Enter transaction details and check its risk score.\n"
            "3. **Bulk Analysis:** Upload a CSV file to analyze multiple transactions.\n"
            "4. **Investigation Center:** Review flagged transactions and update case details.\n"
            "5. **Analytics & Reports:** Explore model performance and reports.\n"
            "6. **Model Center:** View information about your machine learning model.\n\n"
            "Which section would you like me to explain step by step?"
        )

    if "my name is" in q or "thank you" in q or "thanks" in q:
        return (
            "You're very welcome! 😊 I'm happy to help you with FraudShield AI. "
            "Ask me anything about fraud detection or how to use the application."
        )
        
    if "dashboard" in q:
        return (
            "**Let's explore the FraudShield AI Dashboard! 📊**\n\n"
            "The Dashboard gives you an overview of your fraud detection application.\n\n"
            "1. Open Dashboard from the left sidebar.\n"
            "2. Explore the transaction statistics and summaries.\n"
            "3. Check the fraud detection overview.\n"
            "4. Review the charts and available performance information.\n\n"
            "The Dashboard helps you understand the overall activity of FraudShield AI.\n"
            "You can share a screenshot of your Dashboard, and I'll explain each section "
            "one by one."
        )    
    if "analyze transaction" in q:
        return (
            "**How to use Analyze Transaction:**\n\n"
            "1. Open Analyze Transaction from the left sidebar.\n"
            "2. Enter the transaction details requested by the form.\n"
            "3. Click the Analyze button.\n"
            "4. Review the prediction, risk score and explanation.\n"
            "5. If flagged, review the transaction in Investigation Center.\n\n"
            "Remember: A high risk score is a warning, not proof of fraud."
        )
        
        # DASHBOARD
    if "dashboard" in q:
        return (
            "**Welcome to the FraudShield AI Dashboard! 📊**\n\n"
            "The Dashboard gives you an overview of your fraud detection system.\n\n"
            "1. Open Dashboard from the left sidebar.\n"
            "2. Review the transaction statistics and summaries.\n"
            "3. Explore the available charts and fraud detection information.\n"
            "4. Use these insights to understand the overall system activity.\n\n"
            "The Dashboard is your starting point for monitoring FraudShield AI."
        )

    # ANALYZE TRANSACTION
    if any(x in q for x in [
        "analyze transaction", "single transaction",
        "individual transaction", "transaction analysis"
    ]):
        return (
            "**How to use Analyze Transaction 🔍**\n\n"
            "1. Open Analyze Transaction from the sidebar.\n"
            "2. Enter the transaction details requested by the form.\n"
            "3. Click the Analyze or Predict button shown on the page.\n"
            "4. Review the prediction and risk score.\n"
            "5. Read the model explanation to understand the warning signals.\n"
            "6. If flagged, review the case in Investigation Center.\n\n"
            "Remember: A high risk score is a warning, not proof of fraud."
        )

    # BULK ANALYSIS
    if any(x in q for x in [
        "bulk analysis", "bulk upload", "upload csv",
        "csv file", "multiple transactions", "bulk screening"
    ]):
        return (
            "**How to use Bulk Analysis 📁**\n\n"
            "1. Open Bulk Analysis from the sidebar.\n"
            "2. Select the CSV upload option.\n"
            "3. Choose your transaction dataset file.\n"
            "4. Check that the uploaded file is accepted by the application.\n"
            "5. Start the bulk analysis using the displayed button.\n"
            "6. Review the predictions and flagged transactions.\n"
            "7. Use any available report or export option to save the results.\n\n"
            "Important: Use the expected dataset format and never upload real "
            "card numbers, CVVs, passwords or OTPs."
        )

    # INVESTIGATION CENTER
    if any(x in q for x in [
        "investigation center", "investigation",
        "investigate a case", "review a case", "flagged case"
    ]):
        return (
            "**How to use Investigation Center 🕵️**\n\n"
            "1. Open Investigation Center from the sidebar.\n"
            "2. Select a flagged transaction or case.\n"
            "3. Review its prediction, risk score and model explanation.\n"
            "4. Examine the available transaction information.\n"
            "5. Add investigation notes if the option is available.\n"
            "6. Update the case status using the available controls.\n"
            "7. Record the evidence and outcome carefully.\n\n"
            "A flagged transaction must be reviewed by a human investigator."
        )

    # ANALYTICS AND REPORTS
    if any(x in q for x in [
        "analytics", "reports", "performance chart",
        "model performance", "confusion matrix"
    ]):
        return (
            "**How to use Analytics & Reports 📈**\n\n"
            "1. Open Analytics & Reports from the sidebar.\n"
            "2. Explore the available fraud detection charts.\n"
            "3. Review the model performance measurements.\n"
            "4. Understand precision, recall, F1-score and accuracy.\n"
            "5. Compare the available model results where provided.\n"
            "6. Review any available reports.\n\n"
            "For fraud detection, precision and recall are especially important. "
            "Accuracy alone may be misleading when fraud cases are rare."
        )

    # MODEL CENTER
    if any(x in q for x in [
        "model center", "machine learning model",
        "random forest", "model status", "trained model"
    ]):
        return (
            "**How to use Model Center 🧠**\n\n"
            "1. Open Model Center from the sidebar.\n"
            "2. Review the information about the available machine learning model.\n"
            "3. Check the model status and displayed performance information.\n"
            "4. Explore any model comparison or evaluation details available.\n\n"
            "FraudShield currently displays a Random Forest model. "
            "A model prediction is an estimate, not proof of fraud."
        )

    # SETTINGS
    if any(x in q for x in [
        "settings", "application settings",
        "configure the app", "configuration"
    ]):
        return (
            "**How to use Settings ⚙️**\n\n"
            "1. Open Settings from the sidebar.\n"
            "2. Review the configuration options displayed.\n"
            "3. Read the descriptions before changing any option.\n"
            "4. Change only the settings you understand and intend to modify.\n\n"
            "The exact options depend on the settings available in your version."
        )

    # GENERAL APP NAVIGATION
    if any(x in q for x in [
        "features", "app modules", "app sections",
        "navigate", "where can i find"
    ]):
        return (
            "**FraudShield AI – Main Features 🛡️**\n\n"
            "1. Dashboard – View system statistics and summaries.\n"
            "2. Analyze Transaction – Analyze an individual transaction.\n"
            "3. Bulk Analysis – Screen multiple transactions using CSV files.\n"
            "4. Investigation Center – Review flagged cases.\n"
            "5. Analytics & Reports – Explore model performance and reports.\n"
            "6. Model Center – View machine learning model information.\n"
            "7. Settings – Review application configuration options.\n\n"
            "Tell me the name of any section and I can explain how to use it."
        )

    # ABOUT SHIELDBOT
    if any(x in q for x in [
        "what can you do", "who are you",
        "what is fraudshield", "about fraudshield"
    ]):
        return (
            "Hello! 👋 I'm ShieldBot, the assistant inside FraudShield AI.\n\n"
            "I can help explain fraud detection, suspicious transactions, risk scores, "
            "machine learning metrics, datasets and the different application features.\n\n"
            "I can also guide you through the Dashboard, Analyze Transaction, "
            "Bulk Analysis, Investigation Center, Analytics & Reports and Model Center."
        )

    # PRIVACY AND SECURITY
    if any(x in q for x in [
        "is my data safe", "privacy", "data security",
        "is it safe to upload"
    ]):
        return (
            "**Security reminder 🔐**\n\n"
            "Never enter real card numbers, CVVs, OTPs, PINs or passwords "
            "into a student demonstration application.\n\n"
            "Use sample or anonymized transaction data for testing. "
            "Follow your institution's and application's data-handling requirements."
        )   
    for phrases, answer in rules:
        if any(p in q for p in phrases):
            return answer

    return (
        "I want to make sure I answer the question you actually asked. The built-in offline assistant "
        "has limited topic coverage. For open-ended questions and natural conversations, add a Gemini API "
        "key to enable AI mode (see the project README, section **ShieldBot AI mode**). "
        "Meanwhile, try asking about risk scores, suspicious transaction signs, investigations, false "
        "positives, the credit-card dataset, Random Forest, or precision/recall."
    )

def _gemini_answer(question: str, history: list[dict], key: str) -> str:
    contents = []
    for m in history[-10:]:
        role = "user" if m.get("role") == "user" else "model"
        content = str(m.get("content", "")).strip()
        if content:
            contents.append({"role": role, "parts": [{"text": content[:4000]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        timeout=35,
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.65, "maxOutputTokens": 900},
        },
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("The AI service returned no answer.")
    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    if not answer.strip():
        raise RuntimeError("The AI service returned an empty answer.")
    return answer.strip()


def ask(question: str, history: list[dict]) -> tuple[str, str]:
    """Return (answer, source), where source is 'gemini' or 'built-in'."""
    question = (question or "").strip()[:1000]

    if not question:
        return "Please type a question first.", "built-in"

    tx = _transaction_answer(question)
    if tx:
        return tx, "built-in"

    key = gemini_key()

    if key:
        try:
            return _gemini_answer(question, history, key), "gemini"
        except Exception:
            return fallback_answer(question), "built-in"

    return fallback_answer(question), "built-in"

    

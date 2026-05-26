# Phonely Setup Guide — Phoenix Air Agent

## Step 1: Run the backend

```bash
cd phonely-airline
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Step 2: Expose locally with ngrok

```bash
ngrok http 8000
# Copy the HTTPS URL, e.g. https://abc123.ngrok.io
```

## Step 3: Configure Phonely

1. Log in to your Phonely dashboard
2. Create a new Agent
3. Under **Agent Design**, paste this as the agent guideline:

---
You are a friendly airline booking assistant for Phoenix Air.
Your job is to guide callers through booking a flight step by step.
Always be polite, confirm information before moving on, and speak clearly.
If a caller asks about refunds, cancellations, or baggage — answer using your knowledge base.
If a caller asks to speak to a human, transfer them immediately.
---

4. Under **Knowledge Base**, upload `phonely/knowledge-base.md`

## Step 4: Add the API Request block (Workflow)

Create a workflow with a single **API Request** block:

- **Method**: POST
- **URL**: `https://abc123.ngrok.io/api/voice`
- **Body**:
```json
{
  "text": "{{user_input}}",
  "session_id": "{{call_id}}",
  "caller_phone": "{{caller_phone}}"
}
```
- **Store response**:
  - `response` → variable `agent_response`
  - `end_call` → variable `should_end`
  - `transfer` → variable `should_transfer`

## Step 5: Flow logic

- After API Request → **Talk** block: speak `{{agent_response}}`
- Add **Filter**: if `should_end == true` → **End Call** block
- Add **Filter**: if `should_transfer == true` → **Transfer Call** block
- Otherwise → loop back to collect next user input → API Request again

## Step 6: Test

Use Phonely's built-in **Test Call** feature or dial your purchased number.

Test cases to verify:
- Normal booking: JFK → LAX, any future date
- No flights: AAL → YVR
- Policy question: "What is your refund policy?"
- Transfer: "I want to speak to a human"

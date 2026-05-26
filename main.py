import os
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from db.database import (
    init_db, load_session, save_session, load_user_profile, cleanup_session,
    get_booking_by_confirmation, get_recent_judge_logs,
)
from state import empty_state
from graph import airline_graph
from services.timeout_guard import run_with_timeout

load_dotenv()

app = FastAPI(title="Phoenix Air Voice Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    from db.database import get_conn
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
    conn.close()
    if count == 0:
        from db.seed_airports import seed
        seed()

    # Pre-build embedding matrices at startup so first request is fast.
    try:
        from services.rag import _build_matrices, _build_general_matrices
        _build_matrices()
        _build_general_matrices()
        print("[STARTUP] Embedding warmup complete.")
    except Exception as e:
        print(f"[STARTUP] Warmup error (non-fatal): {e}")


# ── Request / Response models ──────────────────────────────────────

class VoiceRequest(BaseModel):
    text:         str
    session_id:   str
    caller_phone: str = ""
    user_id:      Optional[int] = None
    language:     str = "en"   # set by web login or language selection node


class VoiceResponse(BaseModel):
    response:         str
    end_call:         bool  = False
    transfer:         bool  = False
    conv_state:       str   = "IDLE"
    suggestions:      list  = []
    flights:          list  = []
    flight_sort:      str   = ""
    response_type:    str   = "llm"   # "llm" | "hardcoded" | "fallback"
    # Booking data for sidebar cards + payment UI
    departure_city:   str   = ""
    departure_iata:   str   = ""
    destination_city: str   = ""
    destination_iata: str   = ""
    travel_date:      str   = ""
    passenger_name:   str   = ""
    flight_price:     float = 0.0
    airline:          str   = ""
    flight_number:    str   = ""
    confirmation_number: str = ""


class SignupRequest(BaseModel):
    phone:      str
    first_name: str
    last_name:  str
    email:      str = ""
    password:   str


class LoginRequest(BaseModel):
    phone_or_email: str
    password:       str


# ── Background LLM judge ───────────────────────────────────────────

def _run_judge_bg(session_id: str, node_name: str, user_input: str,
                   response: str, context: dict, language: str):
    try:
        from services.llm_judge import judge_response
        judge_response(
            session_id=session_id,
            node_name=node_name,
            user_input=user_input,
            response=response,
            context=context,
            language=language,
        )
    except Exception as e:
        print(f"[JUDGE BG ERROR] {e}")


# ── Main voice endpoint ────────────────────────────────────────────

@app.post("/api/voice", response_model=VoiceResponse)
def voice_endpoint(req: VoiceRequest):
    saved = load_session(req.session_id)

    if saved:
        state = saved
        state["user_input"] = req.text
        state["caller_phone"] = req.caller_phone or state.get("caller_phone", "")
        # Allow web-login user_id to be injected into an existing session
        if req.user_id and not state.get("user_id"):
            state["user_id"] = req.user_id
    else:
        state = empty_state(req.session_id, req.caller_phone,
                            language=req.language or "en",
                            user_id=req.user_id)
        state["user_input"] = req.text

        # Returning customer greeting
        if req.caller_phone:
            profile = load_user_profile(req.caller_phone)
            if profile:
                first     = profile.get("first_name", "")
                last_from = profile.get("last_from", "")
                last_to   = profile.get("last_to", "")
                if first:
                    welcome = f"Welcome back, {first}! "
                    if last_from and last_to:
                        welcome += (
                            f"Last time you flew from {last_from} to {last_to}. "
                            "Are you booking a similar trip today?"
                        )
                    else:
                        welcome += "How can I help you today?"
                    state["response"] = welcome

    # ── First turn: show language selection if no input ────────────
    if state.get("conv_state") in ("IDLE", "SELECTING_LANGUAGE") and not req.text.strip():
        welcome = (
            "Welcome to Phoenix Air!\n"
            "Please select your language / Seleccione su idioma:\n"
            "1. English  2. Español  3. Français  4. हिंदी  "
            "5. 中文  6. العربية  7. Português  8. Deutsch"
        )
        if state.get("response"):
            welcome = state["response"]   # returning customer custom welcome
        state["conv_state"] = "SELECTING_LANGUAGE"
        save_session(req.session_id, state["conv_state"], state)
        return VoiceResponse(response=welcome, conv_state="SELECTING_LANGUAGE")

    # ── Run LangGraph agent (15s timeout, 1 retry) ─────────────────
    fallback_state = {**state, "response": "I'm sorry, that took too long. Please try again.", "end_call": False}
    result = run_with_timeout(
        airline_graph.invoke,
        args=(state,),
        timeout_s=15.0,
        retries=1,
        fallback=fallback_state,
    )

    # ── Persist updated session ────────────────────────────────────
    new_conv_state = result.get("conv_state", "IDLE")
    save_session(req.session_id, new_conv_state, result)

    if new_conv_state == "BOOKING_CONFIRMED":
        cleanup_session(req.session_id)

    # ── Background LLM judge (non-blocking) ───────────────────────
    threading.Thread(
        target=_run_judge_bg,
        args=(
            req.session_id,
            new_conv_state,
            req.text,
            result.get("response", ""),
            {
                "departure_city":  result.get("departure_city"),
                "destination_city": result.get("destination_city"),
                "travel_date":     result.get("travel_date"),
                "conv_state":      new_conv_state,
            },
            result.get("language", "en"),
        ),
        daemon=True,
    ).start()

    _flight = result.get("selected_flight") or {}
    _first  = result.get("passenger_first", "") or ""
    _last   = result.get("passenger_last",  "") or ""
    return VoiceResponse(
        response=result.get("response", "I'm sorry, something went wrong."),
        end_call=result.get("end_call", False),
        transfer=result.get("transfer", False),
        conv_state=new_conv_state,
        suggestions=result.get("suggestions") or [],
        flights=result.get("flights") or [],
        flight_sort=result.get("flight_sort") or "",
        response_type=result.get("response_type", "llm"),
        departure_city=result.get("departure_city", "") or "",
        departure_iata=result.get("departure_iata", "") or "",
        destination_city=result.get("destination_city", "") or "",
        destination_iata=result.get("destination_iata", "") or "",
        travel_date=result.get("travel_date", "") or "",
        passenger_name=f"{_first} {_last}".strip(),
        flight_price=float(_flight.get("price", 0) or 0),
        airline=_flight.get("airline", "") or "",
        flight_number=_flight.get("flightNumber", "") or "",
        confirmation_number=result.get("confirmation", "") or "",
    )


# ── Auth endpoints ─────────────────────────────────────────────────

@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    from services.auth import create_account
    result = create_account(req.phone, req.first_name, req.last_name, req.email, req.password)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Signup failed"))
    return result


@app.post("/api/auth/login")
def login_endpoint(req: LoginRequest):
    from services.auth import login_account
    user = login_account(req.phone_or_email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "success":    True,
        "user_id":    user["id"],
        "first_name": user["first_name"],
        "last_name":  user["last_name"],
        "phone":      user["phone"],
        "email":      user.get("email", ""),
    }


# ── Ticket download ────────────────────────────────────────────────

@app.get("/api/ticket/{confirmation}")
def download_ticket(confirmation: str):
    booking = get_booking_by_confirmation(confirmation)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    from services.pdf_ticket import generate_ticket
    pdf_bytes = generate_ticket(booking)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="ticket_{confirmation}.pdf"'},
    )


# ── Observability ──────────────────────────────────────────────────

@app.get("/api/observability")
def get_observability(limit: int = 100):
    logs = get_recent_judge_logs(limit)
    avg = lambda key: (
        round(sum(l[key] for l in logs if l.get(key)) / max(1, sum(1 for l in logs if l.get(key))), 2)
        if logs else None
    )
    return {
        "total":            len(logs),
        "avg_relevance":    avg("relevance"),
        "avg_accuracy":     avg("accuracy"),
        "avg_helpfulness":  avg("helpfulness"),
        "safety_pass_rate": (
            round(sum(1 for l in logs if l.get("safety") == "pass") / max(1, len(logs)), 2)
            if logs else None
        ),
        "logs": logs,
    }


# ── Health + static ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")


app.mount("/app", StaticFiles(directory="frontend"), name="frontend")

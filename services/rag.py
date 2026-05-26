"""
Policy retrieval — 2 layers:

  L1 (rule-based)  : keyword match → pre-authored short answer (no LLM, instant)
  L2 (embedding)   : dual cosine similarity —
                       Score A: query vs seed questions (question ↔ question)
                       Score B: query vs full policy doc  (question ↔ paragraph)
                     combined = max(A, B) per policy → winner returns full text as LLM context

  Model: all-MiniLM-L6-v2 — shared singleton via services/embedder.py
"""

import numpy as np
from services.embedder import get_embedder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Policy documents ──────────────────────────────────────────────────────────

POLICIES = [
    # ── 1. Refund ─────────────────────────────────────────────────────────────
    {
        "id": "refund_policy",
        "text": (
            "Refund Policy: Tickets booked on Phoenix Air are fully refundable within 24 hours "
            "of purchase, no questions asked. After 24 hours, refunds are available for a fee "
            "of $75 per ticket. Non-refundable tickets may be converted to travel credit valid "
            "for 12 months. Refunds are processed within 7-10 business days to the original payment method."
        ),
        "seed_questions": [
            "Can I get a refund?",
            "Will I get my money back if I cancel?",
            "What is the refund policy?",
            "How do I request a refund?",
            "Is my ticket refundable?",
            "What is the refund fee?",
            "How long does a refund take?",
            "Can I get a full refund on my ticket?",
            "How do I get my money back?",
            "What happens to my money if I cancel?",
        ],
    },
    # ── 2. Flight change ──────────────────────────────────────────────────────
    {
        "id": "change_policy",
        "text": (
            "Flight Change Policy: You may change your flight up to 2 hours before scheduled departure. "
            "Same-day changes cost $50. Changes made more than 7 days in advance are free of charge. "
            "Changes within 2-7 days incur a $35 fee. Your new flight must be within 12 months of "
            "the original booking date. Fare differences apply if the new flight is more expensive."
        ),
        "seed_questions": [
            "Can I change my flight?",
            "How do I reschedule my flight?",
            "What is the fee for changing my flight?",
            "Can I modify my booking?",
            "How much does it cost to change a flight?",
            "Is it free to change my flight?",
            "How late can I change my flight?",
            "Can I switch to a different flight?",
            "Can I move my flight to another date?",
            "What is the flight change fee?",
        ],
    },
    # ── 3. Cancellation ───────────────────────────────────────────────────────
    {
        "id": "cancellation_policy",
        "text": (
            "Cancellation Policy: Cancellations made more than 7 days before departure receive a full "
            "refund minus a $75 processing fee. Cancellations within 7 days of departure are eligible "
            "for 50% travel credit only. No-shows forfeit the full ticket value. To cancel, call our "
            "customer support line or visit our website."
        ),
        "seed_questions": [
            "What happens if I cancel my flight?",
            "What is the cancellation policy?",
            "How do I cancel my booking?",
            "What is the cancellation fee?",
            "Will I be charged for cancelling?",
            "What happens if I don't show up?",
            "Can I cancel my flight last minute?",
            "Do I get a travel credit if I cancel?",
            "How much is the cancellation penalty?",
            "What is a no-show policy?",
        ],
    },
    # ── 4. Baggage ────────────────────────────────────────────────────────────
    {
        "id": "baggage_policy",
        "text": (
            "Baggage Policy: Each passenger may bring one personal item (fits under seat) and one "
            "carry-on bag (max 22x14x9 inches, 25 lbs) free of charge. Checked bags cost $35 for "
            "the first bag, $45 for the second, and $75 for each additional bag. Oversized bags "
            "(over 62 linear inches) incur a $100 fee. Sports equipment fees vary — contact us in advance."
        ),
        "seed_questions": [
            "How many bags can I bring?",
            "What is the baggage allowance?",
            "How much does checked baggage cost?",
            "What is the carry-on size limit?",
            "Can I bring an extra bag?",
            "What is the baggage fee?",
            "How heavy can my bag be?",
            "What are the luggage restrictions?",
            "Is carry-on luggage free?",
            "What if my bag is oversized?",
            "What is the weight limit for checked bags?",
            "How much does a second checked bag cost?",
        ],
    },
    # ── 5. Seat selection ─────────────────────────────────────────────────────
    {
        "id": "seat_policy",
        "text": (
            "Seat Selection Policy: Standard seat selection is free during booking. Preferred seats "
            "(extra legroom, bulkhead, exit rows) cost $25-$50 extra. Seat upgrades to business class "
            "are available at check-in for $150-$300 depending on route length. Seats can be changed "
            "online up to 24 hours before departure at no charge."
        ),
        "seed_questions": [
            "Can I choose my seat?",
            "How much does seat selection cost?",
            "Is seat selection free?",
            "How do I upgrade my seat?",
            "How much is a business class upgrade?",
            "Can I get extra legroom?",
            "Can I change my seat after booking?",
            "What are the seat upgrade options?",
            "How much does an exit row seat cost?",
            "Can I pick a window seat?",
        ],
    },
    # ── 6. Check-in ───────────────────────────────────────────────────────────
    {
        "id": "checkin_policy",
        "text": (
            "Check-in Policy: Online check-in opens 24 hours before departure and closes 1 hour before. "
            "Airport check-in closes 45 minutes before domestic flights and 60 minutes before international. "
            "Passengers arriving after check-in closes may forfeit their seat without refund."
        ),
        "seed_questions": [
            "When does check-in open?",
            "How early do I need to check in?",
            "What is the check-in deadline?",
            "Can I check in online?",
            "When does online check-in close?",
            "How early should I arrive at the airport?",
            "What time does check-in close?",
            "What happens if I miss check-in?",
            "How do I check in for my flight?",
            "When can I get my boarding pass?",
        ],
    },
    # ── 7. Pets ───────────────────────────────────────────────────────────────
    {
        "id": "pet_policy",
        "text": (
            "Pet Policy: Small pets (dog or cat, under 20 lbs including carrier) may travel in the cabin "
            "for a $95 fee each way. The pet must be in an approved soft-sided carrier (max 18x11x11 inches) "
            "that fits under the seat. The carrier counts as your carry-on. Maximum 2 pets per flight. "
            "Larger pets must travel as checked baggage ($150 each way). Pets are not allowed on "
            "international flights longer than 8 hours. Certified service animals travel free with "
            "proper documentation submitted 48 hours in advance."
        ),
        "seed_questions": [
            "Can I bring my dog on the plane?",
            "Can I travel with my cat?",
            "What is the pet policy?",
            "How much does it cost to bring a pet?",
            "Can I bring my pet in the cabin?",
            "What size carrier do I need for my pet?",
            "Are service animals allowed?",
            "Can pets travel in the hold?",
            "How do I book a flight with my pet?",
            "Is there a fee for bringing a pet?",
            "Can I fly internationally with my pet?",
        ],
    },
    # ── 8. Special assistance ─────────────────────────────────────────────────
    {
        "id": "special_assistance_policy",
        "text": (
            "Special Assistance Policy: Wheelchair assistance is available at all airports — request "
            "at least 48 hours before departure. Passengers needing extra boarding time may pre-board. "
            "Medical oxygen is available on request for a $75 fee — notify us at least 72 hours in advance. "
            "Passengers with hearing or vision impairments should notify us at booking. Emotional support "
            "animals travel free with valid documentation. We comply with all disability access regulations."
        ),
        "seed_questions": [
            "Can I get wheelchair assistance?",
            "Do you offer help for disabled passengers?",
            "What special assistance is available?",
            "Can I bring medical equipment on the plane?",
            "Is oxygen available on the flight?",
            "Can I pre-board if I need extra time?",
            "Do you accommodate passengers with disabilities?",
            "Are emotional support animals allowed?",
            "How do I request special assistance?",
            "Do you help passengers who need mobility aid?",
        ],
    },
    # ── 9. Infants and children ───────────────────────────────────────────────
    {
        "id": "infant_child_policy",
        "text": (
            "Infant and Child Policy: Infants under 2 years may travel as lap infants free of charge "
            "on domestic flights, or for $75 on international routes. Children aged 2 and above require "
            "their own seat at the standard fare. Unaccompanied minors aged 5-14 may travel alone with "
            "a $100 service fee each way — our staff escort them at departure and arrival. Children under "
            "5 cannot travel unaccompanied. Car seats and strollers are checked free of charge."
        ),
        "seed_questions": [
            "Can I bring my baby on the flight?",
            "How do infants travel on your airline?",
            "Is there a fee for an infant?",
            "Can my child fly alone?",
            "What is the unaccompanied minor policy?",
            "How old does a child need to be to fly alone?",
            "Can I bring a stroller?",
            "What is the lap infant policy?",
            "Do children need their own seat?",
            "Can I check my car seat for free?",
        ],
    },
    # ── 10. Loyalty / rewards ─────────────────────────────────────────────────
    {
        "id": "loyalty_policy",
        "text": (
            "Phoenix Air Rewards: Earn 1 mile for every $1 spent on Phoenix Air flights. "
            "Status tiers — Silver: 25,000 miles/year (priority boarding, one free checked bag); "
            "Gold: 50,000 miles/year (lounge access, two free checked bags); "
            "Platinum: 100,000 miles/year (all Gold benefits + complimentary upgrades). "
            "Miles expire after 18 months of account inactivity. "
            "Redeem at 15,000 miles = $150 flight credit. Miles can also transfer to hotel partners."
        ),
        "seed_questions": [
            "Do you have a loyalty program?",
            "How do I earn miles?",
            "How do Phoenix Air Rewards work?",
            "How many miles do I need for a free flight?",
            "What are the membership tiers?",
            "Do my miles expire?",
            "How do I redeem my miles?",
            "Can I transfer my miles?",
            "What benefits do I get as a Gold member?",
            "How do I sign up for the rewards program?",
        ],
    },
    # ── 11. Delayed / cancelled flights ───────────────────────────────────────
    {
        "id": "delay_cancellation_policy",
        "text": (
            "Delay and Cancellation Policy (airline-caused): If your flight is delayed more than 3 hours, "
            "we provide a $15 meal voucher. Delays over 6 hours include hotel accommodation if an overnight "
            "stay is required. If Phoenix Air cancels your flight, you receive a full refund or free rebooking "
            "on the next available flight. Compensation travel credit of $50-$200 is issued for significant "
            "delays caused by us. Weather-related delays or cancellations do not qualify for compensation "
            "but we offer free rebooking."
        ),
        "seed_questions": [
            "What happens if my flight is delayed?",
            "What if Phoenix Air cancels my flight?",
            "Do I get compensation for a delayed flight?",
            "What is the delay compensation policy?",
            "Will I get a hotel if my flight is cancelled?",
            "What happens if my flight is cancelled due to weather?",
            "Am I entitled to a refund if the airline cancels?",
            "How do I rebook a cancelled flight?",
            "What do I get if my flight is delayed by 6 hours?",
            "Will the airline pay for my hotel if there is a delay?",
        ],
    },
    # ── 12. Lost / damaged baggage ────────────────────────────────────────────
    {
        "id": "lost_baggage_policy",
        "text": (
            "Lost and Damaged Baggage Policy: Report lost or damaged baggage immediately at the baggage "
            "claim desk before leaving the airport. Phoenix Air searches for missing bags for up to 5 days. "
            "If not found, compensation is up to $3,500 for domestic flights and $1,800 for international. "
            "For delayed baggage, essential items are reimbursed up to $100 per day for a maximum of 3 days. "
            "Claims must be filed within 24 hours for domestic flights and within 7 days for international."
        ),
        "seed_questions": [
            "My bag is lost, what do I do?",
            "What happens if my luggage is lost?",
            "How do I report lost baggage?",
            "What is the compensation for lost baggage?",
            "My bag was damaged, what should I do?",
            "How long does it take to find a lost bag?",
            "What if my bag is delayed?",
            "Can I get reimbursed for essentials if my bag is delayed?",
            "How do I file a lost baggage claim?",
            "What is the deadline to report lost luggage?",
        ],
    },
    # ── 13. In-flight meals ───────────────────────────────────────────────────
    {
        "id": "meal_policy",
        "text": (
            "In-Flight Meal Policy: Complimentary non-alcoholic beverages are served on all flights. "
            "Snacks are provided on flights over 90 minutes. Hot meals are served on flights over 4 hours — "
            "complimentary in business class, available for purchase ($12-$18) in economy. "
            "Special dietary meals (vegetarian, vegan, kosher, halal, gluten-free) are available at no "
            "extra charge — request at least 48 hours before departure. Alcoholic beverages cost $8-$12 "
            "and are available on flights over 2 hours."
        ),
        "seed_questions": [
            "Is there food on the flight?",
            "What meals are served on board?",
            "Do you serve vegetarian meals?",
            "Can I get a kosher meal?",
            "Is food free on Phoenix Air?",
            "Are there snacks on the flight?",
            "Can I get a halal meal?",
            "How do I request a special meal?",
            "Do you serve alcohol on the plane?",
            "Is there a meal on short flights?",
        ],
    },
    # ── 14. Payment ───────────────────────────────────────────────────────────
    {
        "id": "payment_policy",
        "text": (
            "Payment Policy: Phoenix Air accepts Visa, Mastercard, American Express, and Discover cards. "
            "Digital wallets accepted: Apple Pay, Google Pay, and PayPal. Cash payments are accepted only "
            "at airport ticket counters — not online. Payment plans are available for bookings over $500 "
            "(split into 3 monthly installments, interest-free). All online transactions are in USD. "
            "International credit cards are accepted. Booking fees are non-refundable."
        ),
        "seed_questions": [
            "What payment methods do you accept?",
            "Can I pay with PayPal?",
            "Can I pay with Apple Pay?",
            "Do you accept credit cards?",
            "Can I pay in installments?",
            "Is there a payment plan available?",
            "Can I pay cash at the airport?",
            "Do you accept international cards?",
            "What currencies do you accept?",
            "Can I split my payment?",
        ],
    },
    # ── 15. Name correction ────────────────────────────────────────────────────
    {
        "id": "name_correction_policy",
        "text": (
            "Name Correction Policy: Minor spelling corrections (up to 3 characters) are free and can be "
            "done online via Manage Booking or by calling our support team. Legal name changes due to "
            "marriage or divorce require supporting documentation and incur a $25 administrative fee. "
            "The name on your ticket must exactly match your government-issued ID or passport. "
            "Name changes are not permitted after check-in closes. "
            "Full ticket transfers to a different passenger are not allowed."
        ),
        "seed_questions": [
            "How do I correct a spelling mistake on my ticket?",
            "Can I change the name on my booking?",
            "I typed my name wrong — how do I fix it?",
            "What is the name correction policy?",
            "Is there a fee to correct my name?",
            "Can I change my name after booking?",
            "How do I fix a wrong name on my flight?",
            "Can I transfer my ticket to someone else?",
            "My name is misspelled on my boarding pass — what do I do?",
            "How much does a name change cost?",
        ],
    },
    # ── 16. Overbooking ───────────────────────────────────────────────────────
    {
        "id": "overbooking_policy",
        "text": (
            "Overbooking and Denied Boarding Policy: If a flight is overbooked, Phoenix Air first asks for "
            "volunteers to give up their seats in exchange for compensation — a travel voucher up to $600 or "
            "cash equivalent. Volunteers also receive a seat on the next available flight at no charge. "
            "If not enough volunteers come forward, passengers may be involuntarily denied boarding and are "
            "entitled to a full refund plus cash compensation: $300 for delays under 4 hours, $600 for delays "
            "over 4 hours (domestic); $650-$1,350 for international flights. "
            "Priority protection: Phoenix Air Gold and Platinum members, passengers with disabilities, and "
            "unaccompanied minors are the last to be involuntarily bumped."
        ),
        "seed_questions": [
            "What happens if my flight is overbooked?",
            "Can I be bumped from my flight?",
            "What is denied boarding compensation?",
            "Will the airline pay me if I am bumped?",
            "What is overbooking?",
            "Can I volunteer to give up my seat?",
            "What do I get if I give up my seat?",
            "Am I protected against overbooking?",
            "What happens if the airline bumps me involuntarily?",
            "How much compensation for being denied boarding?",
        ],
    },
    # ── 17. Medical travel ────────────────────────────────────────────────────
    {
        "id": "medical_travel_policy",
        "text": (
            "Medical and Health Travel Policy: Pregnant passengers are welcome to fly up to 36 weeks for "
            "single pregnancies and 32 weeks for multiple pregnancies — a doctor's certificate is required "
            "after 28 weeks. Passengers with serious medical conditions should carry a doctor's letter. "
            "FAA-approved portable oxygen concentrators (POCs) are permitted on board with prior approval. "
            "CPAP and BiPAP machines are allowed as carry-on medical devices at no additional charge. "
            "Passengers in rigid plaster casts may require additional seating on long-haul flights. "
            "We recommend consulting your doctor before flying after recent surgery, serious illness, "
            "heart attack, stroke, or if you are at risk for deep vein thrombosis (DVT). "
            "Pacemaker and implant patients: notify the gate agent and request a manual pat-down instead "
            "of the full-body scanner."
        ),
        "seed_questions": [
            "Can I fly while pregnant?",
            "How many weeks pregnant can I fly?",
            "What is the policy for pregnant passengers?",
            "Do I need a doctor's note to fly pregnant?",
            "Can I bring my CPAP machine on the plane?",
            "Can I fly after surgery?",
            "What are the rules for flying with a pacemaker?",
            "Can I bring my portable oxygen concentrator?",
            "Are there restrictions for passengers with medical conditions?",
            "Can I fly with a broken leg in a cast?",
            "Is flying safe if I have a heart condition?",
            "Can I bring medical equipment as carry-on?",
        ],
    },
    # ── 18. Prohibited items ──────────────────────────────────────────────────
    {
        "id": "prohibited_items_policy",
        "text": (
            "Prohibited and Restricted Items Policy: The following are never allowed on any flight: "
            "explosives, flammable liquids (petrol, lighter fluid in large quantities), compressed gases, "
            "radioactive materials, and corrosive substances. "
            "Carry-on prohibited items include: firearms, knives or sharp objects over 4 inches, sporting "
            "bats, and large tools. "
            "Checked bags: firearms must be declared, unloaded, in a locked hard-sided case. "
            "Lithium-ion batteries over 100Wh must travel in carry-on, not checked luggage. "
            "E-cigarettes and vapes must be in carry-on only (not checked). "
            "Hoverboards and e-bikes with large lithium batteries are not accepted on any flight. "
            "Dry ice (max 5.5 lbs) is allowed for packaging perishables with prior airline approval. "
            "Always check TSA.gov or contact us for the full current list before packing."
        ),
        "seed_questions": [
            "What items are not allowed on the plane?",
            "Can I bring a knife in my carry-on?",
            "Can I pack a firearm in checked luggage?",
            "Are lithium batteries allowed on planes?",
            "What is prohibited in my carry-on bag?",
            "Can I bring my e-cigarette on the plane?",
            "Are hoverboards allowed on flights?",
            "What dangerous goods are banned from flights?",
            "Can I bring aerosol cans on the plane?",
            "What are the rules for carrying a gun on a flight?",
            "Can I bring tools in my checked bag?",
            "Are power banks allowed in checked luggage?",
        ],
    },
    # ── 19. Upgrades ──────────────────────────────────────────────────────────
    {
        "id": "upgrade_policy",
        "text": (
            "Upgrade Policy: Economy passengers can bid for a business class upgrade through the Phoenix Air "
            "Upgrade Bid system — bids start at $80 and are submitted up to 24 hours before departure. "
            "Economy Plus (extra legroom seats) upgrades cost $30-$75 depending on route and are available "
            "during booking or at check-in. Phoenix Air Platinum members receive complimentary upgrades "
            "based on availability at check-in. Gold members may use 15,000 miles for an upgrade. "
            "Upgrades cannot be purchased with cash at the gate after check-in closes. "
            "Business class on international routes over 6 hours includes lie-flat seats, premium meals, "
            "priority boarding, and lounge access."
        ),
        "seed_questions": [
            "How do I upgrade to business class?",
            "Can I bid for a business class seat?",
            "How much is a business class upgrade?",
            "Can I use miles to upgrade my seat?",
            "How does the upgrade bid system work?",
            "Is there economy plus seating?",
            "How do I get extra legroom?",
            "Are complimentary upgrades available?",
            "What is included in business class?",
            "Can I upgrade at the airport?",
        ],
    },
    # ── 20. Group bookings ────────────────────────────────────────────────────
    {
        "id": "group_booking_policy",
        "text": (
            "Group Booking Policy: Groups of 10 or more passengers qualify for group fares — typically "
            "5-15% below standard fares. Groups receive flexible name-change windows (names may be "
            "submitted up to 30 days before departure), one dedicated group coordinator, and priority "
            "check-in assistance. A non-refundable deposit is required at booking; full payment is due "
            "60 days before departure. Seat assignments are arranged together where possible. "
            "Groups of 20 or more may qualify for complimentary group boarding priority. "
            "Contact our group desk: groups@phoenixair.com or call 1-800-749-2888 (Mon-Fri 9am-6pm)."
        ),
        "seed_questions": [
            "Can I book flights for a group?",
            "How do I arrange a group booking?",
            "What is the group discount?",
            "How many people qualify for a group fare?",
            "Is there a corporate travel program?",
            "Can I book 20 seats at once?",
            "How do group fares work?",
            "Do groups get priority boarding?",
            "What is the group deposit policy?",
            "Who do I contact for group bookings?",
        ],
    },
]

# ── Keyword pre-filter: maps keyword sets → policy IDs (L1 fast path) ─────────
# ORDER MATTERS — more specific phrases must come before generic ones.
# e.g. "lost bag" before standalone "bag", "airline cancelled" before "cancel".
_KEYWORD_MAP = [
    # ── Specific lost/damaged baggage BEFORE general baggage ──────────
    (["lost bag", "bag was lost", "bag is lost", "bag got lost",
      "missing bag", "bag missing", "damaged bag", "damaged luggage",
      "lost luggage", "luggage lost", "baggage claim",
      "delayed baggage", "bag not arrived"],                        "lost_baggage_policy"),
    # ── Airline-caused delays/cancellations BEFORE passenger cancel ───
    (["airline cancelled", "airline canceled",
      "cancelled by the airline", "canceled by the airline",
      "flight was cancelled", "flight got cancelled",
      "flight delayed", "flight is delayed",
      "delay compensation", "cancelled due to weather",
      "weather cancellation"],                                       "delay_cancellation_policy"),
    # ── Passenger-initiated cancellation ─────────────────────────────
    (["cancel my flight", "cancel my booking", "cancel my ticket",
      "cancellation fee", "cancellation policy",
      "no-show", "no show", "if i cancel"],                         "cancellation_policy"),
    # ── Refund ───────────────────────────────────────────────────────
    (["refund", "money back", "reimburs",
      "get my money back", "want my money"],                        "refund_policy"),
    # ── Flight change ─────────────────────────────────────────────────
    (["change flight", "change my flight", "reschedule",
      "modify flight", "move my flight", "switch my flight",
      "different flight date", "change the date"],                  "change_policy"),
    # ── General baggage (no lost/missing) ────────────────────────────
    (["baggage allowance", "baggage fee", "baggage limit",
      "luggage allowance", "luggage fee", "luggage limit",
      "carry on", "carry-on", "carryon",
      "checked bag", "check my bag", "extra bag",
      "oversize bag", "overweight bag", "bag fee",
      "bag weight", "bag size", "sports equipment"],                "baggage_policy"),
    # ── Seat ─────────────────────────────────────────────────────────
    (["seat", "legroom", "exit row", "bulkhead",
      "seat upgrade", "pick my seat", "choose my seat",
      "window seat", "aisle seat"],                                 "seat_policy"),
    # ── Check-in ─────────────────────────────────────────────────────
    (["check in", "check-in", "checkin", "boarding pass",
      "how early should i arrive", "when to arrive at airport",
      "gate closes", "check in deadline"],                          "checkin_policy"),
    # ── Pets ─────────────────────────────────────────────────────────
    (["pet", "dog", "cat", "animal", "service animal",
      "bring my pet", "fly with pet", "travel with pet"],           "pet_policy"),
    # ── Special assistance ────────────────────────────────────────────
    (["wheelchair", "disability", "disabled", "special assistance",
      "medical equipment", "oxygen on flight", "pre-board",
      "mobility aid"],                                              "special_assistance_policy"),
    # ── Infants / children ────────────────────────────────────────────
    (["infant", "baby", "lap infant", "child fly alone",
      "unaccompanied minor", "stroller", "car seat",
      "travelling with baby", "baby on the flight"],                "infant_child_policy"),
    # ── Loyalty ──────────────────────────────────────────────────────
    (["miles", "rewards", "loyalty", "frequent flyer",
      "phoenix miles", "earn miles", "redeem miles",
      "points program"],                                            "loyalty_policy"),
    # ── Meals ────────────────────────────────────────────────────────
    (["meal on flight", "food on the flight", "special meal",
      "kosher", "halal", "vegetarian meal", "vegan meal",
      "in-flight food", "inflight food"],                           "meal_policy"),
    # ── Payment ──────────────────────────────────────────────────────
    (["payment method", "pay with paypal", "paypal",
      "apple pay", "google pay", "pay with apple",
      "installment", "payment plan", "pay in cash",
      "do you accept credit", "accepted cards"],                    "payment_policy"),
    # ── Name correction ───────────────────────────────────────────────────────
    (["name change", "name correction", "wrong name", "typo in name",
      "correct my name", "fix my name", "spelling mistake on ticket",
      "misspelled name", "change name on ticket", "name on my booking"],
                                                                     "name_correction_policy"),
    # ── Overbooking / denied boarding ─────────────────────────────────────────
    (["overbooked", "overbooking", "bumped from flight", "denied boarding",
      "volunteer to give up", "give up my seat", "involuntary denied",
      "am i protected", "bumped off"],                               "overbooking_policy"),
    # ── Medical / pregnancy ──────────────────────────────────────────────────
    (["pregnant", "pregnancy", "flying pregnant", "weeks pregnant",
      "pacemaker", "cpap", "cpap machine", "medical condition",
      "fly after surgery", "broken leg", "cast on flight",
      "doctor note to fly", "portable oxygen concentrator",
      "medical equipment on plane", "flying with a medical"],        "medical_travel_policy"),
    # ── Prohibited items ──────────────────────────────────────────────────────
    (["prohibited item", "forbidden item", "not allowed on plane",
      "can i bring a knife", "firearms on flight", "gun in luggage",
      "gun in checked", "lithium battery", "hoverboard on plane",
      "e-cigarette on plane", "vape on plane", "dangerous goods",
      "what can i not bring"],                                       "prohibited_items_policy"),
    # ── Upgrades ─────────────────────────────────────────────────────────────
    (["upgrade to business", "business class upgrade", "bid for upgrade",
      "upgrade bid", "economy plus", "upgrade with miles",
      "complimentary upgrade", "how to upgrade my seat"],            "upgrade_policy"),
    # ── Group bookings ────────────────────────────────────────────────────────
    (["group booking", "group travel", "group flight", "group discount",
      "group fare", "book for a group", "10 passengers", "20 passengers",
      "corporate booking", "corporate travel", "book seats for a group"],
                                                                     "group_booking_policy"),
]

# Pre-authored short answers — returned directly at L1, no LLM
_POLICY_SHORT_ANSWERS = {
    "refund_policy": (
        "Tickets are fully refundable within 24 hours of purchase. "
        "After 24 hours a $75 fee applies. Non-refundable tickets become 12-month travel credit. "
        "Refunds take 7-10 business days."
    ),
    "cancellation_policy": (
        "Cancellations 7+ days before departure get a full refund minus a $75 fee. "
        "Within 7 days: 50% travel credit only. No-shows forfeit the full ticket value."
    ),
    "change_policy": (
        "You can change your flight up to 2 hours before departure. "
        "Changes 7+ days ahead are free. Within 2-7 days: $35 fee. Same-day: $50. "
        "Fare differences apply if the new flight costs more."
    ),
    "baggage_policy": (
        "One personal item and one carry-on (max 22x14x9 in, 25 lbs) are free. "
        "Checked bags: $35 first, $45 second, $75 each additional. Oversized bags: $100 extra."
    ),
    "seat_policy": (
        "Standard seats are free to select. Preferred seats (extra legroom, exit rows): $25-$50 extra. "
        "Business class upgrades at check-in: $150-$300 depending on route."
    ),
    "checkin_policy": (
        "Online check-in opens 24 hours before and closes 1 hour before departure. "
        "Airport check-in closes 45 min before domestic, 60 min before international flights."
    ),
    "pet_policy": (
        "Small pets (under 20 lbs with carrier) travel in cabin for $95 each way. "
        "Carrier max size: 18x11x11 inches. Max 2 pets per flight. "
        "Larger pets travel as checked baggage for $150 each way. Service animals fly free."
    ),
    "special_assistance_policy": (
        "Wheelchair assistance is available at all airports — request 48 hours in advance. "
        "Medical oxygen available for $75 (notify 72 hours ahead). "
        "Emotional support animals travel free with documentation."
    ),
    "infant_child_policy": (
        "Infants under 2 travel free on domestic flights as lap infants ($75 internationally). "
        "Unaccompanied minors aged 5-14: $100 service fee each way. "
        "Strollers and car seats are checked free of charge."
    ),
    "loyalty_policy": (
        "Earn 1 mile per $1 spent. Redeem at 15,000 miles = $150 flight credit. "
        "Tiers: Silver (25k miles), Gold (50k), Platinum (100k). Miles expire after 18 months of inactivity."
    ),
    "delay_cancellation_policy": (
        "Delays over 3 hours: $15 meal voucher. Over 6 hours: hotel if overnight stay needed. "
        "Airline-cancelled flights: full refund or free rebooking. Weather delays: free rebooking only."
    ),
    "lost_baggage_policy": (
        "Report lost bags at baggage claim before leaving the airport. "
        "Compensation: up to $3,500 domestic, $1,800 international. "
        "Delayed baggage: essentials reimbursed up to $100/day for 3 days."
    ),
    "meal_policy": (
        "Complimentary snacks and drinks on flights over 90 minutes. "
        "Hot meals on flights over 4 hours (free in business, $12-$18 in economy). "
        "Special dietary meals available free — request 48 hours in advance."
    ),
    "payment_policy": (
        "Accepted: Visa, Mastercard, Amex, Discover, Apple Pay, Google Pay, PayPal. "
        "Cash accepted at airport counters only. "
        "Payment plans available for bookings over $500 (3 monthly installments, interest-free)."
    ),
    "name_correction_policy": (
        "Minor spelling corrections (up to 3 characters) are free via Manage Booking or by calling support. "
        "Legal name changes require documentation and a $25 fee. "
        "Your name must exactly match your government-issued ID."
    ),
    "overbooking_policy": (
        "Volunteers who give up their seat receive a travel voucher up to $600 plus a seat on the next flight. "
        "Involuntary denied boarding: $300-$600 domestic or $650-$1,350 international, plus a full refund."
    ),
    "medical_travel_policy": (
        "Pregnant passengers may fly up to 36 weeks (32 for multiples); doctor's certificate required after 28 weeks. "
        "CPAP machines and FAA-approved oxygen concentrators are allowed as carry-on medical devices. "
        "Consult your doctor before flying after surgery or with serious conditions."
    ),
    "prohibited_items_policy": (
        "Prohibited carry-on: firearms, knives over 4 inches, flammable liquids, explosives. "
        "Lithium batteries over 100Wh must be in carry-on only — not checked luggage. "
        "Hoverboards and e-bikes are not accepted. E-cigarettes must be in carry-on."
    ),
    "upgrade_policy": (
        "Bid for business class from $80 (submit up to 24 hours before departure). "
        "Economy Plus extra legroom: $30-$75 at booking or check-in. "
        "Platinum members get complimentary upgrades; Gold members can upgrade with 15,000 miles."
    ),
    "group_booking_policy": (
        "Groups of 10 or more receive 5-15% off standard fares, flexible name changes, and priority check-in. "
        "Full payment due 60 days before departure. "
        "Contact groups@phoenixair.com or call 1-800-749-2888."
    ),
}

# Rule-based answers for common general airline questions (not Phoenix Air-specific)
# ORDER MATTERS — more specific phrases first.
_GENERAL_RULES = [
    (["wifi", "wi-fi", "internet on the flight", "internet on board", "online on the flight",
      "connect to the internet", "use internet on plane"],
     "Phoenix Air offers in-flight Wi-Fi on most routes. Connect via the onboard portal — pricing and availability vary by aircraft. Premium cabins often receive complimentary access."),
    (["passport", "visa", "travel document", "what id do i need", "identification required",
      "do i need a passport", "do i need a visa"],
     "Domestic flights require a government-issued photo ID (driver's licence or passport). International flights require a valid passport. Some destinations also require a visa — check your destination's requirements at least 6 weeks before travel."),
    (["security", "tsa", "what can i bring through", "liquids rule", "3-1-1", "security screening",
      "airport security", "security check"],
     "At security: remove shoes, belt, and jacket. Laptops go in a separate tray. Liquids must be 100 ml or less in a clear resealable bag (one bag per person). Sharp objects are not allowed in carry-on. TSA PreCheck members use dedicated fast lanes."),
    (["boarding process", "boarding order", "boarding group", "when does boarding start",
      "when does boarding begin", "how does boarding work"],
     "Boarding usually starts 30-45 minutes before departure. Priority boarding goes first (business class, loyalty elites, families with young children). General boarding follows in groups. Have your boarding pass and ID ready at the gate."),
    (["entertainment", "movies on the plane", "movies on the flight", "movies on board",
      "movie on board", "do you have movies", "tv on the plane",
      "seatback screen", "in-flight entertainment", "watch a movie", "what to watch"],
     "Phoenix Air provides personal seatback screens with movies, TV shows, music, and games on flights over 4 hours. Shorter flights may have overhead screens or a streaming app for personal devices. Headphones are available for purchase if you don't bring your own."),
    (["connecting flight", "layover", "transit", "stopover", "how long for connection",
      "minimum connection time", "miss my connection"],
     "Allow at least 60 minutes for domestic connections and 90-120 minutes for international. On a single ticket your bags transfer automatically. If we cause you to miss a connection, we rebook you at no charge. Pack essentials in your carry-on in case of baggage delays."),
    (["track my flight", "flight status", "where is my flight", "is my flight on time",
      "flight tracker", "check flight status", "flight delayed notification"],
     "Track your flight on the Phoenix Air website or app, or via FlightAware / Flightradar24. We send SMS and email updates if you provided contact details at booking. Check airport departure boards for real-time gate information."),
    (["how early should i get to the airport", "how early to arrive", "when to arrive at the airport",
      "how much time before my flight", "2 hours enough", "3 hours enough", "arrive early"],
     "For domestic flights, arrive at least 2 hours before departure. For international flights, allow 3 hours. During peak travel periods add 30-60 extra minutes. If you only have a carry-on and check in online, 90 minutes may be sufficient for domestic."),
    (["cheapest time to book", "best time to buy a flight", "how far in advance should i book",
      "cheapest day to fly", "when to book", "best deal on flights", "save money on flights"],
     "Book domestic flights 4-6 weeks in advance and international flights 2-4 months ahead for the best fares. Mid-week departures (Tuesday-Wednesday) and early morning flights are often cheaper. Sign up for Phoenix Air fare alerts to catch sales."),
    (["trip insurance", "travel insurance", "flight insurance"],
     "Travel insurance covers trip cancellation, medical emergencies, lost luggage, and more. Phoenix Air offers optional travel insurance at checkout. Compare plans at booking to find one that fits your needs."),
    (["jet lag", "time zone", "circadian", "adjust after flying",
      "sleep after flight", "body clock after travel"],
     "Jet lag occurs when you cross multiple time zones faster than your body can adjust. Symptoms (fatigue, poor sleep, difficulty concentrating) typically last about 1 day per time zone crossed. To reduce it: gradually shift your sleep schedule before travel, seek natural light at your destination, stay well-hydrated, and consider melatonin for eastward journeys."),
    (["turbulence", "plane shaking", "rough flight", "bumpy flight",
      "scared of turbulence", "turbulent flight"],
     "Turbulence is normal atmospheric air movement and modern aircraft are engineered to handle it safely — even severe turbulence will not bring down a structurally sound plane. The best safety measure is to keep your seatbelt fastened whenever you are seated. Seats over the wings experience less motion. Deep breathing can help manage anxiety during bumpy patches."),
    (["dim the lights", "lights off during", "cabin lights during",
      "why do they turn off lights", "lights during takeoff",
      "lights during landing", "why lights dimmed"],
     "Airlines dim cabin lights during takeoff and landing so passengers' eyes adjust to outside light levels. This ensures everyone can see clearly in the unlikely event of an emergency evacuation — eyes already adjusted to low light can immediately navigate a dark cabin or exterior."),
    (["defibrillator", "aed", "cardiac arrest on plane",
      "heart attack on plane", "medical emergency on flight",
      "do planes have doctors"],
     "All commercial aircraft are required to carry an Automated External Defibrillator (AED). Flight attendants are trained in CPR and basic first aid. In a medical emergency the crew will ask if any passengers are medical professionals. For serious emergencies, the captain may divert to the nearest suitable airport."),
    (["ear pressure", "ear pain on plane", "popping ears",
      "ears hurt when flying", "blocked ears", "ear barotrauma"],
     "Ear discomfort during takeoff and landing is caused by changing cabin pressure. To relieve it: swallow, yawn, or chew gum. You can also try the Valsalva manoeuvre — pinch your nose and gently blow. A decongestant nasal spray an hour before landing helps if you are congested. Infants benefit from feeding or a dummy to encourage swallowing."),
    (["dvt", "blood clot", "deep vein thrombosis", "blood clots on flight",
      "swollen legs after flight", "economy class syndrome"],
     "Long-haul flights increase the risk of deep vein thrombosis (DVT). To reduce the risk: walk the aisle every 1-2 hours, do calf raises in your seat, stay well-hydrated, wear compression socks, and avoid crossing your legs for long periods. Those with a personal or family history of DVT should consult a doctor before flying long distances."),
    (["fear of flying", "flying anxiety", "nervous about flying",
      "scared to fly", "aerophobia"],
     "Fear of flying is very common — affecting roughly 25% of people. Helpful strategies include learning aviation safety statistics, focusing on crew behaviour as a normalcy cue, practising breathing exercises, or taking a dedicated fear-of-flying course. Sitting over the wings reduces felt turbulence. Speak to your GP about short-term anxiety relief if needed."),
    (["oxygen mask", "cabin pressure drops", "depressurisation",
      "emergency oxygen", "altitude on plane"],
     "If cabin pressure drops, oxygen masks fall from overhead panels automatically — place yours over your nose and mouth before assisting others. Aircraft cabins are pressurised to the equivalent of roughly 6,000-8,000 feet altitude, which is far lower than the actual cruising altitude and comfortable for most passengers."),
]

# ── General knowledge documents (for embedding retrieval — L2 general path) ───

GENERAL_DOCS = [
    {
        "id": "entertainment_doc",
        "text": (
            "In-Flight Entertainment: Phoenix Air provides personal seatback screens on flights over 4 hours "
            "with movies, TV shows, music, and games. Shorter flights have overhead screens or a streaming app "
            "for personal devices (download before boarding). Headphones are sold on board for $5 or bring your own. "
            "In-flight Wi-Fi is available on most routes — connect via the onboard portal (fee applies, free in business class). "
            "USB charging ports and AC power outlets are available at most seats on newer aircraft."
        ),
        "seed_questions": [
            "Is there entertainment on the flight?",
            "Do you have movies on the plane?",
            "Is there a screen on my seat?",
            "Can I watch TV on the flight?",
            "Do you have in-flight entertainment?",
            "What is available to watch during the flight?",
            "Is there Wi-Fi on the plane?",
            "Can I use the internet during the flight?",
            "Do planes have seatback screens?",
            "Are there charging ports on the plane?",
            "Can I stream movies on my phone?",
            "Do I need to bring headphones?",
        ],
    },
    {
        "id": "security_doc",
        "text": (
            "Airport Security: At the security checkpoint, remove shoes, belt, and jacket and place in a tray. "
            "Laptops and tablets must come out of bags separately. Liquids, gels, and aerosols must be 100 ml "
            "or less per container, all fitting in one clear resealable bag (the 3-1-1 rule). "
            "Sharp objects, firearms, and most tools are prohibited in carry-on — pack them in checked luggage. "
            "TSA PreCheck and Global Entry holders use dedicated fast lanes and generally keep shoes and laptops in bags. "
            "Pacemakers and implants: notify the officer and request a manual pat-down instead of the scanner."
        ),
        "seed_questions": [
            "What are the airport security rules?",
            "What do I need to do at security?",
            "Can I bring liquids on the plane?",
            "What is the 3-1-1 rule for liquids?",
            "Can I bring my laptop in my carry-on?",
            "What items are not allowed in carry-on luggage?",
            "How early should I arrive for security?",
            "What is TSA PreCheck?",
            "Do I need to take my shoes off at security?",
            "What is not allowed through airport security?",
            "Can I bring a water bottle through security?",
            "Are power banks allowed on planes?",
        ],
    },
    {
        "id": "boarding_doc",
        "text": (
            "Boarding Process: Boarding begins 30-45 minutes before scheduled departure. "
            "Priority boarding: business and first class, Phoenix Air Platinum and Gold members, "
            "passengers needing extra time, and families with children under 5. "
            "General boarding follows in zones printed on your boarding pass, typically back-to-front. "
            "The boarding gate closes 15 minutes before departure — arrive at the gate on time. "
            "Mobile boarding passes are accepted; have your boarding pass and photo ID ready. "
            "If you miss boarding, speak to a gate agent immediately."
        ),
        "seed_questions": [
            "How does boarding work?",
            "What is the boarding order?",
            "When does boarding start?",
            "How early does boarding begin?",
            "What is priority boarding?",
            "When does the gate close?",
            "What do I need to board the plane?",
            "Do I need a printed boarding pass?",
            "Can I use a mobile boarding pass?",
            "When should I be at the gate?",
            "What if I arrive late to the gate?",
            "How do boarding groups work?",
        ],
    },
    {
        "id": "connecting_flights_doc",
        "text": (
            "Connecting Flights and Layovers: Allow at least 60 minutes for domestic connections and "
            "90-120 minutes for international connections. On a single ticket, checked baggage is transferred "
            "automatically — you do not need to reclaim it between connections. On separate tickets, "
            "you must collect and recheck bags. During a long layover (6+ hours) you may leave the airport "
            "if visa rules allow. Missed connections caused by Phoenix Air are rebooked at no charge on the "
            "next available flight. Pack essentials (medication, charger, change of clothes) in your carry-on "
            "in case of baggage delays."
        ),
        "seed_questions": [
            "What happens during a layover?",
            "How long do I need for a connecting flight?",
            "Will my bags be transferred automatically?",
            "What should I do during a long layover?",
            "What if I miss my connecting flight?",
            "How do I find my connecting gate?",
            "Can I leave the airport during a layover?",
            "How much time do I need between connecting flights?",
            "Do I need to recheck my bags on a connecting flight?",
            "What happens if the first flight is late and I miss my connection?",
            "Is 1 hour enough for a connection?",
            "What is a minimum connection time?",
        ],
    },
    {
        "id": "flight_status_doc",
        "text": (
            "Flight Tracking and Status: Track any Phoenix Air flight in real time on our website or mobile app "
            "under 'Flight Status'. Third-party trackers such as FlightAware and Flightradar24 also provide "
            "live aircraft positions and estimated arrival times. "
            "If you provided your mobile number or email at booking, we send automatic delay notifications. "
            "Gate information and departure board updates are displayed throughout the airport. "
            "For significant delays, check the Phoenix Air app for rebooking options or speak to a gate agent."
        ),
        "seed_questions": [
            "How do I track my flight?",
            "Where is my flight right now?",
            "How do I check flight status?",
            "Is my flight on time?",
            "Can I get real-time flight updates?",
            "How do I know if my flight is delayed?",
            "Where can I see live flight information?",
            "How do I find out my gate number?",
            "Will I get a notification if my flight is delayed?",
            "How do I check if a flight is on schedule?",
        ],
    },
    {
        "id": "international_travel_doc",
        "text": (
            "International Travel Requirements: A valid passport is required for all international flights. "
            "Ensure your passport is valid for at least 6 months beyond your travel dates — many countries require this. "
            "Some destinations require a visa — apply well in advance at the destination country's embassy or consulate. "
            "Many countries offer e-Visa or visa-on-arrival options. "
            "Arrive at least 3 hours before an international departure. "
            "Upon arrival you will pass through immigration (passport control) and customs — declare any goods required. "
            "Check entry requirements for your specific destination at iata.org/timatic or your government's travel advisory."
        ),
        "seed_questions": [
            "What documents do I need for international travel?",
            "Do I need a visa?",
            "How early should I arrive for an international flight?",
            "What happens at customs?",
            "What is passport control?",
            "Do I need travel insurance for international flights?",
            "Is my passport valid enough for international travel?",
            "Do I need a visa for my destination?",
            "What is an eVisa?",
            "What should I declare at customs?",
            "How long before an international flight should I check in?",
            "Can I travel internationally without a visa?",
        ],
    },
    {
        "id": "booking_tips_doc",
        "text": (
            "Flight Booking Tips: Book domestic flights 4-6 weeks in advance and international flights "
            "2-4 months ahead for the best fares. Mid-week departures (Tuesday and Wednesday) and early-morning "
            "or late-night flights are typically cheaper than weekend or prime-time slots. "
            "Being flexible with your dates by 1-2 days can save significantly. "
            "Sign up for Phoenix Air fare alerts and newsletters to catch flash sales. "
            "Book directly on the Phoenix Air website for the best prices and flexibility on changes. "
            "Check whether the base fare includes baggage — budget fares often add fees for checked bags."
        ),
        "seed_questions": [
            "When is the cheapest time to book a flight?",
            "How far in advance should I book?",
            "How do I find the cheapest flights?",
            "What day is cheapest to fly?",
            "When are flights usually on sale?",
            "How do I get a good deal on flights?",
            "Should I book early or wait for a sale?",
            "What time of year is cheapest to fly?",
            "How do I save money on flights?",
            "Is it cheaper to book direct with the airline?",
            "How do fare alerts work?",
            "When do airlines release cheap seats?",
        ],
    },
    {
        "id": "airport_arrival_doc",
        "text": (
            "How Early to Arrive at the Airport: For domestic flights, plan to arrive at least 2 hours before "
            "departure. For international flights, arrive at least 3 hours before. "
            "If you have checked baggage, add an extra 20-30 minutes. "
            "During peak periods (holidays, summer weekends) add another 30-60 minutes. "
            "TSA PreCheck or expedited security members may need only 90 minutes for domestic. "
            "Factor in parking, shuttle, and check-in queue times. Arriving late can mean a missed flight — "
            "airlines close check-in counters and gates before the stated departure time."
        ),
        "seed_questions": [
            "How early should I get to the airport?",
            "When should I arrive at the airport?",
            "How much time before my flight should I arrive?",
            "Is 1 hour enough to get through the airport?",
            "How long does airport check-in take?",
            "Is 2 hours early enough for a flight?",
            "How much time do I need at the airport?",
            "Do I need to arrive 3 hours early?",
            "Is 90 minutes enough time at the airport?",
            "How early for an international flight?",
            "What time should I leave for the airport?",
            "Am I too early if I arrive 4 hours before?",
        ],
    },
    {
        "id": "health_on_flights_doc",
        "text": (
            "Health and Wellbeing on Flights: "
            "Jet lag is caused by crossing time zones faster than your body clock can adapt. "
            "Symptoms — fatigue, disrupted sleep, difficulty concentrating — last roughly 1 day per time zone crossed. "
            "Remedies: adjust your sleep schedule a few days before travel, seek natural light at your destination, "
            "stay hydrated, and consider melatonin for eastward journeys. "
            "Deep Vein Thrombosis (DVT): on long-haul flights, walk the aisle every 1-2 hours, do seated calf "
            "raises, wear compression socks, and stay well-hydrated. "
            "Ear pressure: swallow, yawn, or chew gum during descent; try a decongestant spray if you are congested. "
            "Dehydration: cabin humidity is very low (~15%) — drink water regularly and limit alcohol and caffeine. "
            "Motion sickness: sit near the wing, focus on a stable point, and consider ginger tablets or antihistamine. "
            "Fear of flying affects roughly 25% of people — breathing exercises, safety statistics, and speaking to "
            "crew can all help."
        ),
        "seed_questions": [
            "How long does jet lag last?",
            "How do I recover from jet lag quickly?",
            "How do I prevent jet lag after a long flight?",
            "How do I avoid blood clots on a long flight?",
            "What is DVT and can flying cause it?",
            "Why do my ears hurt on the plane?",
            "What should I do if my ears pop on a flight?",
            "How do I stay healthy on a long-haul flight?",
            "Does flying cause dehydration?",
            "How do I deal with fear of flying?",
            "How do I handle motion sickness on a plane?",
            "Is it safe to fly when I have a cold?",
        ],
    },
    {
        "id": "aircraft_safety_doc",
        "text": (
            "Aircraft Safety Features: "
            "All commercial aircraft carry Automated External Defibrillators (AEDs). "
            "Flight attendants are trained in CPR and first aid; in a serious medical emergency they ask for "
            "medical professionals onboard and the captain may divert to the nearest airport. "
            "Cabin lights are dimmed during takeoff and landing so passengers' eyes adjust to outside light levels, "
            "enabling clear vision during an emergency evacuation. "
            "Emergency exits are marked with illuminated signs — locate your nearest exit when boarding. "
            "If cabin pressure drops, oxygen masks fall automatically from overhead panels; "
            "place yours on before helping others. "
            "Life vests are stored under or beside your seat for water landings. "
            "The brace position (head down, hands over neck) minimises injury in emergency landings. "
            "Commercial aircraft have multiple redundant engines, hydraulic systems, and autopilot computers. "
            "Aviation is statistically the safest form of long-distance travel."
        ),
        "seed_questions": [
            "Do planes have defibrillators on board?",
            "Is there medical equipment on the plane?",
            "Why do airlines dim the lights during takeoff and landing?",
            "What should I do in a plane emergency?",
            "Where are the emergency exits on the plane?",
            "What is the brace position?",
            "How do the oxygen masks work on a plane?",
            "Is flying safe?",
            "How safe are commercial aircraft?",
            "What happens if the cabin loses pressure?",
            "Is there a life vest under my seat?",
            "What happens if someone has a cardiac arrest on a plane?",
        ],
    },
    {
        "id": "turbulence_doc",
        "text": (
            "Understanding Turbulence: "
            "Turbulence is irregular air movement caused by atmospheric pressure changes, jet streams, "
            "thunderstorms, and air flowing over mountains or warm ocean currents. "
            "Modern commercial aircraft are designed and stress-tested to withstand forces far exceeding "
            "anything passengers will experience. "
            "Turbulence is categorised as light, moderate, severe, or extreme — severe and extreme are rare. "
            "The primary injury risk is being thrown when not belted — always keep your seatbelt loosely "
            "fastened whenever you are seated. "
            "Clear-air turbulence (CAT) occurs at cruising altitude without visual warning and cannot be "
            "detected on radar, making the seatbelt rule especially important. "
            "Seats over the wings experience less motion than those at the front or tail. "
            "Turbulence cannot cause a structurally sound aircraft to crash."
        ),
        "seed_questions": [
            "Is turbulence dangerous?",
            "Can a plane crash from turbulence?",
            "What causes turbulence on flights?",
            "How bad can turbulence get?",
            "I am scared of turbulence — is the plane safe?",
            "Why is the plane shaking?",
            "What is clear air turbulence?",
            "Where should I sit to avoid turbulence?",
            "Why should I keep my seatbelt on during the flight?",
            "Can turbulence make a plane fall out of the sky?",
            "How do pilots handle turbulence?",
            "Is severe turbulence common on flights?",
        ],
    },
    {
        "id": "airport_facilities_doc",
        "text": (
            "Airport Facilities and Services: "
            "Most major airports offer a wide range of amenities after the security checkpoint. "
            "Dining: restaurants, cafes, and fast food are available in both departures and arrivals. "
            "Duty-free: international travellers can purchase alcohol, perfume, electronics, and more "
            "tax-free before departure — limits apply when entering your destination country. "
            "Lounges: available to business class passengers, Phoenix Air Gold/Platinum members, and "
            "Priority Pass cardholders — most offer complimentary food, drinks, showers, and quiet seating. "
            "Wi-Fi: most airports offer free or paid Wi-Fi — look for the airport network name or ask staff. "
            "Currency exchange and ATMs are in most terminals. "
            "Pharmacies, convenience stores, and baby care rooms are common at large airports. "
            "Left-luggage storage lets you leave bags for a few hours for a fee. "
            "Charging stations are available at most gates and seating areas."
        ),
        "seed_questions": [
            "Are there restaurants at the airport?",
            "Is there Wi-Fi at the airport?",
            "What shops are available at the airport?",
            "Are there lounges at the airport?",
            "How do I access an airport lounge?",
            "Can I buy duty-free goods?",
            "What is duty-free shopping?",
            "Is there a pharmacy at the airport?",
            "Can I store my bags at the airport?",
            "Where can I exchange money at the airport?",
            "Are there phone charging points at the airport?",
            "Can I sleep at the airport overnight?",
        ],
    },
    {
        "id": "first_time_flyer_doc",
        "text": (
            "Tips for First-Time Flyers: "
            "Arrive early — at least 2 hours for domestic and 3 hours for international flights. "
            "Wear comfortable, loose-fitting clothing. "
            "Pack essentials (medication, valuables, and a change of clothes) in your carry-on, not checked luggage. "
            "At security: have your boarding pass and ID ready, remove shoes and jacket, and place liquids in a clear bag. "
            "Follow departure board signs to your gate — boarding begins 30-45 minutes before departure. "
            "On the plane: fasten your seatbelt as soon as you sit down, follow crew instructions, and drink water regularly. "
            "Turbulence is normal and the aircraft is safe — focus on your breathing if you feel anxious. "
            "After landing, remain seated until the seatbelt sign turns off before standing. "
            "At a new destination, follow signs to immigration, baggage claim, and customs."
        ),
        "seed_questions": [
            "What should I know for my first time flying?",
            "Tips for a first-time flyer?",
            "What do I do when I arrive at the airport for the first time?",
            "What happens at the airport step by step?",
            "I have never flown before — what should I expect?",
            "What should I pack for my first flight?",
            "How does the boarding process work for a new flyer?",
            "Is flying scary for the first time?",
            "What do I need to know before my first flight?",
            "What is it like to fly for the first time?",
        ],
    },
]

# ── Retrieval layer ───────────────────────────────────────────────────────────
#
# Primary:  sentence-transformers (all-MiniLM-L6-v2) — semantic cosine similarity
# Fallback: TF-IDF (sklearn)                          — used when transformers unavailable
#
# Both use dual-matching: query vs seed questions + query vs full doc, take max.
# Thresholds differ: embeddings score 0.45+, TF-IDF scores 0.10+.

EMBED_POL_THRESHOLD  = 0.65   # policy: real policy questions score 0.80+; false positives ~0.55
EMBED_GEN_THRESHOLD  = 0.45   # general: broader topics need lower bar
TFIDF_POL_THRESHOLD  = 0.25   # TF-IDF fallback for policy
TFIDF_GEN_THRESHOLD  = 0.12   # TF-IDF fallback for general
EMBED_THRESHOLD      = EMBED_POL_THRESHOLD   # backward compat alias

# ── Embedding matrices (built lazily if sentence-transformers is available) ───

_SEED_INDEX:      list[str] = []
_seed_matrix                = None
_doc_matrix                 = None

_GEN_SEED_INDEX:  list[str] = []
_gen_seed_matrix            = None
_gen_doc_matrix             = None

# ── TF-IDF retrievers (built once at module load — sklearn only, never crashes) ─

def _make_tfidf(corpus_ids: list[str], corpus_texts: list[str]):
    """Returns (vectorizer, matrix, ids) ready for cosine_similarity queries."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(corpus_texts)
    return vec, mat, corpus_ids


def _tfidf_retrieve(question: str, vec, mat, ids: list[str],
                    items: list[dict], id_key: str, text_key: str,
                    threshold: float, label: str) -> tuple[str, float]:
    """TF-IDF retrieval. ids may contain regular IDs (seeds) or '__doc__'+id (docs)."""
    q_vec = vec.transform([question.lower()])
    sims  = cosine_similarity(q_vec, mat).flatten()

    best_per_item: dict[str, float] = {}
    for i, iid in enumerate(ids):
        real_id = iid[7:] if iid.startswith("__doc__") else iid
        s = float(sims[i])
        if s > best_per_item.get(real_id, -1.0):
            best_per_item[real_id] = s

    best_score = -1.0
    best_item  = None
    for item in items:
        iid   = item[id_key]
        score = best_per_item.get(iid, 0.0)
        print(f"[{label} TFIDF] {iid}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_item  = item

    if best_score < threshold or best_item is None:
        print(f"[{label} TFIDF] no match -- best {best_score:.3f} < {threshold}")
        return "", best_score

    print(f"[{label} TFIDF] WINNER: '{best_item[id_key]}' ({best_score:.3f})")
    return best_item[text_key], best_score


# Build policy TF-IDF corpus: seed questions ONLY (not full doc texts).
# Doc texts contain many topics and cause false-positive matches for unrelated questions.
_pol_tfidf_ids:   list[str] = []
_pol_tfidf_texts: list[str] = []
for _p in POLICIES:
    for _q in _p["seed_questions"]:
        _pol_tfidf_ids.append(_p["id"])
        _pol_tfidf_texts.append(_q.lower())

_pol_tfidf_vec, _pol_tfidf_mat, _ = _make_tfidf(_pol_tfidf_ids, _pol_tfidf_texts)
print(f"[RAG TFIDF] Policy TF-IDF built -- {len(_pol_tfidf_texts)} seed questions")

# Build general TF-IDF corpus
_gen_tfidf_ids:   list[str] = []
_gen_tfidf_texts: list[str] = []
for _d in GENERAL_DOCS:
    for _q in _d["seed_questions"]:
        _gen_tfidf_ids.append(_d["id"])
        _gen_tfidf_texts.append(_q.lower())
    _gen_tfidf_ids.append("__doc__" + _d["id"])
    _gen_tfidf_texts.append(_d["text"].lower())

_gen_tfidf_vec, _gen_tfidf_mat, _ = _make_tfidf(_gen_tfidf_ids, _gen_tfidf_texts)
print(f"[GEN TFIDF] General TF-IDF built -- {len(_gen_tfidf_texts)} entries")


def _build_matrices():
    global _seed_matrix, _doc_matrix, _SEED_INDEX
    if _seed_matrix is not None:
        return
    embedder = get_embedder()
    if embedder is None:
        return

    seed_sentences = []
    for policy in POLICIES:
        for q in policy["seed_questions"]:
            _SEED_INDEX.append(policy["id"])
            seed_sentences.append(q)
    _seed_matrix = embedder.encode(seed_sentences, normalize_embeddings=True)
    _doc_matrix  = embedder.encode([p["text"] for p in POLICIES], normalize_embeddings=True)
    print(f"[RAG EMBED] Policy embedding matrices built -- {len(seed_sentences)} seeds + {len(POLICIES)} docs")


def _build_general_matrices():
    global _gen_seed_matrix, _gen_doc_matrix, _GEN_SEED_INDEX
    if _gen_seed_matrix is not None:
        return
    embedder = get_embedder()
    if embedder is None:
        return

    seed_sentences = []
    for doc in GENERAL_DOCS:
        for q in doc["seed_questions"]:
            _GEN_SEED_INDEX.append(doc["id"])
            seed_sentences.append(q)
    _gen_seed_matrix = embedder.encode(seed_sentences, normalize_embeddings=True)
    _gen_doc_matrix  = embedder.encode([d["text"] for d in GENERAL_DOCS], normalize_embeddings=True)
    print(f"[GEN EMBED] General embedding matrices built -- {len(seed_sentences)} seeds + {len(GENERAL_DOCS)} docs")


# ── Public API ────────────────────────────────────────────────────────────────

def query_policy_rule_based(question: str) -> str | None:
    """L1: keyword match → pre-authored short answer. Returns None if no match."""
    q = question.lower()
    for keywords, policy_id in _KEYWORD_MAP:
        if any(kw in q for kw in keywords):
            return _POLICY_SHORT_ANSWERS.get(policy_id)
    return None


def query_general_rule_based(question: str) -> str | None:
    """Rule-based answer for common general airline questions. Returns None if no match."""
    q = question.lower()
    for keywords, answer in _GENERAL_RULES:
        if any(kw in q for kw in keywords):
            return answer
    return None


def query_policy_with_score(question: str) -> tuple[str, float]:
    """
    L2 retrieval for policy docs.
    Primary: sentence-transformer embedding (cosine, threshold 0.45).
    Fallback: TF-IDF (sklearn, threshold 0.10) when transformers unavailable.
    Both use dual-matching: query vs seed questions + query vs full doc, take max.
    """
    try:
        _build_matrices()
        embedder = get_embedder()

        if embedder is not None and _seed_matrix is not None:
            # ── Embedding path ────────────────────────────────────────────────
            q_vec            = embedder.encode([question], normalize_embeddings=True)[0]
            seed_scores_flat = np.dot(_seed_matrix, q_vec)
            policy_seed_best: dict[str, float] = {}
            for i, pid in enumerate(_SEED_INDEX):
                s = float(seed_scores_flat[i])
                if s > policy_seed_best.get(pid, -1.0):
                    policy_seed_best[pid] = s
            doc_scores_flat = np.dot(_doc_matrix, q_vec)
            best_score, best_policy = -1.0, None
            for j, policy in enumerate(POLICIES):
                pid      = policy["id"]
                combined = max(policy_seed_best.get(pid, 0.0), float(doc_scores_flat[j]))
                print(f"[RAG EMBED] {pid}: combined={combined:.3f}")
                if combined > best_score:
                    best_score, best_policy = combined, policy
            if best_score >= EMBED_POL_THRESHOLD and best_policy:
                print(f"[RAG EMBED] WINNER: '{best_policy['id']}' ({best_score:.3f})")
                return best_policy["text"], best_score
            print(f"[RAG EMBED] no match -- best {best_score:.3f} < {EMBED_POL_THRESHOLD}")
            # Fall through to TF-IDF

        # ── TF-IDF fallback (or secondary when embedding misses) ──────────────
        return _tfidf_retrieve(
            question, _pol_tfidf_vec, _pol_tfidf_mat, _pol_tfidf_ids,
            POLICIES, "id", "text", TFIDF_POL_THRESHOLD, "RAG",
        )

    except Exception as e:
        print(f"[RAG] retrieval error: {e}")
        return "", 0.0


def query_general_with_score(question: str) -> tuple[str, float]:
    """
    L2 retrieval for general airline knowledge docs.
    Primary: sentence-transformer embedding.
    Fallback: TF-IDF when transformers unavailable.
    """
    try:
        _build_general_matrices()
        embedder = get_embedder()

        if embedder is not None and _gen_seed_matrix is not None:
            # ── Embedding path ────────────────────────────────────────────────
            q_vec            = embedder.encode([question], normalize_embeddings=True)[0]
            seed_scores_flat = np.dot(_gen_seed_matrix, q_vec)
            gen_seed_best: dict[str, float] = {}
            for i, gid in enumerate(_GEN_SEED_INDEX):
                s = float(seed_scores_flat[i])
                if s > gen_seed_best.get(gid, -1.0):
                    gen_seed_best[gid] = s
            doc_scores_flat = np.dot(_gen_doc_matrix, q_vec)
            best_score, best_doc = -1.0, None
            for j, doc in enumerate(GENERAL_DOCS):
                gid      = doc["id"]
                combined = max(gen_seed_best.get(gid, 0.0), float(doc_scores_flat[j]))
                print(f"[GEN EMBED] {gid}: combined={combined:.3f}")
                if combined > best_score:
                    best_score, best_doc = combined, doc
            if best_score >= EMBED_GEN_THRESHOLD and best_doc:
                print(f"[GEN EMBED] WINNER: '{best_doc['id']}' ({best_score:.3f})")
                return best_doc["text"], best_score
            print(f"[GEN EMBED] no match -- best {best_score:.3f} < {EMBED_GEN_THRESHOLD}")
            return "", best_score

        # ── TF-IDF fallback ───────────────────────────────────────────────────
        return _tfidf_retrieve(
            question, _gen_tfidf_vec, _gen_tfidf_mat, _gen_tfidf_ids,
            GENERAL_DOCS, "id", "text", TFIDF_GEN_THRESHOLD, "GEN",
        )

    except Exception as e:
        print(f"[GEN] retrieval error: {e}")
        return "", 0.0

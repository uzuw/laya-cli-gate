from laya import Router
import time
router = Router()
state = {"subject": "Duplicate charge on invoice 4411", "body": "Billed twice for March. Refund please. Thinking of cancelling."}
questions = {
 "owner": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges/refunds", "technical": "errors", "account": "login/security", "other": "none fits"}},
 "refund": {"type": "noul", "instructions": "Refund requested?"},
 "cancel": {"type": "noul", "instructions": "Threatening to cancel?"},
 "spam": {"type": "noul", "instructions": "Is this spam?"},
 "urgent": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "med", "high"]},
 "sentiment": {"type": "score", "instructions": "Sentiment?", "criteria": ["neg", "neu", "pos"]},
 "bug": {"type": "noul", "instructions": "Reports a bug?"},
 "vip": {"type": "noul", "instructions": "Is VIP customer?"},
 "escalate": {"type": "noul", "instructions": "Needs human?"},
 "priority": {"type": "choice", "instructions": "Priority?", "criteria": {"p1": "urgent money loss", "p2": "normal", "p3": "low"}},
}
t = time.time()
d = router.predict(state, questions)
dt = (time.time() - t) * 1000
for k, v in d["answers"].items():
    print(k, "->", v.get("choice", v.get("noul", v.get("score"))), f"{v.get('confidence',0):.2f}")
print(f"{len(questions)} questions in {dt:.0f}ms ({dt/len(questions):.1f}ms/q)")

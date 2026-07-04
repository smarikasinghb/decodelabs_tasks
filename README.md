# DecodeLabs AI Internship — Task Repository

**Intern:** Smarika Singh  
**Batch:** 2026 | **Track:** Artificial Intelligence  
**Institute:** MITS-DU, Gwalior

---

## Project 1 — Rule-Based AI Chatbot

### Overview
A terminal-based chatbot built using **pure Python** — no ML libraries, no external dependencies. The system simulates intelligent conversation through deterministic, rule-based decision-making.

### Architecture: IPO Model

INPUT (Raw User Text)
  → Sanitization: .lower().strip()
  → Intent Matching: keyword → intent dictionary lookup O(1)
  → OUTPUT: Randomized response from intent pool

### Why Dictionary over If-Elif?

| Approach | Complexity | Scalability |
|---|---|---|
| If-Elif Ladder | O(n) — linear | Degrades with more rules |
| Dictionary .get() | O(1) — constant | Instant regardless of size |

### Features
- Input sanitization (case + whitespace normalization)
- Keyword-based intent matching
- Response variation using randomized reply pools
- Graceful exit on multiple commands
- Default fallback for unrecognized inputs

### How to Run

python3 rule_based_chatbot.py

### File Structure

decodelabs_tasks/
├── README.md
└── project1/
    └── rule_based_chatbot.py

---

*Built as part of the DecodeLabs AI Industrial Training Program — Batch 2026.*
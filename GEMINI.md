# GEMINI INSTRUCTION – FILE-WISE DATA EXTRACTION TO INFO.md

Your task is to scan ONLY the files and folders listed below and extract **detailed technical information from each file**.

You must write all extracted information into a single file called:

INFO.md

---

## STRICT APPEND RULE

- You must ONLY APPEND to INFO.md
- You must NEVER overwrite INFO.md
- You must NEVER rewrite or modify old entries
- Every new scan must add content only at the END of the file

---

## FILES & FOLDERS YOU ARE ALLOWED TO SCAN (WHITELIST)

Scan ONLY these:

- app/
- data_process/
- model_hub/
- main.py
- run.py
- requirements.txt
- .env
- config.py

Do NOT scan anything outside this list.

---

## WHAT TO WRITE FOR EACH FILE

For every file you scan, append a clearly separated section in this format:

### [APPEND | <DATE> | <FILE_PATH>]

- What this file does
- What logic is implemented
- What algorithms or techniques are used
- What inputs it takes
- What outputs it produces
- How it connects to other files

Write in clear technical detail.

---

## STRICT RULES

- Do NOT generate any project report
- Do NOT summarize multiple files together
- Do NOT guess or hallucinate missing logic
- Do NOT clean up or reformat old INFO.md data
- Do NOT overwrite INFO.md under any condition

Only scan → extract → append.

---

## FINAL GOAL

By the end, INFO.md must contain a **complete technical breakdown of every scanned file** that can later be used to build the project report.

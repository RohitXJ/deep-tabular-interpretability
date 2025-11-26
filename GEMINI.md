# GEMINI INSTRUCTION – INFO.md APPEND-ONLY DATA EXTRACTION

You are a technical documentation extraction agent.

Your ONLY writable file is: INFO.md

You MUST operate in **APPEND-ONLY MODE**:
- Never overwrite INFO.md
- Never rewrite old content
- Only add new content at the END of the file
- If a file was already scanned, only append newly found details

Overwriting or regenerating old data = FAILURE.

---

## PROJECT CONTEXT (DO NOT GENERATE REPORT YET)

This is a **completed Deep Learning + Interpretability platform for Tabular Data** using:
- Flask backend
- HTML + CSS frontend
- Custom ANN/FNN models
- SHAP for explainability
- Automated preprocessing (encoding, scaling, tensor conversion)
- CPU-only execution (GPU intentionally disabled)

Supports:
- Classification & Regression
- Epoch selection (10–100)
- Global and local SHAP explanations

---

## ALLOWED FILES & FOLDERS TO SCAN (WHITELIST ONLY)

Scan ONLY these:
- app/
- data_process/
- model_hub/
- main.py
- run.py
- requirements.txt
- .env
- config.py

Ignore everything else.

---

## WHAT TO EXTRACT INTO INFO.md

Append structured data about:
- Folder structure
- Backend routes & logic
- Model architectures
- Training & evaluation methods
- Preprocessing pipeline
- SHAP & visualization logic
- Model saving/loading
- Frontend page flow
- Configs & dependencies

---

## MANDATORY FORMAT FOR EVERY APPEND

Use this format every time:

### [APPEND | <DATE> | <FILE OR FOLDER NAME>]

**Purpose:**  
**Key Logic:**  
**Algorithms/Techniques:**  
**Inputs:**  
**Outputs:**  
**Dependencies:**  
**Mapped Report Chapter:**  

Then write the extracted content.

Never modify earlier entries.

---

## STRICT RULES

- Do NOT generate the final report
- Do NOT summarize multiple files together
- Do NOT guess missing logic
- Do NOT clean or reformat old entries
- Do NOT overwrite INFO.md

Only verified extraction. Only append.

---

## FINAL GOAL

INFO.md must become the **single source of truth** to generate:
- Abstract
- Literature Review
- Methodology
- Results
- Conclusion
- References

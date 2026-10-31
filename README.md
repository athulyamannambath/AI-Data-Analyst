# AI Data Analyst

Ask questions about a CSV dataset in plain English and get a number, a table or a chart back.

```
question -> local LLM (Ollama) -> JSON analysis plan -> validator -> Pandas engine -> formatter -> answer
```

The LLM only **plans** the analysis. It never writes or runs code, and it never sees raw data rows.
Every plan is checked by a validator before Pandas executes it with a fixed set of operations.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) (free, runs the model on your own machine)
- About 2 GB of disk and 8 GB of RAM for the default model (`qwen2.5:3b`)

## Setup (Windows PowerShell)

**1. Download the datasets** from Kaggle (login required) and put the unzipped `.csv` files directly in `backend/data/`:

| Dataset key | Kaggle page | File name expected |
|---|---|---|
| `superstore` | vivek468/superstore-dataset-final | `Sample - Superstore.csv` |
| `students` | spscientist/students-performance-in-exams | `StudentsPerformance.csv` |
| `hr_attrition` | pavansubhasht/ibm-hr-analytics-attrition-dataset | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| `netflix` (optional) | shivamb/netflix-shows | `netflix_titles.csv` |

A dataset only appears in the app when its file is present. File names must match `backend/data/datasets.json`.

**2. Create the Python environment**

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

**3. Install the model**

```powershell
ollama pull qwen2.5:3b
```

Have 16 GB of RAM or more? `qwen2.5:7b` gives noticeably better results.

**4. Create `backend/.env`** (copy `.env.example`):

```
OLLAMA_MODEL=qwen2.5:3b
```

**5. Check everything works**

```powershell
python -m pytest -q tests
```

**6. Start the API** (Ollama must be running; look for its tray icon or run `ollama serve`)

```powershell
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** to try the API in your browser. The first question after startup can take up to a minute while the model loads; later ones take a few seconds.

Frontend developers: see [API.md](API.md).

## Try it from the command line

```powershell
python try_llm.py superstore "total sales by region as a bar chart"
```

Prints the LLM's plan, the validated plan and the result.

## Measuring accuracy

`backend/tests/test_questions.json` holds test questions. For each one, the correct answer is computed directly with Pandas and compared with the app's answer.

```powershell
python evaluate.py                         # all questions
python evaluate.py --dataset students      # one dataset
python evaluate.py --ids hr-01 hr-02       # specific questions
python evaluate.py --model qwen2.5:7b      # compare another model
python evaluate.py --save results.json     # keep the results
```

`python pandas_checks.py` prints the Pandas answers for a few questions so you can check by hand.

| Dataset | Passed | Total | Notes |
|---|---|---|---|
| Superstore | 14 | 15 | sup-13 fails (see limitations) |
| IBM HR | 13 | 14 | hr-13 fails (see limitations) |
| Students | 12 | 12 | |
| **All** | **39** | **41** | **95%** |

Model: `qwen2.5:3b`. Average 12 seconds per question; the first question after switching dataset takes about a minute.
The question list was used while tuning the system, so treat 95% as the score on familiar questions. Add questions of your own to `backend/tests/test_questions.json` to test how well it generalises.

## Project structure

```
ai-data-analyst/
├── README.md
├── API.md                       # for the frontend developer
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes
│   │   ├── pipeline.py          # question -> answer, one function
│   │   ├── dataset_manager.py   # loads CSVs, schema, aliases
│   │   ├── llm_service.py       # Ollama client, prompt, JSON parsing
│   │   ├── validator.py         # allowlist checks on the plan
│   │   ├── repair.py            # fixes known, unambiguous model mistakes
│   │   ├── engine.py            # runs plans with Pandas
│   │   └── formatter.py         # number / table / chart + explanation
│   ├── data/
│   │   ├── datasets.json        # dataset registry (files, dates, notes, aliases)
│   │   └── *.csv                # you download these
│   ├── tests/                   # unit tests + test_questions.json
│   ├── evaluate.py              # accuracy evaluation
│   ├── try_llm.py               # one-question command-line check
│   ├── pandas_checks.py         # reference answers computed in Pandas
│   ├── requirements.txt
│   └── .env                     # model name (not committed)
└── frontend/                    # Member 2
```

## How the pieces work

**Plans, not code.** The model replies with JSON such as:

```json
{"operation": "groupby_aggregate", "group_by": "Region", "column": "Sales",
 "aggregation": "sum", "sort": "desc", "output": "chart", "chart_type": "bar"}
```

**Operations supported:** `count_rows`, `missing_values`, `aggregate` (sum, mean, median, min, max, count, nunique), `groupby_aggregate`, `top_n`, `time_trend` (day, month, quarter, year). Any of them can take up to 5 filters (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`).

**Safety**
- The model sees column names, types, short notes, category labels (columns with 12 or fewer values) and date ranges. It never sees raw rows.
- Model output is parsed as JSON and never executed.
- The validator rejects anything outside the allowlists: unknown operations or fields, unknown columns, `sum` on text, a trend on a non-date column, too many filters, and so on.
- Column names are matched loosely (case, spaces, underscores) and through per-dataset aliases (for example `salary` means `MonthlyIncome`).
- Plans pass through a small **repair** step for known, unambiguous model mistakes, using only validated columns. For example, "top 5 states by sales" written as a list of single orders is turned into a ranking of states by total (or average) sales. The `plan` in every answer shows what actually ran.
- The question is passed to the model as data, with instructions to ignore any request in it to change the rules.
- Explanations are written from templates using the computed values, so the text cannot contradict the numbers.

**Adding a dataset:** put the CSV in `backend/data/` and add an entry to `datasets.json` (file name, label, description, date columns with formats, column notes, aliases). No code changes are needed.

## Known limitations

- Only the six operations above. No joins, percentages, correlations, forecasts or custom formulas. Unsupported questions get a short explanation instead of an answer.
- One dataset per question.
- Charts support one grouping column. A grouping by two columns is returned as a table.
- Tables are capped at 100 rows. Questions are limited to 500 characters.
- Columns that hold comma-separated lists (Netflix `country`, `cast`, `listed_in`) cannot be split and counted yet.
- Two questions in the test set still fail with the 3B model. "Show the top 3 states by total sales" is sometimes answered with raw order rows instead of a ranking of states, and "How many employees left the company in each department?" sometimes drops the "left" condition (the answer then counts all employees per department). A larger model (`qwen2.5:7b`) is the first thing to try for both.
- A small local model sometimes misreads a question. The answer's `plan` field shows what was actually calculated, so wrong answers can be spotted. Larger models help.
- Local models are slow on a CPU: expect 4 to 13 seconds per question. The first question after startup, and the first one after switching to a different dataset, takes up to a minute (the model has to read the new dataset's instructions once).

## Troubleshooting

| Problem | Fix |
|---|---|
| "I can't reach Ollama" | Start the Ollama app or run `ollama serve`. |
| "The model ... is not installed" | Run `ollama list`, then `ollama pull <model>`. The name must match `OLLAMA_MODEL` in `.env`. |
| `try_llm.py` prints nothing | The file is empty or an old copy. Re-copy it from the repository. |
| A dataset is missing from the dropdown | Its CSV is not in `backend/data/`, or the file name does not match `datasets.json`. |
| `pytest` hangs or is slow | Run `python -m pytest -q tests`. Projects inside OneDrive are slow; move the folder somewhere like `C:\dev\`. |
| `ModuleNotFoundError: No module named 'app'` | Run commands from the `backend/` folder. |
| "no such column" for a word users commonly say | Add it to the dataset's `aliases` in `datasets.json`. |

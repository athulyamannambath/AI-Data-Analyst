# API guide for the frontend

Base URL (local development): `http://127.0.0.1:8000`
Interactive docs where you can try every call: `http://127.0.0.1:8000/docs`

CORS is open, so a frontend on another port (for example `localhost:3000` or `localhost:5173`) can call the API directly.

Start the backend: see [README.md](README.md). The Ollama app must be running.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |
| GET | `/datasets` | Datasets for the dropdown |
| GET | `/datasets/{name}/preview` | Schema and sample rows for the preview panel |
| POST | `/ask` | Ask a question about a dataset |

### GET /datasets

```json
[
  {"name": "superstore", "label": "Superstore Sales", "description": "Retail orders from a US superstore..."},
  {"name": "students", "label": "Students Performance", "description": "Exam scores of 1000 students..."}
]
```

Only datasets whose CSV file is installed are listed. Use `name` in the other calls.

### GET /datasets/{name}/preview

```json
{
  "name": "students",
  "label": "Students Performance",
  "description": "Exam scores of 1000 students...",
  "rows": 1000,
  "column_count": 8,
  "columns": [
    {"name": "gender", "type": "text", "missing": 0, "note": ""},
    {"name": "math score", "type": "number", "missing": 0, "note": "Math exam score, 0-100"}
  ],
  "sample": [
    {"gender": "female", "race/ethnicity": "group B", "math score": 72, "reading score": 72, "writing score": 74}
  ]
}
```

`type` is one of `number`, `text`, `date`, `boolean`. `sample` holds the first 5 rows (shortened here). Returns HTTP 404 with `{"detail": "..."}` for an unknown dataset.

### POST /ask

Request:

```json
{"dataset": "superstore", "question": "total sales by region as a bar chart"}
```

The response is **always HTTP 200** with the same set of keys, whatever happened. Look at `status`:

| `status` | Meaning | Show the user |
|---|---|---|
| `success` | The question was answered | the result (see below) and `explanation` |
| `clarification` | The question cannot be answered as asked (for example, no such column, or a prediction) | `message` |
| `error` | Something failed (Ollama not running, a bad plan, no matching rows, empty question) | `error` |

Keys that do not apply are `null`:

```
status, dataset, question, result_type, label, value, columns, rows, row_count,
truncated, chart, explanation, message, error, plan
```

Only a malformed request body (for example, a missing `question` field) returns HTTP 422.

#### `result_type` for a successful answer

**`number`** uses `label`, `value` and `explanation`:

```json
{
  "status": "success", "result_type": "number",
  "label": "Number of rows", "value": 237,
  "explanation": "There are 237 rows. Filter: Attrition == Yes.",
  "plan": {"operation": "count_rows", "filters": [{"column": "Attrition", "op": "==", "value": "Yes"}]}
}
```

**`table`** uses `columns`, `rows`, `row_count`, `truncated` and `explanation`:

```json
{
  "status": "success", "result_type": "table",
  "columns": ["Region", "Sales (sum)"],
  "rows": [["West", 725457.82], ["East", 678781.24], ["Central", 501239.89], ["South", 391721.91]],
  "row_count": 4, "truncated": false,
  "explanation": "Total Sales by Region, sorted high to low. West comes first at 725,457.82."
}
```

Tables are capped at 100 rows. If `truncated` is `true`, `row_count` is the real total and you should say that only the first rows are shown.

**`chart`** has everything a table has, plus a `chart` object:

```json
"chart": {"type": "bar", "x": "Region", "y": "Sales (sum)"}
```

- `type` is `bar`, `line` or `pie`.
- `x` and `y` are names from `columns`. Each row is `[x value, y value]`.
- Line charts (trends over time) are in chronological order. Pie charts are only sent for 8 slices or fewer.
- If a chart is not possible, the answer comes back as a `table` instead, so always handle both.

**`clarification`:**

```json
{"status": "clarification", "message": "I can't make predictions, but I can show the historical sales trend."}
```

**`error`:**

```json
{"status": "error", "error": "I can't reach Ollama. Open the Ollama app (or run 'ollama serve') and try again."}
```

#### `plan`

On success, `plan` shows exactly what was calculated. It is useful for a "How was this calculated?" expander, and for spotting a wrong answer.

## Things to build into the UI

- **Loading indicator.** Answers take about 5 to 15 seconds. The first question after the backend starts, and the first one after the user switches to a different dataset, can take up to a minute. Show a message such as "Getting ready for this dataset..." if a request takes more than 15 seconds. Set the request timeout to at least 200 seconds.
- **Disable the Submit button** while a question is running. Questions are processed one at a time.
- **Handle all three statuses.** `clarification` and `error` are normal outcomes, not crashes. Show them as friendly messages.
- **Questions are limited to 500 characters.** Longer ones return `status: "error"`.
- **Numbers.** `value` and table cells are plain numbers rounded to 2 decimals. Format them for display (thousands separators) in the frontend.
- **Dates** in tables are text such as `2016-01` (month) or `2016-01-05`.

## Example call (JavaScript)

```js
async function ask(dataset, question) {
  const res = await fetch("http://127.0.0.1:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, question }),
  });
  if (!res.ok) throw new Error("Request failed: " + res.status);
  const data = await res.json();

  switch (data.status) {
    case "success":       return renderResult(data);   // number, table or chart
    case "clarification": return showNotice(data.message);
    default:              return showError(data.error);
  }
}
```

## Questions that work well

- "What is the total sales?" / "What is the average profit by segment?"
- "Total profit by category as a bar chart"
- "Show the monthly sales trend"
- "Top 5 sub-categories by total sales"
- "How many employees left the company?"
- "Average math score for students who completed the test preparation course"

Good mock data for building the UI before the backend is ready: copy the example responses above.

# Detailed Change Log

This file records the experiment-page and backend fixes made in this workspace, along with the checks run afterward.

## Problem Addressed

The experiment page expected an asynchronous API: it sent a request, expected a `job_id`, and then polled a status URL. The backend instead ran the population simulation during the original request and returned a completed result without a `job_id`. As a result, the page could report that the backend had not returned a job ID, and a long simulation held the initial HTTP request open.

The frontend production build also failed because the generation-metrics request had an inferred `unknown` response type. Additionally, the experiment page used a hard-coded API URL and did not consistently show FastAPI error details.

## Backend Changes

File: `backend/main.py`

- Added a single-worker `ThreadPoolExecutor` to run the population script outside the initial HTTP request.
- Added an in-memory job registry protected during creation and lookup by a thread lock. A job stores its status, submitted parameters, progress values, result, or error.
- Changed `POST /api/experiments/population` to validate and accept the configured population size and generation count, create a UUID job ID, schedule the run, and return immediately.
- Added `GET /api/experiments/population/{job_id}` for the experiment page to retrieve a job's current status.
- Prevented overlapping population experiments. A second start request while a job is running receives HTTP `409` with a readable detail message.
- Kept the subprocess timeout at 15 minutes. Non-zero exit codes retain the last 5,000 characters of stdout and stderr in the job error, and unexpected exceptions are reported as failed jobs instead of leaving the frontend waiting indefinitely.
- On success, the job response includes the request parameters and generation metrics loaded from `results/population/generation_stats.csv`.
- Unknown or expired job IDs return HTTP `404`.

The request limits remain population size `2..100` and generations `1..50`. A start request looks like:

```http
POST /api/experiments/population
Content-Type: application/json

{"population_size": 2, "generations": 1}
```

The initial response now has the shape:

```json
{
	"status": "running",
	"job_id": "<uuid>",
	"parameters": {
		"population_size": 2,
		"generations": 1
	},
	"progress": {
		"completed_generations": 0,
		"total_generations": 1
	}
}
```

The page polls `GET /api/experiments/population/<uuid>` until the status is `completed` or `failed`. A successful status response contains `result.generation_metrics`; a failed response contains an error message and, when available, subprocess stderr/stdout.

### Backend Job-State Limitation

Job records are held in process memory. Restarting the backend clears existing job IDs and status; this implementation does not persist jobs or resume simulations after a restart. The one-worker executor and running-job check intentionally allow one population experiment at a time.

## Frontend Changes

Files: `frontend/src/pages/ExperimentPage.tsx`, `frontend/src/api/client.ts`

- Exported the shared API URL from the API client. The experiment page now honors `VITE_API_URL` and uses `http://localhost:8000` only as the fallback, matching the rest of the frontend.
- Kept the page's start-and-poll workflow and aligned it with the backend's job ID and status response.
- Added handling for FastAPI's `detail` error format as well as job-level `error.message`, so errors such as a concurrent-run conflict are shown instead of a generic start/status failure.
- Displays captured stderr when a background experiment fails, preserving line breaks in a scrollable/preformatted diagnostic area.
- Replaced the percentage bar with an indeterminate progress element. The backend reports completion but does not stream per-generation updates while the simulation is running, so showing a changing numeric percentage would have been misleading.
- Added an explicit return type for `getGenerationMetrics()` using the existing `GenerationMetric` domain type. This resolves the `unknown` TypeScript error in `PopulationPage.tsx` without changing that page's behavior.

## Verification Results

- Frontend production build (`npm run build`): passed, including TypeScript compilation and Vite bundling.
- Focused population tests (`python -m pytest tests/test_m4_population.py -q`): 28 passed.
- Full Python suite (`python -m pytest -q`): 85 passed and 1 failed. The failure is `test_run_experiment_rejects_silent_heuristic_fallback` in `tests/test_daily_generation_runtime.py`. That test expects no saved trained forecaster to be available, but this workspace contains forecast model artifacts, so the helper returns a trained forecaster instead of raising. This failure is separate from the population API/page changes.
- Frontend lint (`npm run lint`): completed with one warning in `frontend/src/contexts/ExperimentContext.tsx` about Fast Refresh and mixed component/non-component exports.
- Live service checks: backend health returned HTTP `200`, and the Vite frontend returned HTTP `200`.
- Live experiment check: submitted population size `2` and `1` generation. The API returned a job ID, the status endpoint reported `completed`, and one generation of metrics was returned. The run also wrote the normal population result artifacts under `results/population/`.

## Running the Services

From the repository root, start the backend with:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal, start the frontend with:

```powershell
cd frontend
npm run dev
```

The verified local addresses were `http://127.0.0.1:8000` for the backend and `http://localhost:5173/` for the frontend.
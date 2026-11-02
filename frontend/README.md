# Frontend (React + Vite)

Single-page interface for the job-scam analysis system. A user pastes a job posting, fills in a structured posting, or pastes a recruiter chat. The page shows the verdict and risk score, the reasons in plain language, the warning signs found, the evidence, and what to do next. Technical details (raw classifier probability, threshold, model name) are collapsed under "How this was decided".

## Run

Start the backend first (from the project root: `python -m backend.app`), then:

    cd frontend
    npm install
    npm run dev

Open the address shown (usually http://localhost:5173). In development, requests to `/api` are proxied to the backend at http://127.0.0.1:5000 (see `vite.config.js`).

If the page says "Could not reach the analysis server. Please try again in a moment.", the backend is not running or has stopped.

## Sample data

`src/services/api.js` has `USE_MOCK = false`, so the app talks to the real backend. Setting it to `true` returns sample data without the backend, which is only useful for layout work.

## Tests and build

    npm test          # Node tests for the view logic and rendering
    npm run build

## Structure

- `src/App.jsx`: chat, composer, report panel
- `src/JobReport.jsx`: job report
- `src/components/GuidanceCards.jsx`: next-step guidance cards
- `src/services/api.js`: API calls (`/api/analyze-job`, `/api/analyze-chat`, `/api/demo-examples`) and shared error handling
- `src/services/reportView.js`, `chatView.js`, `conversation.js`: how reports, chats and pasted conversations are turned into what the user sees
- `src/styles.css`: styling, light and dark mode

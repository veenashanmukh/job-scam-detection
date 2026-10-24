# Frontend (React + Vite)

Chatbot-style interface for the Job Scam Assistant.

## Run
    cd frontend
    npm install
    npm run dev

Open the address shown (usually http://localhost:5173).

## Sample data
`src/api.js` has `USE_MOCK = true`, so every result is sample data and the app works without the backend.
To connect the Flask backend, set `USE_MOCK = false`. In dev, requests to `/api` are proxied to http://127.0.0.1:5000 (see `vite.config.js`).

## Structure
- `src/App.jsx`: chat, composer, report panel, timeline
- `src/api.js`: API calls and sample data (response shape follows the team JSON format)
- `src/styles.css`: styling, light and dark mode

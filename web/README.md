# recipe-recsys web app

React + TypeScript + Vite front end for the recipe-recsys API.

- **Search:** a request in any language is parsed by the local LLM; the parsed
  constraints are shown as chips above the results.
- **For you:** pick dietary restrictions and recipes you have cooked; recommendations
  update as you go (iALS fold-in, popularity when there is no history).

In Docker the app is built in its own stage and served by the API on
http://localhost:8000 (`docker compose up app` from the repository root).

For development with hot reload, keep the API running and start Vite, which
proxies `/api` to port 8000:

```bash
npm install
npm run dev    # http://localhost:5173
npm run lint
npm run build
```

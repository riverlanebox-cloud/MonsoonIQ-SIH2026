# Deploying MonsoonIQ

There are two ways to run the console.

| | Live API (`make serve`, Docker) | Static snapshot (Vercel) |
|---|---|---|
| Server | FastAPI + trained models | none, files only |
| Dates | any archived day, any lead | the 585 archived days, leads 1–5 |
| Numbers | computed per request | computed once at build time by the same handlers |
| Not available | — | `/explain`, live endpoint probes on the API page |

## Vercel (static snapshot)

Vercel caps a serverless function far below the size of numpy + pandas + scipy +
scikit-learn + the models (about 340 MB), so the site is deployed as static files.

1. In Vercel, **Add New → Project**, import `MonsoonIQ-SIH2026` from GitHub.
2. Keep the defaults; `vercel.json` in the repo sets everything:
   * install: `npm ci --prefix frontend`
   * build: `bash scripts/vercel_build.sh`
   * output: `frontend/dist`
3. Deploy. Every push to `main` redeploys; pull requests get preview URLs.

`scripts/vercel_build.sh`:

1. installs the minimal Python set in `requirements-vercel.txt` into `.vercel-pydeps/`;
2. runs `scripts/export_static.py`, which loads the committed models and archive and
   calls the API handlers for every archived day and lead. Output, about 50 MB:
   * shared payloads (`timeline.json`, `events.json`, verification JSON, report PDF, …);
   * `day/<date>.json.gz`: console, district detail for all 53 districts,
     bulletins (English and Hindi) and the CSV export, for leads 1–5;
3. builds the console with `VITE_STATIC=1`, so `frontend/src/api.js` reads
   `/snapshot/...` instead of calling the API.

The export takes about 7 minutes on 2 cores (it runs one process per core).

To try the static build locally:

```bash
bash scripts/vercel_build.sh
npx vite preview --outDir frontend/dist     # or any static file server
```

## Live API

`make serve` (or `docker compose up --build`) runs FastAPI on port 8000 and serves the
built console at `/`. Any host that runs a Python container (Render, Railway, Fly.io, a
VM) works; allow about 1 GB of memory.

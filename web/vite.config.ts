import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the API runs separately (`docker compose up app`); in the Docker
// image the API serves the built app itself, so both share an origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})

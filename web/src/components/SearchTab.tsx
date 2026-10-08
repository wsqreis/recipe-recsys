import { useState, type FormEvent } from 'react'
import { api, type SearchResponse } from '../api'
import { restrictionLabel } from '../text'
import { RecipeCard } from './RecipeCard'

const EXAMPLES = [
  'algo sem lactose, com frango, em 20 min',
  'vegan chocolate cake, no nuts please',
  'sopa de legumes sem cogumelo pronta em meia hora',
  "I can't have milk or eggs, what can I bake?",
]

interface Props {
  onOpen: (id: number) => void
}

export function SearchTab({ onOpen }: Props) {
  const [query, setQuery] = useState('')
  const [useLlm, setUseLlm] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<SearchResponse | null>(null)

  async function run(text: string) {
    if (!text.trim()) return
    setQuery(text)
    setLoading(true)
    setError(null)
    try {
      setResponse(await api.search(text, useLlm))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    void run(query)
  }

  const parsed = response?.parsed
  return (
    <section>
      <form className="search" onSubmit={submit}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What do you feel like cooking? Any language, any restriction."
          aria-label="Recipe request"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>
      <div className="options">
        <label>
          <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          Parse with the local LLM (Ollama)
        </label>
        <span className="examples">
          Try:{' '}
          {EXAMPLES.map((example) => (
            <button key={example} className="link" onClick={() => void run(example)}>
              {example}
            </button>
          ))}
        </span>
      </div>

      {error && <p className="error">{error}</p>}
      {response?.llm_error && <p className="warning">{response.llm_error}</p>}

      {parsed && (
        <div className="parsed">
          <span className="label">{response.llm_used ? 'The LLM understood' : 'Keywords found'}:</span>
          {parsed.restrictions.map((r) => (
            <span
              key={r}
              className="chip restriction"
              title={
                response.keyword_restrictions.includes(r)
                  ? 'Also found by the keyword safety net'
                  : 'Found by the LLM'
              }
            >
              {restrictionLabel(r)}
            </span>
          ))}
          {parsed.include.map((i) => (
            <span key={i} className="chip include">with {i}</span>
          ))}
          {parsed.exclude.map((i) => (
            <span key={i} className="chip exclude">without {i}</span>
          ))}
          {parsed.max_minutes !== null && (
            <span className="chip time">≤ {parsed.max_minutes} min</span>
          )}
          {parsed.query && <span className="chip query">“{parsed.query}”</span>}
        </div>
      )}

      {response && response.results.length === 0 && (
        <p className="empty">No recipe satisfies every constraint.</p>
      )}
      <div className="grid">
        {response?.results.map((recipe) => (
          <RecipeCard key={recipe.recipe_id} recipe={recipe} onOpen={onOpen} />
        ))}
      </div>
    </section>
  )
}

import { useEffect, useState } from 'react'
import { api, type RecipeCard as Recipe, type RecommendResponse } from '../api'
import { restrictionLabel, titleCase } from '../text'
import { RecipeCard } from './RecipeCard'

interface Props {
  onOpen: (id: number) => void
}

export function ForYouTab({ onOpen }: Props) {
  const [restrictions, setRestrictions] = useState<string[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [cooked, setCooked] = useState<Recipe[]>([])
  const [lookup, setLookup] = useState('')
  const [matches, setMatches] = useState<Recipe[]>([])
  const [response, setResponse] = useState<RecommendResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.restrictions().then(setRestrictions, (e: Error) => setError(e.message))
  }, [])

  // Name lookup as the user types, debounced; short inputs show nothing.
  const lookingUp = lookup.trim().length >= 3
  const visibleMatches = lookingUp ? matches : []
  useEffect(() => {
    if (!lookingUp) return
    const timer = setTimeout(() => {
      api.findRecipes(lookup).then(setMatches, (e: Error) => setError(e.message))
    }, 250)
    return () => clearTimeout(timer)
  }, [lookup, lookingUp])

  // Recommendations refresh whenever the history or the restrictions change.
  useEffect(() => {
    api
      .recommend(
        cooked.map((r) => r.recipe_id),
        selected,
      )
      .then(setResponse, (e: Error) => setError(e.message))
  }, [cooked, selected])

  function toggle(name: string) {
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]))
  }

  function add(recipe: Recipe) {
    setCooked((c) => (c.some((r) => r.recipe_id === recipe.recipe_id) ? c : [...c, recipe]))
    setLookup('')
    setMatches([])
  }

  return (
    <section className="for-you">
      <div className="panel">
        <h4>Dietary restrictions</h4>
        <div className="toggles">
          {restrictions.map((name) => (
            <label key={name} className={selected.includes(name) ? 'on' : ''}>
              <input
                type="checkbox"
                checked={selected.includes(name)}
                onChange={() => toggle(name)}
              />
              {restrictionLabel(name)}
            </label>
          ))}
        </div>

        <h4>Recipes you have cooked</h4>
        <input
          value={lookup}
          onChange={(e) => setLookup(e.target.value)}
          placeholder="Find a recipe by name, e.g. banana bread"
          aria-label="Find a recipe you have cooked"
        />
        {visibleMatches.length > 0 && (
          <ul className="matches">
            {visibleMatches.map((recipe) => (
              <li key={recipe.recipe_id}>
                <button className="link" onClick={() => add(recipe)}>
                  + {titleCase(recipe.name)}
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="cooked">
          {cooked.map((recipe) => (
            <span key={recipe.recipe_id} className="chip include">
              {titleCase(recipe.name)}
              <button
                className="remove"
                aria-label={`Remove ${recipe.name}`}
                onClick={() => setCooked((c) => c.filter((r) => r.recipe_id !== recipe.recipe_id))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {response && (
        <p className="strategy">
          {response.strategy === 'personalized'
            ? `Personalized from ${response.used_history.length} recipe(s) (iALS fold-in, no retraining).`
            : 'Most popular recipes: with no history, nothing beats popularity. Add recipes you have cooked to personalize.'}
          {selected.length > 0 &&
            ` Restrictions (${selected.map(restrictionLabel).join(', ')}) are a hard filter.`}
        </p>
      )}
      <div className="grid">
        {response?.results.map((recipe) => (
          <RecipeCard key={recipe.recipe_id} recipe={recipe} onOpen={onOpen} />
        ))}
      </div>
    </section>
  )
}

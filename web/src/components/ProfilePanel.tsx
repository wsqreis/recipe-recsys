import { useEffect, useState } from 'react'
import { api, type RecipeCard as Recipe } from '../api'
import { restrictionLabel, titleCase } from '../text'

export interface Profile {
  restrictions: string[]
  cooked: Recipe[]
}

interface Props {
  profile: Profile
  onChange: (profile: Profile) => void
}

/** Dietary restrictions and cooked recipes, shared by the "For you" and "Week" tabs. */
export function ProfilePanel({ profile, onChange }: Props) {
  const [options, setOptions] = useState<string[]>([])
  const [lookup, setLookup] = useState('')
  const [matches, setMatches] = useState<Recipe[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.restrictions().then(setOptions, (e: Error) => setError(e.message))
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

  function toggle(name: string) {
    const { restrictions } = profile
    onChange({
      ...profile,
      restrictions: restrictions.includes(name)
        ? restrictions.filter((x) => x !== name)
        : [...restrictions, name],
    })
  }

  function add(recipe: Recipe) {
    if (!profile.cooked.some((r) => r.recipe_id === recipe.recipe_id)) {
      onChange({ ...profile, cooked: [...profile.cooked, recipe] })
    }
    setLookup('')
  }

  function remove(id: number) {
    onChange({ ...profile, cooked: profile.cooked.filter((r) => r.recipe_id !== id) })
  }

  return (
    <div className="panel">
      <h4>Dietary restrictions</h4>
      <div className="toggles">
        {options.map((name) => (
          <label key={name} className={profile.restrictions.includes(name) ? 'on' : ''}>
            <input
              type="checkbox"
              checked={profile.restrictions.includes(name)}
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
        {profile.cooked.map((recipe) => (
          <span key={recipe.recipe_id} className="chip include">
            {titleCase(recipe.name)}
            <button
              className="remove"
              aria-label={`Remove ${recipe.name}`}
              onClick={() => remove(recipe.recipe_id)}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

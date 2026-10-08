import { useEffect, useState } from 'react'
import { api, type RecipeDetail } from '../api'
import { titleCase } from '../text'

interface Props {
  id: number
  onClose: () => void
}

export function RecipeModal({ id, onClose }: Props) {
  const [recipe, setRecipe] = useState<RecipeDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.recipe(id).then(setRecipe, (e: Error) => setError(e.message))
  }, [id])

  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [onClose])

  return (
    <div className="backdrop" onClick={onClose}>
      <div className="modal" role="dialog" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose} aria-label="Close">
          ×
        </button>
        {error && <p className="error">{error}</p>}
        {!recipe && !error && <p>Loading…</p>}
        {recipe && (
          <>
            <h2>{titleCase(recipe.name)}</h2>
            <p className="meta">
              {recipe.minutes} min
              {recipe.calories !== null && <> · {Math.round(recipe.calories)} kcal</>}
            </p>
            {recipe.description && <p className="description">{recipe.description}</p>}
            <h4>Ingredients</h4>
            <ul>
              {recipe.ingredients.map((ingredient) => (
                <li key={ingredient}>{ingredient}</li>
              ))}
            </ul>
            <h4>Steps</h4>
            <ol>
              {recipe.steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          </>
        )}
      </div>
    </div>
  )
}

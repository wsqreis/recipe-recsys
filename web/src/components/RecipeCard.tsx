import type { RecipeCard as Recipe } from '../api'
import { titleCase } from '../text'

interface Props {
  recipe: Recipe
  onOpen: (id: number) => void
}

export function RecipeCard({ recipe, onOpen }: Props) {
  const shown = recipe.ingredients.slice(0, 6)
  const more = recipe.ingredients.length - shown.length
  return (
    <button className="card" onClick={() => onOpen(recipe.recipe_id)}>
      <h3>{titleCase(recipe.name)}</h3>
      <p className="meta">
        {recipe.minutes} min
        {recipe.calories !== null && <> · {Math.round(recipe.calories)} kcal</>}
        {' · '}
        {recipe.ingredients.length} ingredients
      </p>
      <p className="ingredients">
        {shown.join(', ')}
        {more > 0 && `, +${more} more`}
      </p>
    </button>
  )
}

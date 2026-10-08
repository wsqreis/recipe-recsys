import { useEffect, useState } from 'react'
import { api, type MenuResponse } from '../api'
import { type Profile, ProfilePanel } from './ProfilePanel'
import { RecipeCard } from './RecipeCard'

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

interface Props {
  profile: Profile
  onProfileChange: (profile: Profile) => void
  onOpen: (id: number) => void
}

export function WeekTab({ profile, onProfileChange, onOpen }: Props) {
  const [variety, setVariety] = useState(0.5)
  const [reuse, setReuse] = useState(0.3)
  const [maxCalories, setMaxCalories] = useState<number | null>(null)
  const [maxMinutes, setMaxMinutes] = useState<number | null>(null)
  const [mainDishes, setMainDishes] = useState(true)
  const [menu, setMenu] = useState<MenuResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Re-plan when anything changes; sliders are debounced so dragging stays smooth.
  useEffect(() => {
    const timer = setTimeout(() => {
      api
        .menu({
          cooked: profile.cooked.map((r) => r.recipe_id),
          restrictions: profile.restrictions,
          variety,
          reuse,
          max_calories: maxCalories,
          max_minutes: maxMinutes,
          main_dishes_only: mainDishes,
        })
        .then(setMenu, (e: Error) => setError(e.message))
    }, 200)
    return () => clearTimeout(timer)
  }, [profile, variety, reuse, maxCalories, maxMinutes, mainDishes])

  return (
    <section className="week">
      <ProfilePanel profile={profile} onChange={onProfileChange} />

      <div className="panel controls">
        <label>
          Variety <strong>{variety.toFixed(2)}</strong>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={variety}
            onChange={(e) => setVariety(Number(e.target.value))}
          />
          <small>Penalizes dishes similar to ones already in the week.</small>
        </label>
        <label>
          Ingredient reuse <strong>{reuse.toFixed(2)}</strong>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={reuse}
            onChange={(e) => setReuse(Number(e.target.value))}
          />
          <small>Rewards recipes whose ingredients are already on the list.</small>
        </label>
        <label>
          Max calories per meal
          <select
            value={maxCalories ?? ''}
            onChange={(e) => setMaxCalories(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">no limit</option>
            {[400, 600, 800].map((v) => (
              <option key={v} value={v}>
                {v} kcal
              </option>
            ))}
          </select>
        </label>
        <label>
          Max time per meal
          <select
            value={maxMinutes ?? ''}
            onChange={(e) => setMaxMinutes(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">no limit</option>
            {[20, 30, 45, 60].map((v) => (
              <option key={v} value={v}>
                {v} min
              </option>
            ))}
          </select>
        </label>
        <label className="inline">
          <input
            type="checkbox"
            checked={mainDishes}
            onChange={(e) => setMainDishes(e.target.checked)}
          />
          Main dishes only
          <small>Recipes tagged as main dishes on Food.com.</small>
        </label>
      </div>

      {error && <p className="error">{error}</p>}
      {menu && (
        <>
          <p className="strategy">
            {menu.strategy === 'personalized'
              ? `Planned from your ${menu.used_history.length} recipe(s).`
              : 'Planned from popular recipes (add recipes you have cooked to personalize).'}{' '}
            {menu.mean_calories !== null && `Average ${Math.round(menu.mean_calories)} kcal per meal. `}
            Similarity within the week: {menu.mean_similarity.toFixed(2)} (lower is more varied).
          </p>
          {menu.days.length === 0 && <p className="empty">No recipe satisfies every constraint.</p>}
          <div className="grid">
            {menu.days.map((recipe, i) => (
              <div key={recipe.recipe_id} className="day">
                <span className="day-name">{DAYS[i % 7]}</span>
                <RecipeCard recipe={recipe} onOpen={onOpen} />
              </div>
            ))}
          </div>
          {menu.shopping_list.length > 0 && (
            <div className="panel shopping">
              <h4>Shopping list ({menu.shopping_list.length} items, pantry staples left out)</h4>
              <ul>
                {menu.shopping_list.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  )
}

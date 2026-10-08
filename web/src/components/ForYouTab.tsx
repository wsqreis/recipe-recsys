import { useEffect, useState } from 'react'
import { api, type RecommendResponse } from '../api'
import { restrictionLabel } from '../text'
import { type Profile, ProfilePanel } from './ProfilePanel'
import { RecipeCard } from './RecipeCard'

interface Props {
  profile: Profile
  onProfileChange: (profile: Profile) => void
  onOpen: (id: number) => void
}

export function ForYouTab({ profile, onProfileChange, onOpen }: Props) {
  const [response, setResponse] = useState<RecommendResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Recommendations refresh whenever the history or the restrictions change.
  useEffect(() => {
    api
      .recommend(
        profile.cooked.map((r) => r.recipe_id),
        profile.restrictions,
      )
      .then(setResponse, (e: Error) => setError(e.message))
  }, [profile])

  return (
    <section className="for-you">
      <ProfilePanel profile={profile} onChange={onProfileChange} />
      {error && <p className="error">{error}</p>}
      {response && (
        <p className="strategy">
          {response.strategy === 'personalized'
            ? `Personalized from ${response.used_history.length} recipe(s) (iALS fold-in, no retraining).`
            : 'Most popular recipes: with no history, nothing beats popularity. Add recipes you have cooked to personalize.'}
          {profile.restrictions.length > 0 &&
            ` Restrictions (${profile.restrictions.map(restrictionLabel).join(', ')}) are a hard filter.`}
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

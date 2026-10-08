import { useState } from 'react'
import { ForYouTab } from './components/ForYouTab'
import type { Profile } from './components/ProfilePanel'
import { RecipeModal } from './components/RecipeModal'
import { SearchTab } from './components/SearchTab'
import { WeekTab } from './components/WeekTab'

type Tab = 'search' | 'for-you' | 'week'

const TABS: [Tab, string][] = [
  ['search', 'Search'],
  ['for-you', 'For you'],
  ['week', 'Week'],
]

export default function App() {
  const [tab, setTab] = useState<Tab>('search')
  const [openId, setOpenId] = useState<number | null>(null)
  // One profile (restrictions + cooked recipes) shared by "For you" and "Week".
  const [profile, setProfile] = useState<Profile>({ restrictions: [], cooked: [] })

  return (
    <div className="app">
      <header>
        <h1>recipe-recsys</h1>
        <p>
          231k Food.com recipes. Dietary restrictions are a hard filter applied after ranking, never
          a suggestion.
        </p>
        <nav>
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {/* Tabs stay mounted so switching keeps their state. */}
        <div hidden={tab !== 'search'}>
          <SearchTab onOpen={setOpenId} />
        </div>
        <div hidden={tab !== 'for-you'}>
          <ForYouTab profile={profile} onProfileChange={setProfile} onOpen={setOpenId} />
        </div>
        <div hidden={tab !== 'week'}>
          <WeekTab profile={profile} onProfileChange={setProfile} onOpen={setOpenId} />
        </div>
      </main>
      {openId !== null && <RecipeModal id={openId} onClose={() => setOpenId(null)} />}
    </div>
  )
}

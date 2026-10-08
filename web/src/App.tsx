import { useState } from 'react'
import { ForYouTab } from './components/ForYouTab'
import { RecipeModal } from './components/RecipeModal'
import { SearchTab } from './components/SearchTab'

type Tab = 'search' | 'for-you'

export default function App() {
  const [tab, setTab] = useState<Tab>('search')
  const [openId, setOpenId] = useState<number | null>(null)

  return (
    <div className="app">
      <header>
        <h1>recipe-recsys</h1>
        <p>
          231k Food.com recipes. Dietary restrictions are a hard filter applied after ranking, never
          a suggestion.
        </p>
        <nav>
          <button className={tab === 'search' ? 'active' : ''} onClick={() => setTab('search')}>
            Search
          </button>
          <button className={tab === 'for-you' ? 'active' : ''} onClick={() => setTab('for-you')}>
            For you
          </button>
        </nav>
      </header>
      <main>
        {/* Both tabs stay mounted so switching keeps their state. */}
        <div hidden={tab !== 'search'}>
          <SearchTab onOpen={setOpenId} />
        </div>
        <div hidden={tab !== 'for-you'}>
          <ForYouTab onOpen={setOpenId} />
        </div>
      </main>
      {openId !== null && <RecipeModal id={openId} onClose={() => setOpenId(null)} />}
    </div>
  )
}

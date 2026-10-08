// Typed client for the recipe-recsys API (src/recipe_recsys/api.py).

export interface RecipeCard {
  recipe_id: number
  name: string
  minutes: number
  ingredients: string[]
  calories: number | null
  score: number | null
}

export interface RecipeDetail extends RecipeCard {
  description: string | null
  steps: string[]
  tags: string[]
}

export interface ParsedQuery {
  restrictions: string[]
  include: string[]
  exclude: string[]
  max_minutes: number | null
  query: string
}

export interface SearchResponse {
  parsed: ParsedQuery
  keyword_restrictions: string[]
  llm_used: boolean
  llm_error: string | null
  results: RecipeCard[]
}

export interface RecommendResponse {
  strategy: 'personalized' | 'popularity'
  used_history: number[]
  ignored_history: number[]
  results: RecipeCard[]
}

export interface MenuRequest {
  cooked: number[]
  restrictions: string[]
  variety: number
  reuse: number
  max_calories: number | null
  max_minutes: number | null
  main_dishes_only: boolean
}

export interface MenuResponse {
  strategy: 'personalized' | 'popularity'
  used_history: number[]
  ignored_history: number[]
  days: RecipeCard[]
  shopping_list: string[]
  mean_calories: number | null
  mean_similarity: number
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  restrictions: () => request<string[]>('/api/restrictions'),
  search: (query: string, useLlm: boolean) =>
    request<SearchResponse>('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query, use_llm: useLlm, n: 12 }),
    }),
  recommend: (cooked: number[], restrictions: string[]) =>
    request<RecommendResponse>('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({ cooked, restrictions, n: 12 }),
    }),
  menu: (body: MenuRequest) =>
    request<MenuResponse>('/api/menu', { method: 'POST', body: JSON.stringify(body) }),
  findRecipes: (q: string) =>
    request<RecipeCard[]>(`/api/recipes?q=${encodeURIComponent(q)}&n=8`),
  recipe: (id: number) => request<RecipeDetail>(`/api/recipes/${id}`),
}

// Restriction names in the API -> what a person reads.
const RESTRICTION_LABELS: Record<string, string> = {
  vegetarian: 'vegetarian',
  vegan: 'vegan',
  lactose: 'dairy-free',
  egg: 'egg-free',
  gluten: 'gluten-free',
  nuts: 'nut-free',
}

export function restrictionLabel(name: string): string {
  return RESTRICTION_LABELS[name] ?? name
}

// Food.com names are lowercase with collapsed punctuation ("quick   easy chicken").
export function titleCase(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ')
}

import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(() => {
    // Load from localStorage or default to false (light mode)
    const saved = localStorage.getItem('dark-mode-preference')
    console.log('ThemeProvider: Initial isDark from localStorage:', saved === 'true')
    return saved === 'true'
  })

  useEffect(() => {
    // Update dark mode on document
    const root = document.documentElement
    console.log('ThemeProvider: Dark mode =', isDark)
    if (isDark) {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    // Save preference
    localStorage.setItem('dark-mode-preference', isDark.toString())
  }, [isDark])

  const toggleDark = () => {
    console.log('ThemeProvider: toggle called, isDark before:', isDark)
    setIsDark(prev => {
      console.log('ThemeProvider: setting isDark from', prev, 'to', !prev)
      return !prev
    })
  }

  return (
    <ThemeContext.Provider value={{ isDark, toggleDark }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}
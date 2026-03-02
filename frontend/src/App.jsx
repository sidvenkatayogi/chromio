import { Routes, Route } from 'react-router-dom'
import PaletteGenerator from './components/PaletteGenerator'
import AuthPage from './components/AuthPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<PaletteGenerator />} />
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/signup" element={<AuthPage mode="signup" />} />
    </Routes>
  )
}

export default App
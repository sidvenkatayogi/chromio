import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { authAPI } from '../utils/api'

function AuthPage({ mode = 'login' }) {
  const navigate = useNavigate()
  const isLogin = mode === 'login'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    // Form validation
    if (!email || !password) {
      setError('Please fill in all required fields')
      setLoading(false)
      return
    }

    if (!isLogin && (!confirmPassword)) {
      setError('Please fill in all fields')
      setLoading(false)
      return
    }

    if (!isLogin && password !== confirmPassword) {
      setError('Passwords do not match')
      setLoading(false)
      return
    }

    try {
      const response = isLogin
        ? await authAPI.login(email, password)
        : await authAPI.signup(email, password)

      // Store access token in localStorage
      const accessToken = response.data.tokens.access_token
      if (accessToken) {
        localStorage.setItem('access_token', accessToken)
      }

      // Navigate to home on success
      navigate('/')
    } catch (err) {
      let errorMessage = 'Something has gone wrong. Please try again later'

      if (err?.response?.data?.message)
        errorMessage = err.response.data.message

      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleAuth = () => {
    // TODO: Implement Google OAuth
    console.log('Google auth clicked')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-950 to-blue-950 px-4">
      <div className="w-full max-w-md">
        <div className="bg-gray-800 rounded-2xl shadow-xl p-8">
          {/* Header */}
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-white mb-2">
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h1>
            <p className="text-gray-300">
              {isLogin ? 'Sign in to continue' : 'Join Chromio today'}
            </p>
          </div>

          {/* Toggle Login/Signup */}
          <div className="flex mb-6 bg-gray-700 rounded-lg p-1">
            <Link
              to="/login"
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all text-center ${
                isLogin
                  ? 'bg-gray-600 text-white shadow-md'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Login
            </Link>
            <Link
              to="/signup"
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all text-center ${
                !isLogin
                  ? 'bg-gray-600 text-white shadow-md'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Sign Up
            </Link>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-200 mb-1">
                Email
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 border border-gray-600 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
                placeholder="Enter your email"
                required
              />
            </div>

            {/* Password field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-200 mb-1">
                Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 border border-gray-600 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
                placeholder={
                  isLogin ? 'Enter your password' : 'Create a password'
                }
                required
                minLength={6}
              />
              {!isLogin && (
                <p className="text-xs text-gray-400 mt-1">
                  Must be at least 6 characters
                </p>
              )}
            </div>

            {/* Confirm password - only for signup */}
            {!isLogin && (
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-200 mb-1">
                  Confirm Password
                </label>
                <input
                  type="password"
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-600 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
                  placeholder="Confirm your password"
                  required
                  minLength={6}
                />
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Create Account'}
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center my-6">
            <div className="flex-1 border-t border-gray-600"></div>
            <span className="px-4 text-sm text-gray-400">or</span>
            <div className="flex-1 border-t border-gray-600"></div>
          </div>

          {/* Google OAuth button */}
          <button
            onClick={handleGoogleAuth}
            className="w-full flex items-center justify-center gap-3 bg-gray-700 border border-gray-600 hover:border-gray-500 text-white font-medium py-3 px-4 rounded-lg transition-all hover:bg-gray-600"
          >
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" className="w-5 h-5">
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
            </svg>
            Continue with Google
          </button>

          {/* Footer text */}
          <p className="text-center text-sm text-gray-300 mt-6">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <Link
              to={isLogin ? '/signup' : '/login'}
              className="text-purple-400 hover:text-purple-300 font-medium"
            >
              {isLogin ? 'Sign up' : 'Sign in'}
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}

export default AuthPage
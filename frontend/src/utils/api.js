import axios from 'axios'

// Create axios instance with default config
const api = axios.create({
  baseURL: import.meta.env.VITE_SERVER_URL,
  headers: {
    'Content-Type': 'application/json',
  }
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const accessToken = localStorage.getItem('access_token')
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      // Could implement refresh logic here
      localStorage.removeItem('access_token')

      // redirect to login if not already on login or signup
      if (typeof window !== 'undefined' && 
          (window.location.pathname != '/login' && window.location.pathname != '/signup' )) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api

export const authAPI = {
  login: (email, password) =>
    api.post('/api/v1/auth/signin', { email, password }),

  signup: (email, password) =>
    api.post('/api/v1/auth/signup', { email, password }),
}
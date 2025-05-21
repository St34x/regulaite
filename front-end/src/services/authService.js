import axios from "axios";
import { jwtDecode } from "jwt-decode";

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8090';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Authentication service functions
const authService = {
  // Register a new user
  register: async (userData) => {
    try {
      const response = await api.post('/auth/register', userData);
      return response.data;
    } catch (error) {
      throw error.response?.data || { detail: 'Registration failed' };
    }
  },

  // Login user
  login: async (email, password) => {
    try {
      // For OAuth2 password flow, we need to send data as form data
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await api.post('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // Store tokens in local storage
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('refreshToken', response.data.refresh_token);
        localStorage.setItem('tokenType', response.data.token_type);
        
        // Store decoded user data
        const userData = jwtDecode(response.data.access_token);
        if (userData) {
          localStorage.setItem('userData', JSON.stringify(userData));
        }
      }

      return response.data;
    } catch (error) {
      throw error.response?.data || { detail: 'Login failed' };
    }
  },

  // Logout user
  logout: async () => {
    try {
      const refreshToken = localStorage.getItem('refreshToken');
      if (refreshToken) {
        await api.post('/auth/logout', { refresh_token: refreshToken });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless of API success
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('tokenType');
      localStorage.removeItem('userData');
    }
  },

  // Get current user profile from API
  getCurrentUser: async () => {
    try {
      const response = await api.get('/auth/me');
      return response.data;
    } catch (error) {
      throw error.response?.data || { detail: 'Failed to get user profile' };
    }
  },
  
  // Get current user data from locally stored token
  getCurrentUserData: () => {
    try {
      const userData = localStorage.getItem('userData');
      if (userData) {
        return JSON.parse(userData);
      }
      
      // If userData is not stored, try to decode from token
      const token = localStorage.getItem('token');
      if (token) {
        try {
          // Use jwtDecode to decode the token
          const data = jwtDecode(token);
          
          // Extract user_id from the token using common JWT fields
          // Try multiple fields that could contain the user ID
          const userId = data.user_id || data.sub || data.id || data.userId || null;
          
          // Create a standardized user data object
          const standardizedData = {
            ...data,
            user_id: userId
          };
          
          // Store the standardized data
          localStorage.setItem('userData', JSON.stringify(standardizedData));
          return standardizedData;
        } catch (e) {
          console.error('Error decoding token:', e);
        }
      }
      
      return null;
    } catch (e) {
      console.error('Error getting user data:', e);
      return null;
    }
  },

  // Refresh access token
  refreshToken: async () => {
    try {
      const refreshToken = localStorage.getItem('refreshToken');
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await api.post('/auth/refresh', { refresh_token: refreshToken });
      
      // Update stored tokens
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('refreshToken', response.data.refresh_token);
      }
      
      return response.data;
    } catch (error) {
      // If refresh fails, logout
      authService.logout();
      throw error.response?.data || { detail: 'Token refresh failed' };
    }
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },

  // Get auth token
  getToken: () => {
    return localStorage.getItem('token');
  }
};

export default authService; 

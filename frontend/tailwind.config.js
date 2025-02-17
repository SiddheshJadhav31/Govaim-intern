/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1a365d',
          light: '#2a4365',
          dark: '#102a4c'
        },
        secondary: {
          DEFAULT: '#2c7a7b',
          light: '#319795',
          dark: '#285e61'
        },
        accent: {
          DEFAULT: '#d4af37',
          light: '#ecc94b',
          dark: '#b7791f'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif']
      }
    },
  },
  plugins: [],
}
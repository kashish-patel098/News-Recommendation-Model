/** @type {import('next').NextConfig} */
const nextConfig = {
  // Expose NEXT_PUBLIC_API_URL to browser code.
  // Set this env var in Vercel / Railway / docker-compose to point at your API.
  // Defaults to localhost for local dev.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};

export default nextConfig;

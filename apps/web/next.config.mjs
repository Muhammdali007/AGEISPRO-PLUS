/** @type {import('next').NextConfig} */
const apiProxyTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";

const nextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiProxyTarget}/:path*`
      }
    ];
  },
  async redirects() {
    return [
      {
        source: "/overview",
        destination: "/dashboard",
        permanent: false
      },
      {
        source: "/dashboard/overview",
        destination: "/dashboard",
        permanent: false
      },
      {
        source: "/cameras/:path*",
        destination: "/dashboard/cameras/:path*",
        permanent: false
      },
      {
        source: "/incidents/:path*",
        destination: "/dashboard/incidents/:path*",
        permanent: false
      },
      {
        source: "/persons/:path*",
        destination: "/dashboard/persons/:path*",
        permanent: false
      },
      {
        source: "/analytics/:path*",
        destination: "/dashboard/analytics/:path*",
        permanent: false
      },
      {
        source: "/users/:path*",
        destination: "/dashboard/users/:path*",
        permanent: false
      },
      {
        source: "/user-management/:path*",
        destination: "/dashboard/users/:path*",
        permanent: false
      },
      {
        source: "/dashboard/user-management/:path*",
        destination: "/dashboard/users/:path*",
        permanent: false
      }
    ];
  }
};

export default nextConfig;

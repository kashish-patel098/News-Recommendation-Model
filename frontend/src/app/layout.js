import './globals.css';

export const metadata = {
  title: 'Nexus // AI News Intelligence',
  description: 'Premium AI-driven news recommendation engine',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

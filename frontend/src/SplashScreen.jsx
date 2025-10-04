import { useEffect } from 'react';

function SplashScreen({ onComplete }) {
  useEffect(() => {
    // Set a timeout to call the completion handler after 3000ms
    const timer = setTimeout(onComplete, 3000);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      // Deep blue background for professional look
      background: 'linear-gradient(135deg, #004d99 0%, #002d59 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      fontFamily: "'Segoe UI', Arial, sans-serif"
    }}>
      <div style={{
        animation: 'fadeInScale 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
        marginBottom: '2rem',
        padding: '0',
      }}>
        <img
          src="https://tse4.mm.bing.net/th/id/OIP.a4U2HKvF_-rYHIu__oOQXgHaES?pid=Api&P=0&h=180" // Placeholder SAP-like image
          alt="SAP Logo"
          style={{
            width: '220px', // Increased size for prominence
            height: 'auto',
            display: 'block',
            // REMOVED FILTER: Now the image's original blue/white will be displayed
            animation: 'pulse 2s ease-in-out infinite'
          }}
        />
      </div>
      
      <div style={{
        width: '240px',
        height: '6px',
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        borderRadius: '3px',
        overflow: 'hidden',
        marginTop: '2.5rem',
        marginBottom: '1rem'
      }}>
        <div style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#87cefa', /* Light blue loader color */
          animation: 'loading 2s ease-in-out infinite'
        }}></div>
      </div>
      <p style={{
        color: 'white',
        fontSize: '1.25rem',
        fontWeight: '600',
        animation: 'fadeIn 1s ease-out'
      }}>
        Procurement Analytics Platform
      </p>
      <p style={{
        color: 'rgba(255, 255, 255, 0.8)',
        fontSize: '0.85rem',
        marginTop: '0.5rem',
        animation: 'fadeIn 1.5s ease-out'
      }}>
        Loading secure services...
      </p>

      <style>{`
        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.6);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.03);
          }
        }
        @keyframes loading {
          0% {
            transform: translateX(-100%) scaleX(0.1);
          }
          50% {
            transform: translateX(0%) scaleX(0.8);
          }
          100% {
            transform: translateX(100%) scaleX(0.1);
          }
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

export default SplashScreen;

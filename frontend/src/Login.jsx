import { useState } from 'react';
import { useAuth } from './AuthContext'; // CORRECTED IMPORT PATH
import { doc, setDoc, serverTimestamp } from 'firebase/firestore'; 
import { db } from './firebaseConfig'; // CORRECTED IMPORT PATH

function Login() {
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);
  const [loginError, setLoginError] = useState(null); // State for error messages

  /**
   * Saves or updates the user's profile in the 'users' collection.
   * Uses the user.uid as the document ID and merge:true to prevent overwriting
   * and handle both signup and login in one function.
   */
  const saveUserProfile = async (user) => {
    // The 'users' collection stores one document per user, using their UID as the doc ID.
    const userRef = doc(db, 'users', user.uid);
    
    await setDoc(userRef, {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName,
      photoURL: user.photoURL,
      // Use serverTimestamp for fields that should be updated on every login
      lastLogin: serverTimestamp(),
      // The createdAt time is either fetched from auth metadata or set on first write
      createdAt: user.metadata.creationTime ? new Date(user.metadata.creationTime).toISOString() : serverTimestamp()
    }, { merge: true }); // CRITICAL: merge: true ensures we don't overwrite existing data
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    setLoginError(null);

    try {
      const result = await login(); // Authenticates user via Google
      const user = result.user;

      // 1. Store/Update user details in Firestore
      await saveUserProfile(user);

    } catch (error) {
      console.error('Login failed:', error);
      // Check if the error is due to the user closing the popup
      if (error.code !== 'auth/popup-closed-by-user' && error.code !== 'auth/cancelled-popup-request') {
          setLoginError('Authentication failed. Please check your network connection and try again.');
      }
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #e8f4ff 0%, #d1e9ff 50%, #b3deff 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '1.5rem',
        boxShadow: '0 25px 50px -12px rgba(0, 112, 243, 0.25)',
        border: '1px solid #b3deff',
        padding: '3rem',
        maxWidth: '450px',
        width: '90%',
        textAlign: 'center'
      }}>
        <img
          src="https://tse4.mm.bing.net/th/id/OIP.a4U2HKvF_-rYHIu__oOQXgHaES?pid=Api&P=0&h=180"
          alt="SAP Logo"
          style={{ width: '120px', marginBottom: '2rem' }}
        />

        <h1 style={{
          fontSize: '2rem',
          fontWeight: 'bold',
          background: 'linear-gradient(135deg, #0070f3, #003d82)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '0.5rem'
        }}>
          Welcome Back
        </h1>

        <p style={{
          color: '#64748b',
          marginBottom: '2rem',
          fontSize: '0.95rem'
        }}>
          Sign in to access your Procurement Analytics Dashboard
        </p>

        {loginError && (
          <div style={{
            backgroundColor: '#fee2e2',
            color: '#991b1b',
            padding: '0.75rem',
            borderRadius: '0.5rem',
            marginBottom: '1rem',
            fontSize: '0.9rem',
            border: '1px solid #fca5a5'
          }}>
            {loginError}
          </div>
        )}

        <button
          onClick={handleGoogleLogin}
          disabled={loading}
          style={{
            width: '100%',
            padding: '1rem',
            background: loading ? '#94a3b8' : 'linear-gradient(135deg, #0070f3, #003d82)',
            color: 'white',
            border: 'none',
            borderRadius: '0.75rem',
            fontSize: '1rem',
            fontWeight: '600',
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            transition: 'all 0.2s',
            boxShadow: loading ? 'none' : '0 4px 12px rgba(0, 112, 243, 0.4)',
            marginBottom: '1rem'
          }}
          onMouseEnter={(e) => {
            if (!loading) e.target.style.transform = 'translateY(-2px)';
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'translateY(0)';
          }}
        >
          {loading ? (
            <>
              <div style={{
                width: '20px',
                height: '20px',
                border: '3px solid rgba(255,255,255,0.3)',
                borderTopColor: 'white',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
              Signing in...
            </>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Sign in with Google
            </>
          )}
        </button>

        <div style={{
          marginTop: '2rem',
          padding: '1rem',
          backgroundColor: '#f8fafc',
          borderRadius: '0.75rem',
          border: '1px solid #e2e8f0'
        }}>
          <p style={{
            fontSize: '0.75rem',
            color: '#64748b',
            margin: 0,
            lineHeight: '1.5'
          }}>
            Secure authentication powered by Firebase
          </p>
        </div>
      </div>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default Login;

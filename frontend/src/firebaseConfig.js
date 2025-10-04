import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// IMPORTANT: Replace these placeholder values with your actual Firebase project configuration.
const firebaseConfig = {
  apiKey: "AIzaSyB6XQiSm0X8X3-gBylZ_pc4Pudf83qEJec",
  authDomain: "ai-based-spend-classification.firebaseapp.com",
  projectId: "ai-based-spend-classification",
  storageBucket: "ai-based-spend-classification.firebasestorage.app",
  messagingSenderId: "964062022601",
  appId: "1:964062022601:web:bf732afae55066a691b618",
  measurementId: "G-W5SWMFVNJB"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize services and export them
export const auth = getAuth(app);
export const db = getFirestore(app); // <-- Used by firebaseService.js

export default app;

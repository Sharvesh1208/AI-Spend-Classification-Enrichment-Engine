import {
  collection,
  addDoc,
  query,
  where,
  getDocs,
  deleteDoc,
  doc,
  orderBy,
  Timestamp, // <-- Use native Firestore Timestamp
} from 'firebase/firestore';
import { db } from './firebaseConfig';

// --- Firestore Collection References ---
// Using 'classifications' for primary data and 'activityLogs' for auditing
const CLASSIFICATIONS_COLLECTION = 'classifications';
const ACTIVITY_COLLECTION = 'activityLogs';

/**
 * Saves a set of classification results to Firestore under the user's ID.
 *
 * @param {string} userId - The unique ID of the current user.
 * @param {Array<Object>} results - The classification results array from the API.
 * @param {string} fileName - The name of the file or a descriptor for the text input.
 * @returns {Promise<string>} The ID of the newly created classification document.
 */
export const saveClassificationResults = async (userId, results, fileName) => {
  try {
    const classificationData = {
      userId: userId,
      fileName: fileName,
      results: results,
      totalRecords: results.length,
      createdAt: Timestamp.now(), // Store as native Firestore Timestamp
      updatedAt: Timestamp.now(),
    };

    const docRef = await addDoc(collection(db, CLASSIFICATIONS_COLLECTION), classificationData);
    console.log('Classification saved with ID: ', docRef.id);
    return docRef.id;
  } catch (error) {
    console.error('Error saving classification: ', error);
    throw error;
  }
};

/**
 * Retrieves all saved classification history for a specific user, ordered by creation date.
 *
 * @param {string} userId - The unique ID of the current user.
 * @returns {Promise<Array<Object>>} An array of classification history objects.
 */
export const getUserClassifications = async (userId) => {
  try {
    const q = query(
      collection(db, CLASSIFICATIONS_COLLECTION),
      where('userId', '==', userId),
      orderBy('createdAt', 'desc') // Order by the native Timestamp
    );

    const querySnapshot = await getDocs(q);
    const classifications = [];

    querySnapshot.forEach((doc) => {
      // Convert the Firestore Timestamp back to a readable ISO string for display in React
      classifications.push({
        id: doc.id,
        ...doc.data(),
        createdAt: doc.data().createdAt?.toDate().toISOString()
      });
    });

    return classifications;
  } catch (error) {
    console.error('Error getting classifications: ', error);
    throw error;
  }
};

/**
 * Deletes a specific classification history document.
 *
 * @param {string} classificationId - The ID of the classification document to delete.
 * @returns {Promise<void>}
 */
export const deleteClassification = async (classificationId) => {
  try {
    await deleteDoc(doc(db, CLASSIFICATIONS_COLLECTION, classificationId));
    console.log('Classification deleted successfully');
  } catch (error) {
    console.error('Error deleting classification: ', error);
    throw error;
  }
};

/**
 * Logs user activity to a separate collection for auditing.
 *
 * @param {string} userId - The unique ID of the current user.
 * @param {string} activityType - A string describing the type of activity (e.g., 'login', 'file_upload').
 * @param {string} description - A detailed description of the activity.
 * @returns {Promise<void>}
 */
export const logUserActivity = async (userId, activityType, description) => {
  try {
    await addDoc(collection(db, ACTIVITY_COLLECTION), {
      userId: userId,
      type: activityType,
      description: description,
      timestamp: Timestamp.now() // Store as native Timestamp
    });
  } catch (error) {
    console.error('Error logging activity: ', error);
    // Logging failures should generally not interrupt the main application flow
  }
};

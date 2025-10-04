import { useState, useEffect, useRef, useCallback } from "react";
import * as Chart from 'chart.js';
import { useAuth } from './AuthContext'; // Adjusted import path
import UserAvatar from './UserAvatar'; // Adjusted import path
import {
  saveClassificationResults,
  getUserClassifications,
  deleteClassification,
  logUserActivity
} from './firebaseService'; // Adjusted import path

function Dashboard() {
  const { currentUser, logout } = useAuth();
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [inputMode, setInputMode] = useState("csv"); // "csv" or "text"
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const [showCharts, setShowCharts] = useState(true);
  const [expandedRows, setExpandedRows] = useState(new Set());
  const [filterMode, setFilterMode] = useState("all");
  const [filterVendor, setFilterVendor] = useState("all");
  const [filterCategory, setFilterCategory] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [savedClassifications, setSavedClassifications] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [savingToFirebase, setSavingToFirebase] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  // State for Custom Confirmation/Alert Modal (Replacing alert() and window.confirm())
  const [confirmation, setConfirmation] = useState(null);

  const pieChartRef = useRef(null);
  const barChartRef = useRef(null);
  const lineChartRef = useRef(null);
  const trendChartRef = useRef(null);
  const spendPieRef = useRef(null);
  const categoryBarRef = useRef(null);

  const pieChartInstance = useRef(null);
  const barChartInstance = useRef(null);
  const lineChartInstance = useRef(null);
  const trendChartInstance = useRef(null);
  const spendPieInstance = useRef(null);
  const categoryBarInstance = useRef(null);

  const checkApiHealth = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/health");
      const data = await res.json();
      setApiStatus(data);
    } catch (err) {
      setApiStatus({ status: "disconnected", error: "Cannot connect to API" });
    }
  };

  useEffect(() => {
    checkApiHealth();
    // Ensure currentUser and its uid are available before calling loadUserClassifications and logUserActivity
    if (currentUser?.uid) {
      loadUserClassifications();
      logUserActivity(currentUser.uid, 'dashboard_view', 'User viewed dashboard');
    }
  }, [currentUser]); // Added currentUser to dependency array

  const loadUserClassifications = async () => {
    // Check for UID before attempting to load data
    if (!currentUser?.uid) return;
    try {
      const classifications = await getUserClassifications(currentUser.uid);
      setSavedClassifications(classifications);
    } catch (error) {
      console.error('Error loading classifications:', error);
      // Using custom state to display alert message
      setConfirmation({
        title: "History Error",
        message: "Failed to load history. Please check the console for details.",
        onConfirm: () => setConfirmation(null)
      });
    }
  };

  const handleUpload = async () => {
    if (inputMode === "csv" && !file) {
      // Replaced alert()
      setConfirmation({
        title: "Missing File",
        message: "Please upload a CSV file before classifying.",
        onConfirm: () => setConfirmation(null)
      });
      return;
    }

    if (inputMode === "text" && !textInput.trim()) {
      // Replaced alert()
      setConfirmation({
        title: "Missing Text",
        message: "Please enter text for classification.",
        onConfirm: () => setConfirmation(null)
      });
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let res;
      let fileName = "text-input";

      if (inputMode === "csv") {
        const formData = new FormData();
        formData.append("file", file);
        fileName = file.name;

        res = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          body: formData,
        });
      } else {
        // Text input mode - send as JSON
        res = await fetch("http://127.0.0.1:8000/predict-text", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: textInput }),
        });
        fileName = `text-input-${new Date().toISOString().split('T')[0]}`;
      }

      const data = await res.json();

      if (data.error) {
        setError(data.error);
        setResults([]);
      } else if (data.results) {
        setResults(data.results);
        setError(null);
        setActiveTab("analytics");

        setSavingToFirebase(true);
        try {
          // currentUser is guaranteed to exist here if the user reached the dashboard
          await saveClassificationResults(currentUser.uid, data.results, fileName);
          await logUserActivity(currentUser.uid, inputMode === "csv" ? 'file_upload' : 'text_classification',
            inputMode === "csv" ? `Uploaded ${fileName}` : 'Classified text input');
          await loadUserClassifications();
        } catch (fbError) {
          console.error('Firebase save error:', fbError);
        } finally {
          setSavingToFirebase(false);
        }
      } else if (Array.isArray(data)) {
        setResults(data);
        setError(null);
        setActiveTab("analytics");
      } else {
        setError("Unexpected response format from API");
        setResults([]);
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logUserActivity(currentUser.uid, 'logout', 'User logged out');
      await logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleDeleteClassification = (classificationId) => {
    // Replaced window.confirm() with custom modal
    setConfirmation({
      title: "Confirm Deletion",
      message: "Are you sure you want to permanently delete this classification record?",
      onConfirm: async () => {
        try {
          await deleteClassification(classificationId);
          await loadUserClassifications();
          setConfirmation(null);
        } catch (error) {
          console.error('Error deleting classification:', error);
          // Replaced alert()
          setConfirmation({
            title: "Deletion Error",
            message: "Failed to delete classification record.",
            onConfirm: () => setConfirmation(null)
          });
        }
      },
      onCancel: () => setConfirmation(null)
    });
  };

  const loadSavedClassification = (classification) => {
    setResults(classification.results);
    setShowHistory(false);
    setActiveTab("analytics");
    // Ensure charts are created after data is set and tab is active
    setTimeout(() => {
        if (activeTab === "analytics" && classification.results.length > 0) {
            createCharts();
        }
    }, 0);
  };

  const formatAmount = (amount) => {
    if (!amount) return "N/A";
    if (typeof amount === "number") {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(amount);
    }
    return amount;
  };

  const getVendorField = (result) => {
    return result.normalized_vendor || result.predicted_vendor || "N/A";
  };

  const getModeColor = (mode) => {
    switch(mode) {
      case "normalization": return { backgroundColor: "#d1fae5", color: "#065f46", border: "1px solid #10b981" };
      case "missing-data": return { backgroundColor: "#fef3c7", color: "#92400e", border: "1px solid #f59e0b" };
      case "error": return { backgroundColor: "#fee2e2", color: "#991b1b", border: "1px solid #ef4444" };
      default: return { backgroundColor: "#e2e8f0", color: "#475569", border: "1px solid #94a3b8" };
    }
  };

  const getModeIcon = (mode) => {
    switch(mode) {
      case "normalization": return "✓";
      case "missing-data": return "⚠";
      case "error": return "✕";
      default: return "⏱";
    }
  };

  const toggleRow = useCallback((index) => {
    setExpandedRows(prev => {
      const newExpanded = new Set(prev);
      if (newExpanded.has(index)) {
        newExpanded.delete(index);
      } else {
        newExpanded.add(index);
      }
      return newExpanded;
    });
  }, []);

  const filteredResults = results.filter(result => {
    if (filterMode !== "all" && result.mode !== filterMode) return false;
    if (filterVendor !== "all" && getVendorField(result) !== filterVendor) return false;
    if (filterCategory !== "all" && result.predicted_category !== filterCategory) return false;
    if (searchQuery && !result.original_text?.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = filteredResults.length > 0 ? {
    total: filteredResults.length,
    withVendor: filteredResults.filter(r => getVendorField(r) !== "N/A").length,
    totalAmount: filteredResults.reduce((acc, r) => acc + (r.amount || 0), 0),
    avgVendorConfidence: filteredResults.filter(r => r.vendor_confidence).reduce((acc, r) => acc + r.vendor_confidence, 0) / filteredResults.filter(r => r.vendor_confidence).length || 0,
    uniqueVendors: new Set(filteredResults.map(r => getVendorField(r)).filter(v => v !== "N/A")).size,
    uniqueCategories: new Set(filteredResults.map(r => r.predicted_category).filter(c => c)).size
  } : null;

  const uniqueVendors = [...new Set(results.map(r => getVendorField(r)))].filter(v => v !== "N/A").sort();
  const uniqueCategories = [...new Set(results.map(r => r.predicted_category))].filter(c => c).sort();

  const generateSpendByCategory = () => {
    const categorySpend = {};
    filteredResults.forEach(result => {
      const category = result.predicted_category || "Uncategorized";
      categorySpend[category] = (categorySpend[category] || 0) + (result.amount || 0);
    });

    const sorted = Object.entries(categorySpend).sort((a, b) => b[1] - a[1]);
    const colors = ['#0066cc', '#0088ff', '#33aaff', '#66ccff', '#0052a3', '#004080', '#003366', '#005599'];

    return {
      labels: sorted.map(([cat]) => cat),
      datasets: [{
        data: sorted.map(([, amt]) => amt),
        backgroundColor: colors.slice(0, sorted.length),
        borderColor: '#ffffff',
        borderWidth: 2
      }]
    };
  };

  const generateTopVendorsByCategory = () => {
    const categoryVendors = {};

    filteredResults.forEach(result => {
      const category = result.predicted_category || "Other";
      const vendor = getVendorField(result);
      if (vendor !== "N/A") {
        if (!categoryVendors[category]) categoryVendors[category] = {};
        categoryVendors[category][vendor] = (categoryVendors[category][vendor] || 0) + (result.amount || 0);
      }
    });

    const topCategories = Object.entries(categoryVendors)
      .map(([cat, vendors]) => ({
        category: cat,
        totalSpend: Object.values(vendors).reduce((a, b) => a + b, 0)
      }))
      .sort((a, b) => b.totalSpend - a.totalSpend)
      .slice(0, 5);

    return {
      labels: topCategories.map(c => c.category),
      datasets: [{
        label: 'Spend by Category',
        data: topCategories.map(c => c.totalSpend),
        backgroundColor: 'rgba(0, 102, 204, 0.8)',
        borderColor: 'rgba(0, 102, 204, 1)',
        borderWidth: 1,
        borderRadius: 4
      }]
    };
  };

  const createCharts = () => {
    Chart.Chart.register(...Chart.registerables);

    if (spendPieRef.current) {
      if (spendPieInstance.current) spendPieInstance.current.destroy();
      const spendData = generateSpendByCategory();
      spendPieInstance.current = new Chart.Chart(spendPieRef.current, {
        type: 'doughnut',
        data: spendData,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
              labels: {
                usePointStyle: true,
                padding: 15,
                font: { size: 12, family: "'Segoe UI', sans-serif" },
                generateLabels: function(chart) {
                  const data = chart.data;
                  return data.labels.map((label, i) => {
                    const value = data.datasets[0].data[i];
                    const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                    const percentage = ((value / total) * 100).toFixed(1);
                    return {
                      text: `${label}: ${percentage}% ($${(value/1000).toFixed(0)}K)`,
                      fillStyle: data.datasets[0].backgroundColor[i],
                      pointStyle: 'circle'
                    };
                  });
                }
              }
            },
            tooltip: {
              callbacks: {
                label: (context) => {
                  const value = context.parsed;
                  return `$${(value/1000).toFixed(2)}K`;
                }
              }
            }
          }
        }
      });
    }

    if (categoryBarRef.current) {
      if (categoryBarInstance.current) categoryBarInstance.current.destroy();
      categoryBarInstance.current = new Chart.Chart(categoryBarRef.current, {
        type: 'bar',
        data: generateTopVendorsByCategory(),
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (context) => `$${(context.parsed.y/1000).toFixed(0)}K`
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: (value) => `$${(value/1000).toFixed(0)}K`,
                font: { size: 11 }
              },
              grid: { color: '#e2e8f0' }
            },
            x: {
              ticks: { font: { size: 11 } },
              grid: { display: false }
            }
          }
        }
      });
    }

    if (lineChartRef.current) {
      if (lineChartInstance.current) lineChartInstance.current.destroy();
      const confidenceRanges = { '0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0 };
      filteredResults.forEach(result => {
        if (result.vendor_confidence) {
          const conf = result.vendor_confidence * 100;
          if (conf < 20) confidenceRanges['0-20%']++;
          else if (conf < 40) confidenceRanges['20-40%']++;
          else if (conf < 60) confidenceRanges['40-60%']++;
          else if (conf < 80) confidenceRanges['60-80%']++;
          else confidenceRanges['80-100%']++;
        }
      });

      lineChartInstance.current = new Chart.Chart(lineChartRef.current, {
        type: 'line',
        data: {
          labels: Object.keys(confidenceRanges),
          datasets: [{
            label: 'Confidence Distribution',
            data: Object.values(confidenceRanges),
            borderColor: '#0066cc',
            backgroundColor: 'rgba(0, 102, 204, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#0066cc'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { font: { size: 11 } },
              grid: { color: '#e2e8f0' }
            },
            x: {
              ticks: { font: { size: 11 } },
              grid: { display: false }
            }
          }
        }
      });
    }

    if (trendChartRef.current) {
      if (trendChartInstance.current) trendChartInstance.current.destroy();
      const modeData = { normalization: 0, 'missing-data': 0, error: 0 };
      filteredResults.forEach(result => {
        if (result.mode) modeData[result.mode] = (modeData[result.mode] || 0) + 1;
      });

      trendChartInstance.current = new Chart.Chart(trendChartRef.current, {
        type: 'bar',
        data: {
          labels: ['Normalized', 'Missing Data', 'Errors'],
          datasets: [{
            data: [modeData.normalization, modeData['missing-data'], modeData.error],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderRadius: 6,
            barThickness: 60
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { font: { size: 11 } },
              grid: { color: '#e2e8f0' }
            },
            x: {
              ticks: { font: { size: 11 } },
              grid: { display: false }
            }
          }
        }
      });
    }
  };

  useEffect(() => {
    if (activeTab === "analytics" && filteredResults.length > 0) {
      setTimeout(createCharts, 100);
    }
    return () => {
      [spendPieInstance, categoryBarInstance, lineChartInstance, trendChartInstance].forEach(ref => {
        if (ref.current) ref.current.destroy();
      });
    };
  }, [activeTab, filteredResults]);

  const handleExportCSV = () => {
    const csvContent = [
      ['Original Text', 'Enriched Description', 'Mode', 'Vendor', 'Category', 'Product', 'Amount', 'Quantity'].join(','),
      ...filteredResults.map(r => [
        `"${(r.original_text || '').replace(/"/g, '""')}"`,
        `"${(r.enriched_description || '').replace(/"/g, '""')}"`,
        r.mode || '',
        `"${getVendorField(r).replace(/"/g, '""')}"`,
        `"${(r.predicted_category || '').replace(/"/g, '""')}"`,
        `"${(r.product || '').replace(/"/g, '""')}"`,
        r.amount || '',
        r.quantity || ''
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `procurement_analytics_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  // Custom Confirmation Modal Component
  const ConfirmationModal = () => {
    if (!confirmation) return null;

    const isAlert = !confirmation.onCancel;
    const { title, message, onConfirm } = confirmation;

    return (
      <div style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 2000 // Higher than history modal
      }}>
        <div style={{
          backgroundColor: "white",
          borderRadius: "12px",
          padding: "2rem",
          maxWidth: "400px",
          width: "90%",
          boxShadow: "0 10px 30px rgba(0,0,0,0.3)"
        }}>
          <h3 style={{ margin: "0 0 1rem 0", color: "#0f172a", fontSize: "1.25rem" }}>{title}</h3>
          <p style={{ margin: "0 0 1.5rem 0", color: "#475569" }}>{message}</p>
          <div style={{ display: "flex", justifyContent: isAlert ? "center" : "space-around", gap: "1rem" }}>
            {!isAlert && (
              <button
                onClick={confirmation.onCancel}
                style={{
                  padding: "0.75rem 1.5rem",
                  border: "1px solid #cbd5e1",
                  background: "white",
                  color: "#475569",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontWeight: "600",
                  transition: "background-color 0.2s"
                }}
              >
                Cancel
              </button>
            )}
            <button
              onClick={onConfirm}
              style={{
                padding: "0.75rem 1.5rem",
                border: "none",
                background: isAlert ? "#0066cc" : "#ef4444",
                color: "white",
                borderRadius: "8px",
                cursor: "pointer",
                fontWeight: "600",
                transition: "background-color 0.2s"
              }}
            >
              {isAlert ? "OK" : "Confirm Delete"}
            </button>
          </div>
        </div>
      </div>
    );
  };


  return (
    <div style={{
      minHeight: "100vh",
      background: "#f8fafc",
      fontFamily: "'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif"
    }}>
      <ConfirmationModal />
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0066cc 0%, #0052a3 100%)",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        borderBottom: "3px solid #004080"
      }}>
        <div style={{ maxWidth: "1600px", margin: "0 auto", padding: "1rem 2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
              <img src="https://tse4.mm.bing.net/th/id/OIP.a4U2HKvF_-rYHIu__oOQXgHaES?pid=Api&P=0&h=180" alt="SAP" style={{ height: "45px" }} />
              <div>
                <h1 style={{ fontSize: "1.75rem", fontWeight: "700", color: "white", margin: 0, letterSpacing: "-0.5px" }}>
                  Procurement Analytics
                </h1>
                <p style={{ fontSize: "0.875rem", color: "rgba(255,255,255,0.8)", margin: "0.25rem 0 0 0" }}>
                  AI-Powered Spend Analysis & Classification
                </p>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
              <div style={{
                display: "flex",
                alignItems: "center",
                padding: "0.5rem 1rem",
                borderRadius: "20px",
                backgroundColor: apiStatus?.status === "healthy" ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)",
                border: `1px solid ${apiStatus?.status === "healthy" ? "#10b981" : "#ef4444"}`
              }}>
                <div style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  marginRight: "0.5rem",
                  backgroundColor: apiStatus?.status === "healthy" ? "#10b981" : "#ef4444"
                }}></div>
                <span style={{ fontSize: "0.875rem", fontWeight: "500", color: "white" }}>
                  {apiStatus?.status === "healthy" ? "API Connected" : "API Offline"}
                </span>
              </div>
              <div style={{ position: "relative" }}>
                <div onClick={() => setShowUserMenu(!showUserMenu)} style={{ cursor: "pointer" }}>
                  <UserAvatar user={currentUser} />
                </div>
                {showUserMenu && (
                  <div style={{
                    position: "absolute",
                    top: "60px",
                    right: 0,
                    backgroundColor: "white",
                    borderRadius: "8px",
                    boxShadow: "0 10px 40px rgba(0,0,0,0.2)",
                    border: "1px solid #e2e8f0",
                    minWidth: "220px",
                    zIndex: 1000
                  }}>
                    <div style={{ padding: "1rem", borderBottom: "1px solid #e2e8f0" }}>
                      <p style={{ margin: 0, fontWeight: "600", fontSize: "0.9rem", color: "#0f172a" }}>
                        {currentUser?.displayName || "User"}
                      </p>
                      <p style={{ margin: "0.25rem 0 0 0", fontSize: "0.8rem", color: "#64748b" }}>
                        {currentUser?.email || "N/A"}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        setShowHistory(true);
                        setShowUserMenu(false);
                      }}
                      style={{
                        width: "100%",
                        padding: "0.75rem 1rem",
                        border: "none",
                        background: "none",
                        textAlign: "left",
                        cursor: "pointer",
                        fontSize: "0.875rem",
                        color: "#334155",
                        borderBottom: "1px solid #e2e8f0"
                      }}
                      onMouseEnter={(e) => e.target.style.backgroundColor = "#f1f5f9"}
                      onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                    >
                      View History
                    </button>
                    <button
                      onClick={() => {
                        setShowUserMenu(false);
                        handleLogout();
                      }}
                      style={{
                        width: "100%",
                        padding: "0.75rem 1rem",
                        border: "none",
                        background: "none",
                        textAlign: "left",
                        cursor: "pointer",
                        fontSize: "0.875rem",
                        color: "#ef4444"
                      }}
                      onMouseEnter={(e) => e.target.style.backgroundColor = "#fef2f2"}
                      onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: "1600px", margin: "0 auto", padding: "2rem" }}>
        {/* Tabs */}
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "2rem", borderBottom: "2px solid #e2e8f0" }}>
          {[
            { id: "overview", label: "Upload", icon: "📤" },
            { id: "analytics", label: "Analytics", icon: "📊", disabled: results.length === 0 },
            { id: "data", label: "Data Table", icon: "📋", disabled: results.length === 0 }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => !tab.disabled && setActiveTab(tab.id)}
              disabled={tab.disabled}
              style={{
                padding: "0.75rem 1.5rem",
                border: "none",
                background: activeTab === tab.id ? "#0066cc" : "transparent",
                color: activeTab === tab.id ? "white" : tab.disabled ? "#cbd5e1" : "#475569",
                fontWeight: "600",
                fontSize: "0.95rem",
                cursor: tab.disabled ? "not-allowed" : "pointer",
                borderRadius: "8px 8px 0 0",
                transition: "all 0.2s",
                opacity: tab.disabled ? 0.5 : 1
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* History Modal */}
        {showHistory && (
          <div style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.6)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 1000
          }} onClick={() => setShowHistory(false)}>
            <div style={{
              backgroundColor: "white",
              borderRadius: "12px",
              padding: "2rem",
              maxWidth: "900px",
              width: "90%",
              maxHeight: "80vh",
              overflow: "auto",
              boxShadow: "0 20px 60px rgba(0,0,0,0.3)"
            }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ marginBottom: "1.5rem", color: "#0f172a", fontSize: "1.5rem" }}>Classification History</h2>
              {savedClassifications.length === 0 ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "#64748b" }}>
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📊</div>
                  <p>No saved classifications yet</p>
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  {savedClassifications.map((classification) => (
                    <div key={classification.id} style={{
                      border: "1px solid #e5e7eb",
                      borderRadius: "8px",
                      padding: "1.25rem",
                      backgroundColor: "#fafafa",
                      transition: "box-shadow 0.2s"
                    }}
                    onMouseEnter={e => e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.05)'}
                    onMouseLeave={e => e.currentTarget.style.boxShadow = 'none'}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div style={{ flex: 1 }}>
                          <p style={{ fontWeight: "600", margin: 0, fontSize: "1rem", color: "#0f172a" }}>
                            {classification.fileName}
                          </p>
                          <p style={{ fontSize: "0.875rem", color: "#0066cc", margin: "0.5rem 0", fontWeight: "500" }}>
                            {classification.totalRecords} records processed
                          </p>
                          <p style={{ fontSize: "0.75rem", color: "#94a3b8", margin: 0 }}>
                            {/* Display creation time using the ISO string saved in firebaseService.js */}
                            Created: {new Date(classification.createdAt).toLocaleString()}
                          </p>
                          {/* Display the associated User ID for debugging/verification */}
                          <p style={{ fontSize: "0.65rem", color: "#94a3b8", margin: "0.5rem 0 0 0" }}>
                            User ID: {classification.userId}
                          </p>
                        </div>
                        <div style={{ display: "flex", gap: "0.75rem" }}>
                          <button
                            onClick={() => loadSavedClassification(classification)}
                            style={{
                              padding: "0.6rem 1.25rem",
                              backgroundColor: "#0066cc",
                              color: "white",
                              border: "none",
                              borderRadius: "6px",
                              cursor: "pointer",
                              fontWeight: "500",
                              fontSize: "0.875rem",
                              boxShadow: "0 2px 6px rgba(0, 102, 204, 0.3)"
                            }}
                            onMouseEnter={e => e.currentTarget.style.backgroundColor = '#0052a3'}
                            onMouseLeave={e => e.currentTarget.style.backgroundColor = '#0066cc'}
                          >
                            Load
                          </button>
                          <button
                            onClick={() => handleDeleteClassification(classification.id)}
                            style={{
                              padding: "0.6rem 1.25rem",
                              backgroundColor: "#ef4444",
                              color: "white",
                              border: "none",
                              borderRadius: "6px",
                              cursor: "pointer",
                              fontWeight: "500",
                              fontSize: "0.875rem"
                            }}
                            onMouseEnter={e => e.currentTarget.style.backgroundColor = '#cc2929'}
                            onMouseLeave={e => e.currentTarget.style.backgroundColor = '#ef4444'}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div style={{
            backgroundColor: "white",
            borderRadius: "12px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            border: "1px solid #e2e8f0",
            overflow: "hidden"
          }}>
            <div style={{
              background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
              padding: "1.5rem 2rem",
              borderBottom: "1px solid #e2e8f0"
            }}>
              <h2 style={{ margin: 0, color: "#0f172a", fontSize: "1.25rem", fontWeight: "600" }}>Upload Data</h2>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "#64748b" }}>
                Upload procurement data via CSV file or enter text directly for AI-powered classification
              </p>
            </div>

            {/* Input Mode Toggle */}
            <div style={{ padding: "1.5rem 2rem", borderBottom: "1px solid #e2e8f0", backgroundColor: "#fafafa" }}>
              <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
                <button
                  onClick={() => setInputMode("csv")}
                  style={{
                    padding: "0.75rem 2rem",
                    border: inputMode === "csv" ? "2px solid #0066cc" : "2px solid #cbd5e1",
                    background: inputMode === "csv" ? "#e0f2fe" : "white",
                    color: inputMode === "csv" ? "#0066cc" : "#64748b",
                    borderRadius: "8px",
                    fontWeight: "600",
                    fontSize: "0.95rem",
                    cursor: "pointer",
                    transition: "all 0.2s"
                  }}
                >
                  📁 CSV File Upload
                </button>
                <button
                  onClick={() => setInputMode("text")}
                  style={{
                    padding: "0.75rem 2rem",
                    border: inputMode === "text" ? "2px solid #0066cc" : "2px solid #cbd5e1",
                    background: inputMode === "text" ? "#e0f2fe" : "white",
                    color: inputMode === "text" ? "#0066cc" : "#64748b",
                    borderRadius: "8px",
                    fontWeight: "600",
                    fontSize: "0.95rem",
                    cursor: "pointer",
                    transition: "all 0.2s"
                  }}
                >
                  ✍️ Text Input
                </button>
              </div>
            </div>

            <div style={{ padding: "2.5rem" }}>
              {inputMode === "csv" ? (
                // CSV Upload Mode
                <div style={{
                  border: "2px dashed #cbd5e1",
                  borderRadius: "12px",
                  padding: "3rem",
                  textAlign: "center",
                  backgroundColor: "#fafafa",
                  transition: "all 0.3s"
                }}>
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📁</div>
                  <p style={{ fontSize: "1.1rem", fontWeight: "600", color: "#0f172a", marginBottom: "0.5rem" }}>
                    Choose CSV File
                  </p>
                  <p style={{ fontSize: "0.875rem", color: "#64748b", marginBottom: "1.5rem" }}>
                    Upload a CSV file containing procurement data for batch classification
                  </p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setFile(e.target.files[0])}
                    style={{
                      display: "block",
                      width: "100%",
                      marginBottom: "1.5rem",
                      padding: "0.75rem",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      fontSize: "0.95rem"
                    }}
                  />
                  {file && (
                    <div style={{
                      backgroundColor: "#e0f2fe",
                      border: "1px solid #0066cc",
                      borderRadius: "8px",
                      padding: "1rem",
                      marginBottom: "1.5rem",
                      display: "inline-block"
                    }}>
                      <p style={{ margin: 0, fontWeight: "600", color: "#0066cc" }}>
                        {file.name}
                      </p>
                      <p style={{ margin: "0.25rem 0 0 0", fontSize: "0.875rem", color: "#0369a1" }}>
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                // Text Input Mode
                <div>
                  <div style={{ marginBottom: "1.5rem" }}>
                    <label style={{
                      display: "block",
                      marginBottom: "0.75rem",
                      fontWeight: "600",
                      color: "#0f172a",
                      fontSize: "1rem"
                    }}>
                      Enter Procurement Description
                    </label>
                    <p style={{ fontSize: "0.875rem", color: "#64748b", marginBottom: "1rem" }}>
                      Type or paste procurement text for instant classification. You can enter multiple lines.
                    </p>
                    <textarea
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      placeholder="Example: Office supplies from Staples including pens, notebooks, and printer paper - Invoice #12345, Amount: $250.00"
                      style={{
                        width: "100%",
                        minHeight: "200px",
                        padding: "1rem",
                        border: "2px solid #cbd5e1",
                        borderRadius: "8px",
                        fontSize: "0.95rem",
                        fontFamily: "'Segoe UI', sans-serif",
                        resize: "vertical",
                        lineHeight: "1.6"
                      }}
                    />
                    {textInput.trim() && (
                      <div style={{
                        marginTop: "1rem",
                        padding: "0.75rem 1rem",
                        backgroundColor: "#e0f2fe",
                        border: "1px solid #0066cc",
                        borderRadius: "6px",
                        fontSize: "0.875rem",
                        color: "#0369a1"
                      }}>
                        ✓ {textInput.length} characters ready for classification
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div style={{ textAlign: "center", marginTop: "2rem" }}>
                <button
                  onClick={handleUpload}
                  disabled={loading || (inputMode === "csv" ? !file : !textInput.trim()) || savingToFirebase}
                  style={{
                    background: loading || (inputMode === "csv" ? !file : !textInput.trim()) || savingToFirebase
                      ? "#cbd5e1"
                      : "linear-gradient(135deg, #0066cc, #0052a3)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "1rem 2.5rem",
                    fontWeight: "600",
                    fontSize: "1rem",
                    cursor: loading || (inputMode === "csv" ? !file : !textInput.trim()) || savingToFirebase ? "not-allowed" : "pointer",
                    transition: "all 0.2s",
                    boxShadow: loading || (inputMode === "csv" ? !file : !textInput.trim()) || savingToFirebase ? "none" : "0 4px 12px rgba(0, 102, 204, 0.3)"
                  }}
                >
                  {loading ? "Processing..." : savingToFirebase ? "Saving..." : inputMode === "csv" ? "Upload & Classify" : "Classify Text"}
                </button>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div style={{
            backgroundColor: "#fee2e2",
            border: "1px solid #ef4444",
            color: "#991b1b",
            padding: "1rem 1.5rem",
            borderRadius: "8px",
            marginBottom: "2rem",
            fontWeight: "500"
          }}>
            Error: {error}
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === "analytics" && stats && (
          <>
            {/* KPI Cards */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
              gap: "1.5rem",
              marginBottom: "2rem"
            }}>
              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                padding: "1.5rem",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                borderLeft: "4px solid #0066cc"
              }}>
                <p style={{ fontSize: "0.875rem", color: "#64748b", margin: 0, fontWeight: "500" }}>
                  Total Records
                </p>
                <p style={{ fontSize: "2.25rem", fontWeight: "700", color: "#0066cc", margin: "0.5rem 0 0 0" }}>
                  {stats.total.toLocaleString()}
                </p>
              </div>
              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                padding: "1.5rem",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                borderLeft: "4px solid #10b981"
              }}>
                <p style={{ fontSize: "0.875rem", color: "#64748b", margin: 0, fontWeight: "500" }}>
                  Unique Vendors
                </p>
                <p style={{ fontSize: "2.25rem", fontWeight: "700", color: "#10b981", margin: "0.5rem 0 0 0" }}>
                  {stats.uniqueVendors}
                </p>
              </div>
              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                padding: "1.5rem",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                borderLeft: "4px solid #8b5cf6"
              }}>
                <p style={{ fontSize: "0.875rem", color: "#64748b", margin: 0, fontWeight: "500" }}>
                  Total Spend
                </p>
                <p style={{ fontSize: "2.25rem", fontWeight: "700", color: "#8b5cf6", margin: "0.5rem 0 0 0" }}>
                  ${(stats.totalAmount / 1000000).toFixed(2)}M
                </p>
              </div>
              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                padding: "1.5rem",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                borderLeft: "4px solid #f59e0b"
              }}>
                <p style={{ fontSize: "0.875rem", color: "#64748b", margin: 0, fontWeight: "500" }}>
                  Avg Confidence
                </p>
                <p style={{ fontSize: "2.25rem", fontWeight: "700", color: "#f59e0b", margin: "0.5rem 0 0 0" }}>
                  {(stats.avgVendorConfidence * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            {/* Filters */}
            <div style={{
              backgroundColor: "white",
              borderRadius: "12px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
              border: "1px solid #e2e8f0",
              marginBottom: "2rem",
              overflow: "hidden"
            }}>
              <div style={{
                background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
                padding: "1.25rem 2rem",
                borderBottom: "1px solid #e2e8f0"
              }}>
                <h3 style={{ margin: 0, color: "#0f172a", fontSize: "1.1rem", fontWeight: "600" }}>
                  Filters & Controls
                </h3>
              </div>
              <div style={{ padding: "1.5rem 2rem" }}>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: "1.25rem",
                  marginBottom: "1.5rem"
                }}>
                  <div>
                    <label style={{
                      display: "block",
                      marginBottom: "0.5rem",
                      fontWeight: "600",
                      color: "#334155",
                      fontSize: "0.875rem"
                    }}>
                      Search
                    </label>
                    <input
                      type="text"
                      placeholder="Search descriptions..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      style={{
                        width: "100%",
                        padding: "0.625rem",
                        borderRadius: "6px",
                        border: "1px solid #cbd5e1",
                        fontSize: "0.875rem"
                      }}
                    />
                  </div>
                  <div>
                    <label style={{
                      display: "block",
                      marginBottom: "0.5rem",
                      fontWeight: "600",
                      color: "#334155",
                      fontSize: "0.875rem"
                    }}>
                      Processing Mode
                    </label>
                    <select
                      value={filterMode}
                      onChange={(e) => setFilterMode(e.target.value)}
                      style={{
                        width: "100%",
                        padding: "0.625rem",
                        borderRadius: "6px",
                        border: "1px solid #cbd5e1",
                        fontSize: "0.875rem"
                      }}
                    >
                      <option value="all">All Modes</option>
                      <option value="normalization">Normalization</option>
                      <option value="missing-data">Missing Data</option>
                      <option value="error">Errors</option>
                    </select>
                  </div>
                  <div>
                    <label style={{
                      display: "block",
                      marginBottom: "0.5rem",
                      fontWeight: "600",
                      color: "#334155",
                      fontSize: "0.875rem"
                    }}>
                      Vendor
                    </label>
                    <select
                      value={filterVendor}
                      onChange={(e) => setFilterVendor(e.target.value)}
                      style={{
                        width: "100%",
                        padding: "0.625rem",
                        borderRadius: "6px",
                        border: "1px solid #cbd5e1",
                        fontSize: "0.875rem"
                      }}
                    >
                      <option value="all">All Vendors</option>
                      {uniqueVendors.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </div>
                  <div>
                    <label style={{
                      display: "block",
                      marginBottom: "0.5rem",
                      fontWeight: "600",
                      color: "#334155",
                      fontSize: "0.875rem"
                    }}>
                      Category
                    </label>
                    <select
                      value={filterCategory}
                      onChange={(e) => setFilterCategory(e.target.value)}
                      style={{
                        width: "100%",
                        padding: "0.625rem",
                        borderRadius: "6px",
                        border: "1px solid #cbd5e1",
                        fontSize: "0.875rem"
                      }}
                    >
                      <option value="all">All Categories</option>
                      {uniqueCategories.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                  <button
                    onClick={handleExportCSV}
                    style={{
                      background: "linear-gradient(135deg, #10b981, #059669)",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      padding: "0.625rem 1.5rem",
                      fontWeight: "600",
                      cursor: "pointer",
                      fontSize: "0.875rem",
                      boxShadow: "0 2px 6px rgba(16, 185, 129, 0.3)"
                    }}
                  >
                    Export to CSV
                  </button>
                  <button
                    onClick={() => {
                      setFilterMode("all");
                      setFilterVendor("all");
                      setFilterCategory("all");
                      setSearchQuery("");
                    }}
                    style={{
                      background: "#64748b",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      padding: "0.625rem 1.5rem",
                      fontWeight: "600",
                      cursor: "pointer",
                      fontSize: "0.875rem"
                    }}
                  >
                    Reset Filters
                  </button>
                </div>
              </div>
            </div>

            {/* Charts Grid */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(450px, 1fr))",
              gap: "1.5rem",
              marginBottom: "2rem"
            }}>
              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                overflow: "hidden"
              }}>
                <div style={{
                  background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
                  padding: "1rem 1.5rem",
                  borderBottom: "1px solid #e2e8f0"
                }}>
                  <h3 style={{ margin: 0, color: "#0f172a", fontSize: "1rem", fontWeight: "600" }}>
                    Spend Distribution by Category
                  </h3>
                </div>
                <div style={{ padding: "2rem", height: "380px" }}>
                  <canvas ref={spendPieRef}></canvas>
                </div>
              </div>

              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                overflow: "hidden"
              }}>
                <div style={{
                  background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
                  padding: "1rem 1.5rem",
                  borderBottom: "1px solid #e2e8f0"
                }}>
                  <h3 style={{ margin: 0, color: "#0f172a", fontSize: "1rem", fontWeight: "600" }}>
                    Top 5 Categories by Spend
                  </h3>
                </div>
                <div style={{ padding: "2rem", height: "380px" }}>
                  <canvas ref={categoryBarRef}></canvas>
                </div>
              </div>

              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                overflow: "hidden"
              }}>
                <div style={{
                  background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
                  padding: "1rem 1.5rem",
                  borderBottom: "1px solid #e2e8f0"
                }}>
                  <h3 style={{ margin: 0, color: "#0f172a", fontSize: "1rem", fontWeight: "600" }}>
                    Classification Confidence Distribution
                  </h3>
                </div>
                <div style={{ padding: "2rem", height: "380px" }}>
                  <canvas ref={lineChartRef}></canvas>
                </div>
              </div>

              <div style={{
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                border: "1px solid #e2e8f0",
                overflow: "hidden"
              }}>
                <div style={{
                  background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
                  padding: "1rem 1.5rem",
                  borderBottom: "1px solid #e2e8f0"
                }}>
                  <h3 style={{ margin: 0, color: "#0f172a", fontSize: "1rem", fontWeight: "600" }}>
                    Processing Mode Analysis
                  </h3>
                </div>
                <div style={{ padding: "2rem", height: "380px" }}>
                  <canvas ref={trendChartRef}></canvas>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Data Table Tab */}
        {activeTab === "data" && stats && (
          <div style={{
            backgroundColor: "white",
            borderRadius: "12px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            border: "1px solid #e2e8f0",
            overflow: "hidden"
          }}>
            <div style={{
              background: "linear-gradient(135deg, #f8fafc, #f1f5f9)",
              padding: "1.25rem 2rem",
              borderBottom: "1px solid #e2e8f0",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center"
            }}>
              <h2 style={{ margin: 0, color: "#0f172a", fontSize: "1.25rem", fontWeight: "600" }}>
                Classification Results ({filteredResults.length} records)
              </h2>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ backgroundColor: "#f8fafc" }}>
                    {['Original Text', 'Enriched Description', 'Vendor', 'Category', 'Amount/Qty'].map(h => (
                      <th key={h} style={{
                        padding: "1rem 1.5rem",
                        textAlign: "left",
                        fontSize: "0.8rem",
                        fontWeight: "700",
                        color: "#475569",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px"
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.map((result, index) => (
                    <tr
                      key={index}
                      style={{
                        cursor: "pointer",
                        backgroundColor: expandedRows.has(index) ? "#f8fafc" : "white"
                      }}
                      onClick={() => toggleRow(index)}
                    >
                      <td style={{
                        padding: "1rem 1.5rem",
                        fontSize: "0.875rem",
                        borderBottom: "1px solid #e2e8f0",
                        maxWidth: "250px"
                      }}>
                        <div style={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: expandedRows.has(index) ? "normal" : "nowrap",
                          color: "#334155"
                        }}>
                          {result.original_text || "N/A"}
                        </div>
                        <span style={{
                          display: "inline-flex",
                          padding: "0.25rem 0.625rem",
                          fontSize: "0.7rem",
                          fontWeight: "600",
                          borderRadius: "12px",
                          marginTop: "0.5rem",
                          alignItems: "center",
                          gap: "0.25rem",
                          ...getModeColor(result.mode)
                        }}>
                          {getModeIcon(result.mode)} {result.mode}
                        </span>
                      </td>
                      <td style={{
                        padding: "1rem 1.5rem",
                        fontSize: "0.875rem",
                        borderBottom: "1px solid #e2e8f0",
                        maxWidth: "300px"
                      }}>
                        <div style={{
                          backgroundColor: "#eff6ff",
                          border: "1px solid #bfdbfe",
                          borderRadius: "6px",
                          padding: "0.75rem",
                          fontStyle: "italic",
                          color: "#1e40af",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: expandedRows.has(index) ? "normal" : "nowrap",
                          fontSize: "0.85rem"
                        }}>
                          {result.enriched_description || "No description available"}
                        </div>
                      </td>
                      <td style={{
                        padding: "1rem 1.5rem",
                        fontSize: "0.875rem",
                        borderBottom: "1px solid #e2e8f0"
                      }}>
                        <div style={{ fontWeight: "600", color: "#0f172a" }}>
                          {getVendorField(result)}
                        </div>
                        {result.vendor_confidence && (
                          <div style={{
                            fontSize: "0.75rem",
                            color: "#0066cc",
                            marginTop: "0.25rem",
                            fontWeight: "500"
                          }}>
                            {(result.vendor_confidence * 100).toFixed(0)}% confidence
                          </div>
                        )}
                      </td>
                      <td style={{
                        padding: "1rem 1.5rem",
                        fontSize: "0.875rem",
                        borderBottom: "1px solid #e2e8f0",
                        maxWidth: "200px"
                      }}>
                        <div style={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: expandedRows.has(index) ? "normal" : "nowrap",
                          color: "#334155",
                          fontWeight: "500"
                        }}>
                          {result.predicted_category || "N/A"}
                        </div>
                      </td>
                      <td style={{
                        padding: "1rem 1.5rem",
                        fontSize: "0.875rem",
                        borderBottom: "1px solid #e2e8f0"
                      }}>
                        {result.amount && (
                          <div style={{ fontWeight: "700", color: "#10b981" }}>
                            {formatAmount(result.amount)}
                          </div>
                        )}
                        {result.quantity && (
                          <div style={{
                            fontWeight: "600",
                            color: "#0066cc",
                            marginTop: "0.25rem",
                            fontSize: "0.8rem"
                          }}>
                            {result.quantity} units
                          </div>
                        )}
                        {!result.amount && !result.quantity && (
                          <span style={{ color: "#94a3b8" }}>N/A</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;

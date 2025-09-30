import { useState, useEffect, useRef } from "react";
import * as Chart from 'chart.js';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const [showChart, setShowChart] = useState(false);
  const [expandedRows, setExpandedRows] = useState(new Set());
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

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
  }, []);

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload a CSV file");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
        setResults([]);
      } else if (data.results) {
        setResults(data.results);
        setError(null);
      } else if (Array.isArray(data)) {
        setResults(data);
        setError(null);
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
      case "normalization": return { backgroundColor: "#dcfce7", color: "#166534", border: "1px solid #bbf7d0" };
      case "missing-data": return { backgroundColor: "#fef3c7", color: "#92400e", border: "1px solid #fde68a" };
      case "error": return { backgroundColor: "#fee2e2", color: "#991b1b", border: "1px solid #fecaca" };
      default: return { backgroundColor: "#f1f5f9", color: "#475569", border: "1px solid #cbd5e1" };
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

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return "#10b981";
    if (confidence >= 0.6) return "#0070f3";
    if (confidence >= 0.4) return "#f59e0b";
    return "#ef4444";
  };

  const toggleRow = (index) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  const stats = results.length > 0 ? {
    total: results.length,
    withVendor: results.filter(r => getVendorField(r) !== "N/A").length,
    withAmount: results.filter(r => r.amount).length,
    withQuantity: results.filter(r => r.quantity).length,
    withEnrichedDesc: results.filter(r => r.enriched_description && r.enriched_description !== "No data available").length,
    avgVendorConfidence: results.filter(r => r.vendor_confidence).reduce((acc, r) => acc + r.vendor_confidence, 0) / results.filter(r => r.vendor_confidence).length || 0
  } : null;

  const generateChartData = () => {
    const categoryData = {};
    const categoryAmounts = {};

    results.forEach(result => {
      const category = result.predicted_category || "Other";
      const amount = result.amount || 0;

      if (categoryData[category]) {
        categoryData[category]++;
        categoryAmounts[category] += amount;
      } else {
        categoryData[category] = 1;
        categoryAmounts[category] = amount;
      }
    });

    const labels = Object.keys(categoryData);
    const counts = Object.values(categoryData);

    const colors = [
      '#0070f3', '#00a1c9', '#f5a623', '#7ed321', '#d0021b',
      '#4a90e2', '#8b572a', '#50e3c2', '#bd10e0', '#9013fe',
      '#007aff', '#ff6900', '#fcb900', '#00d084', '#eb144c'
    ];

    return {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: colors.slice(0, labels.length),
        borderColor: colors.slice(0, labels.length).map(color => color + '40'),
        borderWidth: 2,
        hoverOffset: 4
      }],
      amounts: categoryAmounts
    };
  };

  const createChart = () => {
    if (!chartRef.current || results.length === 0) return;

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const chartData = generateChartData();
    Chart.Chart.register(...Chart.registerables);

    chartInstance.current = new Chart.Chart(chartRef.current, {
      type: 'pie',
      data: {
        labels: chartData.labels,
        datasets: chartData.datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              usePointStyle: true,
              padding: 20,
              font: { size: 12 },
              generateLabels: function(chart) {
                const data = chart.data;
                return data.labels.map((label, i) => {
                  const count = data.datasets[0].data[i];
                  const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                  const percentage = ((count / total) * 100).toFixed(1);
                  const amount = chartData.amounts[label];

                  return {
                    text: `${label} (${percentage}%) ${amount > 0 ? `- $${amount.toLocaleString()}` : ''}`,
                    fillStyle: data.datasets[0].backgroundColor[i],
                    strokeStyle: data.datasets[0].borderColor[i],
                    pointStyle: 'circle'
                  };
                });
              }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const count = context.raw;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = ((count / total) * 100).toFixed(1);
                const amount = chartData.amounts[label];

                return [
                  `${label}: ${count} records (${percentage}%)`,
                  amount > 0 ? `Total Amount: $${amount.toLocaleString()}` : 'No amount data'
                ];
              }
            }
          }
        }
      }
    });
  };

  useEffect(() => {
    if (showChart && results.length > 0) {
      setTimeout(createChart, 100);
    }
  }, [showChart, results]);

  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  const handleExportCSV = () => {
    const csvContent = [
      ['Original Text', 'Enriched Description', 'Mode', 'Vendor', 'Category', 'Product', 'Amount', 'Quantity', 'Vendor Confidence', 'Category Confidence'].join(','),
      ...results.map(r => [
        `"${(r.original_text || '').replace(/"/g, '""')}"`,
        `"${(r.enriched_description || '').replace(/"/g, '""')}"`,
        r.mode || '',
        `"${getVendorField(r).replace(/"/g, '""')}"`,
        `"${(r.predicted_category || '').replace(/"/g, '""')}"`,
        `"${(r.product || '').replace(/"/g, '""')}"`,
        r.amount || '',
        r.quantity || '',
        r.vendor_confidence || '',
        r.category_confidence || ''
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'classified_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const containerStyle = {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #e8f4ff 0%, #d1e9ff 50%, #b3deff 100%)",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
  };

  const mainContentStyle = {
    maxWidth: "1400px",
    margin: "0 auto",
    padding: "2rem 1.5rem"
  };

  const headerStyle = {
    textAlign: "center",
    marginBottom: "3rem",
    position: "relative"
  };

  const sapLogoStyle = {
    position: "absolute",
    top: "-1rem",
    left: "0",
    width: "120px",
    height: "auto",
    zIndex: 10
  };

  const titleStyle = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    background: "linear-gradient(135deg, #0070f3, #003d82, #001a3d)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    marginBottom: "1rem"
  };

  const cardStyle = {
    backgroundColor: "white",
    borderRadius: "1rem",
    boxShadow: "0 25px 50px -12px rgba(0, 112, 243, 0.15)",
    border: "1px solid #b3deff",
    marginBottom: "2rem",
    overflow: "hidden"
  };

  const buttonStyle = {
    background: "linear-gradient(135deg, #0070f3, #003d82)",
    color: "white",
    border: "none",
    borderRadius: "0.75rem",
    padding: "0.75rem 2rem",
    fontWeight: "600",
    cursor: "pointer",
    boxShadow: "0 10px 15px -3px rgba(0, 112, 243, 0.3)",
    transition: "all 0.2s",
    margin: "0 0.5rem"
  };

  const secondaryButtonStyle = {
    backgroundColor: "#f8fafc",
    color: "#003d82",
    border: "1px solid #0070f3",
    borderRadius: "0.75rem",
    padding: "0.75rem 1.5rem",
    fontWeight: "600",
    cursor: "pointer",
    transition: "all 0.2s",
    margin: "0 0.5rem"
  };

  return (
    <div style={containerStyle}>
      <div style={mainContentStyle}>
        {/* Header */}
        <div style={headerStyle}>
          <div style={sapLogoStyle}>
            <img
              src="https://tse4.mm.bing.net/th/id/OIP.a4U2HKvF_-rYHIu__oOQXgHaES?pid=Api&P=0&h=180"
              alt="SAP Labs"
              style={{ width: "100%", height: "auto" }}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "center", marginBottom: "1rem", marginTop: "2rem" }}>
            <div style={{
              padding: "0.75rem",
              background: "linear-gradient(135deg, #0070f3, #003d82)",
              borderRadius: "1rem",
              boxShadow: "0 10px 15px -3px rgba(0, 112, 243, 0.3)"
            }}>
              <span style={{ fontSize: "2rem" }}>📊</span>
            </div>
          </div>
          <h1 style={titleStyle}>Procurement Spend Classifier</h1>
          <p style={{ fontSize: "1.25rem", color: "#003d82", maxWidth: "32rem", margin: "0 auto" }}>
            Advanced AI-powered classification with enriched descriptions
          </p>
        </div>

        {/* API Status */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: "2rem" }}>
          <div style={{
            display: "inline-flex",
            alignItems: "center",
            padding: "0.5rem 1rem",
            borderRadius: "9999px",
            fontWeight: "500",
            boxShadow: "0 1px 3px 0 rgba(0, 112, 243, 0.1)",
            ...(apiStatus?.status === "healthy"
              ? { backgroundColor: "#dcfce7", color: "#166534", border: "1px solid #bbf7d0" }
              : { backgroundColor: "#fee2e2", color: "#991b1b", border: "1px solid #fecaca" })
          }}>
            <div style={{
              width: "0.5rem",
              height: "0.5rem",
              borderRadius: "50%",
              marginRight: "0.75rem",
              backgroundColor: apiStatus?.status === "healthy" ? "#10b981" : "#ef4444"
            }}></div>
            <span>API: {apiStatus?.status === "healthy" ? "Connected" : "Disconnected"}</span>
            {apiStatus?.enriched_descriptions && (
              <span style={{ marginLeft: "0.5rem", fontSize: "0.75rem", opacity: 0.8 }}>
                (✨ Enriched Descriptions Enabled)
              </span>
            )}
          </div>
        </div>

        {/* Upload Section */}
        <div style={cardStyle}>
          <div style={{
            background: "linear-gradient(135deg, #e8f4ff, #d1e9ff)",
            padding: "1.5rem",
            borderBottom: "1px solid #b3deff"
          }}>
            <h2 style={{ fontSize: "1.125rem", fontWeight: "600", color: "#003d82", margin: 0 }}>
              📤 Upload CSV File
            </h2>
          </div>

          <div style={{ padding: "1.5rem" }}>
            <div style={{
              border: "2px dashed #0070f3",
              borderRadius: "0.75rem",
              padding: "3rem 2rem",
              textAlign: "center",
              backgroundColor: "#fefefe"
            }}>
              <span style={{ fontSize: "3rem", display: "block", marginBottom: "1rem" }}>📄</span>

              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
                style={{
                  display: "block",
                  width: "100%",
                  fontSize: "0.875rem",
                  marginBottom: "1rem"
                }}
              />

              {file && (
                <div style={{
                  backgroundColor: "#e8f4ff",
                  borderRadius: "0.5rem",
                  padding: "0.75rem",
                  display: "inline-flex",
                  alignItems: "center",
                  marginBottom: "1rem"
                }}>
                  <span>{file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                </div>
              )}

              <div style={{ display: "flex", justifyContent: "center", gap: "1rem" }}>
                <button
                  onClick={handleUpload}
                  disabled={loading || !file}
                  style={{
                    ...buttonStyle,
                    opacity: loading || !file ? 0.5 : 1,
                    cursor: loading || !file ? "not-allowed" : "pointer"
                  }}
                >
                  {loading ? "Processing..." : "📤 Upload & Classify"}
                </button>

                <button onClick={checkApiHealth} style={secondaryButtonStyle}>
                  🎯 Test API
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div style={{
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            color: "#991b1b",
            padding: "1rem 1.5rem",
            borderRadius: "0.75rem",
            marginBottom: "2rem"
          }}>
            ❌ Error: {error}
          </div>
        )}

        {/* Stats Cards */}
        {stats && (
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "1rem",
            marginBottom: "2rem"
          }}>
            <div style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              boxShadow: "0 1px 3px 0 rgba(0, 112, 243, 0.1)",
              border: "1px solid #b3deff",
              padding: "1.5rem"
            }}>
              <p style={{ fontSize: "0.875rem", color: "#0070f3", margin: "0 0 0.25rem 0" }}>Total</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#003d82", margin: 0 }}>{stats.total}</p>
            </div>
            <div style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              boxShadow: "0 1px 3px 0 rgba(0, 112, 243, 0.1)",
              border: "1px solid #b3deff",
              padding: "1.5rem"
            }}>
              <p style={{ fontSize: "0.875rem", color: "#0070f3", margin: "0 0 0.25rem 0" }}>Vendors</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#059669", margin: 0 }}>{stats.withVendor}</p>
            </div>
            <div style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              boxShadow: "0 1px 3px 0 rgba(0, 112, 243, 0.1)",
              border: "1px solid #b3deff",
              padding: "1.5rem"
            }}>
              <p style={{ fontSize: "0.875rem", color: "#0070f3", margin: "0 0 0.25rem 0" }}>Enriched</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#7c3aed", margin: 0 }}>{stats.withEnrichedDesc}</p>
            </div>
            <div style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              boxShadow: "0 1px 3px 0 rgba(0, 112, 243, 0.1)",
              border: "1px solid #b3deff",
              padding: "1.5rem"
            }}>
              <p style={{ fontSize: "0.875rem", color: "#0070f3", margin: "0 0 0.25rem 0" }}>Confidence</p>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#003d82", margin: 0 }}>{(stats.avgVendorConfidence * 100).toFixed(0)}%</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results.length > 0 && !loading && (
          <div>
            <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginBottom: "2rem" }}>
              <button
                onClick={() => setShowChart(!showChart)}
                style={{
                  ...buttonStyle,
                  background: showChart ? "linear-gradient(135deg, #059669, #16a34a)" : "linear-gradient(135deg, #0070f3, #003d82)"
                }}
              >
                📊 {showChart ? "Hide Chart" : "Visualize Data"}
              </button>
              <button onClick={handleExportCSV} style={secondaryButtonStyle}>
                📥 Export Results
              </button>
            </div>

            {/* Chart */}
            {showChart && (
              <div style={{ ...cardStyle, marginBottom: "2rem" }}>
                <div style={{
                  background: "linear-gradient(135deg, #e8f4ff, #d1e9ff)",
                  padding: "1.5rem",
                  borderBottom: "1px solid #b3deff"
                }}>
                  <h2 style={{ fontSize: "1.25rem", fontWeight: "bold", color: "#003d82", margin: 0 }}>
                    📊 Category Distribution
                  </h2>
                </div>
                <div style={{ padding: "2rem" }}>
                  <div style={{ position: "relative", height: "400px" }}>
                    <canvas ref={chartRef}></canvas>
                  </div>
                </div>
              </div>
            )}

            {/* Results Table */}
            <div style={cardStyle}>
              <div style={{
                background: "linear-gradient(135deg, #e8f4ff, #d1e9ff)",
                padding: "1.5rem",
                borderBottom: "1px solid #b3deff"
              }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: "bold", color: "#003d82", margin: 0 }}>
                  Classification Results ({results.length})
                </h2>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{
                        backgroundColor: "#e8f4ff",
                        padding: "1rem",
                        textAlign: "left",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                        color: "#003d82",
                        borderBottom: "1px solid #b3deff"
                      }}>TEXT</th>
                      <th style={{
                        backgroundColor: "#e8f4ff",
                        padding: "1rem",
                        textAlign: "left",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                        color: "#003d82",
                        borderBottom: "1px solid #b3deff"
                      }}>ENRICHED DESCRIPTION</th>
                      <th style={{
                        backgroundColor: "#e8f4ff",
                        padding: "1rem",
                        textAlign: "left",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                        color: "#003d82",
                        borderBottom: "1px solid #b3deff"
                      }}>VENDOR</th>
                      <th style={{
                        backgroundColor: "#e8f4ff",
                        padding: "1rem",
                        textAlign: "left",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                        color: "#003d82",
                        borderBottom: "1px solid #b3deff"
                      }}>CATEGORY</th>
                      <th style={{
                        backgroundColor: "#e8f4ff",
                        padding: "1rem",
                        textAlign: "left",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                        color: "#003d82",
                        borderBottom: "1px solid #b3deff"
                      }}>AMOUNT / QTY</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={index} style={{
                        cursor: "pointer",
                        transition: "background-color 0.15s"
                      }}
                      onClick={() => toggleRow(index)}>
                        <td style={{
                          padding: "1rem",
                          fontSize: "0.875rem",
                          borderBottom: "1px solid #f1f5f9",
                          maxWidth: "200px"
                        }}>
                          <div style={{
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: expandedRows.has(index) ? "normal" : "nowrap",
                            fontWeight: "500"
                          }}>
                            {result.original_text || "N/A"}
                          </div>
                          <span style={{
                            display: "inline-flex",
                            alignItems: "center",
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.65rem",
                            fontWeight: "600",
                            borderRadius: "9999px",
                            marginTop: "0.5rem",
                            ...getModeColor(result.mode)
                          }}>
                            {getModeIcon(result.mode)} {result.mode}
                          </span>
                        </td>
                        <td style={{
                          padding: "1rem",
                          fontSize: "0.875rem",
                          borderBottom: "1px solid #f1f5f9",
                          maxWidth: "250px"
                        }}>
                          <div style={{
                            backgroundColor: "#f0f9ff",
                            border: "1px solid #bfdbfe",
                            borderRadius: "0.5rem",
                            padding: "0.75rem",
                            fontStyle: "italic",
                            color: "#1e40af",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: expandedRows.has(index) ? "normal" : "nowrap"
                          }}>
                            ✨ {result.enriched_description || "No enriched description"}
                          </div>
                        </td>
                        <td style={{
                          padding: "1rem",
                          fontSize: "0.875rem",
                          borderBottom: "1px solid #f1f5f9"
                        }}>
                          <div style={{ fontWeight: "500" }}>{getVendorField(result)}</div>
                          {result.vendor_confidence && (
                            <div style={{ fontSize: "0.75rem", color: "#0070f3" }}>
                              {(result.vendor_confidence * 100).toFixed(0)}% confidence
                            </div>
                          )}
                        </td>
                        <td style={{
                          padding: "1rem",
                          fontSize: "0.875rem",
                          borderBottom: "1px solid #f1f5f9",
                          maxWidth: "200px"
                        }}>
                          <div style={{
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: expandedRows.has(index) ? "normal" : "nowrap"
                          }}>
                            {result.predicted_category || "N/A"}
                          </div>
                        </td>
                        <td style={{
                          padding: "1rem",
                          fontSize: "0.875rem",
                          borderBottom: "1px solid #f1f5f9"
                        }}>
                          {result.amount && (
                            <div style={{ fontWeight: "bold", color: "#059669" }}>
                              💰 {formatAmount(result.amount)}
                            </div>
                          )}
                          {result.quantity && (
                            <div style={{ fontWeight: "bold", color: "#0070f3", marginTop: "0.25rem" }}>
                              📦 {result.quantity} units
                            </div>
                          )}
                          {!result.amount && !result.quantity && "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        <style>
          {`
            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
            tr:hover {
              background-color: #e8f4ff !important;
            }
            button:hover:not(:disabled) {
              transform: translateY(-1px);
              box-shadow: 0 20px 25px -5px rgba(0, 112, 243, 0.2);
            }
          `}
        </style>
      </div>
    </div>
  );
}

export default App;
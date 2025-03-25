import { useState } from "react";
import { Upload, FileUp, AlertCircle, BarChart as BarChartIcon, Loader2 } from "lucide-react";
import Papa from 'papaparse';
import BarChart from "../components/BarChart";
import PieChart from "../components/PieChart";
import ScatterPlot from "../components/ScatterPlot";
import Histogram from "../components/Histogram";
import BoxPlot from "../components/BoxPlot";

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [visualizations, setVisualizations] = useState([]);
  const [data, setData] = useState([]);
  const [isFormSubmitted, setIsFormSubmitted] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (file) => {
    setError(null);
    setVisualizations([]);

    // Check file type
    const fileType = file.name.split(".").pop()?.toLowerCase();
    if (fileType !== "csv" && fileType !== "xlsx" && fileType !== "xls") {
      setError("Please upload a CSV or Excel file");
      return;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size should be less than 10MB");
      return;
    }

    setFile(file);
  };

  const handleAnalyzeDataset = async () => {
    if (!file) {
      setError("Please select a CSV or Excel file");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // First, parse the file locally to get data for visualization
      const reader = new FileReader();
      reader.onload = async (event) => {
        try {
          if (file.name.endsWith('.csv')) {
            // Use Papaparse for CSV files
            Papa.parse(event.target.result, {
              header: true,
              dynamicTyping: true,
              complete: (results) => {
                setData(results.data);
                
                // Now send to backend for analysis
                sendToBackend(formData);
              },
              error: (err) => {
                throw new Error(`Error parsing file: ${err.message}`);
              }
            });
          } else {
            // For Excel files, we'd need to use a library like xlsx
            // For simplicity in this implementation, we'll focus on CSV
            setError("Excel parsing is not implemented in this demo");
            setIsLoading(false);
          }
        } catch (err) {
          setError(err.message || "Error reading file");
          setIsLoading(false);
        }
      };
      
      reader.readAsText(file);
      
    } catch (err) {
      setError(err.message || "An error occurred while processing the file");
      setIsLoading(false);
    }
  };
  
  const sendToBackend = async (formData) => {
    try {
      const response = await fetch("http://localhost:8000/analyze-dataset", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const result = await response.json();
      setVisualizations(result.visualizations || []);
      setIsFormSubmitted(true);
    } catch (err) {
      setError(err.message || "An error occurred while analyzing the dataset");
    } finally {
      setIsLoading(false);
    }
  };

  const renderVisualization = (viz, index) => {
    if (!data.length) return null; // Wait until data is loaded

    switch (viz.type) {
      case "Bar Chart":
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <h3 className="font-medium text-lg mb-2">{viz.title || "Bar Chart"}</h3>
            <BarChart data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      case "Pie Chart":
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <h3 className="font-medium text-lg mb-2">{viz.title || "Pie Chart"}</h3>
            <PieChart data={data} xColumn={viz.x_column} />
          </div>
        );
      case "Scatter Plot":
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <h3 className="font-medium text-lg mb-2">{viz.title || "Scatter Plot"}</h3>
            <ScatterPlot data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      case "Histogram":
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <h3 className="font-medium text-lg mb-2">{viz.title || "Histogram"}</h3>
            <Histogram data={data} xColumn={viz.x_column} />
          </div>
        );
      case "Box Plot":
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <h3 className="font-medium text-lg mb-2">{viz.title || "Box Plot"}</h3>
            <BoxPlot data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      default:
        return (
          <div key={index} className="chart-container bg-white p-4 rounded-lg shadow-md">
            <p>Unsupported visualization type: {viz.type}</p>
          </div>
        );
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <header className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Data Analysis Dashboard</h1>
        {isFormSubmitted && (
          <button 
            className="btn-secondary"
            onClick={() => setIsFormSubmitted(false)}
          >
            Upload New Dataset
          </button>
        )}
      </header>

      {!isFormSubmitted ? (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Upload Your Dataset</h2>
            <p className="text-gray-600 mb-6">
              Upload your CSV or Excel file to analyze and visualize your data. The system will automatically generate relevant visualizations.
            </p>
            
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center ${
                isDragging ? "border-primary bg-primary/5" : "border-gray-200"
              } transition-colors duration-200`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="bg-primary/10 p-4 rounded-full">
                  <Upload className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <p className="text-lg font-medium">{file ? file.name : "Drag and drop your file here"}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    {file
                      ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type}`
                      : "CSV and Excel files supported (max 10MB)"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-500">or</span>
                  <label className="cursor-pointer text-primary hover:text-primary/80 font-medium">
                    Browse files
                    <input
                      type="file"
                      className="hidden"
                      accept=".csv,.xlsx,.xls"
                      onChange={handleFileChange}
                    />
                  </label>
                </div>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 mt-4 text-red-600 bg-red-50 p-3 rounded-md">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            )}

            <div className="mt-6 flex justify-end">
              <button
                onClick={handleAnalyzeDataset}
                disabled={!file || isLoading}
                className="bg-primary text-white px-4 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <BarChartIcon className="h-5 w-5" />
                    Analyze Dataset
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <h2 className="text-xl font-semibold mb-4">Analysis Results: {file?.name}</h2>
          {visualizations.length === 0 ? (
            <div className="bg-white p-6 rounded-lg shadow-md text-center">
              <p className="text-gray-600">No visualizations were generated for this dataset.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {visualizations.map((viz, index) => renderVisualization(viz, index))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dashboard;

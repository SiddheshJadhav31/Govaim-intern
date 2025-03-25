import { Search, Filter, ArrowUpRight, Upload, Loader2, AlertCircle, FileUp } from 'lucide-react';
import { useState } from 'react';
import Papa from 'papaparse';

const LegislativeTracker = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredBills, setFilteredBills] = useState([]);
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFormSubmitted, setIsFormSubmitted] = useState(false);
  const [data, setData] = useState([]);

  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    
    const filtered = data.filter(bill => 
      bill.title?.toLowerCase().includes(term) ||
      bill.number?.toLowerCase().includes(term) ||
      bill.summary?.toLowerCase().includes(term) ||
      bill.tags?.some(tag => tag.toLowerCase().includes(term))
    );
    
    setFilteredBills(filtered);
  };

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

  const handleAnalyzeLegislation = async () => {
    if (!file) {
      setError("Please select a CSV or Excel file");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_name", "Random Forest"); // Using Random Forest as default model

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
                setFilteredBills(results.data);
                
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
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const result = await response.json();
      setIsFormSubmitted(true);
      console.log("Legislation analysis successful:", result);

    } catch (err) {
      setError(err.message || "An error occurred while analyzing the legislation data");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <header className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Legislative Tracker</h1>
        {isFormSubmitted && (
          <button 
            className="btn-secondary"
            onClick={() => setIsFormSubmitted(false)}
          >
            Upload New Data
          </button>
        )}
      </header>

      {!isFormSubmitted ? (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Upload Legislative Data</h2>
            <p className="text-gray-600 mb-6">
              Upload your CSV or Excel file containing legislative data. The system will analyze and predict bill passage probability.
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
                onClick={handleAnalyzeLegislation}
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
                    <FileUp className="h-5 w-5" />
                    Analyze Legislation
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <div className="flex items-center gap-2">
              <Search className="h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search bills..."
                value={searchTerm}
                onChange={handleSearch}
                className="flex-1 p-2 border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <Filter className="h-5 w-5 text-gray-400 cursor-pointer" />
            </div>
          </div>

          <div className="grid gap-4">
            {filteredBills.map((bill, index) => (
              <div key={index} className="bg-white p-4 rounded-lg shadow-md">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-medium text-lg">{bill.title || 'Untitled Bill'}</h3>
                    <p className="text-sm text-gray-500">{bill.number || 'No number'}</p>
                  </div>
                  <ArrowUpRight className="h-5 w-5 text-gray-400" />
                </div>
                <p className="mt-2 text-gray-600">{bill.summary || 'No summary available'}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {bill.tags?.map((tag, tagIndex) => (
                    <span
                      key={tagIndex}
                      className="px-2 py-1 bg-gray-100 text-gray-600 text-sm rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LegislativeTracker;
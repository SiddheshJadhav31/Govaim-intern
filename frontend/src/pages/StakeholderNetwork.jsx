import { Search, Filter, UserPlus, Mail, Phone, Upload, Loader2, AlertCircle, FileUp } from 'lucide-react';
import { useState } from 'react';
import Papa from 'papaparse';

const StakeholderNetwork = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredStakeholders, setFilteredStakeholders] = useState([]);
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFormSubmitted, setIsFormSubmitted] = useState(false);
  const [data, setData] = useState([]);

  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    
    const filtered = data.filter(stakeholder => 
      stakeholder.name?.toLowerCase().includes(term) ||
      stakeholder.type?.toLowerCase().includes(term) ||
      stakeholder.role?.toLowerCase().includes(term)
    );
    
    setFilteredStakeholders(filtered);
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

  const handleAnalyzeNetwork = async () => {
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
                setFilteredStakeholders(results.data);
                
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
      // Add the required parameters for network graph generation
      formData.append("from_col", "from");
      formData.append("to_col", "to");
      formData.append("amount_col", "amount");

      const response = await fetch("http://localhost:8000/generate-graph/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const result = await response.json();
      setIsFormSubmitted(true);
      console.log("Network analysis successful:", result);

    } catch (err) {
      setError(err.message || "An error occurred while analyzing the network data");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Stakeholder Network</h1>
          <p className="text-gray-600 mt-2">Manage and analyze key relationships</p>
        </div>
        <div className="flex gap-3">
          {isFormSubmitted && (
            <button 
              className="btn-secondary flex items-center gap-2"
              onClick={() => setIsFormSubmitted(false)}
            >
              <Upload size={20} />
              Upload New Data
            </button>
          )}
          <button className="btn-primary flex items-center gap-2">
            <UserPlus size={20} />
            Add Stakeholder
          </button>
        </div>
      </header>

      {!isFormSubmitted ? (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Upload Network Data</h2>
            <p className="text-gray-600 mb-6">
              Upload your CSV or Excel file containing stakeholder network data. The system will analyze relationships and generate a network visualization.
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
                onClick={handleAnalyzeNetwork}
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
                    Analyze Network
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 card">
            <div className="flex gap-4 mb-6">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
                <input
                  type="text"
                  placeholder="Search stakeholders..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20"
                  value={searchTerm}
                  onChange={handleSearch}
                />
              </div>
              <button className="flex items-center gap-2 px-4 py-2 border border-gray-200 rounded-lg hover:bg-gray-50">
                <Filter size={20} />
                Filters
              </button>
            </div>

            <div className="space-y-4">
              {filteredStakeholders.map((stakeholder, index) => (
                <StakeholderCard key={index} {...stakeholder} />
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Network Statistics</h2>
              <div className="space-y-4">
                <StatItem label="Total Stakeholders" value={data.length.toString()} />
                <StatItem label="Active Relationships" value="89" />
                <StatItem label="Influence Score" value="7.8/10" />
                <StatItem label="Recent Interactions" value="24" />
              </div>
            </div>

            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
              <div className="space-y-2">
                <button className="w-full text-left px-4 py-2 hover:bg-gray-50 rounded-lg flex items-center gap-2">
                  <Mail size={20} className="text-gray-500" />
                  Send Email
                </button>
                <button className="w-full text-left px-4 py-2 hover:bg-gray-50 rounded-lg flex items-center gap-2">
                  <Phone size={20} className="text-gray-500" />
                  Schedule Call
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const StakeholderCard = ({ name, role, organization, influence, tags, lastContact }) => (
  <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
    <div className="flex items-start justify-between mb-2">
      <div>
        <h3 className="font-semibold text-lg">{name}</h3>
        <p className="text-gray-600">{role} at {organization}</p>
      </div>
      <div className="text-right">
        <p className="text-sm text-gray-600">Influence Score</p>
        <p className="font-semibold text-primary">{influence}/10</p>
      </div>
    </div>

    <div className="flex items-center justify-between mt-4">
      <div className="flex gap-2">
        {tags.map((tag, index) => (
          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-sm">
            {tag}
          </span>
        ))}
      </div>
      <p className="text-sm text-gray-500">Last Contact: {lastContact}</p>
    </div>
  </div>
);

const StatItem = ({ label, value }) => (
  <div className="flex justify-between items-center">
    <span className="text-gray-600">{label}</span>
    <span className="font-semibold">{value}</span>
  </div>
);

const stakeholders = [
  {
    name: "Sarah Johnson",
    role: "Committee Chair",
    organization: "House Energy Committee",
    influence: 8.5,
    tags: ["Energy", "Climate"],
    lastContact: "2 days ago"
  },
  {
    name: "Michael Chen",
    role: "Policy Director",
    organization: "Tech Innovation Council",
    influence: 7.2,
    tags: ["Technology", "Innovation"],
    lastContact: "1 week ago"
  },
  {
    name: "Amanda Rodriguez",
    role: "Senior Advisor",
    organization: "Department of Commerce",
    influence: 8.9,
    tags: ["Trade", "Economics"],
    lastContact: "3 days ago"
  }
];

export default StakeholderNetwork;
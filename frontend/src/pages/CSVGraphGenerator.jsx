"use client"

import { useState } from "react"
import { Upload, FileUp, AlertCircle, BarChart } from "lucide-react"

const CSVGraphGenerator = () => {
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [columnNames, setColumnNames] = useState([])
  const [fromColumn, setFromColumn] = useState("")
  const [toColumn, setToColumn] = useState("")
  const [amountColumn, setAmountColumn] = useState("")
  const [graphImage, setGraphImage] = useState(null)
  const [step, setStep] = useState(1) // 1: Upload, 2: Select Columns, 3: View Graph

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    validateAndSetFile(droppedFile)
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0])
    }
  }

  const validateAndSetFile = (file) => {
    setError(null)
    setGraphImage(null)

    // Check file type
    const fileType = file.name.split(".").pop()?.toLowerCase()
    if (fileType !== "csv") {
      setError("Please upload a CSV file")
      return
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size should be less than 10MB")
      return
    }

    setFile(file)
  }

  const handleUploadFile = async () => {
    if (!file) {
      setError("Please select a CSV file")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to upload file")
      }

      const data = await response.json()
      setColumnNames(data.columns || [])
      setStep(2) // Move to column selection step
    } catch (err) {
      setError(err.message || "An error occurred while uploading the file")
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateGraph = async () => {
    if (!fromColumn || !toColumn || !amountColumn) {
      setError("Please select all required columns")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("from_col", fromColumn)
      formData.append("to_col", toColumn)
      formData.append("amount_col", amountColumn)

      const response = await fetch("http://localhost:8000/generate-graph/", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to generate graph")
      }

      const data = await response.json()
      setGraphImage(data.graph_url)
      setStep(3) // Move to graph view step
    } catch (err) {
      setError(err.message || "An error occurred while generating the graph")
    } finally {
      setIsLoading(false)
    }
  }

  const renderUploadStep = () => (
    <div className="space-y-6">
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center ${
          isDragging ? "border-blue-500 bg-blue-50" : "border-gray-200"
        } transition-colors duration-200`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="bg-blue-100 p-4 rounded-full">
            <Upload className="h-8 w-8 text-blue-500" />
          </div>
          <div>
            <p className="text-lg font-medium">{file ? file.name : "Drag and drop your CSV file here"}</p>
            <p className="text-sm text-gray-500 mt-1">
              {file
                ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type}`
                : "Only CSV files are supported (max 10MB)"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">or</span>
          </div>
          <label htmlFor="file-upload">
            <div className="bg-blue-500 hover:bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg cursor-pointer transition-colors">
              Browse Files
            </div>
            <input id="file-upload" type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
          </label>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleUploadFile}
          disabled={!file || isLoading}
          className={`px-6 py-2 rounded-lg font-semibold ${
            file && !isLoading
              ? "bg-blue-500 hover:bg-blue-600 text-white"
              : "bg-gray-200 text-gray-500 cursor-not-allowed"
          } transition-colors`}
        >
          {isLoading ? "Uploading..." : "Upload CSV"}
        </button>
      </div>
    </div>
  )

  const renderColumnSelectionStep = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg">
        <p className="text-blue-700 font-medium">File uploaded successfully!</p>
        <p className="text-sm text-blue-600 mt-1">Please select the columns for graph generation</p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">From Column (Source)</label>
          <select
            value={fromColumn}
            onChange={(e) => setFromColumn(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select source column</option>
            {columnNames.map((column) => (
              <option key={`from-${column}`} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">To Column (Destination)</label>
          <select
            value={toColumn}
            onChange={(e) => setToColumn(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select destination column</option>
            {columnNames.map((column) => (
              <option key={`to-${column}`} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Amount Column (Value)</label>
          <select
            value={amountColumn}
            onChange={(e) => setAmountColumn(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select amount column</option>
            {columnNames.map((column) => (
              <option key={`amount-${column}`} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => {
            setStep(1)
            setColumnNames([])
            setFromColumn("")
            setToColumn("")
            setAmountColumn("")
          }}
          className="px-6 py-2 rounded-lg font-semibold border border-gray-300 hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={handleGenerateGraph}
          disabled={!fromColumn || !toColumn || !amountColumn || isLoading}
          className={`px-6 py-2 rounded-lg font-semibold ${
            fromColumn && toColumn && amountColumn && !isLoading
              ? "bg-blue-500 hover:bg-blue-600 text-white"
              : "bg-gray-200 text-gray-500 cursor-not-allowed"
          } transition-colors`}
        >
          {isLoading ? "Generating..." : "Generate Graph"}
        </button>
      </div>
    </div>
  )

  const renderGraphViewStep = () => (
    <div className="space-y-6">
      <div className="bg-green-50 p-4 rounded-lg">
        <p className="text-green-700 font-medium">Graph generated successfully!</p>
      </div>

      <div className="border rounded-lg p-4 bg-white">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart className="h-5 w-5" />
          Transaction Network Graph
        </h3>

        {graphImage && (
          <div className="flex justify-center">
            <img
              src={graphImage || "/placeholder.svg"}
              alt="Transaction Network Graph"
              className="max-w-full rounded-lg border"
            />
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => setStep(2)}
          className="px-6 py-2 rounded-lg font-semibold border border-gray-300 hover:bg-gray-50"
        >
          Back to Column Selection
        </button>
        <button
          onClick={() => {
            setFile(null)
            setColumnNames([])
            setFromColumn("")
            setToColumn("")
            setAmountColumn("")
            setGraphImage(null)
            setStep(1)
          }}
          className="px-6 py-2 rounded-lg font-semibold bg-blue-500 hover:bg-blue-600 text-white"
        >
          Upload New File
        </button>
      </div>
    </div>
  )

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Transaction Network Graph</h1>
          <p className="text-gray-600 mt-2">Upload CSV data to visualize transaction networks</p>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <FileUp className="h-5 w-5" />
            <h2 className="text-xl font-semibold">CSV Data Processor</h2>
          </div>
          <p className="text-gray-600 mt-1">
            Upload your CSV file and select columns to generate a transaction network graph
          </p>
        </div>
        <div className="p-6">
          {/* Progress Steps */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 flex items-center justify-center rounded-full ${
                    step >= 1 ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-500"
                  }`}
                >
                  1
                </div>
                <span className="text-sm mt-1">Upload CSV</span>
              </div>
              <div className={`flex-1 h-1 mx-2 ${step >= 2 ? "bg-blue-500" : "bg-gray-200"}`}></div>
              <div className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 flex items-center justify-center rounded-full ${
                    step >= 2 ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-500"
                  }`}
                >
                  2
                </div>
                <span className="text-sm mt-1">Select Columns</span>
              </div>
              <div className={`flex-1 h-1 mx-2 ${step >= 3 ? "bg-blue-500" : "bg-gray-200"}`}></div>
              <div className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 flex items-center justify-center rounded-full ${
                    step >= 3 ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-500"
                  }`}
                >
                  3
                </div>
                <span className="text-sm mt-1">View Graph</span>
              </div>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-md mb-6">
              <AlertCircle className="h-5 w-5" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {step === 1 && renderUploadStep()}
          {step === 2 && renderColumnSelectionStep()}
          {step === 3 && renderGraphViewStep()}
        </div>
      </div>
    </div>
  )
}

export default CSVGraphGenerator


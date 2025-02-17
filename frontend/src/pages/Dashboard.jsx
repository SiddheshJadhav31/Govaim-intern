import React, { useState, useEffect } from "react";
import * as d3 from "d3";
import Papa from "papaparse";
import visualizations from "../../public/response.json";
import BarChart from "../components/BarChart";
import PieChart from "../components/PieChart";
import ScatterPlot from "../components/ScatterPlot";
import Histogram from "../components/Histogram";
import BoxPlot from "../components/BoxPlot";

const Dashboard = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    Papa.parse("/titanic.csv", {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: (result) => setData(result.data),
    });
  }, []);

  const renderVisualization = (viz, index) => {
    if (!data.length) return null; // Wait until data is loaded

    switch (viz.type) {
      case "Bar Chart":
        return (
          <div key={index} className="chart-container">
            <BarChart data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      case "Pie Chart":
        return (
          <div key={index} className="chart-container">
            <PieChart data={data} xColumn={viz.x_column} />
          </div>
        );
      case "Scatter Plot":
        return (
          <div key={index} className="chart-container">
            <ScatterPlot data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      case "Histogram":
        return (
          <div key={index} className="chart-container">
            <Histogram data={data} xColumn={viz.x_column} />
          </div>
        );
      case "Box Plot":
        return (
          <div key={index} className="chart-container">
            <BoxPlot data={data} xColumn={viz.x_column} yColumn={viz.y_column} />
          </div>
        );
      default:
        return (
          <div key={index} className="chart-container">
            <p>Unsupported visualization type: {viz.type}</p>
          </div>
        );
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {visualizations.visualizations.map((viz, index) => renderVisualization(viz, index))}
      </div>
    </div>
  );
};

export default Dashboard;

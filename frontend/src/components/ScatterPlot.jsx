import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const ScatterPlot = ({ data, xColumn, yColumn }) => {
  const ref = useRef();

  useEffect(() => {
    if (!data.length || !xColumn || !yColumn) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 50, left: 60 };
    const width = 450 - margin.left - margin.right;
    const height = 320 - margin.top - margin.bottom;

    const xScale = d3.scaleLinear().domain(d3.extent(data, d => d[xColumn])).nice().range([0, width]);
    const yScale = d3.scaleLinear().domain(d3.extent(data, d => d[yColumn])).nice().range([height, 0]);

    const svgContainer = svg
      .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X Axis
    svgContainer.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    // Y Axis
    svgContainer.append("g").call(d3.axisLeft(yScale));

    // Dots
    svgContainer.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d[xColumn]))
      .attr("cy", d => yScale(d[yColumn]))
      .attr("r", 4)
      .attr("fill", "steelblue")
      .attr("opacity", 0.8);

  }, [data, xColumn, yColumn]);

  return (
    <div className="p-4 bg-white shadow-lg rounded-2xl transition-all duration-300 hover:bg-gray-100">
      <svg ref={ref} width="100%" height="350"></svg>
    </div>
  );
};

export default ScatterPlot;

import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const BarChart = ({ data, xColumn, yColumn }) => {
  const ref = useRef();

  useEffect(() => {
    if (!data.length || !xColumn || !yColumn) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove(); // Clear previous render

    const margin = { top: 20, right: 20, bottom: 50, left: 60 };
    const width = 450 - margin.left - margin.right;
    const height = 320 - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
      .domain(data.map(d => d[xColumn]))
      .range([0, width])
      .padding(0.4);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d[yColumn]) * 1.1]) // Adding 10% padding
      .range([height, 0]);

    const svgContainer = svg
      .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X Axis
    svgContainer.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .attr("transform", "rotate(-30)")
      .style("text-anchor", "end");

    // Y Axis
    svgContainer.append("g").call(d3.axisLeft(yScale));

    // Bars with transition
    svgContainer.selectAll(".bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d[xColumn]))
      .attr("width", xScale.bandwidth())
      .attr("y", height)
      .attr("height", 0)
      .attr("fill", "steelblue")
      .transition()
      .duration(800)
      .attr("y", d => yScale(d[yColumn]))
      .attr("height", d => height - yScale(d[yColumn]));

    // Add labels on bars
    svgContainer.selectAll(".label")
      .data(data)
      .enter()
      .append("text")
      .attr("x", d => xScale(d[xColumn]) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d[yColumn]) - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "black")
      .attr("font-size", "12px")
      .text(d => d[yColumn]);

  }, [data, xColumn, yColumn]);

  return (
    <div className="p-4 bg-white shadow-lg rounded-2xl transition-all duration-300 hover:bg-gray-100">
      <svg ref={ref} width="100%" height="350"></svg>
    </div>
  );
};

export default BarChart;

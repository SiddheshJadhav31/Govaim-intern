import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const Histogram = ({ data, xColumn, bins = 10 }) => {
  const ref = useRef();

  useEffect(() => {
    if (!data.length) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = 450 - margin.left - margin.right;
    const height = 320 - margin.top - margin.bottom;

    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d[xColumn]))
      .nice()
      .range([0, width]);

    const histogram = d3.histogram().domain(xScale.domain()).thresholds(xScale.ticks(bins));
    const binsData = histogram(data.map(d => d[xColumn]));

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(binsData, d => d.length)])
      .nice()
      .range([height, 0]);

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

    // Bars
    svgContainer.selectAll("rect")
      .data(binsData)
      .enter()
      .append("rect")
      .attr("x", d => xScale(d.x0))
      .attr("width", d => xScale(d.x1) - xScale(d.x0) - 1)
      .attr("y", height)
      .attr("height", 0)
      .attr("fill", "steelblue")
      .transition()
      .duration(800)
      .attr("y", d => yScale(d.length))
      .attr("height", d => height - yScale(d.length));

  }, [data, xColumn, bins]);

  return (
    <div className="p-4 bg-white shadow-lg rounded-2xl transition-all duration-300 hover:bg-gray-100">
      <svg ref={ref} width="100%" height="350"></svg>
    </div>
  );
};

export default Histogram;

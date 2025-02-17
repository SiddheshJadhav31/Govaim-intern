import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const BoxPlot = ({ data, xColumn }) => {
  const ref = useRef();

  useEffect(() => {
    if (!data.length) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = 450 - margin.left - margin.right;
    const height = 320 - margin.top - margin.bottom;

    const values = data.map(d => d[xColumn]).sort(d3.ascending);
    const q1 = d3.quantile(values, 0.25);
    const median = d3.quantile(values, 0.5);
    const q3 = d3.quantile(values, 0.75);
    const min = d3.min(values);
    const max = d3.max(values);

    const xScale = d3.scaleLinear().domain([min, max]).range([0, width]);

    const svgContainer = svg
      .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    svgContainer.append("line")
      .attr("x1", xScale(min))
      .attr("x2", xScale(max))
      .attr("y1", height / 2)
      .attr("y2", height / 2)
      .attr("stroke", "black");

    svgContainer.append("rect")
      .attr("x", xScale(q1))
      .attr("width", xScale(q3) - xScale(q1))
      .attr("y", height / 4)
      .attr("height", height / 2)
      .attr("fill", "lightblue")
      .attr("stroke", "black");

    svgContainer.append("line")
      .attr("x1", xScale(median))
      .attr("x2", xScale(median))
      .attr("y1", height / 4)
      .attr("y2", height * 0.75)
      .attr("stroke", "black");

  }, [data, xColumn]);

  return (
    <div className="p-4 bg-white shadow-lg rounded-2xl transition-all duration-300 hover:bg-gray-100">
      <svg ref={ref} width="100%" height="350"></svg>
    </div>
  );
};

export default BoxPlot;

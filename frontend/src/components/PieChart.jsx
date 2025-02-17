import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const PieChart = ({ data, xColumn }) => {
  const ref = useRef();

  useEffect(() => {
    if (!data.length) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 300;
    const height = 300;
    const radius = Math.min(width, height) / 2;
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const pieData = Array.from(d3.rollup(data, v => v.length, d => d[xColumn]), ([key, value]) => ({ key, value }));
    const total = d3.sum(pieData, d => d.value);

    const pie = d3.pie().value(d => d.value);
    const arc = d3.arc().innerRadius(0).outerRadius(radius);
    const arcHover = d3.arc().innerRadius(0).outerRadius(radius + 10); // Expand on hover

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`);

    const paths = g
      .selectAll("path")
      .data(pie(pieData))
      .enter()
      .append("path")
      .attr("d", arc)
      .attr("fill", d => color(d.data.key))
      .style("cursor", "pointer")
      .on("mouseover", function (_, d) {
        d3.select(this).transition().duration(200).attr("d", arcHover);
      })
      .on("mouseout", function (_, d) {
        d3.select(this).transition().duration(200).attr("d", arc);
      });

    // Add labels
    g.selectAll("text")
      .data(pie(pieData))
      .enter()
      .append("text")
      .attr("transform", d => `translate(${arc.centroid(d)})`)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .attr("fill", "white")
      .text(d => `${Math.round((d.data.value / total) * 100)}%`);

  }, [data, xColumn]);

  return (
    <div className="p-4 bg-white shadow-lg rounded-2xl transition-all duration-300 hover:bg-gray-100">
      <svg ref={ref} width="100%" height="300"></svg>
    </div>
  );
};

export default PieChart;

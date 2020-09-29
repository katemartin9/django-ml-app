function getLabels(data) {
    var arr = []
    for (const [key, value] of Object.entries(data)) {
        if (arr.includes(value.index) === false) {
            arr.push(value.index);
        }
    }
    return arr
}

fetch('http://127.0.0.1:8000/regml/model-options/boston project/api')
    .then(response => {
      return response.json();
    })
    .then(data => {
      const corrData = data[0].output
      const linearData = data[1].output

      const margin = { top: 30, right: 20, bottom: 60, left: 50 };
      const width = 350 - margin.left - margin.right;
      const height = 350 - margin.top - margin.bottom;

      const corrMatrix = d3
        .select("body")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      const columns = getLabels(corrData);

      // Build X scales and axis:
      var x = d3
          .scaleBand()
          .range([ 0, width])
          .domain(columns)
          .padding(0.01);

      corrMatrix.append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "xaxis")
          .call(d3.axisBottom(x))

      // Build X scales and axis:
      var y = d3
          .scaleBand()
          .range([height, 0])
          .domain(columns)
          .padding(0.01);

      corrMatrix
          .append("g")
          .attr("class", "yaxis")
          .call(d3.axisLeft(y));

      // Build color scale
      var myColor = d3
          .scaleLinear()
          .range(["#ffe2e2", "#800000"])
          .domain([0,1])

      corrMatrix
          .selectAll()
          .data(corrData)
          .enter()
          .append("rect")
          .attr("x", function(d) { return x(d.index) })
          .attr("y", function(d) { return y(d.variable) })
          .attr("width", x.bandwidth() )
          .attr("height", y.bandwidth() )
          .style("fill", function(d) { return myColor(d.value)} );

      corrMatrix
          .select(".xaxis")
          .selectAll('text')
          .attr("text-anchor", "start")
          .attr("dx", "1.3em")
          .attr("dy", "-.3em")
          .attr("transform", "rotate(70)")
    });


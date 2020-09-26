fetch('http://127.0.0.1:8000/regml/model-options/boston project/api')
    .then(response => {
      return response.json();
    })
    .then(data => {
      console.log(data);
      const margin = { top: 30, right: 20, bottom: 30, left: 50 };
      const width = 300 - margin.left - margin.right;
      const height = 300 - margin.top - margin.bottom;

      function createSvg() {
      d3
      .select("body")
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    }
      var corrMatrix = createSvg()
      var linePlots = createSvg()
      debugger;
    });


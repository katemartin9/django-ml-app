function getPointCategoryName(point, dimension) {
    var series = point.series,
        isY = dimension === 'y',
        axis = series[isY ? 'yAxis' : 'xAxis'];
    return axis.categories[point[isY ? 'y' : 'x']];
}

var CORRMATRIX = CORRMATRIX || (function() {
    var _args = {};
    return {
        init: function(Args) {
            _args = Args;
        },
        plotData : function () {
            let source = _args[0];
        fetch(`${source}/api`)
            .then(response => {
              return response.json();
            })
            .then(data => {
                const corrData = data[0];

                const axisVals = corrData["output"]
                    .map(element => element.index)
                    .filter((item, i, ar) => ar.indexOf(item) === i)
                    .sort();
                const innerVals = corrData["output"].map(element => [axisVals.indexOf(element.index),
                    axisVals.indexOf(element.variable), Math.round(element.value * 10) / 10])

                // corr matrix plot
                Highcharts.chart('matrix_container', {

                    chart: {
                        type: 'heatmap',
                        marginTop: 40,
                        marginBottom: 80,
                        plotBorderWidth: 1
                    },

                    title: {
                        text: 'Feature Correlation Matrix'
                    },

                    xAxis: {
                        categories: axisVals
                    },

                    yAxis: {
                        categories: axisVals,
                        title: null,
                        reversed: false
                    },

                    accessibility: {
                        point: {
                            descriptionFormatter: function (point) {
                                var ix = point.index + 1,
                                    xName = getPointCategoryName(point, 'x'),
                                    yName = getPointCategoryName(point, 'y'),
                                    val = point.value;
                                return ix + '. ' + xName + ' sales ' + yName + ', ' + val + '.';
                            }
                        }
                    },

                    colorAxis: {
                        min: -1,
                        max: 1,
                           stops: [
                                  [0.1, "#fa0202"],
                                  [0.5, "#ffffff"],
                                  [0.9, "#076611"]
                              ],
                    },

                    legend: {
                        align: 'right',
                        layout: 'vertical',
                        margin: 0,
                        verticalAlign: 'top',
                        y: 25,
                        symbolHeight: 280
                    },

                    tooltip: {
                        formatter: function () {
                            return '<b>' + getPointCategoryName(this.point, 'x') + '</b> <br><b>' +
                                this.point.value + '</b> <br><b>' + getPointCategoryName(this.point, 'y') + '</b>';
                        }
                    },

                    series: [{
                        name: 'Corr Matrix',
                        borderWidth: 0.2,
                        data: innerVals,
                        dataLabels: {
                            enabled: true,
                            color: '#000000'
                        }
                    }],

                    responsive: {
                        rules: [{
                            condition: {
                                maxWidth: 500
                            },
                            chartOptions: {
                                yAxis: {
                                    labels: {
                                        formatter: function () {
                                            return this.value.charAt(0);
                                        }
                                    }
                                }
                            }
                        }]
                    }

                });

            })

            }
        }

    }())
var dataset;
var Borough = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
    Crime_Type = ["Assault", "Burglary", "Housing", "Murder", "Robbery", "Shooting"];

var margin = {top:100, right:75, bottom:70, left:125};

// calculate width and height based on window size
var w = Math.max(Math.min(window.innerWidth, 1000), 500) - margin.left - margin.right - 20,
    gridSize = Math.floor(w / Crime_Type.length),
    h = gridSize * (Borough.length+2);

// Reset the overall font size
var newFontSize = w * 62.5 / 900;
d3.select("html")
    .style("font-size", newFontSize + "%");

// Create Svg container
var svg = d3.select("#heatmap")
    .append("svg")
    .attr("width", w + margin.top + margin.bottom + 200)
    .attr("height", h + margin.left + margin.right + 25)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Add the title
svg.append("text")
    .attr("class", "Title")
    .attr("x", 20)
    .attr("y", -50)
    .attr("fill", "#000")
    .attr("text-anchor", "center")
    .attr("font-weight", "bold")
    .text("Visualizing Crimes in New York City");

// Add the x Axis title
svg.append("text")
    .attr("class", "xAxis")
    .attr("x", w)
    .attr("y", h*0.73)
    .attr("fill", "#000")
    .attr('font-size', '1.5em')
    .attr("text-anchor", "right")
    .attr("font-weight", "bold")
    .text("Crime Type");

// Add the y Axis title
svg.append("text")
    .attr("class", "yAxis")
    .attr("x", -85)
    .attr("y", -10)
    .attr("fill", "#000")
    .attr('font-size', '1.5em')
    .attr("text-anchor", "left")
    .attr("font-weight", "bold")
    .text("Borough");

// Add the labels for the different boroughs
var boroughLabels = svg.selectAll(".boroughLabel")
    .data(Borough)
    .enter()
    .append("text")
    .text(function(d) { return d; })
    .attr("x", 0)
    .attr("y", function(d, i) { return i * gridSize; })
    .style("text-anchor", "end")
    .attr("transform", "translate(-6," + gridSize / 1.5 + ")");

// Add the labels for crime types
var crimeLabels = svg.selectAll(".crimeLabel")
    .data(Crime_Type)
    .enter()
    .append("text")
    .text(function(d) { return d; })
    .attr("x", function(d, i) { return i * gridSize; })
    .attr("y", h - (w * 0.305))
    .style("text-anchor", "middle")
    .attr("transform", "translate(" + gridSize / 2 + ", -6)");


// Load the data
d3.csv("heatmap.csv").then(data => {
    data.forEach(function(d) {
        d.Bronx = +d.Bronx;
        d.Brooklyn = +d.Brooklyn;
        d.Manhattan = +d.Manhattan;
        d.Queens = +d.Queens;
        d['Staten Island'] = +d['Staten Island'];
        d.Year = +d.Year;
    });

    dataset = data;

    // Group data by Year
    var nest = d3.nest()
        .key(function(d) { return d.Year; })
        .entries(dataset);

    // Array of Years in the data
    var years = nest.map(function(d) { return d.key; });
    var currentYearIndex = 0;

    // Color scale used as a dummy because I was not able to get the scaleSequential
    //    along with Interpolate to work
    var color = d3.scaleQuantize()
        .domain([1, 10])
        .range(d3.schemeBlues[9]);

    var format = d3.format("");

    // Create the Year dropdown menu
    var yearMenu = d3.select("#yearDropdown");
    yearMenu
        .append("select")
        .attr("id", "yearMenu")
        .selectAll("option")
        .data(years)
        .enter()
        .append("option")
        .attr("value", function(d, i) { return i; })
        .text(function(d) { return d; });

    // Function to create the initial heatmap
    var drawHeatmap = function(CrimeArray) {

        for(var row = 0; row < CrimeArray.length; row++){
            var heatmap = svg.selectAll(".crimes")
                .data(CrimeArray[row])
                .enter()
                .append("rect")
                .attr("x", function(d, i) {
                    return (row) * gridSize; })
                .attr("y", function(d, i) {
                    return (i) * gridSize; })
                .attr('rx', 15)
                .attr('ry', 15)
                .attr('id', function(d, i){
                    return row.toString()+i.toString();
                })
                .attr("class", "crime bordered")
                .attr("width", gridSize)
                .attr("height", gridSize)
                .attr("stroke", "white")
                .attr("stroke-opacity", 0.5)
                .attr("fill", function(d){
                    return testColorScale(d);})
                .attr("opacity", function(d){
                    return 1;
                });
        };
    };
    // Temporary variable to hold the year we are looking for
    var temp = years[currentYearIndex];

    // Find all the data related to the year in temp
    var selectYear = nest.find(function(d) {
        return d.key === temp;
    });

    // A function to return an array with just the specific crimes we want to find
    function getArrayForCrime(inputData, outputArray,  crimeType){
        for(var i = 0; i < inputData.length; i++){
            if(inputData[i]['Crime Type'] === crimeType){
                outputArray.push(inputData[i].Bronx);
                outputArray.push(inputData[i].Brooklyn);
                outputArray.push(inputData[i].Manhattan);
                outputArray.push(inputData[i].Queens);
                outputArray.push(inputData[i]['Staten Island']);
            };
        };
    };

    // Initializing the arrays for the different types of crimes
    var murderArray = [], robberyArray = [],
        assaultArray = [], burglaryArray = [],
        shootingArray = [],housingArray = [];

    // A function to return an array with all crime types in a specified year
    function getAllCrimes(CrimeArray, data){

        var murderArray = [], robberyArray = [],
            assaultArray = [], burglaryArray = [],
            shootingArray = [],housingArray = [];

        var CrimeArray = [assaultArray, burglaryArray,
            housingArray, murderArray,
            robberyArray, shootingArray];

        getArrayForCrime(data.values, murderArray, 'Murder');
        getArrayForCrime(data.values, robberyArray, 'Robbery');
        getArrayForCrime(data.values, assaultArray, 'Assault');
        getArrayForCrime(data.values, burglaryArray, 'Burglary');
        getArrayForCrime(data.values, shootingArray, 'Shooting');
        getArrayForCrime(data.values, housingArray, 'Housing');

        return CrimeArray;
    };

    // Initializing the CrimeArray
    var CrimeArray = [];

    // Populating the CrimeArray with all crime data for a given year
    CrimeArray = getAllCrimes(CrimeArray, selectYear);

    var flattenedCrimeArray = [];

    // Flatten tje CrimeArray to be a single array instead of an Array of Arrays
    for(var L = 0; L < CrimeArray.length; L++){
        flattenedCrimeArray = flattenedCrimeArray.concat(CrimeArray[L]);
    };

    // Setting legendValues to the flattenedCrimeArray as to not cause issues in other areas
    var legendValues = flattenedCrimeArray;

    // I was not able to use d3.extent so I just got the min and max
    var range = Math.max(...legendValues) - Math.min(...legendValues);

    // Creating a variable to hold the amount I would need to go up for my threshold based on the
    //   range of the data
    var legendStepValue = Math.floor(range/9);

    // Initializing this array with a 0 prior to filling
    var legendTickValues = [0];

    // Filling the legendTickValues array with the calculated values
    for(var step = 0; step < 8; step++){
        legendTickValues.push(Math.round((Math.min(...legendValues) + (step * legendStepValue)+(legendStepValue/4))));
    };

    // Add the numbers to the legend as the thresholds
    svg.selectAll(".crimes")
        .data(legendTickValues)
        .enter()
        .append('text')
        .attr('class', 'legendNumbers')
        .attr("x", function(d,i){return  i * 85;})
        .attr("y", 835)
        .attr("fill", "#000")
        .attr("text-anchor", "start")
        .attr("font-weight", "bold")
        .text(function(d){return d});

    // This function is used to update the values of the already present tick values
    //    I had issues when this was not broken out into its own function
    function addTickValues(flattenedCrimeArray){

        var legendValues = flattenedCrimeArray;

        var range = Math.max(...legendValues) - Math.min(...legendValues);

        var legendStepValue = Math.floor(range/9);
        var legendTickValues = [0];

        for(var step = 0; step < 8; step++){
            legendTickValues.push(Math.round((Math.min(...legendValues) + (step * legendStepValue)+(legendStepValue/6))));
        };

        // Process the current threshold values and update them accordingly
        svg.selectAll(".legendNumbers")
            .text(function(d, i){
                d = legendTickValues[i];
                return d;});
    };


    var legend = g => {
        const x = d3.scaleLinear()
            .domain(d3.extent(color.domain()))
            .rangeRound([0,600]);

        var testColorScale = d3.scaleSequential(d3["interpolate" + "RdPu"])
            .domain([0, Math.max(...flattenedCrimeArray)-20])
            .clamp(true);

        g.selectAll("rect")
            .data(color.range().map(d => color.invertExtent(d)))
            .enter()
            .append("rect")
            .attr("height", (gridSize*0.5))
            .attr("x", function(d, i){
                return 86*i;})
            .attr("width",function(d) {
                return (86);
            })
            .attr('rx', 6)
            .attr('ry', 6)
            .attr("fill", function(d, i){
                return testColorScale(legendTickValues[i]);
            })
            .exit();

        //Title for legend
        g.append("text")
            .attr("class", "caption")
            .attr("x", x.range()[0])
            .attr("y", -6)
            .attr("fill", "#000")
            .attr("text-anchor", "start")
            .attr("font-weight", "bold")
            .text("No. of Crimes");

        // Add the axis to the bottom of the screen and removing the domain line
        g.call(d3.axisBottom(x)
            .tickSize(gridSize*Math.pow(9,-1))
            .tickFormat(format)
            .ticks(9)
            .tickValues([25,50]))
            .select(".domain")
            .remove();
    };

    // Adding a new group to the SVG where we will add the legend
    svg.append("g")
        .attr("transform", "translate(0,750)")
        .call(legend);

    // The color scale used
    var testColorScale = d3.scaleSequential(d3["interpolate" + "RdPu"])
        .domain([0, Math.max(...flattenedCrimeArray)-20])
        .clamp(true);

    // Initial heatmap generation
    drawHeatmap(CrimeArray);

    // Function to update the heatmap
    var updateHeatmap = function(thisData) {

        // I had to define the color scale inside here as well as I was having issues
        var testColorScale = d3.scaleSequential(d3["interpolate" + "RdPu"])
            .domain([0, Math.max(...flattenedCrimeArray)-20])
            .clamp(true);

        // Update the values of the thresholds
        addTickValues(thisData);

        // Update the colors on the heat map based on the passed in data
        var heatmap = svg.selectAll(".crime")
            .data(thisData)
            .transition()
            .duration(500)
            .style("fill", function(d){
                return testColorScale(d);
            });
    };

    // Run update when the drop down has been used to select a year
    yearMenu.on("change", function() {
        // Get the year that was selected
        var selectedYear = d3.select(this)
            .select("select")
            .property("value");
        currentYearIndex = +selectedYear;
        var temp = years[currentYearIndex];

        // Get the data associated with that year
        var selectYear = nest.find(function(d) {
            return d.key === temp;
        });

        // Resetting the CrimeArray so we do no have overlapping values
        var CrimeArray = [];

        // Getting all the crime data for the selected year
        var CrimeArray = getAllCrimes(CrimeArray, selectYear);
        var flattenedCrimeArray = [];

        // Flatten the CrimeArray from an Array of Arrays into a single array
        for(var L = 0; L < CrimeArray.length; L++){
            flattenedCrimeArray = flattenedCrimeArray.concat(CrimeArray[L]);
        }

        // Update the heatmap
        updateHeatmap(thisData = flattenedCrimeArray);
    });

    // This will allow the navigation based on the press of 'prev' or 'next'
    d3.selectAll(".nav").on("click", function() {
        if(d3.select(this).classed("prev")) {
            if(currentYearIndex === 0) {
                currentYearIndex = years.length-1;
            } else {
                currentYearIndex--;
            }
        } else if(d3.select(this).classed("next")) {
            if(currentYearIndex === years.length-1) {
                currentYearIndex = 0;
            } else {
                currentYearIndex++;
            }
        }
        d3.select("#yearMenu").property("value", currentYearIndex);
        var temp = years[currentYearIndex];

        // Find the data associated with the year selected
        var selectYear = nest.find(function(d) {
            return d.key === temp;
        });

        // Resetting the value of CrimeArray to prevent overlapping data
        var CrimeArray = [];

        // Populating CrimeArray with the crime data for the selected year
        var CrimeArray = getAllCrimes(CrimeArray, selectYear);
        var flattenedCrimeArray = [];

        // Flatten CrimeArray from an Array of Arrays to one single Array
        for(var L = 0; L < CrimeArray.length; L++){
            flattenedCrimeArray = flattenedCrimeArray.concat(CrimeArray[L]);
        }

        // Update the heatmap
        updateHeatmap(flattenedCrimeArray);
    })
})
var tempData = [
    {country: 'Bangladesh',
        population_2012: 105905297,
        growth: {year_2013:42488,
            year_2014:934,
            year_2015:52633,
            year_2016:112822,
            year_2017:160792}},

    {country: 'Ethopia',
        population_2012: 75656319,
        growth: {year_2013:1606010,
            year_2014:1606705,
            year_2015:1600666,
            year_2016:1590077,
            year_2017:1580805}},

    {country: 'Kenya',
        population_2012: 33007327,
        growth: {year_2013:705153,
            year_2014:703994,
            year_2015:699906,
            year_2016:694295,
            year_2017:687910}},

    {country: 'Afghanistan',
        population_2012: 23280573,
        growth: {year_2013:717151,
            year_2014:706082,
            year_2015:665025,
            year_2016:616262,
            year_2017:573643}},

    {country: 'Morocco',
        population_2012: 13619520,
        growth: {year_2013:11862,
            year_2014:7997,
            year_2015:391,
            year_2016:-8820,
            year_2017:-17029}}
];

var Country_Data = {};
var listOfCounties = {};

// Population Country_Data with the copied in data and modify to be more usable
for(var key in tempData) {
    Country_Data[tempData[key]['country']] = {
        'Initial_Population': tempData[key]['population_2012'],
        'Growth_Per_Year': {
            '2013' : {"Population_Change_This_Year" : tempData[key]['growth']['year_2013'],
                "Total_Population_This_Year" : tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'],
                "Population_Growth_Percentage" : (((tempData[key]['growth']['year_2013']) / (tempData[key]['population_2012'])) * 100)},

            '2014' : {"Population_Change_This_Year" : tempData[key]['growth']['year_2014'],
                "Total_Population_This_Year" : tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'],
                "Population_Growth_Percentage" : (((tempData[key]['growth']['year_2014']) / (tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'])) * 100)},

            '2015' : {"Population_Change_This_Year" : tempData[key]['growth']['year_2015'],
                "Total_Population_This_Year" : tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'] + tempData[key]['growth']['year_2015'],
                "Population_Growth_Percentage" : (((tempData[key]['growth']['year_2015']) / (tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'])) * 100)},

            '2016' : {"Population_Change_This_Year" : tempData[key]['growth']['year_2016'],
                "Total_Population_This_Year" : tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'] + tempData[key]['growth']['year_2015'] + tempData[key]['growth']['year_2016'],
                "Population_Growth_Percentage" : (((tempData[key]['growth']['year_2016']) / (tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'] + tempData[key]['growth']['year_2015'])) * 100)},

            '2017' : {"Population_Change_This_Year" : tempData[key]['growth']['year_2017'],
                "Total_Population_This_Year" : tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'] + tempData[key]['growth']['year_2015'] + tempData[key]['growth']['year_2016'] + tempData[key]['growth']['year_2017'],
                "Population_Growth_Percentage" : (((tempData[key]['growth']['year_2017']) / (tempData[key]['population_2012'] + tempData[key]['growth']['year_2013'] + tempData[key]['growth']['year_2014'] + tempData[key]['growth']['year_2015'] + tempData[key]['growth']['year_2016'])) * 100)}
        },

        'Total_Population': tempData[key]['population_2012'] +
            tempData[key]['growth']['year_2013'] +
            tempData[key]['growth']['year_2014'] +
            tempData[key]['growth']['year_2015'] +
            tempData[key]['growth']['year_2016'] +
            tempData[key]['growth']['year_2017']
    };
    listOfCounties[key] = {"Name": tempData[key]['country'],
        "Total_Population": tempData[key]['population_2012'] +
            tempData[key]['growth']['year_2013'] +
            tempData[key]['growth']['year_2014'] +
            tempData[key]['growth']['year_2015'] +
            tempData[key]['growth']['year_2016'] +
            tempData[key]['growth']['year_2017']};
};

//  Just to print out all the data to verify integrity
//
// console.log("Here is the list of countries: " + listOfCounties);
// for(var key in listOfCounties){
//     console.log("key: " + listOfCounties[key]);
//     console.log("listOfCounties[key].Name: " + listOfCounties[key].Name);
//     console.log("listOfCounties[key].Total_Population: " + listOfCounties[key].Total_Population);
//
// }
// for(var key in Country_Data ){
//     " + key + "
//     console.log("Here is the key: " + key);
//     console.log("Here is Country_Data[" + key + "]: " + Country_Data[key]);
//     console.log("Here is Country_Data[" + key + "]['Initial_Population']: " + Country_Data[key]['Initial_Population']);
//     console.log("Here is Country_Data[" + key + "]['Growth_Per_Year']" + Country_Data[key]['Growth_Per_Year']);
//     for(var newKey in Country_Data[key]['Growth_Per_Year']){
//         for(var lastKey in Country_Data[key]['Growth_Per_Year'][newKey]){
//             console.log("Here is Country_Data["+key+"]['Growth_Per_Year']["+newKey+"]["+lastKey+"]: " + Country_Data[key]['Growth_Per_Year'][newKey][lastKey]);
//         }
//     }
//     console.log("Here is Country_Data[key]['Total_Population']: " + Country_Data[key]['Total_Population']);
// };

var margin = {top: 200, right: 600, bottom: 30, left: 150},
    width = 1800 - margin.left - margin.right,
    height = 720 - margin.top - margin.bottom,
    tooltipHeight = 300,
    tooltipWidth = 300;

function MainChart(MainChartCountryData, MainChartCountryList, Margin, mainWidth, mainHeight, ){

    function GetListOfCountryNames(passedInList, outputListOfCountryNames){
        for(var tempKey in passedInList){
            outputListOfCountryNames.push(passedInList[tempKey].Name);
        }
        return outputListOfCountryNames;
    };

    function GetListOfCountryPopulations(outputListOfPopulations){
        for(var tempKey in MainChartCountryList){
            outputListOfPopulations.push(MainChartCountryList[tempKey].Total_Population)
        };

        return outputListOfPopulations;
    };

    function CreateListOfLists(listToReturn){
        for(var i = 0; i < countryPopulations.length; i++){
            listToReturn.push([countryNames[i],countryPopulations[i]]);
        };
        return listToReturn;
    };

    function CreateListOfPopulationGrowth(listOfPopulationGrowthToReturn){

        for(var anotherTempKey in Country_Data){
            var tempList = [];
            for(var theLastTempKey in Country_Data[anotherTempKey]['Growth_Per_Year']){
                tempList.push(Country_Data[anotherTempKey]['Growth_Per_Year'][theLastTempKey].Population_Growth_Percentage);
            }
            listOfPopulationGrowthToReturn.push(tempList);
        };
    };

    function CreateToolTipData(listOfYears, listOfPopGrowth, dataForToolTip, tableForExtent){
        for(var temp = 0; temp < listOfPopulationGrowth.length; temp++){
            var tempTable = [];
            var tempTableForGrowth = []
            for(var k = 0; k < listOfPopulationGrowth[temp].length; k++){
                tempTableForGrowth.push(listOfPopulationGrowth[temp][k]);
                tempTable.push([listOfYears[k],listOfPopulationGrowth[temp][k]])
            }
            dataForTooltip.push(tempTable);
            tableForExtent.push(tempTableForGrowth);
        }
    };

    var formatXAxis = d3.format(",.2r");

    var extentTableForTooltip = [];
    var listOfPopulationGrowth = [];
    var countryPopulations = [];
    var countryNames = [];
    var listOfNamesAndPops = [];
    var listOfYears = [2013,2014,2015,2016,2017];
    var dataForTooltip = [];

    GetListOfCountryNames(MainChartCountryList, countryNames);
    GetListOfCountryPopulations(countryPopulations);
    CreateListOfPopulationGrowth(listOfPopulationGrowth);
    CreateListOfLists(listOfNamesAndPops);
    CreateToolTipData(listOfYears, listOfPopulationGrowth, dataForTooltip, extentTableForTooltip);

    // Generate a scale using the country names
    var scaleY = d3.scalePoint()
        .domain(countryNames)
        .range([20, (mainHeight-60)]);

    var scaleX = d3.scaleLinear()
        .domain([0,(d3.extent(countryPopulations)[1]*1.1)])
        .range([0, mainWidth])
        .clamp(true);

    // This is the xAxis which is going horizontal and is the population
    var xAxis = d3.axisBottom(scaleX)
        .ticks(8)
        .tickFormat(formatXAxis)
        .tickSize(0)
        .tickSizeInner(0)
        .tickSizeOuter(0)
        .tickPadding(15);

    // This is the yAxis which is going vertically and is the name of the counties
    var yAxis = d3.axisLeft(scaleY)
        .tickPadding(10)
        .tickSize(0)
        .tickSizeInner(0)
        .tickSizeOuter(0);


// GO HERE
    // https://stackoverflow.com/questions/43904643/add-chart-to-tooltip-in-d3

    var svg = d3.select("body").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    //svg.call(toolTip);

    var xAxis = svg.append("g")
        .attr("class", "xAxis")
        .attr("transform", "translate(0," + (height-30) + ")")
        .call(xAxis);

    // Define yAxis and generate it
    var yAxis = svg.append("g")
        .attr("class", "yAxis")
        .call(yAxis);

    // Remove the line from the axis
    yAxis.select(".domain").remove();

    svg.selectAll(".bar")
        .data(listOfNamesAndPops)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", 0)
        .attr('rx', 10)
        .attr('ry', 10)
        .attr('stroke', 'black')
        .attr('stroke-width', 3)
        .attr("Name", function(d){return d[0];})
        .attr("Total_Population", function(d){return d[1];})
        .attr("height", function(d,i){
            return 40;})
        .attr("y", function(d) {return scaleY(d[0]) - 18; })
        .attr("width", function(d) {
            return scaleX(d[1]); })
        .on('mouseover', function(d,i) {

            toolTipChart(1200,200, dataForTooltip[i], extentTableForTooltip[i]);
        })
        .on('mouseout', function(d){
           d3.select('svg').select('#tempSVG').remove();
        })

};


MainChart(Country_Data, listOfCounties, margin, width, height);

function toolTipChart(moveX, moveY, tooltipData, tooltipExtent){
    var specifiedCountryDataForScatterPlot = [];
    var tipSVG = d3.select("svg")
        .append("svg")
        .attr('id', 'tempSVG')
        .attr("width", 900)
        .attr("height", 900)
        .attr('transform', 'translate(' + moveX + "," + moveY + ")");


    var toolTipWidth = 500;
    var tooltipHeight = 400;
    var tooltipPadding = 40;
    var tooltipMargin = {top: 20, right: 20, bottom: 20, left:20}
    var toolTipYears = [2013,2014,2015,2016,2017];

    var tooltipXScale = d3.scaleLinear()
        .domain(d3.extent(toolTipYears))
        .range([tooltipPadding, (toolTipWidth - tooltipPadding * 2)-20]);

    var test = d3.extent(tooltipExtent);
    var tooltipYScale = d3.scaleLinear()
        .domain(d3.extent(tooltipExtent))
        .range([(tooltipHeight - tooltipPadding - 20), tooltipPadding]);

    var line = d3.line()
        .x(function(d,i){
            return tooltipXScale(d[0])+40;
        })
        .y(function(d,i){
            return tooltipYScale(d[1]);
        });

    var tooltipXAxis = d3.axisBottom().scale(tooltipXScale).ticks(5).tickPadding(4);

    var tooltipYAxis = d3.axisLeft().scale(tooltipYScale).ticks(10);

    d3.select("svg").selectAll('#tempSVG').append('path')
        .datum(tooltipData)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 10)
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("d", line);

    d3.select("svg").selectAll('#tempSVG')
        .append("text")
        .attr('class', 'xAxis')
        .attr('transform', 'translate(445,405)')
        .text("Year");

    d3.select("svg").selectAll('#tempSVG')
        .append("text")
        .attr('class', 'yAxis')
        .attr('transform', 'translate(0,40)')
        .text("Pct %");

    tipSVG.append('g')
        .attr('class', 'tooltip X Axis')
        .attr('transform', 'translate(40,' + (tooltipHeight - tooltipPadding) + ")")
        .call(tooltipXAxis);

    tipSVG.append('g')
        .attr('class', 'tooltip y axis')
        .attr('transform', 'translate(' + (tooltipPadding + 10) + ", 10)")
        .call(tooltipYAxis)
        .append('text')
        .text('Y AXIS');

}


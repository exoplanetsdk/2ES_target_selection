// TODO urgent: HD 5 gibt nen Fehler weil G2/3V nicht verstanden wird!
// TODO: Add t_obs "min", "sec", "hour" switch!
// TODO: simbad query: changing SpRes leads to new simbad call!
// TODO urgent: Einzelne Variablen verÃ¤ndern und <Enter> geht. Mehrere verÃ¤ndern und Button drÃ¼cken -> Fehler! Rechne erst entsprechende
// Variablen um (Schritt-fÃ¼r-Schritt) und verwende dann die Werte fÃ¼r RV Precision.
function getManualValues() {
//
// returns Object with values in input fields on website
//
   return {
      "Vmag": [parseFloat(document.getElementById("vmag").value)],
      "Jmag": [parseFloat(document.getElementById("jmag").value)],
      "Temp": [parseFloat(document.getElementById("temp").value)],
      "SpRes": parseFloat(document.getElementById("SpRes").value),
      "SNRVmag":  parseFloat(document.getElementById("snrVmag").value),
      "SNRJmag":  parseFloat(document.getElementById("snrJmag").value),
      "MiDia": parseFloat(document.getElementById("MiDia").value),
      "TObs": parseFloat(document.getElementById("tObs").value),
   };
}

function toSNR() {
//
// return new snr from vmag
//
   var newValues = getManualValues();
   if (magNaN != "jmag" && magnitude == "vmag") {
      var mag = "Vmag";
   } else if (magNaN != "vmag" && magnitude == "jmag") {
      var mag = "Jmag";
   }
   // SNR = 100 * 10^[0.2(9mag-V)] * (MiDia/4m) * (TObs / 10 min)^0.5
   newValues["SNR"+mag] = standVals["snr"+mag]
      * (newValues["MiDia"]/standVals["miDia"])
      * Math.pow(10, 0.2 * (standVals[magnitude]-newValues[mag]))
      * Math.sqrt(newValues["TObs"]/standVals["tObs"]);
   Values = newValues;
   return Values["SNR"+mag];
}

function calcNewTObs() {
//
// return new observing time and update Values object
//
   var newValues = getManualValues();
   if (magnitude == "vmag") {
      var mag = "Vmag";
   } else {
      var mag = "Jmag";
   }
   // m = -2.5 log10(F);  F = 10^{-0.4m}
   // tobs = 10 min * (10^[-0.2(9mag-V)])^2 * (4m/MiDia)^2 * (SNR/100)^2
   newValues["TObs"] = standVals["tObs"]
                       * Math.pow(standVals["miDia"]/newValues["MiDia"], 2)
                       * Math.pow(10, -0.4 * (standVals[magnitude]-newValues[mag]))
                       * Math.pow(newValues["SNR"+mag]/standVals["snr"+mag], 2);
   Values = newValues;
   return Values["TObs"];
}

function updateSlave(writeToHtml) {
    if (document.querySelector(".master > input").name == 'snr') {
        queryVariable('TObs', writeToHtml)
    } else if (document.querySelector(".master > input").name == 'observingtime') {
        queryVariable('SNR', writeToHtml)
    }
}

function coloring(x) {
    x.style.backgroundColor = coloringColor;
    setTimeout(function() {
              x.style.backgroundColor = "unset";
    }, coloringTime); // reset color after 'coloringTime' seconds
}

function queryVariable(variable, writeToHtml) {
//
// check for new tObs or SNR and update input field with color
//
   if (variable == "SNR") {
      var newVar = toSNR();
      if (writeToHtml) {
         if (magNaN != "vmag" && magnitude == "jmag") {
            coloring(document.getElementById("inputTableSnrJmag"))
            document.getElementById("snrJmag").value = newVar.toFixed(2);
            document.getElementById("snrJmag").parentElement.style.display = ""
            document.getElementById("snrVmag").parentElement.style.display = "none"
         }
         else if (magNaN != "jmag" && magnitude == "vmag") {
            coloring(document.getElementById("inputTableSnrVmag"))
            document.getElementById("snrVmag").value = newVar.toFixed(2);
            document.getElementById("snrVmag").parentElement.style.display = ""
            document.getElementById("snrJmag").parentElement.style.display = "none"
         }
      }
   }
   // if snr and/or MiDia  where changed, recalculate tobs:
   else if (variable == "TObs") {
      var newVar = calcNewTObs();
      if (writeToHtml) {
         if (magNaN != "vmag" && magnitude == "jmag") {
            coloring(document.getElementById("tObs").parentElement);
            document.getElementById("tObs").value = newVar.toFixed(2);
         }
         else if (magNaN != "jmag" && magnitude == "vmag") {
            coloring(document.getElementById("tObs").parentElement);
            document.getElementById("tObs").value = newVar.toFixed(2);
         }
      }
   }
}

function PrepareDownload(values, fileEnding="") {
   const delim = ",";
   var output = "Id,Coordinates,Spectral Type,Temperature,Vmag,Jmag,";
   // add to header the wavelengths provided
   var i=0
   for (; i<Wavelengths["Lambda"].length; i++) {
      output += "RV_Prec("+Wavelengths["Lambda"][i]+" +- "+Wavelengths["Lambda_err"][i]+")";
      output += delim;
   }
   // add custom/full wavelength range
   RVwindow = [parseFloat(document.getElementById("Lam_min").value) || Wavelengths["Lambda"][0],
               parseFloat(document.getElementById("Lam_max").value) || Wavelengths["Lambda"].slice(-1)]
   output += "RV_Prec("+RVwindow[0]+" - "+RVwindow[1]+")" + "\n";

   for (let i=0; i < values["SpT"].length; i++) {
      output += values["Id"][i] + delim + values["Coord"][i] + delim;
      output += values["SpT"][i] + delim + values["Temp"][i] + delim;
      output += values["Vmag"][i] + delim + values["Jmag"][i] + delim;
      for (let j=0; j < values["RVPrecision"][i].length; j++) {
         output += values["RVPrecision"][i][j];
         output += delim;
      }
      output += GetCustomRVP(values, i);
      output += "\n";
   }
   return output
}

function SpectralToTemperature(classification, id, coord, SpT) {
   let simpleSpT = simplifySpT(SpT);
   for (let classirow of classification) {
      // check if this element from classification.csv is simbad's spectral type.
      // note: check for includes and not '==' in case we have e.g. A8I which from
      // classification.csv should be A8Ia0, A8Ia, A8Ib and thus includes A8I, too
      if (simplifySpT(classirow['Stellar Type']).includes(simpleSpT)) {
         var stellarTemp = parseFloat(classirow['Temp K']);
         return stellarTemp;
      }
      // if both of the upper cases did not apply:
      // Error Code -2: Spectral type not understood!
   }
   document.getElementById("simbad_err").innerHTML += "<br>Spectral Type of <br>"+id+" ("+coord+")<br>was not understood resp. found!";
   document.getElementById("simbad_err").innerHTML += "<br>Simbad returned as its Spectral Type "+SpT;
   throw new Error("ERROR in fetching spectral type from simbad!");
}

function GetWavelengths(allRows) {
   var lam_min = [];
   var lam_max = [];
   var Lams = [];
   var Lams_err = [];

   for (let i=0; i<allRows.length; i++) {
      var row = allRows[i];
      lam_min.push(parseFloat(row['Lam_min']));
      lam_max.push(parseFloat(row['Lam_max']));
      Lams.push((lam_min[i] + lam_max[i]) / 2.0);
      Lams_err.push(Lams[i] - lam_min[i]);
   }
   return {"Lambda":Lams, "Lambda_err": Lams_err};
}

function GetTemperatures(header) {
//
// Get temperatures from given header of file
//
   var temperatures = [];
   for (let i=0; i<header.length; i++) {
      temperatures.push(parseFloat(header[i]));
   }
   return temperatures;
}

function GetRVPrecision(allRows, temperatures) {
//
// Get RV Precision from file. Save as Object with Temps as keys
//
   var rVPrecision = {};
   for (let i=0; i < temperatures.length; i++) {
      rVPrecision[temperatures[i]] = [];
   }
   for (let i=0; i< allRows.length; i++) {
      let row = allRows[i];
      for (let j=0; j<temperatures.length; j++) {
         rVPrecision[temperatures[j]].push(parseFloat(row[temperatures[j]]));
      }
   }
   return rVPrecision;
}

//////////////////////////////
// Call GetOptional from here, too! ---> cleanest solution. apply to every possible function!
//////////////////////////////
function GetOptional(fromListener) {
   let lam_min_cut = parseFloat(document.getElementById("Lam_min").value);
   let lam_max_cut = parseFloat(document.getElementById("Lam_max").value);
   if (lam_min_cut > lam_max_cut) {
      document.getElementById("simbad_err").innerHTML = "Î»<sub>min</sub> bigger than Î»<sub>max</sub>!"
      // return "undefined", to be used in makeTrace for empty optional wavelength range
      return;
   }
   if (lam_min_cut == lam_max_cut) {
      document.getElementById("simbad_err").innerHTML = "Î»<sub>min</sub> and Î»<sub>max</sub> are equal but should not!"
      // return "undefined", to be used in makeTrace for empty optional wavelength range
      return;
   }
   // remove comments if the above if's did not happen!
   document.getElementById("simbad_err").innerHTML = "";
   optional = [lam_min_cut, lam_max_cut];
   if (fromListener && xvals.length != 0) {
      if (optional[0] < Math.min(...Wavelengths["Lambda"]))
          optional[0] = Math.min(...Wavelengths["Lambda"]);
      if (optional[1] > Math.max(...Wavelengths["Lambda"]))
          optional[1] = Math.max(...Wavelengths["Lambda"]);
      makeTrace(xvals, xerrs, yvals);
      updateTable(values, customRVPrecision);
   }
}

function GetCustomRVP(values, k=0) {
   var customRVP = [];
   let precisions = values["RVPrecision"][k];
   let wavelengths = Wavelengths["Lambda"];
   let RVwindow = optional || [NaN, NaN];

   if (isNaN(RVwindow[0]) || RVwindow[1]<Math.min(...wavelengths)) {
      RVwindow[0] = Math.min(...wavelengths)
   }
   if (isNaN(RVwindow[1]) || RVwindow[1]>Math.max(...wavelengths)) {
      RVwindow[1] = Math.max(...wavelengths)
   }

   for (let i=0; i < wavelengths.length; i++) {
      if (wavelengths[i] >= RVwindow[0] && wavelengths[i] <= RVwindow[1]) {
         customRVP.push(precisions[i]);
      }
   }
   if (customRVP.length == 0) {
      document.getElementById("simbad_err").innerHTML = "No value lays in selected wavelength range!";
      return;
   }

   let sum = 0;
   for (let x of customRVP) {
       if (!isNaN(x)) sum += 1. / (x*x);
   }
   console.log(1. / Math.sqrt(sum))
   return 1. / Math.sqrt(sum);
}

function rvPrecision(allRows, values, writeToHtml) {
//
// Get the RV Precision and prepare final dataset for plot/download
//
   if (magnitude == "jmag") {
      var mag = "Jmag";
   } else if (magnitude == "vmag") {
      var mag = "Vmag";
   }

   var header = Object.keys(allRows[0]);
   header.pop(); header.pop(); // pop the Lam_max and Lam_min entry in csv

   // Get Wavelengths as an object. Keys 'Lambda' and 'Lambda_err', global var
   Wavelengths = GetWavelengths(allRows);
   // Get Temperatures as a list, global var(s)
   Temperatures = GetTemperatures(header);
   Tmin = Math.min(...Temperatures);
   Tmax = Math.max(...Temperatures);
   // Get RV Precision as Object. Keys: Temperatures, global var
   RVPrecision = GetRVPrecision(allRows, Temperatures);
   // prepare array for RV precision entries
   //// das hier ist nur fuer vmag definiert. nochmal ueberpruefen?
   values["RVPrecision"] = new Array(values['Vmag'].length);
   // Normalize SNR and Spectral Resolution
   values["SNR"+mag]   = standVals["snr"+mag] / values["SNR"+mag];
   values["SpRes"] = Math.pow(standVals["spRes"] / values["SpRes"], 1.5);
   for (let i=0; i<values[mag].length; i++) {
      // interpolate values to given temperature if inside temp range. Check for and interpolate in case.
      let testingTheOutput = checkInput(RVPrecision, Temperatures, values["Temp"][i]);
      //values["RVPrecision"][i] = checkInput(RVPrecision, Temperatures, values["Temp"][i]);
      values["RVPrecision"][i] = testingTheOutput;

      // Check if Temperature is in desired range. else -> Temperature in range. else RV Precision eq. 0
      // TODO: Move following if condition to checkInput() function!
      if (values["RVPrecision"][i][0] == -1) {
         if (writeToHtml && values["Temp"][i] < Tmin) {
            document.getElementById("simbad_err").innerHTML += "<br>(*)temperature: "+values["Temp"][i]+" smaller than "+Tmin;
         }
         else if (!writeToHtml && values["Temp"][i] < Tmin) {
            document.getElementById("simbad_err").innerHTML += "<br>(*)"+values["Id"][i]+" ("+values["Coord"][i]+") ";
            document.getElementById("simbad_err").innerHTML += "temperature: "+values["Temp"][i]+" smaller than "+Tmin;
         }
         else if (writeToHtml && values["Temp"][i] > Tmax) {
            document.getElementById("simbad_err").innerHTML += "<br>(*)temperature: "+values["Temp"][i]+" bigger than "+Tmax;
         }
         else if (!writeToHtml && values["Temp"][i] > Tmax) {
            document.getElementById("simbad_err").innerHTML += "<br>(*)"+values["Id"][i]+" ("+values["Coord"][i]+") ";
            document.getElementById("simbad_err").innerHTML += "temperature: "+values["Temp"][i]+" bigger than "+Tmax;
         }
         continue;
      }
      // normalize with SNR and spectral Resolution for each RVPrecision(Temperature) value
      for (let j=0; j<values["RVPrecision"][i].length; j++) {
         values["RVPrecision"][i][j] *= values["SNR"+mag] * values["SpRes"];
      }
   }
   if (writeToHtml && values["RVPrecision"][0].includes(-1)) {
      // error posted above; we exit here to prevent further unwanted postprocessing
      return;
   }
   else {
      // if length == 1, thus i=0 only! and RV Precision not -1 (error as above), then:
      if (writeToHtml) {
         // firstly get optional param to get additional table element for custom wavelength range
         GetOptional(false);
         let customRVPrecision = GetCustomRVP(values);
         // firstly update table for resized height
         updateTable(values, customRVPrecision);
         // update global values. optional should come before xvals, etc.!
         xvals = Wavelengths["Lambda"];
         xerrs = Wavelengths["Lambda_err"];
         yvals = values['RVPrecision'][0];
         makeTrace(xvals, xerrs, yvals, customRVPrecision, ...optional);
      }
      // not erroneous temperature and more than one object
      else {
         var output = PrepareDownload(values, mag)
         download("output"+mag+".csv", output)
         csv_output.innerHTML = output
      }
   }
}

function analyseSimbadQuery(URLRows, classificationRows, writeToHtml) {
   // URLrows are rows from simbad query which is returned as csv file
   // classificationRows are rows from classification.csv file
   var header = Object.keys(URLRows[0]);

   var variables = {
      "SpT":   new Array(URLRows.length -1),
      "Vmag":  new Array(URLRows.length -1),
      "Jmag":  new Array(URLRows.length -1),
      "Id":    new Array(URLRows.length -1),
      "Coord": new Array(URLRows.length -1),
      "Temp":  new Array(URLRows.length -1),
      "SpRes": parseFloat(document.getElementById("SpRes").value),
      "SNRVmag":   parseFloat(document.getElementById("snrVmag").value),
      "SNRJmag":   parseFloat(document.getElementById("snrJmag").value),
      "MiDia": parseFloat(document.getElementById("MiDia").value),
      "TObs":  parseFloat(document.getElementById("tObs").value),
   };
   for (let i=0; i<URLRows.length; i++) {
      // TODO: Das writeToHTml aus SpectralToTemperature rausnehmen und hier hinmachen!
      // Dann den File output definieren!
      var row = URLRows[i];
      variables["SpT"][i] = (row["SpT"] || "").split(' ')[0];
      variables["Vmag"][i] = parseFloat(row["Vmag"]);
      variables["Jmag"][i] = parseFloat(row["Jmag"]);
      variables["Id"][i] = row["Name"];
      variables["Coord"][i] = row["Coo"];
      variables["Temp"][i] = row["Temperature"] || SpectralToTemperature(classificationRows, variables["Id"][i], variables["Coord"][i], variables["SpT"][i]);

      //if (magNaN != "vmag" && !isNaN(variables["Jmag"][i]) && (isNaN(variables["Vmag"][i]) || variables["Vmag"][i] >= variables["Jmag"][i])) {
      if (magNaN != "vmag" && !isNaN(variables["Jmag"][i]) && (isNaN(variables["Vmag"][i]) || variables["Temp"][i] <= 4000)) {
         magnitude = "jmag";
         if (writeToHtml) {
         document.getElementById("Magnitude-jband").selected = "true";
         document.getElementById("jmag").style.display = "";
         document.getElementById("vmag").style.display = "none";
         document.getElementById("snrJmag").parentElement.style.display = ""
         document.getElementById("snrVmag").parentElement.style.display = "none"
         if ('snr' == document.querySelector('.master > input').name) togglemaster(document.getElementById("snrJmag"))
         }
      } else { //if (magNaN != "jmag" && !isNaN(variables["Vmag"][i]) && (isNaN(variables["Jmag"][i]) || variables["Jmag"][i] > variables["Vmag"][i])) {
         magnitude = "vmag";
         if (writeToHtml) {
         document.getElementById("Magnitude-vband").selected = "true";
         // magNaN = "";
         document.getElementById("jmag").style.display = "none";
         document.getElementById("vmag").style.display = "";
         document.getElementById("snrVmag").parentElement.style.display = ""
         document.getElementById("snrJmag").parentElement.style.display = "none"
         if ('snr' == document.querySelector('.master > input').name) togglemaster(document.getElementById("snrVmag"))
         }
      }
   }

    nan2emptystr = x => isNaN(x) ? "" : x
    isnewtemp = document.getElementById("temp").value != nan2emptystr(variables["Temp"][0]);
    isnewvmag = document.getElementById("vmag").value != nan2emptystr(variables["Vmag"][0]);
    isnewjmag = document.getElementById("jmag").value != nan2emptystr(variables["Jmag"][0]);
    document.getElementById("temp").value = variables["Temp"][0];
    document.getElementById("vmag").value = variables["Vmag"][0];
    document.getElementById("jmag").value = variables["Jmag"][0];
    updateSlave(writeToHtml)

    if (isnewtemp) coloring(document.getElementById("temp").parentElement);
    if (isnewvmag) coloring(document.getElementById("vmag").parentElement);
    if (isnewjmag) coloring(document.getElementById("jmag").parentElement);

   magNaN = "";
   // load lookup tables RVP(lam,T) and compute RV precision
   if (magnitude == "vmag") {
      Plotly.d3.csv("dataVmag.csv", function(allRows) {
         rvPrecision(allRows, variables, writeToHtml);
      });
   } else if (magnitude == "jmag") {
      Plotly.d3.csv("dataJmag.csv", function(allRows) {
         rvPrecision(allRows, variables, writeToHtml);
      });
   }
}

function querySimbad(url, writeToHtml) {
   const baseURL = 'https://simbad.u-strasbg.fr/simbad/sim-script?script=';
   // output console=off script=off \n format object "%SP" \n query [... to be added]
   // output  = 'output console=off script=off error=off' //
   // error=off means: ignore objects that don't exist
   //const output = 'output%20console=off%20script=off%20error=off%0D';
   const output = 'output%20console=off%20script=off%20error=off%0D';
   // format  = 'format object "%SP\n%FLUXLIST(V;F)"'
   const format = 'format%20object%20%22%25IDLIST(1),%2527COO(A%20D),%25SP,%25FLUXLIST(V%3BF),%25FLUXLIST(J%3BF)%22';
   // produce a header for later csv analysis
   const header = 'echodata%20Name,Coo,SpT,Vmag,Jmag';
   const URL = [baseURL, output, format, header, url].join("%0A")
   console.log("Simbad Query: '", URL, "'");
   // reset the simbad error field
   document.getElementById("simbad_err").innerHTML = "";
   // simbad returns file in csv format. read the csv and analyse simbad (call function)
   Plotly.d3.csv(URL, function(URLRows) {
      URLRows.pop() // simbad appends an empty line
      // check if output is actually only fault. possible only if all inputs are invalid
      if (URLRows === undefined || URLRows.length == 0 || Object.keys(URLRows[0])[0].slice(2, 7) == "error") {
         document.getElementById("simbad_err").innerHTML = "Given object(s) not found by simbad!";
         return;
      }
      Plotly.d3.csv("classification.csv", function(classiRows) {
         analyseSimbadQuery(URLRows, classiRows, writeToHtml);
      });
   });
}


function queryBatch(csvfileobj) {
    Plotly.d3.csv(URL.createObjectURL(csvfileobj), function(URLRows) {
//        URLRows.pop() // d3.csv appends a newline
        var writeToHtml = URLRows.length < 2
        // check if output is actually only fault. possible only if all inputs are invalid
        if (URLRows === undefined || URLRows.length == 0 || Object.keys(URLRows[0])[0].slice(2, 7) == "error") {
            document.getElementById("simbad_err").innerHTML = "Given object(s) not found by simbad!"
            return
        }
        Plotly.d3.csv("classification.csv", function(classiRows) {
            analyseSimbadQuery(URLRows, classiRows, writeToHtml)
        })
    })
}


function fileUpload() {
   const file = document.getElementById("file").files[0];
   //const file = input.target.files[0];
   const reader = new FileReader();

   reader.onload = (event) => {
      const file = event.target.result;
      const allLines = file.split(/\r\n|\n/);
      // the variable writeToHtml decides, if we have multiple lines in the uploaded file
      // and thus decide if we want to change parameters in the html dom!
      var writeToHtml = allLines.length < 2;
      var querystr = allLines.join("%0A");
      // remove the %0A from the end to prevent an empty line and query simbad
      querySimbad(querystr, writeToHtml);
   };

   reader.onerror = (event) => {
      alert(event.target.error.name);
   };
   reader.readAsText(file);
}

function keyDown(event) {
   if (event.key === 'Enter') {
      // Cancel the default action, if needed
      event.preventDefault();
      if (["snrVmag", "snrJmag", "MiDia", "vmag", "jmag", "tObs"].includes(event.target.id)) {
          //updateSlave(true)
      }

      // mimic submit click
      queryButton.style.backgroundColor = '#666'
      setTimeout(function() {
          queryButton.style.backgroundColor = "";
      }, 100);

      inputFunction();
   }
}

function simbadFieldInput() {
   var co_query = document.getElementById("co_query");
   var id_query = document.getElementById("id_query");
   var query = "";
   if (id_query.checked) {
      // query = 'query id '
      query += 'query%20id%20';
   } else if (co_query.checked) {
      // query = 'query id '
      query += 'query%20coo%20';
   } else {
      document.getElementById("simbad_err").innerHTML = "Neither 'identifier' nor 'coordinates' selected! Select one and repeat the query!";
      return;
   }
   document.getElementById("simbad_err").innerHTML = "";
   let simbad = document.getElementById("simbadField").value; // content to search for
   query += simbad;
   // call function with input and writeToHtml = true (only one element)
   querySimbad(query, true);
}

function whichMagBand() {
//
// check which selection was set for the magnitude band
//
   var Selection = document.getElementById("Magnitude");
   var Vselect = document.getElementById("vmag");
   var Jselect = document.getElementById("jmag");
   var SNRVselect = document.getElementById("inputTableSnrVmag");
   var SNRJselect = document.getElementById("inputTableSnrJmag");

   updateSlave(true)

   if (Selection[0].selected) {
      Vselect.style.display = "";
      SNRVselect.style.display = "";
      Jselect.style.display = "none";
      SNRJselect.style.display = "none";
      magnitude = "vmag";
      if (xvals.length > 0) {
         magNaN = "vmag";
         inputFunction();
      }
   } else if (Selection[1].selected) {
      Vselect.style.display = "none";
      SNRVselect.style.display = "none";
      Jselect.style.display = "";
      SNRJselect.style.display = "";
      magnitude = "jmag";
      if (xvals.length > 0) {
         magNaN = "jmag";
         inputFunction();
      }
   }
}

function whichRadio() {
//
// check which radio button was selected in 'Query selection'
//
   var fileInput = document.getElementById("fileInput");
   var manually = document.getElementById("setManually");
   var idQuery = document.getElementById("id_query");
   var coQuery = document.getElementById("co_query");

   document.getElementById("temp").readOnly = !manually.checked;
   document.getElementById("vmag").readOnly = !manually.checked;
   document.getElementById("jmag").readOnly = !manually.checked;
   document.getElementById("simbadField").readOnly = !(idQuery.checked || coQuery.checked);
   simbadField.placeholder = coQuery.checked? "Simbad coords" : "Simbad resolver";

   if (fileInput.checked) {
      inputSelection = "fileInput";
      // remove j and v mag independent of which is selected!
      if (magNaN != "vmag" && magnitude=="jmag") {
      } else if (magNaN != "jmag" && magnitude=="vmag") {
      }
   }
   else if (fileParam.checked) {
      inputSelection = "fileParam"
   }
   else if (idQuery.checked || coQuery.checked) {
      inputSelection = "idcoQuery";
      // switch from input to show field
      if (magNaN != "vmag" && magnitude=="jmag") {
         document.getElementById("jmag").style="visibility:visible";
      } else if (magNaN != "jmag" && magnitude=="vmag") {
         document.getElementById("vmag").style="visibility:visible";
      }
   }
   else if (manually.checked) {
      inputSelection = "manual";
      //document.getElementById("inputTableTemp").style="visibility:visible";
      if (magNaN != "jmag" && magnitude=="vmag") {
         document.getElementById("vmag").style.display = "";
      } else if (magNaN != "vmag" && magnitude=="jmag") {
         document.getElementById("jmag").style.display = "";
      }
   }
}

function inputFunction() {
    updateSlave(true)
    csv_output.innerHTML = ""

    document.getElementById("simbad_err").innerHTML = "";
    if (inputSelection == "") {
        document.getElementById("simbad_err").innerHTML = "No type of query selected! Select one from above and repeat the query!";
    } else if (inputSelection == "idcoQuery") {
        simbadFieldInput();
    } else if (inputSelection == "fileInput") {
        fileUpload();
        coloring(outputfile);
    } else if (inputSelection == "fileParam") {
        queryBatch(document.getElementById("file").files[0])
        coloring(outputfile);
    } else if (inputSelection == "manual") {
        var variables = getManualValues();
        if (magnitude == "jmag") {
            Plotly.d3.csv("dataJmag.csv", function(loadedRows) {
                let allRows = loadedRows;
                rvPrecision(allRows, variables, true);
            });
        } else {
            Plotly.d3.csv("dataVmag.csv", function(loadedRows) {
                let allRows = loadedRows;
                rvPrecision(allRows, variables, true);
            });
        }
    }
}

function togglemaster(elem) {
    var x = document.querySelector('.master > input')
    if (x) x.parentElement.classList.remove('master');
    (elem || event.target).parentElement.classList.add('master');
}

//// check for mouse clicks, focus out events and enter key events ////


//////////////////// Magnitude selection event listeners ////////////////////
document.getElementById("Magnitude").addEventListener("change", whichMagBand);


//////////////////// radio button event listeners ////////////////////
document.getElementById("fileInput").addEventListener("change", whichRadio);
document.getElementById("fileParam").addEventListener("change", whichRadio);
document.getElementById("id_query").addEventListener("change", whichRadio);
document.getElementById("co_query").addEventListener("change", whichRadio);
document.getElementById("setManually").addEventListener("change", whichRadio);

//////////////////// value event listeners ////////////////////
// TODO: In Simbad abfrage den Values dict Ã¼berschreiben
document.getElementById("temp").addEventListener("keydown", keyDown);
document.getElementById("vmag").addEventListener("keydown", keyDown);
document.getElementById("jmag").addEventListener("keydown", keyDown);
document.getElementById("MiDia").addEventListener("keydown", keyDown);
document.getElementById("SpRes").addEventListener("keydown", keyDown);
document.getElementById("snrVmag").addEventListener("keydown", keyDown);
document.getElementById("snrJmag").addEventListener("keydown", keyDown);
document.getElementById("tObs").addEventListener("keydown", keyDown);
document.getElementById("Lam_min").addEventListener("keydown", keyDown);
document.getElementById("Lam_max").addEventListener("keydown", keyDown);

//////////////////// button event listeners ////////////////////
document.getElementById("queryButton").addEventListener("click", inputFunction);
document.getElementById("simbadField").addEventListener("keydown", keyDown);

// onload plot the default values
whichRadio()
inputFunction()

